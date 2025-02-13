/*
 * Copyright 2021 Garena Online Private Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ENVPOOL_CORE_ASYNC_ENVPOOL_H_
#define ENVPOOL_CORE_ASYNC_ENVPOOL_H_

#include <algorithm>
#include <atomic>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "ThreadPool.h"
#include "ale/vector/action_buffer_queue.h"
#include "ale/vector/state_buffer_queue.h"
#include "ale/vector/array.h"
#include "ale/vector/envpool.h"
#include "ale/vector/spec.h"

/**
 * Async EnvPool
 *
 * batch-action -> action buffer queue -> threadpool -> state buffer queue
 *
 * ThreadPool is tailored with EnvPool, so here we don't use the existing
 * third_party ThreadPool (which is really slow).
 */
template <typename Env>
class AsyncEnvPool : public EnvPool<typename Env::Spec> {
 protected:
  std::size_t num_envs_;
  std::size_t batch_;
  std::size_t max_num_players_;
  std::size_t num_threads_;
  bool is_sync_;
  std::atomic<int> stop_;
  std::atomic<std::size_t> stepping_env_num_;
  std::vector<std::thread> workers_;
  std::unique_ptr<ActionBufferQueue> action_buffer_queue_;
  std::unique_ptr<StateBufferQueue> state_buffer_queue_;
  std::vector<std::unique_ptr<Env>> envs_;
  std::vector<std::atomic<int>> stepping_env_;
  std::chrono::duration<double> dur_send_, dur_recv_, dur_send_all_;

  template <typename V>
  void SendImpl(V&& action) {
    int* env_id = static_cast<int*>(action[0].Data());
    int shared_offset = action[0].Shape(0);
    std::vector<ActionSlice> actions;
    std::shared_ptr<std::vector<Array>> action_batch =
        std::make_shared<std::vector<Array>>(std::forward<V>(action));
    for (int i = 0; i < shared_offset; ++i) {
      int eid = env_id[i];
      envs_[eid]->SetAction(action_batch, i);
      actions.emplace_back(ActionSlice{
          .env_id = eid,
          .order = is_sync_ ? i : -1,
          .force_reset = false,
      });
    }
    if (is_sync_) {
      stepping_env_num_ += shared_offset;
    }
    // add to abq
    auto start = std::chrono::system_clock::now();
    action_buffer_queue_->EnqueueBulk(actions);
    dur_send_ += std::chrono::system_clock::now() - start;
  }

 public:
  using Spec = typename Env::Spec;
  using Action = typename Env::Action;
  using State = typename Env::State;
  using ActionSlice = typename ActionBufferQueue::ActionSlice;

  explicit AsyncEnvPool(const Spec& spec)
      : EnvPool<Spec>(spec),
        num_envs_(spec.config["num_envs"_]),
        batch_(spec.config["batch_size"_] <= 0 ? num_envs_
                                               : spec.config["batch_size"_]),
        max_num_players_(spec.config["max_num_players"_]),
        num_threads_(spec.config["num_threads"_]),
        is_sync_(batch_ == num_envs_ && max_num_players_ == 1),
        stop_(0),
        stepping_env_num_(0),
        action_buffer_queue_(new ActionBufferQueue(num_envs_)),
        state_buffer_queue_(new StateBufferQueue(
            batch_, num_envs_, max_num_players_,
            spec.state_spec.template AllValues<ShapeSpec>())),
        envs_(num_envs_) {
    std::size_t processor_count = std::thread::hardware_concurrency();
    ThreadPool init_pool(std::min(processor_count, num_envs_));
    std::vector<std::future<void>> result;
    for (std::size_t i = 0; i < num_envs_; ++i) {
      result.emplace_back(init_pool.enqueue(
          [i, spec, this] { envs_[i].reset(new Env(spec, i)); }));
    }
    for (auto& f : result) {
      f.get();
    }
    if (num_threads_ == 0) {
      num_threads_ = std::min(batch_, processor_count);
    }
    for (std::size_t i = 0; i < num_threads_; ++i) {
      workers_.emplace_back([this] {
        for (;;) {
          ActionSlice raw_action = action_buffer_queue_->Dequeue();
          if (stop_ == 1) {
            break;
          }
          int env_id = raw_action.env_id;
          int order = raw_action.order;
          bool reset = raw_action.force_reset || envs_[env_id]->IsDone();
          envs_[env_id]->EnvStep(state_buffer_queue_.get(), order, reset);
        }
      });
    }
    if (spec.config["thread_affinity_offset"_] >= 0) {
      std::size_t thread_affinity_offset =
          spec.config["thread_affinity_offset"_];
      for (std::size_t tid = 0; tid < num_threads_; ++tid) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        std::size_t cid = (thread_affinity_offset + tid) % processor_count;
        CPU_SET(cid, &cpuset);
        pthread_setaffinity_np(workers_[tid].native_handle(), sizeof(cpu_set_t),
                               &cpuset);
      }
    }
  }

  ~AsyncEnvPool() override {
    stop_ = 1;
    // LOG(INFO) << "envpool send: " << dur_send_.count();
    // LOG(INFO) << "envpool recv: " << dur_recv_.count();
    // send n actions to clear threadpool
    std::vector<ActionSlice> empty_actions(workers_.size());
    action_buffer_queue_->EnqueueBulk(empty_actions);
    for (auto& worker : workers_) {
      worker.join();
    }
  }

  void Send(const Action& action) {
    SendImpl(action.template AllValues<Array>());
  }
  void Send(const std::vector<Array>& action) override { SendImpl(action); }
  void Send(std::vector<Array>&& action) override { SendImpl(action); }

  std::vector<Array> Recv() override {
    int additional_wait = 0;
    if (is_sync_ && stepping_env_num_ < batch_) {
      additional_wait = batch_ - stepping_env_num_;
    }
    auto start = std::chrono::system_clock::now();
    auto ret = state_buffer_queue_->Wait(additional_wait);
    dur_recv_ += std::chrono::system_clock::now() - start;
    if (is_sync_) {
      stepping_env_num_ -= ret[0].Shape(0);
    }
    return ret;
  }

  void Reset(const Array& env_ids) override {
    TArray<int> tenv_ids(env_ids);
    int shared_offset = tenv_ids.Shape(0);
    std::vector<ActionSlice> actions(shared_offset);
    for (int i = 0; i < shared_offset; ++i) {
      actions[i].force_reset = true;
      actions[i].env_id = tenv_ids[i];
      actions[i].order = is_sync_ ? i : -1;
    }
    if (is_sync_) {
      stepping_env_num_ += shared_offset;
    }
    action_buffer_queue_->EnqueueBulk(actions);
  }
};

#endif  // ENVPOOL_CORE_ASYNC_ENVPOOL_H_
