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

#ifndef ENVPOOL_CORE_ENV_H_
#define ENVPOOL_CORE_ENV_H_

#include <memory>
#include <random>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "ale/vector/env_spec.h"
#include "ale/vector/state_buffer_queue.h"

template <typename Dtype>
struct InitializeHelper {
  static void Init(Array* arr) {}
};

template <typename Dtype>
struct InitializeHelper<Container<Dtype>> {
  static void Init(Array* arr) {
    auto* carr = reinterpret_cast<Container<Dtype>*>(arr->Data());
    for (std::size_t i = 0; i < arr->size; ++i) {
      new (carr + i) Container<Dtype>(nullptr);
    }
  }
};

template <typename Spec>
void InplaceInitialize(const Spec& spec, Array* arr) {
  InitializeHelper<typename Spec::dtype>::Init(arr);
}

template <typename SpecTuple>
struct SpecToTArray;

template <typename... Args>
struct SpecToTArray<std::tuple<Args...>> {
  using Type = std::tuple<TArray<typename Args::dtype>...>;
};

/**
 * Single RL environment abstraction.
 */
template <typename EnvSpec>
class Env {
 protected:
  int max_num_players_;
  EnvSpec spec_;
  int env_id_, seed_;
  std::mt19937 gen_;

 private:
  StateBufferQueue* sbq_;
  int order_, current_step_{-1};
  bool is_single_player_;
  StateBuffer::WritableSlice slice_;
  // for parsing single env action from input action batch
  std::vector<ShapeSpec> action_specs_;
  std::vector<bool> is_player_action_;
  std::shared_ptr<std::vector<Array>> action_batch_;
  std::vector<Array> raw_action_;
  int env_index_;

 public:
  using Spec = EnvSpec;
  using State =
      Dict<typename EnvSpec::StateKeys,
           typename SpecToTArray<typename EnvSpec::StateSpec::Values>::Type>;
  using Action =
      Dict<typename EnvSpec::ActionKeys,
           typename SpecToTArray<typename EnvSpec::ActionSpec::Values>::Type>;

  Env(const EnvSpec& spec, int env_id)
      : max_num_players_(spec.config["max_num_players"_]),
        spec_(spec),
        env_id_(env_id),
        seed_(spec.config["seed"_] + env_id),
        gen_(seed_),
        is_single_player_(max_num_players_ == 1),
        action_specs_(spec.action_spec.template AllValues<ShapeSpec>()),
        is_player_action_(Transform(action_specs_, [](const ShapeSpec& s) {
          return (!s.shape.empty() && s.shape[0] == -1);
        })) {
    slice_.done_write = [] { LOG(INFO) << "Use `Allocate` to write state."; };
  }

  virtual ~Env() = default;

  void SetAction(std::shared_ptr<std::vector<Array>> action_batch,
                 int env_index) {
    action_batch_ = std::move(action_batch);
    env_index_ = env_index;
  }

  void ParseAction() {
    raw_action_.clear();
    std::size_t action_size = action_batch_->size();
    if (is_single_player_) {
      for (std::size_t i = 0; i < action_size; ++i) {
        if (is_player_action_[i]) {
          raw_action_.emplace_back(
              (*action_batch_)[i].Slice(env_index_, env_index_ + 1));
        } else {
          raw_action_.emplace_back((*action_batch_)[i][env_index_]);
        }
      }
    } else {
      std::vector<int> env_player_index;
      int* player_env_id = static_cast<int*>((*action_batch_)[1].Data());
      int player_offset = (*action_batch_)[1].Shape(0);
      for (int i = 0; i < player_offset; ++i) {
        if (player_env_id[i] == env_id_) {
          env_player_index.push_back(i);
        }
      }
      int player_num = env_player_index.size();
      bool continuous = false;
      int start = 0;
      int end = 0;
      if (player_num > 0) {
        start = env_player_index[0];
        end = env_player_index[player_num - 1] + 1;
        continuous = (player_num == end - start);
      }
      for (std::size_t i = 0; i < action_size; ++i) {
        if (is_player_action_[i]) {
          if (continuous) {
            raw_action_.emplace_back((*action_batch_)[i].Slice(start, end));
          } else {
            action_specs_[i].shape[0] = player_num;
            Array arr(action_specs_[i]);
            for (int j = 0; j < player_num; ++j) {
              int player_index = env_player_index[j];
              arr[j].Assign((*action_batch_)[i][player_index]);
            }
            raw_action_.emplace_back(std::move(arr));
          }
        } else {
          raw_action_.emplace_back((*action_batch_)[i][env_index_]);
        }
      }
    }
  }

  void EnvStep(StateBufferQueue* sbq, int order, bool reset) {
    PreProcess(sbq, order, reset);
    if (reset) {
      Reset();
    } else {
      ParseAction();
      Step(Action(std::move(raw_action_)));
      raw_action_.clear();
    }
    PostProcess();
  }

  virtual void Reset() { throw std::runtime_error("reset not implemented"); }
  virtual void Step(const Action& action) {
    throw std::runtime_error("step not implemented");
  }
  virtual bool IsDone() { throw std::runtime_error("is_done not implemented"); }

 protected:
  void PreProcess(StateBufferQueue* sbq, int order, bool reset) {
    sbq_ = sbq;
    order_ = order;
    if (reset) {
      current_step_ = 0;
    } else {
      ++current_step_;
    }
  }

  void PostProcess() {
    slice_.done_write();
    // action_batch_.reset();
  }

  State Allocate(int player_num = 1) {
    slice_ = sbq_->Allocate(player_num, order_);
    State state(slice_.arr);
    bool done = IsDone();
    int max_episode_steps = spec_.config["max_episode_steps"_];
    state["done"_] = done;
    state["discount"_] = static_cast<float>(!done);
    // dm_env.StepType.FIRST == 0
    // dm_env.StepType.MID == 1
    // dm_env.StepType.LAST == 2
    state["step_type"_] = current_step_ == 0 ? 0 : done ? 2 : 1;
    state["trunc"_] = done && (current_step_ >= max_episode_steps);
    state["info:env_id"_] = env_id_;
    state["elapsed_step"_] = current_step_;
    int* player_env_id(static_cast<int*>(state["info:players.env_id"_].Data()));
    for (int i = 0; i < player_num; ++i) {
      player_env_id[i] = env_id_;
    }
    // Inplace initialize all container fields
    int i = 0;
    std::apply(
        [&](auto&&... spec) {
          (InplaceInitialize(spec, &slice_.arr[i++]), ...);
        },
        spec_.state_spec.AllValues());
    return state;
  }
};

#endif  // ENVPOOL_CORE_ENV_H_
