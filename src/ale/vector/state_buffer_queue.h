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

#ifndef ENVPOOL_CORE_STATE_BUFFER_QUEUE_H_
#define ENVPOOL_CORE_STATE_BUFFER_QUEUE_H_

#include <algorithm>
#include <cstdint>
#include <list>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "array.h"
#include "circular_buffer.h"
#include "spec.h"
#include "state_buffer.h"
#include "lightweightsemaphore.h"

class StateBufferQueue {
 protected:
  std::size_t batch_;
  std::size_t max_num_players_;
  std::vector<bool> is_player_state_;
  std::vector<ShapeSpec> specs_;
  std::size_t queue_size_;
  std::vector<std::unique_ptr<StateBuffer>> queue_;
  std::atomic<uint64_t> alloc_count_, done_ptr_, alloc_tail_;

  // Create stock statebuffers in a background thread
  CircularBuffer<std::unique_ptr<StateBuffer>> stock_buffer_;
  std::vector<std::thread> create_buffer_thread_;
  std::atomic<bool> quit_;

 public:
  StateBufferQueue(std::size_t batch_env, std::size_t num_envs,
                   std::size_t max_num_players,
                   const std::vector<ShapeSpec>& specs)
      : batch_(batch_env),
        max_num_players_(max_num_players),
        is_player_state_(Transform(specs,
                                   [](const ShapeSpec& s) {
                                     return (!s.shape.empty() &&
                                             s.shape[0] == -1);
                                   })),
        specs_(Transform(specs,
                         [=](ShapeSpec s) {
                           if (!s.shape.empty() && s.shape[0] == -1) {
                             // If first dim is num_players
                             s.shape[0] = batch_ * max_num_players_;
                             return s;
                           }
                           return s.Batch(batch_);
                         })),
        // two times enough buffer for all the envs
        queue_size_((num_envs / batch_env + 2) * 2),
        queue_(queue_size_),  // circular buffer
        alloc_count_(0),
        done_ptr_(0),
        stock_buffer_((num_envs / batch_env + 2) * 2),
        quit_(false) {
    // Only initialize first half of the buffer
    // At the consumption of each block, the first consumping thread
    // will allocate a new state buffer and append to the tail.
    // alloc_tail_ = num_envs / batch_env + 2;
    for (auto& q : queue_) {
      q = std::make_unique<StateBuffer>(batch_, max_num_players_, specs_,
                                        is_player_state_);
    }
    std::size_t processor_count = std::thread::hardware_concurrency();
    // hardcode here :(
    std::size_t create_buffer_thread_num = std::max(1UL, processor_count / 64);
    for (std::size_t i = 0; i < create_buffer_thread_num; ++i) {
      create_buffer_thread_.emplace_back(std::thread([&]() {
        while (true) {
          stock_buffer_.Put(std::make_unique<StateBuffer>(
              batch_, max_num_players_, specs_, is_player_state_));
          if (quit_) {
            break;
          }
        }
      }));
    }
  }

  ~StateBufferQueue() {
    // stop the thread
    quit_ = true;
    for (std::size_t i = 0; i < create_buffer_thread_.size(); ++i) {
      stock_buffer_.Get();
    }
    for (auto& t : create_buffer_thread_) {
      t.join();
    }
  }

  /**
   * Allocate slice of memory for the current env to write.
   * This function is used from the producer side.
   * It is safe to access from multiple threads.
   */
  StateBuffer::WritableSlice Allocate(std::size_t num_players, int order = -1) {
    std::size_t pos = alloc_count_.fetch_add(1);
    std::size_t offset = (pos / batch_) % queue_size_;
    // if (pos % batch_ == 0) {
    //   // At the time a new statebuffer is accessed, the first visitor
    //   allocate
    //   // a new state buffer and put it at the back of the queue.
    //   std::size_t insert_pos = alloc_tail_.fetch_add(1);
    //   std::size_t insert_offset = insert_pos % queue_size_;
    //   queue_[insert_offset].reset(
    //       new StateBuffer(batch_, max_num_players_, specs_,
    //       is_player_state_));
    // }
    return queue_[offset]->Allocate(num_players, order);
  }

  /**
   * Wait for the state buffer at the head to be ready.
   * This function can only be accessed from one thread.
   *
   * BIG CAVEATE:
   * Wait should be accessed from only one thread.
   * If Wait is accessed from multiple threads, it is only safe if the finish
   * time of each state buffer is in the same order as the allocation time.
   */
  std::vector<Array> Wait(std::size_t additional_done_count = 0) {
    std::unique_ptr<StateBuffer> newbuf = stock_buffer_.Get();
    std::size_t pos = done_ptr_.fetch_add(1);
    std::size_t offset = pos % queue_size_;
    auto arr = queue_[offset]->Wait(additional_done_count);
    if (additional_done_count > 0) {
      // move pointer to the next block
      alloc_count_.fetch_add(additional_done_count);
    }
    std::swap(queue_[offset], newbuf);
    return arr;
  }
};

#endif  // ENVPOOL_CORE_STATE_BUFFER_QUEUE_H_
