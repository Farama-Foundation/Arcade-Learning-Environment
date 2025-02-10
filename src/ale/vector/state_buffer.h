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

#ifndef ENVPOOL_CORE_STATE_BUFFER_H_
#define ENVPOOL_CORE_STATE_BUFFER_H_

#ifndef MOODYCAMEL_DELETE_FUNCTION
#define MOODYCAMEL_DELETE_FUNCTION = delete
#endif

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <functional>
#include <utility>
#include <vector>

#include "ale/vector/array.h"
#include "ale/vector/dict.h"
#include "ale/vector/spec.h"
#include "lightweightsemaphore.h"

/**
 * Buffer of a batch of states, which is used as an intermediate storage device
 * for the environments to write their state outputs of each step.
 * There's a quota for how many envs' results are stored in this buffer,
 * which is controlled by the batch arguments in the constructor.
 */
class StateBuffer {
 protected:
  std::size_t batch_;
  std::size_t max_num_players_;
  std::vector<Array> arrays_;
  std::vector<bool> is_player_state_;
  std::atomic<uint64_t> offsets_{0};
  std::atomic<std::size_t> alloc_count_{0};
  std::atomic<std::size_t> done_count_{0};
  moodycamel::LightweightSemaphore sem_;

 public:
  /**
   * Return type of StateBuffer.Allocate is a slice of each state arrays that
   * can be written by the caller. When writing is done, the caller should
   * invoke done write.
   */
  struct WritableSlice {
    std::vector<Array> arr;
    std::function<void()> done_write;
  };

  /**
   * Create a StateBuffer instance with the player_specs and shared_specs
   * provided.
   */
  StateBuffer(std::size_t batch, std::size_t max_num_players,
              const std::vector<ShapeSpec>& specs,
              std::vector<bool> is_player_state)
      : batch_(batch),
        max_num_players_(max_num_players),
        arrays_(MakeArray(specs)),
        is_player_state_(std::move(is_player_state)) {}

  /**
   * Tries to allocate a piece of memory without lock.
   * If this buffer runs out of quota, an out_of_range exception is thrown.
   * Externally, caller has to catch the exception and handle accordingly.
   */
  WritableSlice Allocate(std::size_t num_players, int order = -1) {
    DCHECK_LE(num_players, max_num_players_);
    std::size_t alloc_count = alloc_count_.fetch_add(1);
    if (alloc_count < batch_) {
      // Make a increment atomically on two uint32_t simultaneously
      // This avoids lock
      uint64_t increment = static_cast<uint64_t>(num_players) << 32 | 1;
      uint64_t offsets = offsets_.fetch_add(increment);
      uint32_t player_offset = offsets >> 32;
      uint32_t shared_offset = offsets;
      DCHECK_LE((std::size_t)shared_offset + 1, batch_);
      DCHECK_LE((std::size_t)(player_offset + num_players),
                batch_ * max_num_players_);
      if (order != -1 && max_num_players_ == 1) {
        // single player with sync setting: return ordered data
        player_offset = shared_offset = order;
      }
      std::vector<Array> state;
      state.reserve(arrays_.size());
      for (std::size_t i = 0; i < arrays_.size(); ++i) {
        const Array& a = arrays_[i];
        if (is_player_state_[i]) {
          state.emplace_back(
              a.Slice(player_offset, player_offset + num_players));
        } else {
          state.emplace_back(a[shared_offset]);
        }
      }
      return WritableSlice{.arr = std::move(state),
                           .done_write = [this]() { Done(); }};
    }
    DLOG(INFO) << "Allocation failed, continue to the next block of memory";
    throw std::out_of_range("StateBuffer out of storage");
  }

  [[nodiscard]] std::pair<uint32_t, uint32_t> Offsets() const {
    uint32_t player_offset = offsets_ >> 32;
    uint32_t shared_offset = offsets_;
    return {player_offset, shared_offset};
  }

  /**
   * When the allocated memory has been filled, the user of the memory will
   * call this callback to notify StateBuffer that its part has been written.
   */
  void Done(std::size_t num = 1) {
    std::size_t done_count = done_count_.fetch_add(num);
    if (done_count + num == batch_) {
      sem_.signal();
    }
  }

  /**
   * Blocks until the entire buffer is ready, aka, all quota has been
   * distributed out, and all user has called done.
   */
  std::vector<Array> Wait(std::size_t additional_done_count = 0) {
    if (additional_done_count > 0) {
      Done(additional_done_count);
    }
    while (!sem_.wait()) {
    }
    // when things are all done, compact the buffer.
    uint64_t offsets = offsets_;
    uint32_t player_offset = (offsets >> 32);
    uint32_t shared_offset = offsets;
    DCHECK_EQ((std::size_t)shared_offset, batch_ - additional_done_count);
    std::vector<Array> ret;
    ret.reserve(arrays_.size());
    for (std::size_t i = 0; i < arrays_.size(); ++i) {
      const Array& a = arrays_[i];
      if (is_player_state_[i]) {
        ret.emplace_back(a.Truncate(player_offset));
      } else {
        ret.emplace_back(a.Truncate(shared_offset));
      }
    }
    return ret;
  }
};

#endif  // ENVPOOL_CORE_STATE_BUFFER_H_
