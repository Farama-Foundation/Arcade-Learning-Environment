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

#ifndef ENVPOOL_CORE_CIRCULAR_BUFFER_H_
#define ENVPOOL_CORE_CIRCULAR_BUFFER_H_

#ifndef MOODYCAMEL_DELETE_FUNCTION
#define MOODYCAMEL_DELETE_FUNCTION = delete
#endif

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include "lightweightsemaphore.h"

template <typename V>
class CircularBuffer {
 protected:
  std::size_t size_;
  moodycamel::LightweightSemaphore sem_get_;
  moodycamel::LightweightSemaphore sem_put_;
  std::vector<V> buffer_;
  std::atomic<uint64_t> head_;
  std::atomic<uint64_t> tail_;

 public:
  explicit CircularBuffer(std::size_t size)
      : size_(size), sem_put_(size), buffer_(size), head_(0), tail_(0) {}

  template <typename T>
  void Put(T&& v) {
    while (!sem_put_.wait()) {
    }
    uint64_t tail = tail_.fetch_add(1);
    auto offset = tail % size_;
    buffer_[offset] = std::forward<T>(v);
    sem_get_.signal();
  }

  V Get() {
    while (!sem_get_.wait()) {
    }
    uint64_t head = head_.fetch_add(1);
    auto offset = head % size_;
    V v = std::move(buffer_[offset]);
    sem_put_.signal();
    return v;
  }
};

#endif  // ENVPOOL_CORE_CIRCULAR_BUFFER_H_
