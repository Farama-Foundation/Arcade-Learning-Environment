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

#ifndef ENVPOOL_CORE_ENVPOOL_H_
#define ENVPOOL_CORE_ENVPOOL_H_

#include <utility>
#include <vector>

#include "ale/vector/env_spec.h"

/**
 * Templated subclass of EnvPool, to be overrided by the real EnvPool.
 */
template <typename EnvSpec>
class EnvPool {
 public:
  EnvSpec spec;
  using Spec = EnvSpec;
  using State = NamedVector<typename EnvSpec::StateKeys, std::vector<Array>>;
  using Action = NamedVector<typename EnvSpec::ActionKeys, std::vector<Array>>;
  explicit EnvPool(EnvSpec spec) : spec(std::move(spec)) {}
  virtual ~EnvPool() = default;

 protected:
  virtual void Send(const std::vector<Array>& action) {
    throw std::runtime_error("send not implemented");
  }
  virtual void Send(std::vector<Array>&& action) {
    throw std::runtime_error("send not implemented");
  }
  virtual std::vector<Array> Recv() {
    throw std::runtime_error("recv not implemented");
  }
  virtual void Reset(const Array& env_ids) {
    throw std::runtime_error("reset not implemented");
  }
};

#endif  // ENVPOOL_CORE_ENVPOOL_H_
