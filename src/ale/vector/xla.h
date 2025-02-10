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

#ifndef ENVPOOL_CORE_XLA_H_
#define ENVPOOL_CORE_XLA_H_

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

#include "ale/vector/array.h"
#include "ale/vector/xla_template.h"

template <typename D>
constexpr bool is_container_v = false;  // NOLINT
template <typename D>
constexpr bool is_container_v<Container<D>> = true;  // NOLINT
template <typename... T>
constexpr bool HasContainerType(std::tuple<T...> /*unused*/) {
  return (is_container_v<typename T::dtype> || ...);
}
bool HasDynamicDim(const std::vector<int>& shape) {
  return std::any_of(shape.begin() + 1, shape.end(),
                     [](int s) { return s == -1; });
}
template <typename... T>
bool HasDynamicDim(const std::tuple<T...>& state_spec) {
  bool dyn = false;
  std::apply([&](auto&&... spec) { dyn = (HasDynamicDim(spec.shape) || ...); },
             state_spec);
  return dyn;
}

template <typename Dtype>
Array CpuBufferToArray(const void* buffer, ::Spec<Dtype> spec, int batch_size,
                       int max_num_players) {
  if (!spec.shape.empty() &&
      spec.shape[0] == -1) {  // If first dim is max_num_players
    spec.shape[0] = max_num_players * batch_size;
  } else {
    spec = spec.Batch(batch_size);
  }
  Array ret(spec);
  ret.Assign(reinterpret_cast<const Dtype*>(buffer), ret.size);
  return ret;
}

template <typename Dtype>
Array GpuBufferToArray(cudaStream_t stream, const void* buffer,
                       ::Spec<Dtype> spec, int batch_size,
                       int max_num_players) {
  if (!spec.shape.empty() &&
      spec.shape[0] == -1) {  // If first dim is max_num_players
    spec.shape[0] = max_num_players * batch_size;
  } else {
    spec = spec.Batch(batch_size);
  }
  Array ret(spec);
  cudaMemcpyAsync(ret.Data(), buffer, ret.size * ret.element_size,
                  cudaMemcpyDeviceToHost, stream);
  return ret;
}

template <typename Dtype>
::Spec<Dtype> NormalizeSpec(const ::Spec<Dtype>& spec, int batch_size,
                            int max_num_players) {
  std::vector<int> shape({0});
  if (!spec.shape.empty() && spec.shape[0] == -1) {
    shape[0] = batch_size * max_num_players;
    shape.insert(shape.end(), spec.shape.begin() + 1, spec.shape.end());
  } else {
    shape[0] = batch_size;
    shape.insert(shape.end(), spec.shape.begin(), spec.shape.end());
  }
  return ::Spec<Dtype>(shape);
}

/**
 * If Spec is a container, the xla interface should be disabled.
 */
template <typename D>
::Spec<D> NormalizeSpec(const ::Spec<Container<D>>& spec, int batch_size,
                        int max_num_players) {
  std::vector<int> shape({0});
  if (!spec.shape.empty() && spec.shape[0] == -1) {
    shape[0] = batch_size * max_num_players;
    shape.insert(shape.end(), spec.shape.begin() + 1, spec.shape.end());
  } else {
    shape[0] = batch_size;
    shape.insert(shape.end(), spec.shape.begin(), spec.shape.end());
  }
  return ::Spec<D>(shape);
}

template <typename EnvPool>
struct XlaSend {
  using In =
      std::array<void*, std::tuple_size_v<typename EnvPool::Action::Keys>>;
  using Out = std::array<void*, 0>;

  static decltype(auto) InSpecs(EnvPool* envpool) {
    int batch_size = envpool->spec.config["batch_size"_];
    int max_num_players = envpool->spec.config["max_num_players"_];
    return std::apply(
        [&](auto&&... s) {
          return std::make_tuple(
              NormalizeSpec(s, batch_size, max_num_players)...);
        },
        envpool->spec.action_spec.AllValues());
  }

  static decltype(auto) OutSpecs(EnvPool* envpool) { return std::tuple<>(); }

  static void Cpu(EnvPool* envpool, const In& in, const Out& out) {
    std::vector<Array> action;
    action.reserve(std::tuple_size_v<typename EnvPool::Action::Keys>);
    int batch_size = envpool->spec.config["batch_size"_];
    int max_num_players = envpool->spec.config["max_num_players"_];
    auto action_spec = envpool->spec.action_spec.AllValues();
    std::size_t index = 0;
    std::apply(
        [&](auto&&... spec) {
          ((action.emplace_back(CpuBufferToArray(in[index++], spec, batch_size,
                                                 max_num_players))),
           ...);
        },
        action_spec);
    envpool->Send(action);
  }

  static void Gpu(EnvPool* envpool, cudaStream_t stream, const In& in,
                  const Out& out) {
    std::vector<Array> action;
    action.reserve(std::tuple_size_v<typename EnvPool::Action::Keys>);
    int batch_size = envpool->spec.config["batch_size"_];
    int max_num_players = envpool->spec.config["max_num_players"_];
    auto action_spec = envpool->spec.action_spec.AllValues();
    std::size_t index = 0;
    std::apply(
        [&](auto&&... spec) {
          ((action.emplace_back(GpuBufferToArray(stream, in[index++], spec,
                                                 batch_size, max_num_players))),
           ...);
        },
        action_spec);
    cudaStreamSynchronize(stream);
    envpool->Send(action);
  }
};

template <typename EnvPool>
struct XlaRecv {
  using In = std::array<void*, 0>;
  using Out =
      std::array<void*, std::tuple_size_v<typename EnvPool::State::Keys>>;

  static decltype(auto) InSpecs(EnvPool* envpool) { return std::tuple<>(); }

  static decltype(auto) OutSpecs(EnvPool* envpool) {
    int batch_size = envpool->spec.config["batch_size"_];
    int max_num_players = envpool->spec.config["max_num_players"_];
    return std::apply(
        [&](auto&&... s) {
          return std::make_tuple(
              NormalizeSpec(s, batch_size, max_num_players)...);
        },
        envpool->spec.state_spec.AllValues());
  }

  static void Cpu(EnvPool* envpool, const In& in, const Out& out) {
    int batch_size = envpool->spec.config["batch_size"_];
    int max_num_players = envpool->spec.config["max_num_players"_];
    std::vector<Array> recv = envpool->Recv();
    for (std::size_t i = 0; i < recv.size(); ++i) {
      CHECK_LE(recv[i].Shape(0), (std::size_t)batch_size * max_num_players);
      std::memcpy(out[i], recv[i].Data(), recv[i].size * recv[i].element_size);
    }
  }

  static void Gpu(EnvPool* envpool, cudaStream_t stream, const In& in,
                  const Out& out) {
    int batch_size = envpool->spec.config["batch_size"_];
    int max_num_players = envpool->spec.config["max_num_players"_];
    std::vector<Array> recv = envpool->Recv();
    for (std::size_t i = 0; i < recv.size(); ++i) {
      CHECK_LE(recv[i].Shape(0), (std::size_t)batch_size * max_num_players);
      cudaMemcpyAsync(out[i], recv[i].Data(),
                      recv[i].size * recv[i].element_size,
                      cudaMemcpyHostToDevice, stream);
    }
  }
};

#endif  // ENVPOOL_CORE_XLA_H_
