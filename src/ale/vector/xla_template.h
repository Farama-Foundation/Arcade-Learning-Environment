/*
 * Copyright 2022 Garena Online Private Limited
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

#ifndef ENVPOOL_CORE_XLA_TEMPLATE_H_
#define ENVPOOL_CORE_XLA_TEMPLATE_H_

#include <cuda_runtime_api.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <tuple>
#include <vector>

namespace py = pybind11;

template <typename Spec>
static auto SpecToTuple(const Spec& spec) {
  return std::make_tuple(py::dtype::of<typename Spec::dtype>(), spec.shape);
}

template <std::size_t N>
void ToArray(const void** raw, std::array<void*, N>* array) {
  int i = 0;
  std::apply([&](auto&&... a) { ((a = const_cast<void*>(raw[i++])), ...); },
             *array);
}

template <std::size_t N>
void ToArray(void** raw, std::array<void*, N>* array) {
  int i = 0;
  std::apply([&](auto&&... a) { ((a = raw[i++]), ...); }, *array);
}

template <typename Class, typename CC>
struct CustomCall {
  using InSpecs =
      typename std::invoke_result<decltype(CC::InSpecs), Class*>::type;
  using OutSpecs =
      typename std::invoke_result<decltype(CC::OutSpecs), Class*>::type;
  using In = std::array<void*, std::tuple_size_v<InSpecs>>;
  using Out = std::array<void*, std::tuple_size_v<OutSpecs>>;

  static py::bytes Handle(Class* obj) {
    return py::bytes(
        std::string(reinterpret_cast<const char*>(&obj), sizeof(Class*)));
  }

  static void Cpu(void* out, const void** in) {
    Class* obj = *reinterpret_cast<Class**>(const_cast<void*>(in[0]));
    in += 1;
    In in_arr;
    Out out_arr;
    ToArray(in, &in_arr);
    if (std::tuple_size<Out>::value == 0) {
      std::memcpy(out, &obj, sizeof(Class*));
    } else {
      void** outs = reinterpret_cast<void**>(out);
      std::memcpy(outs[0], &obj, sizeof(Class*));
      ToArray(outs + 1, &out_arr);
    }
    CC::Cpu(obj, in_arr, out_arr);
  }

  static void Gpu(cudaStream_t stream, void** buffers, const char* opaque,
                  std::size_t opaque_len) {
    Class* obj = *reinterpret_cast<Class**>(const_cast<char*>(opaque));
    buffers += 1;
    In in_arr;
    Out out_arr;
    ToArray(buffers, &in_arr);
    buffers += std::tuple_size<In>::value;
    buffers += 1;
    ToArray(buffers, &out_arr);
    CC::Gpu(obj, stream, in_arr, out_arr);
  }

  static auto Specs(Class* obj) {
    auto handle_spec =
        std::make_tuple(SpecToTuple(Spec<uint8_t>({sizeof(Class*)})));
    auto in_specs = CC::InSpecs(obj);
    auto in = std::apply(
        [&](auto&&... a) { return std::make_tuple(SpecToTuple(a)...); },
        in_specs);
    auto out_specs = CC::OutSpecs(obj);
    auto out = std::apply(
        [&](auto&&... a) { return std::make_tuple(SpecToTuple(a)...); },
        out_specs);
    return std::make_tuple(std::tuple_cat(handle_spec, in),
                           std::tuple_cat(handle_spec, out));
  }

  static auto Capsules() {
    return std::make_tuple(
        py::capsule(reinterpret_cast<void*>(Cpu), "xla._CUSTOM_CALL_TARGET"),
        py::capsule(reinterpret_cast<void*>(Gpu), "xla._CUSTOM_CALL_TARGET"));
  }

  static auto Xla(Class* obj) {
    return std::make_tuple(Handle(obj), Specs(obj), Capsules());
  }
};

#endif  // ENVPOOL_CORE_XLA_TEMPLATE_H_
