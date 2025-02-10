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

#ifndef ENVPOOL_CORE_PY_ENVPOOL_H_
#define ENVPOOL_CORE_PY_ENVPOOL_H_

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <exception>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "ale/vector/envpool.h"
#include "ale/vector/xla.h"

namespace py = pybind11;

/**
 * Convert Array to py::array, with py::capsule
 */
template <typename dtype>
struct ArrayToNumpyHelper {
  static py::array Convert(const Array& a) {
    auto* ptr = new std::shared_ptr<char>(a.SharedPtr());
    auto capsule = py::capsule(ptr, [](void* ptr) {
      delete reinterpret_cast<std::shared_ptr<char>*>(ptr);
    });
    return py::array(a.Shape(), reinterpret_cast<dtype*>(a.Data()), capsule);
  }
};

template <typename dtype>
struct ArrayToNumpyHelper<Container<dtype>> {
  using UniquePtr = Container<dtype>;
  static py::array Convert(const Array& a) {
    auto* ptr_arr = reinterpret_cast<UniquePtr*>(a.Data());
    auto* ptr =
        new std::unique_ptr<py::object[]>(new py::object[a.size]);  // NOLINT
    auto capsule = py::capsule(ptr, [](void* ptr) {
      delete reinterpret_cast<std::unique_ptr<py::object[]>*>(ptr);  // NOLINT
    });
    for (std::size_t i = 0; i < a.size; ++i) {
      auto* inner_ptr = new UniquePtr(std::move(ptr_arr[i]));
      (ptr_arr + i)->~UniquePtr();
      auto capsule = py::capsule(inner_ptr, [](void* inner_ptr) {
        delete reinterpret_cast<UniquePtr*>(inner_ptr);
      });
      if (*inner_ptr == nullptr) {
        (*ptr)[i] = py::none();
      } else {
        (*ptr)[i] =
            py::array((*inner_ptr)->Shape(),
                      reinterpret_cast<dtype*>((*inner_ptr)->Data()), capsule);
      }
    }
    return {py::dtype("object"), a.Shape(),
            reinterpret_cast<py::object*>(ptr->get()), capsule};
  }
};

template <typename dtype>
Array NumpyToArray(const py::array& arr) {
  using ArrayT = py::array_t<dtype, py::array::c_style | py::array::forcecast>;
  ArrayT arr_t(arr);
  ShapeSpec spec(arr_t.itemsize(),
                 std::vector<int>(arr_t.shape(), arr_t.shape() + arr_t.ndim()));
  return {spec, reinterpret_cast<char*>(arr_t.mutable_data())};
}

template <typename dtype>
Array NumpyToArrayIncRef(const py::array& arr) {
  using ArrayT = py::array_t<dtype, py::array::c_style | py::array::forcecast>;
  auto* arr_ptr = new ArrayT(arr);
  ShapeSpec spec(
      arr_ptr->itemsize(),
      std::vector<int>(arr_ptr->shape(), arr_ptr->shape() + arr_ptr->ndim()));
  return Array(spec, reinterpret_cast<char*>(arr_ptr->mutable_data()),
               [arr_ptr](char* p) {
                 py::gil_scoped_acquire acquire;
                 delete arr_ptr;
               });
}

template <typename Spec>
struct SpecTupleHelper {
  static decltype(auto) Make(const Spec& spec) {
    return std::make_tuple(py::dtype::of<typename Spec::dtype>(), spec.shape,
                           spec.bounds, spec.elementwise_bounds);
  }
};

/**
 * For Container type, it is converted a numpy array of numpy array.
 * The spec itself describes the shape of the outer array, the inner_spec
 * contains the spec of the inner array.
 * Therefore the shape returned to python side has the format
 * (outer_shape, inner_shape).
 */
template <typename dtype>
struct SpecTupleHelper<Spec<Container<dtype>>> {
  static decltype(auto) Make(const Spec<Container<dtype>>& spec) {
    return std::make_tuple(py::dtype::of<dtype>(),
                           std::make_tuple(spec.shape, spec.inner_spec.shape),
                           spec.inner_spec.bounds,
                           spec.inner_spec.elementwise_bounds);
  }
};

template <typename... Spec>
decltype(auto) ExportSpecs(const std::tuple<Spec...>& specs) {
  return std::apply(
      [&](auto&&... spec) {
        return std::make_tuple(SpecTupleHelper<Spec>::Make(spec)...);
      },
      specs);
}

template <typename EnvSpec>
class PyEnvSpec : public EnvSpec {
 public:
  using StateSpecT =
      decltype(ExportSpecs(std::declval<typename EnvSpec::StateSpec>()));
  using ActionSpecT =
      decltype(ExportSpecs(std::declval<typename EnvSpec::ActionSpec>()));

  StateSpecT py_state_spec;
  ActionSpecT py_action_spec;
  typename EnvSpec::ConfigValues py_config_values;
  static std::vector<std::string> py_config_keys;
  static std::vector<std::string> py_state_keys;
  static std::vector<std::string> py_action_keys;
  static typename EnvSpec::ConfigValues py_default_config_values;

  explicit PyEnvSpec(const typename EnvSpec::ConfigValues& conf)
      : EnvSpec(conf),
        py_state_spec(ExportSpecs(EnvSpec::state_spec)),
        py_action_spec(ExportSpecs(EnvSpec::action_spec)),
        py_config_values(EnvSpec::config.AllValues()) {}
};
template <typename EnvSpec>
std::vector<std::string> PyEnvSpec<EnvSpec>::py_config_keys =
    EnvSpec::Config::AllKeys();
template <typename EnvSpec>
std::vector<std::string> PyEnvSpec<EnvSpec>::py_state_keys =
    EnvSpec::StateSpec::AllKeys();
template <typename EnvSpec>
std::vector<std::string> PyEnvSpec<EnvSpec>::py_action_keys =
    EnvSpec::ActionSpec::AllKeys();
template <typename EnvSpec>
typename EnvSpec::ConfigValues PyEnvSpec<EnvSpec>::py_default_config_values =
    EnvSpec::kDefaultConfig.AllValues();

/**
 * Bind specs to arrs, and return py::array in ret
 */
template <typename... Spec>
void ToNumpy(const std::vector<Array>& arrs, const std::tuple<Spec...>& specs,
             std::vector<py::array>* ret) {
  std::size_t index = 0;
  std::apply(
      [&](auto&&... spec) {
        (ret->emplace_back(
             ArrayToNumpyHelper<typename Spec::dtype>::Convert(arrs[index++])),
         ...);
      },
      specs);
}

template <typename... Spec>
void ToArray(const std::vector<py::array>& py_arrs,
             const std::tuple<Spec...>& specs, std::vector<Array>* ret) {
  std::size_t index = 0;
  std::apply(
      [&](auto&&... spec) {
        (ret->emplace_back(
             NumpyToArrayIncRef<typename Spec::dtype>(py_arrs[index++])),
         ...);
      },
      specs);
}

/**
 * Templated subclass of EnvPool,
 * to be overrided by the real EnvPool.
 */
template <typename EnvPool>
class PyEnvPool : public EnvPool {
 public:
  using PySpec = PyEnvSpec<typename EnvPool::Spec>;

  PySpec py_spec;
  static std::vector<std::string> py_state_keys;
  static std::vector<std::string> py_action_keys;

  explicit PyEnvPool(const PySpec& py_spec)
      : EnvPool(py_spec), py_spec(py_spec) {}

  /**
   * get xla functions
   */
  auto Xla() {
    if (HasContainerType(EnvPool::spec.state_spec)) {
      throw std::runtime_error(
          "State of this env has dynamic shaped container, xla is disabled");
    }
    if (HasDynamicDim(EnvPool::spec.state_spec)) {
      throw std::runtime_error(
          "State of this env has dynamic (-1) shape, xla is disabled");
    }
    if (EnvPool::spec.config["max_num_players"_] != 1) {
      throw std::runtime_error(
          "Xla is not available for multiplayer environment.");
    }
    return std::make_tuple(
        std::make_tuple("recv",
                        CustomCall<EnvPool, XlaRecv<EnvPool>>::Xla(this)),
        std::make_tuple("send",
                        CustomCall<EnvPool, XlaSend<EnvPool>>::Xla(this)));
  }

  /**
   * py api
   */
  void PySend(const std::vector<py::array>& action) {
    std::vector<Array> arr;
    arr.reserve(action.size());
    ToArray(action, py_spec.action_spec, &arr);
    py::gil_scoped_release release;
    EnvPool::Send(arr);  // delegate to the c++ api
  }

  /**
   * py api
   */
  std::vector<py::array> PyRecv() {
    std::vector<Array> arr;
    {
      py::gil_scoped_release release;
      arr = EnvPool::Recv();
      DCHECK_EQ(arr.size(), std::tuple_size_v<typename EnvPool::State::Keys>);
    }
    std::vector<py::array> ret;
    ret.reserve(EnvPool::State::kSize);
    ToNumpy(arr, py_spec.state_spec, &ret);
    return ret;
  }

  /**
   * py api
   */
  void PyReset(const py::array& env_ids) {
    // PyArray arr = PyArray::From<int>(env_ids);
    auto arr = NumpyToArrayIncRef<int>(env_ids);
    py::gil_scoped_release release;
    EnvPool::Reset(arr);
  }
};

template <typename EnvPool>
std::vector<std::string> PyEnvPool<EnvPool>::py_state_keys =
    PyEnvPool<EnvPool>::PySpec::py_state_keys;
template <typename EnvPool>
std::vector<std::string> PyEnvPool<EnvPool>::py_action_keys =
    PyEnvPool<EnvPool>::PySpec::py_action_keys;

py::object abc_meta = py::module::import("abc").attr("ABCMeta");

/**
 * Call this macro in the translation unit of each envpool instance
 * It will register the envpool instance to the registry.
 * The static bool status is local to the translation unit.
 */
#define REGISTER(MODULE, SPEC, ENVPOOL)                              \
  py::class_<SPEC>(MODULE, "_" #SPEC, py::metaclass(abc_meta))       \
      .def(py::init<const typename SPEC::ConfigValues&>())           \
      .def_readonly("_config_values", &SPEC::py_config_values)       \
      .def_readonly("_state_spec", &SPEC::py_state_spec)             \
      .def_readonly("_action_spec", &SPEC::py_action_spec)           \
      .def_readonly_static("_state_keys", &SPEC::py_state_keys)      \
      .def_readonly_static("_action_keys", &SPEC::py_action_keys)    \
      .def_readonly_static("_config_keys", &SPEC::py_config_keys)    \
      .def_readonly_static("_default_config_values",                 \
                           &SPEC::py_default_config_values);         \
  py::class_<ENVPOOL>(MODULE, "_" #ENVPOOL, py::metaclass(abc_meta)) \
      .def(py::init<const SPEC&>())                                  \
      .def_readonly("_spec", &ENVPOOL::py_spec)                      \
      .def("_recv", &ENVPOOL::PyRecv)                                \
      .def("_send", &ENVPOOL::PySend)                                \
      .def("_reset", &ENVPOOL::PyReset)                              \
      .def_readonly_static("_state_keys", &ENVPOOL::py_state_keys)   \
      .def_readonly_static("_action_keys", &ENVPOOL::py_action_keys) \
      .def("_xla", &ENVPOOL::Xla);

#endif  // ENVPOOL_CORE_PY_ENVPOOL_H_
