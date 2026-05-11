#ifndef ALE_VECTOR_PYTHON_INTERFACE_HPP_
#define ALE_VECTOR_PYTHON_INTERFACE_HPP_

#include <nanobind/nanobind.h>

namespace nb = nanobind;

/// Add vector environment bindings to the module
void init_vector_module(nb::module_& m);

#endif  // ALE_VECTOR_PYTHON_INTERFACE_HPP_
