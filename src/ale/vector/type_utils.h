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

#ifndef ENVPOOL_CORE_TYPE_UTILS_H_
#define ENVPOOL_CORE_TYPE_UTILS_H_

#include <functional>
#include <tuple>
#include <type_traits>
#include <vector>

template <class T, class TupleTs>
struct any_match;

template <class T, template <typename...> class Tuple, typename... Ts>
struct any_match<T, Tuple<Ts...>> : std::disjunction<std::is_same<T, Ts>...> {};

template <class T, class TupleTs>
struct all_match;

template <class T, template <typename...> class Tuple, typename... Ts>
struct all_match<T, Tuple<Ts...>> : std::conjunction<std::is_same<T, Ts>...> {};

template <class To, class TupleTs>
struct all_convertible;

template <class To, template <typename...> class Tuple, typename... Fs>
struct all_convertible<To, Tuple<Fs...>>
    : std::conjunction<std::is_convertible<Fs, To>...> {};

template <typename T>
constexpr bool is_tuple_v = false;  // NOLINT
template <typename... types>
constexpr bool is_tuple_v<std::tuple<types...>> = true;  // NOLINT

template <typename T>
constexpr bool is_vector_v = false;  // NOLINT
template <typename VT>
constexpr bool is_vector_v<std::vector<VT>> = true;  // NOLINT

#endif  // ENVPOOL_CORE_TYPE_UTILS_H_
