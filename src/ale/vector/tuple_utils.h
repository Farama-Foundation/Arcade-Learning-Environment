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

#ifndef ENVPOOL_CORE_TUPLE_UTILS_H_
#define ENVPOOL_CORE_TUPLE_UTILS_H_

#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

template <class T, class Tuple>
struct Index;

template <class T, class... Types>
struct Index<T, std::tuple<T, Types...>> {
  static constexpr std::size_t kValue = 0;
};

template <class T, class U, class... Types>
struct Index<T, std::tuple<U, Types...>> {
  static constexpr std::size_t kValue =
      1 + Index<T, std::tuple<Types...>>::kValue;
};

template <class F, class K, class V, std::size_t... I>
decltype(auto) ApplyZip(F&& f, K&& k, V&& v,
                        std::index_sequence<I...> /*unused*/) {
  return std::invoke(std::forward<F>(f),
                     std::make_tuple(I, std::get<I>(std::forward<K>(k)),
                                     std::get<I>(std::forward<V>(v)))...);
}

template <typename... T>
using tuple_cat_t = decltype(std::tuple_cat(std::declval<T>()...));  // NOLINT

template <typename TupleType, typename T, std::size_t... Is>
decltype(auto) TupleFromVectorImpl(std::index_sequence<Is...> /*unused*/,
                                   const std::vector<T>& arguments) {
  return TupleType(arguments[Is]...);
}

template <typename TupleType, typename T, std::size_t... Is>
decltype(auto) TupleFromVectorImpl(std::index_sequence<Is...> /*unused*/,
                                   std::vector<T>&& arguments) {
  return TupleType(std::move(arguments[Is])...);
}

template <typename TupleType, typename V>
decltype(auto) TupleFromVector(V&& arguments) {
  return TupleFromVectorImpl<TupleType>(
      std::make_index_sequence<std::tuple_size_v<TupleType>>{},
      std::forward<V>(arguments));
}

#endif  // ENVPOOL_CORE_TUPLE_UTILS_H_
