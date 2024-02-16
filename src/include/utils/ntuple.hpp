
#ifndef _NTUPLE_TEMPLATE_H
#define _NTUPLE_TEMPLATE_H

#include <tuple>

template <std::size_t N, typename T> struct tuple_repeattype {
  template <typename... Ts>
  using type = typename tuple_repeattype<N - 1, T>::template type<T, Ts...>;
};

template <typename T> struct tuple_repeattype<0, T> {
  template <typename... Ts> using type = typename std::tuple<Ts...>;
};

/**
 * @brief Tuple of `N` items of type `T`.
 */
template <std::size_t N, typename T>
using ntuple = typename tuple_repeattype<N, T>::template type<>;

#endif