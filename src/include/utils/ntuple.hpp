
#ifndef _NTUPLE_TEMPLATE_H
#define _NTUPLE_TEMPLATE_H

#include <tuple>

template <std::size_t N, typename T> struct ntuple_detail {
  template <typename... Ts>
  using type = typename ntuple_detail<N - 1, T>::template type<T, Ts...>;
};

template <typename T> struct ntuple_detail<0, T> {
  template <typename... Ts> using type = typename std::tuple<Ts...>;
};

/**
 * @brief Tuple of `N` items of type `T`.
 */
template <std::size_t N, typename T>
using ntuple = typename ntuple_detail<N, T>::template type<>;

#endif