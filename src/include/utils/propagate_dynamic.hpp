#ifndef _PROPAGATE_EIGEN_DYNAMIC_SIZE
#define _PROPAGATE_EIGEN_DYNAMIC_SIZE

// Go on an adventure to find Eigen::Dynamic
// Should be here (Eigen 3.4)
#include "Eigen/Core"

/**
 * @brief If @c DIMS_IN is the same as @c Eigen::Dynamic , @c value is equal to
 * @c Eigen::Dynamic . Otherwise, @c value is equal to @c DIMS_OUT
 */
template <int DIMS_IN, int DIMS_OUT> struct propagate_dynamic {
  static constexpr int value = DIMS_OUT;
};

template <int DIMS_OUT> struct propagate_dynamic<Eigen::Dynamic, DIMS_OUT> {
  static constexpr int value = Eigen::Dynamic;
};

template<int DIMS_IN, int DIMS_OUT>
inline constexpr int propagate_dynamic_v = propagate_dynamic<DIMS_IN, DIMS_OUT>::value;

#endif