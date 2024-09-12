#include <cmath>
#include <fmt/core.h>
#include <iostream>
#include <numbers>
#include <vector>

#include "Eigen/Dense"

#include "solver/global_optimizer.hpp"
#include "utils/daeo_utils.hpp"

template <typename T, typename XT, typename YT, int YDIMS, int PDIMS>
Eigen::Vector<std::common_type_t<XT, YT>, 2>
xprime(T t, Eigen::Vector<XT, 2> const &x, Eigen::Vector<YT, YDIMS> const &y,
       Eigen::Vector<T, PDIMS> const &p) {
  Eigen::Vector<std::common_type_t<XT, YT>, 2> res(x(1), y(0));
  return res;
}

int main(int argc, char **argv) {
  auto h = [](auto t, auto const &x, auto const &y, auto const &p) {
    using std::pow, std::cos, std::cos;
    using std::numbers::pi;
    pow((y(0) - x(0)) / 3, 2) + 3 * cos(pi / 4 * x(0));
  };

  auto f = [](auto t, auto const &x, auto const &y, auto const &p) {
    return xprime(t, x, y, p);
  };

  GlobalOptimizerSettings<double> gopt_settings;
  gopt_settings.MODE = FIND_ALL_LOCAL_MINIMIZERS;
  gopt_settings.LOGGING_ENABLED = true;
  gopt_settings.TOL_Y = 1.0e-8;

  using gopt_t = UnconstrainedGlobalOptimizer<decltype(h),1, 1, 1>;
  gopt_t gopt(h, gopt_settings);
}