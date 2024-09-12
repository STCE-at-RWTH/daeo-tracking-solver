#include <cmath>
#include <fmt/core.h>
#include <iostream>
#include <numbers>
#include <vector>

#include "Eigen/Dense"

#include "solver/global_optimizer.hpp"
#include "utils/daeo_utils.hpp"

int main(int argc, char **argv) {

  /**
   * Minimize h and h2, which are functions in 1 and two dimensions
   */
  auto h = [](const auto t, const auto x, const auto &y,
              const auto &p) -> auto {
    return (pow(x - y(0), 2) + sin(p(0) * y(0)));
  };

  auto h2 = [](const auto t, const auto x, auto const &y, const auto &p) {
    return (pow((y(0) - 1.0) / p(0), 2) + pow(y(1) / p(1), 2));
  };

  GlobalOptimizerSettings<double> settings_only_minimize;
  settings_only_minimize.MODE = FIND_ALL_LOCAL_MINIMIZERS;
  settings_only_minimize.LOGGING_ENABLED = true;
  settings_only_minimize.TOL_Y = 1.0e-8;

  using opt1_t = UnconstrainedGlobalOptimizer<decltype(h), 1, 1>;
  typename opt1_t::y_t ll {-20.0};
  typename opt1_t::y_t ur {20.0};
  typename opt1_t::y_interval_t domain1 = build_box(ll, ur);
  typename opt1_t::params_t p{5.0};
  opt1_t optimizer(h, settings_only_minimize);
  vector<double> x_tests{1.0, 1.25, 2.5, 3.5, 8.0};
  for (double x0 : x_tests) {
    auto res = optimizer.find_minima_at(0.0, x0, domain1, p);
    fmt::println("results at x={:.2f}:", x0);
    for (auto const &y : res.minima_intervals) {
      fmt::println(" {:: .6f}", y);
    }
  }

  using opt2_t = UnconstrainedGlobalOptimizer<decltype(h2), 2, 2>;
  typename opt2_t::y_interval_t domain2{
      typename opt2_t::interval_t{-20.0, 20.0},
      typename opt2_t::interval_t{-20.0, 20.0}};

  typename opt2_t::params_t p2{1.0, 1.0};
  opt2_t opt2(h2, settings_only_minimize);
  auto res2 = opt2.find_minima_at(0.0, 0.0, domain2, p2);
  fmt::println("{:::.6f}", res2.minima_intervals);
}