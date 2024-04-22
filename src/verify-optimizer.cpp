#include <cmath>
#include <fmt/core.h>
#include <iostream>
#include <numbers>
#include <vector>

#include "Eigen/Dense"
#include "boost/numeric/interval.hpp"

#include "solver/local_optima_solver.hpp"
#include "solver/settings.hpp"

template <typename T>
using boost_interval_transc_t =
    boost::numeric::interval<T, suggested_solver_policies<T>>;
using double_ival = boost_interval_transc_t<double>;

int main(int argc, char **argv) {
  auto h = [](const auto t, const auto x, const auto &y,
              const auto &p) -> auto {
    return (pow(x - y(0), 2) + sin(p(0) * y(0)));
  };

  BNBOptimizerSettings<double> opt_s;
  opt_s.LOGGING_ENABLED = true;
  opt_s.TOL_Y = 1.0e-6;

  using opt_t = BNBLocalOptimizer<decltype(h), double,
                                  suggested_solver_policies<double>, 1, 1>;
  typename opt_t::y_t ll{0.0};
  typename opt_t::y_t ur{6.0};
  typename opt_t::params_t p{5.0};
  opt_t optimizer(h, ll, ur, opt_s);
  auto res = optimizer.find_minima_at(0.0, 1.0, p, false);
  fmt::println("results: {:::.6f}", res.minima_intervals);
}