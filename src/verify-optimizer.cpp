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
    boost::numeric::interval<T, suggested_interval_policies<T>>;
using double_ival = boost_interval_transc_t<double>;

int main(int argc, char **argv) {
  auto h = [](const auto t, const auto x, const auto &y, const auto &p) -> auto{
    return (pow(x - y(0), 2) + sin(p(0) * y(0)));
  };

  auto h2 = [](const auto t, const auto x, auto const &y, const auto &p) {
    return (pow((y(0) - 1.0) / p(0), 2) + pow(y(1) / p(1), 2));
  };

  BNBOptimizerSettings<double> opt_s;
  opt_s.MODE = FIND_ALL_LOCAL_MINIMIZERS;
  opt_s.LOGGING_ENABLED = true;
  opt_s.TOL_Y = 1.0e-8;
  opt_s.MAX_REFINE_ITER = 40;

  using opt_t = BNBOptimizer<decltype(h), double,
                                  suggested_interval_policies<double>, 1, 1>;
  typename opt_t::y_t ll{-20};
  typename opt_t::y_t ur{20.0};
  typename opt_t::params_t p{5.0};
  opt_t optimizer(h, ll, ur, opt_s);
  vector<double> x_tests{1.0, 1.25, 2.5, 3.5, 8.0};
  for (double x0 : x_tests) {
    auto res = optimizer.find_minima_at(0.0, x0, p);
    fmt::println("results at x={:.2f}: {:::.6f}", x0, res.minima_intervals);
  }

  using opt2_t = BNBOptimizer<decltype(h2), double,
                                   suggested_interval_policies<double>, 2, 2>;
  typename opt2_t::y_t ll2{-20, -20};
  typename opt2_t::y_t ur2{20, 20};
  typename opt2_t::params_t p2{1.0, 1.0};
  opt2_t opt2(h2, ll2, ur2, opt_s);
  auto res2 = opt2.find_minima_at(0.0, 0.0, p2);
  fmt::println("{:::.6f}", res2.minima_intervals);
  fmt::println("{::.4e}", res2.minima_intervals[0].unaryExpr([
  ](auto ival) -> auto{ return boost::numeric::width(ival); }));
}