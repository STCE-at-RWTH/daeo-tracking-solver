#include <cmath>
#include <fmt/core.h>
#include <iostream>
#include <numbers>
#include <vector>

#include "Eigen/Dense"
#include "boost/numeric/interval.hpp"

#include "solver/local_optima_solver.hpp"
#include "solver/settings.hpp"

template <typename F, typename G> struct lagrangian_detail {
  F f;
  G g;
  template <typename T, typename X, typename Y, int YDIMS, int PDIMS>
  Y L(T t, X x, Eigen::Vector<Y, YDIMS> const &y,
      Eigen::Vector<T, PDIMS> const &p) const {
    auto idcs = Eigen::seq(1, Eigen::last);
    return f(t, x, y(idcs), p) + y(0) * g(t, x, y(idcs), p);
  }

  template <typename T, typename X, typename Y, int YDIMS, int PDIMS>
  Y dLdy_sqr(T t, X x, Eigen::Vector<Y, YDIMS> const &y,
             Eigen::Vector<T, PDIMS> const &p) const {
    using dco_mode_t = dco::gt1s<Y>;
    using active_t = typename dco_mode_t::type;
    Y res{0};
    Eigen::Vector<active_t, YDIMS> y_active(y.rows());
    for (size_t i = 0; i < y.rows(); i++) {
      dco::value(y_active(i)) = y(i);
    }
    for (size_t i = 0; i < y.rows(); i++) {
      dco::derivative(y_active(i)) = 1.0;
      res += pow(dco::derivative(L(t, x, y_active, p)), 2);
      dco::derivative(y_active(i)) = 0;
    }
    return res;
  }
};

int main(int argc, char **argv) {

  /**
   * Minimize h and h2, which are functions in 1 and two dimensions
   */
  auto h = [](const auto t, const auto x, const auto &y, const auto &p) -> auto{
    return (pow(x - y(0), 2) + sin(p(0) * y(0)));
  };

  auto h2 = [](const auto t, const auto x, auto const &y, const auto &p) {
    return (pow((y(0) - 1.0) / p(0), 2) + pow(y(1) / p(1), 2));
  };

  BNBOptimizerSettings<double> settings_only_minimize;
  settings_only_minimize.MODE = FIND_ALL_LOCAL_MINIMIZERS;
  settings_only_minimize.LOGGING_ENABLED = true;
  settings_only_minimize.TOL_Y = 1.0e-8;

  using opt1_t = BNBOptimizer<decltype(h), double,
                              suggested_interval_policies<double>, 1, 1>;
  typename opt1_t::y_t ll{-20};
  typename opt1_t::y_t ur{20.0};
  typename opt1_t::params_t p{5.0};
  opt1_t optimizer(h, ll, ur, settings_only_minimize);
  vector<double> x_tests{1.0, 1.25, 2.5, 3.5, 8.0};
  // for (double x0 : x_tests) {
  //   auto res = optimizer.find_minima_at(0.0, x0, p);
  //   fmt::println("results at x={:.2f}: {:::.6f}", x0, res.minima_intervals);
  // }

  // using opt2_t = BNBOptimizer<decltype(h2), double,
  //                             suggested_interval_policies<double>, 2, 2>;
  // typename opt2_t::y_t ll2{-20, -20};
  // typename opt2_t::y_t ur2{20, 20};
  // typename opt2_t::params_t p2{1.0, 1.0};
  // opt2_t opt2(h2, ll2, ur2, settings_only_minimize);
  // auto res2 = opt2.find_minima_at(0.0, 0.0, p2);
  // fmt::println("{:::.6f}", res2.minima_intervals);

  BNBOptimizerSettings<double> settings_with_constraint;
  settings_with_constraint.TOL_Y = 1.0e-3;
  settings_with_constraint.LOGGING_ENABLED = false;
  settings_with_constraint.MODE = FIND_ALL_LOCAL_MINIMIZERS;
  settings_with_constraint.RETEST_CRITICAL_POINTS = true;
  settings_with_constraint.MAXITER = 1'000'000; // oh no

  auto h3 = [](auto t, auto x, auto const &y, auto const &p) {
    return p(0) * y(0) + p(1) * y(1);
  };
  auto g = [](auto t, auto x, auto const &y, auto const &p) {
    return pow(y(0), 2) + pow(y(1), 2) - p(2);
  };

  lagrangian_detail<decltype(h3), decltype(g)> const L_detail{h3, g};
  auto L = [](auto t, auto x, auto const &y, auto const &p) {
    return p(0) * y(1) + p(1) * y(2) +
           y(0) * (pow(y(1), 2) + pow(y(2), 2) - p(2));
  };
  auto gradL = [](auto t, auto x, auto const &y, auto const &p) {
    // auto val = p(0) * y(1) + p(1) * y(2) + y(0) * (pow(y(1), 2) + pow(y(2),
    // 2) - p(2));
    auto grady0 = (pow(y(1), 2) + pow(y(2), 2) - p(2));
    auto grady1 = p(0) + 2 * y(1) * y(0);
    auto grady2 = p(1) + 2 * y(2) * y(0);
    return pow(grady0, 2) + pow(grady1, 2) + pow(grady2, 2);
  };

  using opt3_t = BNBOptimizer<decltype(gradL), double,
                              suggested_interval_policies<double>, 3, 3>;
  typename opt3_t::y_t ll3{-2, -2, -2};
  typename opt3_t::y_t ur3{2, 2, 2};
  typename opt3_t::params_t p3{1.0, 1.0, 1.0};
  opt3_t opt3(gradL, ll3, ur3, settings_with_constraint);
  auto res3 = opt3.find_minima_at(1.0, 1.0, p3);
  fmt::println("{:d}", res3.minima_intervals.size());
  for (auto const &y : res3.minima_intervals) {
    fmt::println("{:: .6f}", y);
  }
  // double thold = 1.0e-3;
  // vector<size_t> n(res3.minima_intervals.size());
  // for (size_t i = 0; i < res3.minima_intervals.size(); i++) {
  //   auto const &y = res3.minima_intervals[i].unaryExpr(
  //       [](auto ival) { return boost::numeric::median(ival); });
  //   for (const auto &y1 : res3.minima_intervals) {
  //     double norm1 = (y - y1.unaryExpr([](auto ival) {
  //                      return boost::numeric::median(ival);
  //                    })).norm();
  //     if (norm1 < thold) {
  //       n[i]++;
  //     }
  //   }
  // }
  // fmt::println("{::d}", n);
  fmt::println("{:d}", res3.hessian_test_inconclusive.size());
  for (auto const &y : res3.hessian_test_inconclusive) {
    fmt::println("{:: .6f}", y);
  }
}