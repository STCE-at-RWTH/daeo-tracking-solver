#include <cmath>
#include <fmt/core.h>
#include <iostream>
#include <numbers>
#include <vector>

#include "boost/numeric/interval.hpp"

#include "solver/daeo_solver.hpp"
#include "solver/local_optima_solver.hpp"
#include "solver/logging.hpp"
#include "solver/objective.hpp"
#include "solver/settings.hpp"

using namespace std::numbers;

template <typename T>
using boost_interval_transc_t =
    boost::numeric::interval<T, suggested_solver_policies<T>>;
using double_ival = boost_interval_transc_t<double>;

constexpr int NUM_Y_DIMS = 1;
constexpr int NUM_PARAMS = 4;

template <typename T>
void run_simple_example(DAEOSolverSettings<T> solver_s,
                        BNBOptimizerSettings<T> optimizer_s) {
  auto f = [](const auto t, const auto x, const auto &y, const auto &p) -> auto{
    return -(p(0) + y(0)) * x;
  };

  auto h = [](const auto t, const auto x, const auto &y, const auto &p) -> auto{
    return pow(p(1) - pow(y(0), 2), 2) - (x - p(2)) * sin(y(0) * p(3));
  };

  // using optimizer_t = BNBLocalOptimizer<decltype(h), double,
  // suggested_solver_policies<double>, NUM_Y_DIMS, NUM_PARAMS>;
  using solver_t = DAEOTrackingSolver<decltype(f), decltype(h), double,
                                      NUM_Y_DIMS, NUM_PARAMS>;
  typename solver_t::params_t p(2.0, 1.0, 0.5, pi / 2);

  fmt::println("\n** standard setup **\n");
  for (int i = 0; i < 6; i++) {
    double dt = pow(10.0, -i);
    solver_t solver(f, h, optimizer_s, solver_s);
    solver.solve_daeo(0, 1, dt, 1.0, p,
                      fmt::format("simple_example_10_minus_{:d}", i));
  }

  fmt::println("\n** no events **\n");
  solver_s.EVENT_DETECTION_AND_CORRECTION = false;
  solver_s.ONLY_GLOBAL_OPTIMIZATION = false;
  solver_s.SEARCH_FREQUENCY = std::numeric_limits<size_t>::max();
  for (int i = 0; i < 6; i++) {
    double dt = pow(10.0, -i);
    solver_t solver(f, h, optimizer_s, solver_s);
    solver.solve_daeo(0, 1, dt, 1.0, p,
                      fmt::format("simple_example_10_minus_{:d}_noevents", i));
  }

  fmt::println("\n** only global opt **\n");
  solver_s.ONLY_GLOBAL_OPTIMIZATION = true;
  solver_s.SEARCH_FREQUENCY = 1;
  for (int i = 0; i < 6; i++) {
    double dt = pow(10.0, -i);
    solver_t solver(f, h, optimizer_s, solver_s);
    solver.solve_daeo(
        0, 1, dt, 1.0, p,
        fmt::format("simple_example_10_minus_{:d}_onlyglobal", i));
  }
}

void griewank_example_event_tolerance_study() {

  BNBOptimizerSettings<double> opt_s;
  opt_s.LOGGING_ENABLED = false;

  DAEOSolverSettings<double> settings_lowtol;
  settings_lowtol.NEWTON_EPS = 1.0e-10;
  settings_lowtol.EVENT_EPS = 1.0e-8;
  settings_lowtol.y0_min = 0.0;
  settings_lowtol.y0_max = 6.0;
  settings_lowtol.SEARCH_FREQUENCY = 100;

  auto settings_lowtol_noevents = settings_lowtol;
  settings_lowtol_noevents.EVENT_DETECTION_AND_CORRECTION = false;

  auto settings_hightol = settings_lowtol;
  settings_hightol.EVENT_EPS = 5.0e-4;

  auto settings_hightol_noevents = settings_hightol;
  settings_hightol_noevents.EVENT_DETECTION_AND_CORRECTION = false;

  auto f = [](const auto t, const auto x, const auto &y, const auto &p) -> auto{
    return y(0);
  };

  auto h = [](const auto t, const auto x, const auto &y, const auto &p) -> auto{
    return pow(x - y(0), 2) + sin(p(0) * y(0));
  };

  // using optimizer_t = BNBLocalOptimizer<decltype(h), T,
  // suggested_solver_policies<T>, NUM_Y_DIMS, NUM_PARAMS>;
  using solver_t = DAEOTrackingSolver<decltype(f), decltype(h), double, 1, 1>;
  typename solver_t::params_t p{5.0};

  solver_t solver1(f, h, opt_s, settings_lowtol);
  solver_t solver2(f, h, opt_s, settings_lowtol_noevents);
  solver_t solver3(f, h, opt_s, settings_hightol);
  solver_t solver4(f, h, opt_s, settings_hightol_noevents);

  solver1.solve_daeo(0., 2.0, 0.0005, 1.0, p, "griewank_example_lowtol");
  solver2.solve_daeo(0., 2.0, 0.0005, 1.0, p,
                     "griewank_example_lowtol_noevents");
  solver3.solve_daeo(0., 2.0, 0.0005, 1.0, p, "griewank_example_hightol");
  solver4.solve_daeo(0., 2.0, 0.0005, 1.0, p,
                     "griewank_example_hightol_noevents");
}

void simple_example_perf_study(int N) {
  auto f = [](const auto t, const auto x, const auto &y, const auto &p) -> auto{
    return -(p(0) + y(0)) * x;
  };

  auto h = [](const auto t, const auto x, const auto &y, const auto &p) -> auto{
    return pow(p(1) - pow(y(0), 2), 2) - (x - p(2)) * sin(y(0) * p(3));
  };

  // using optimizer_t = BNBLocalOptimizer<decltype(h), double,
  // suggested_solver_policies<double>, NUM_Y_DIMS, NUM_PARAMS>;
  using solver_t = DAEOTrackingSolver<decltype(f), decltype(h), double,
                                      NUM_Y_DIMS, NUM_PARAMS>;
  typename solver_t::params_t p(2.0, 1.0, 0.5, pi / 2);

  DAEOSolverSettings<double> solver_s;
  solver_s.NEWTON_EPS = 1.0e-8;
  solver_s.EVENT_EPS = 5.0e-6;
  solver_s.y0_min = -6.0;
  solver_s.y0_max = 6.0;

  BNBOptimizerSettings<double> opt_s;
  opt_s.LOGGING_ENABLED = false;

  /**
   * TRACK LOCAL, NEVER REOPTIMIZE
   */
  solver_s.SEARCH_FREQUENCY = std::numeric_limits<size_t>::max();
  solver_s.EVENT_DETECTION_AND_CORRECTION = true;
  solver_s.ONLY_GLOBAL_OPTIMIZATION = false;

  for (int i = 0; i < N; i++) {
    double dt = pow(10.0, -i);
    solver_t solver(f, h, opt_s, solver_s);
    solver.solve_daeo(0, 1, dt, 1.0, p,
                      fmt::format("se_tracking_10_minus{:d}", i));
  }

  /**
   * IGNORE EVENTS, NO GLOBAL SEARCH
   * (shouldn't be much different from tracking with events, bisection is
   * "cheap")
   */
  solver_s.SEARCH_FREQUENCY = std::numeric_limits<size_t>::max();
  solver_s.EVENT_DETECTION_AND_CORRECTION = false;
  solver_s.ONLY_GLOBAL_OPTIMIZATION = false;

  for (int i = 0; i < N; i++) {
    double dt = pow(10.0, -i);
    solver_t solver(f, h, opt_s, solver_s);
    solver.solve_daeo(0, 1, dt, 1.0, p,
                      fmt::format("se_tracking_noevents_10_minus{:d}", i));
  }

  /**
   * TRACK ONLY GLOBAL, ALWAYS REOPTIMIZE
   */
  solver_s.SEARCH_FREQUENCY = 1;
  solver_s.ONLY_GLOBAL_OPTIMIZATION = true;
  solver_s.EVENT_DETECTION_AND_CORRECTION = true;

  for (int i = 0; i < N; i++) {
    double dt = pow(10.0, -i);
    solver_t solver(f, h, opt_s, solver_s);
    solver.solve_daeo(0, 1, dt, 1.0, p,
                      fmt::format("se_tracking_onlyglobal_10_minus{:d}", i));
  }
}

int main(int argc, char **argv) {
  /* Implement the scenario described in
   *  Numerical simulation of differential-algebraic equations
   *    with embedded global optimization criteria [Deussen, Hüser, Naumann]
   */
  fmt::println("double epsilon is {:.6e}",
               std::numeric_limits<double>::epsilon());
  fmt::println("*** optimizer correctness test ***");
  // auto h = [](const auto t, const auto x, const auto &y, const auto &p) -> auto{
  //   return pow(p(1) - pow(y(0), 2), 2) - (x - p(2)) * sin(y(0) * p(3));
  // };
  // BNBOptimizerSettings<double> opt_s;
  // opt_s.LOGGING_ENABLED = false;
  // Eigen::Vector<double, 1> ll {-8.0};
  // Eigen::Vector<double, 1> ur {8.0};
  // BNBLocalOptimizer<decltype(h), double, suggested_solver_policies<double>, 1, 4> opt(h, ll, ur, opt_s);
  // Eigen::Vector<double, 4> p(2.0, 1.0, 0.5, pi / 2);
  // auto res = opt.find_minima_at(0.0, 1.0, p, false);
  // for (auto &y : res.minima_intervals){
  //   fmt::println("{::.4e}", y);
  // }
  
  fmt::println("*** simple example time ***");
  simple_example_perf_study(6);
  fmt::println("*** griewank time ***");
  //griewank_example_event_tolerance_study();
}
