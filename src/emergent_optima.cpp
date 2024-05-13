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
    boost::numeric::interval<T, suggested_interval_policies<T>>;
using double_ival = boost_interval_transc_t<double>;

void griewank_example_event_tolerance_study() {

  BNBOptimizerSettings<double> opt_s;
  opt_s.TOL_Y = 1.0e-10;
  opt_s.LOGGING_ENABLED = false;

  DAEOSolverSettings<double> settings_lowtol;
  settings_lowtol.LINEARIZE_OPTIMIZER_DRIFT = false;
  settings_lowtol.NEWTON_EPS = 1.0e-10;
  settings_lowtol.EVENT_DETECTION_EPS = 1.0e-4;
  settings_lowtol.y0_min = -10.0;
  settings_lowtol.y0_max = 10.0;
  settings_lowtol.SEARCH_FREQUENCY = 100;

  auto settings_lowtol_noevents = settings_lowtol;
  settings_lowtol_noevents.EVENT_DETECTION_AND_CORRECTION = false;

  auto settings_hightol = settings_lowtol;
  //settings_hightol.EVENT_DETECTION_EPS = 5.0e-4;

  auto settings_hightol_noevents = settings_hightol;
  settings_hightol_noevents.EVENT_DETECTION_AND_CORRECTION = false;

  auto f = [](const auto t, const auto x, const auto &y, const auto &p) -> auto{
    return y(0);
  };

  auto h = [](const auto t, const auto x, const auto &y, const auto &p) -> auto{
    return  (pow(x - y(0), 2) + p(0)*sin(p(1) * y(0)));
  };

  // using optimizer_t = BNBLocalOptimizer<decltype(h), T,
  // suggested_solver_policies<T>, NUM_Y_DIMS, NUM_PARAMS>;
  using solver_t = DAEOTrackingSolver<decltype(f), decltype(h), double, 1, 2>;
  typename solver_t::params_t p{1.0, 5.0};

  solver_t solver1(f, h, opt_s, settings_lowtol);
  solver_t solver2(f, h, opt_s, settings_lowtol_noevents);
  solver_t solver3(f, h, opt_s, settings_hightol);
  solver_t solver4(f, h, opt_s, settings_hightol_noevents);

  solver1.solve_daeo(0., 2.0, 0.0001, 1.0, p, "griewank_example_lowtol");
  // solver2.solve_daeo(0., 2.0, 0.0005, 1.0, p, "griewank_example_lowtol_noevents");
  // solver3.solve_daeo(0., 1.25, 0.0001, 1.0, p, "griewank_example_hightol");
  // solver4.solve_daeo(0., 2.0, 0.0005, 1.0, p, "griewank_example_hightol_noevents");
}

int main(int argc, char **argv) {
  /* Implement the scenario described in
   *  Numerical simulation of differential-algebraic equations
   *    with embedded global optimization criteria [Deussen, Hüser, Naumann]
   */
  griewank_example_event_tolerance_study();
}
