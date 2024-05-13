/**
 * @file settings.hpp
 * @author Sasha [fleming@stce.rwth-aachen.de]
 * @brief Defines the configuration and logging options for @ref
 * local_optima_solver.hpp and @ref daeo_solver.hpp
 */

#ifndef _SOLVER_SETTINGS_HPP
#define _SOLVER_SETTINGS_HPP

#include "boost/numeric/interval.hpp"

template <typename T>
using suggested_interval_policies = boost::numeric::interval_lib::policies<
    boost::numeric::interval_lib::save_state<
        boost::numeric::interval_lib::rounded_transc_std<T>>,
    boost::numeric::interval_lib::checking_base<T>>;

enum BNBOptimizerMode{
  FIND_ALL_LOCAL_MINIMIZERS,
  FIND_ONLY_GLOBAL_MINIMIZER
};

template <typename NUMERIC_T> struct BNBOptimizerSettings {
  BNBOptimizerMode MODE = FIND_ALL_LOCAL_MINIMIZERS;
  std::size_t MAXITER = 100'000;
  std::size_t MAX_REFINE_ITER = 1'000;

  NUMERIC_T TOL_Y = 1.0e-8;
  
  bool LOGGING_ENABLED = true;
  bool RETEST_CRITICAL_POINTS = false;
};

template <typename NUMERIC_T> struct DAEOSolverSettings {
  NUMERIC_T y0_min;
  NUMERIC_T y0_max;

  size_t SEARCH_FREQUENCY = 20;
  size_t MAX_NEWTON_ITERATIONS = 30;
  NUMERIC_T NEWTON_EPS = 1.0e-8;
  NUMERIC_T EVENT_DETECTION_EPS = 5.0e-6; // this may be computeable from limits
  NUMERIC_T EVENT_DRIFT_COEFF = 0.1;      // these might always be the same.

  bool TRACK_LOCAL_OPTIMA = true;
  bool EVENT_DETECTION_AND_CORRECTION = true;
  bool LINEARIZE_OPTIMIZER_DRIFT = true;
  bool LOGGING_ENABLED = true;
  bool ONLY_GLOBAL_OPTIMIZATION = false;
};

#endif