/**
 * @file settings.hpp
 * @author Sasha [fleming@stce.rwth-aachen.de]
 * @brief Defines the configuration and logging options for @ref local_optima_bnp.hpp and @ref daeo_solver.hpp
 */

#ifndef _SOLVER_SETTINGS_HPP
#define _SOLVER_SETTINGS_HPP

#include <chrono>
#include <fstream>
#include <ranges>
#include <string>
#include <vector>

#include "boost/numeric/interval.hpp"

template <typename T>
using suggested_solver_policies = boost::numeric::interval_lib::policies<
    boost::numeric::interval_lib::save_state<
        boost::numeric::interval_lib::rounded_transc_std<T>>,
    boost::numeric::interval_lib::checking_base<T>>;

template <typename NUMERIC_T>
struct BNBOptimizerSettings
{
    NUMERIC_T TOL_X;
    NUMERIC_T TOL_Y;

    std::size_t MAXITER = 100;
    std::size_t MAX_REFINE_ITER = 4;

    bool LOGGING_ENABLED = true;
};

template <typename NUMERIC_T>
struct DAEOSolverSettings
{
    NUMERIC_T TOL_T;

    NUMERIC_T y0_min;
    NUMERIC_T y0_max;

    size_t SEARCH_FREQUENCY = 10;
    size_t MAX_NEWTON_ITERATIONS = 20;
    NUMERIC_T NEWTON_EPS = 1.0e-6;

    bool RUN_GLOBAL_OPT_AT_EVENTS = false;
    bool LOGGING_ENABLED = true;
};

#endif