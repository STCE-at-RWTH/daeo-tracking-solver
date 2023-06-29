/**
 * @file settings.hpp
 * @author Sasha [fleming@stce.rwth-aachen.de]
 * @brief Defines the configuration and logging options for @ref local_optima_bnp.hpp
 */

#ifndef _BNB_SETTINGS_HPP
#define _BNB_SETTINGS_HPP

#include <chrono>
#include <filesystem>

#include "boost/numeric/interval.hpp"

template <typename T>
using suggested_solver_policies = boost::numeric::interval_lib::policies<
    boost::numeric::interval_lib::save_state<
        boost::numeric::interval_lib::rounded_transc_std<T>>,
    boost::numeric::interval_lib::checking_base<T>>;

template <typename NUMERIC_T>
struct BNBSolverSettings
{
    NUMERIC_T TOL_X;
    NUMERIC_T TOL_Y;

    std::size_t MAXITER = 10000;
    std::size_t MAX_REFINE_ITER = 4;
};

enum BNB_EVENT
{
    INTERVAL_START,
    INTERVAL_SPLIT,
    INTERVAL_CONVERGED,
    INTERVAL_DISCARD,

    VALUE_TEST_PASS,
    VALUE_TEST_FAIL,

    GRADIENT_TEST_PASS,
    GRADIENT_TEST_FAIL,

    HESSIAN_TEST_PASS_SPD,
    HESSIAN_TEST_PASS_SND,
    HESSIAN_TEST_FAIL
};

template <typename NUMERIC_T>
class BNBSolverLogger
{
    size_t n_active_dims;
    size_t n_outputs;
    size_t n_params;

    std::filesystem::path logs_directory;
};

#endif