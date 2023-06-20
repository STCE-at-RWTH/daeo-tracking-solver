/**
 * @file bnb_settings.hpp
 * @author Sasha [fleming@stce.rwth-aachen.de]
 * @brief Defines the configuration and print options for @ref local_optima_bnp.hpp
 */

#ifndef _BNB_SETTINGS_HPP
#define _BNB_SETTINGS_HPP

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
};

template <typename NUMERIC_T>
class BNBSolverLogger
{
    
};

#endif