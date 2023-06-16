/**
 * @file bnb_settings.hpp
 * @author Sasha [fleming@stce.rwth-aachen.de]
 * @brief Defines the configuration and print options for @ref local_optima_bnp.hpp 
*/

#ifndef _BNB_SETTINGS_HPP
#define _BNB_SETTINGS_HPP

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