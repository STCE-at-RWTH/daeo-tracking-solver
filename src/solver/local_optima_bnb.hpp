/**
 * @file local_optima_bnb.hpp
 * @author Sasha [fleming@stce.rwth-aachen.de]
 * @brief Implementation of a branch and bound solver to find regions containing local optima of f(x; p).
*/
#ifndef _LOCAL_OPTIMA_BNB_HPP // header guard
#define _LOCAL_OPTIMA_BNB_HPP

#include <vector>
#include <tuple>

#include "dco.hpp"

/**
 * @class LocalOptimaSolver
 * @brief Finds regions containing local minima of f(x;p)
 * @tparam OBJECTIVE_T Type of the objective function.
 * @tparam NDIMS Number of dimensions of the active argument to the objective function.
 * @tparam INTERVAL_T Type of the intervals with which to perform interval arithmetic for minimization.
 * @tparam NPARAMS Size of the parameter vector 'p'
 * @tparam P_T Type of the parameters to f(x; p)
*/
template<typename OBJECTIVE_T, std::size_t NDIMS, typename INTERVAL_T, std::size_t NPARAMS, typename P_T>
class LocalOptimaSolver{

    /**
     * @brief The objective function f(x; p) of which to find the minima.
    */
    OBJECTIVE_T& m_objective;

    /**
     * @brief A list of all intervals (which may be N-dimensional) that contain the located local minima. 
    */
    std::vector<std::vector<INTERVAL_T>> m_minima;
    

    void run(std::vector<INTERVAL_T> domain, std::vector<P_T> params)
    {
        
    }
    
};

#endif // header guard