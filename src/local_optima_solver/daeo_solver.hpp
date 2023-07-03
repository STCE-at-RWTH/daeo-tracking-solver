/**
 * @file daeo_solver.hpp
 * @author Sasha [fleming@stce.rwth-aachen.de]
 */
#ifndef _DAEO_SOLVER_HPP
#define _DAEO_SOLVER_HPP

#include "local_optima_solver.hpp"

template <typename XZERO, typename XPRIME, typename OBJECTIVE, typename NUMERIC_T>
class DAEOTrackingSolver
{
    XZERO const &m_initial_condition;
    XPRIME const &m_xprime;
    OBJECTIVE const &m_objective;

    LocalOptimaBNBSolver<OBJECTIVE, NUMERIC_T, suggested_solver_policies> m_local_optimizer;
    



};

#endif