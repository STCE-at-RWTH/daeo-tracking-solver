/**
 * @file daeo_solver.hpp
 * @author Sasha [fleming@stce.rwth-aachen.de]
 */
#ifndef _DAEO_SOLVER_HPP
#define _DAEO_SOLVER_HPP

#include "local_optima_solver.hpp"

template <typename XPRIME, typename OBJECTIVE, typename NUMERIC_T>
class DAEOTrackingSolver
{
    size_t m_stepnum = 0;
    const NUMERIC_T m_dt;

    vector<NUMERIC_T> const &m_x0;
    vector<NUMERIC_T> m_x;

    XPRIME const &m_xprime;
    OBJECTIVE const &m_objective;

    LocalOptimaBNBSolver<OBJECTIVE, NUMERIC_T, suggested_solver_policies<NUMERIC_T>> m_local_optimizer;

public:
    DAEOTrackingSolver(NUMERIC_T t_dt, vector<NUMERIC_T> t_x0, XPRIME t_xprime,
                       OBJECTIVE t_objective, BNBSolverSettings t_local_settings)
        : m_dt(t_dt), m_x0(t_x0), m_xprime(t_xprime), m_objective(t_objective), m_local_optimizer(t_objective, t_local_settings)
    {
    }

    inline NUMERIC_T dt()
    {
        return m_dt;
    }

    inline NUMERIC_T t()
    {
        return m_dt * m_stepnum;
    }

    void step_daeo(size_t N_steps)
    {
    }
};

#endif