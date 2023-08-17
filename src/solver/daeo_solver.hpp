/**
 * @file daeo_solver.hpp
 * @author Sasha [fleming@stce.rwth-aachen.de]
 */
#ifndef _DAEO_SOLVER_HPP
#define _DAEO_SOLVER_HPP

#include <limits>
#include <vector>

#include "boost/numeric/interval.hpp"
#include "dco.hpp"
#include "fmt/format.h"
#include "fmt/ranges.h"

#include "fmt_extensions/interval.hpp"
#include "settings.hpp"
#include "utils/io.hpp"
#include "utils/matrices.hpp"

#include "local_optima_solver.hpp"

using boost::numeric::median;
using std::vector;

template <typename XPRIME, typename OBJECTIVE, typename NUMERIC_T>
class DAEOTrackingSolver
{
    size_t m_stepnum = 0;
    NUMERIC_T m_t = 0;

    vector<NUMERIC_T> const m_x0;
    vector<NUMERIC_T> m_x;
    vector<NUMERIC_T> const m_params;

    XPRIME const &m_xprime;
    OBJECTIVE const &m_objective;
    DAEOSolverSettings<NUMERIC_T> m_settings;

    LocalOptimaBNBSolver<OBJECTIVE, NUMERIC_T, suggested_solver_policies<NUMERIC_T>> m_optimizer;

public:
    DAEOTrackingSolver(vector<NUMERIC_T> &t_x0, XPRIME const &t_xprime, OBJECTIVE const &t_objective, vector<NUMERIC_T> const &t_params,
                       BNBSolverSettings<NUMERIC_T> &t_opt_settings, DAEOSolverSettings<NUMERIC_T> &t_global_settings)
        : m_x0(t_x0), m_xprime(t_xprime), m_objective(t_objective),
          m_params(t_params), m_optimizer(t_objective, t_opt_settings), m_settings(t_global_settings)
    {
        m_optimizer.set_search_domain({{m_settings.y0_min, m_settings.y0_max}});
    }

    inline NUMERIC_T t()
    {
        return m_t;
    }

    inline vector<NUMERIC_T> x()
    {
        return m_x;
    }

    void step_daeo()
    {
        auto minimizer_res = m_optimizer.find_minima_at(m_t, m_x, m_params);

        NUMERIC_T h;
        NUMERIC_T h_star = std::numeric_limits<NUMERIC_T>::max();
        NUMERIC_T y_star;

        for( auto& y : minimizer_res.minima_intervals){
            h = m_objective(m_t, m_x, median(y), m_params);
            if(h < h_star){
                h_star = h;
                y_star = y;
            }
        }


    }
};

#endif