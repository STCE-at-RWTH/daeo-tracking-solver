/**
 * @file daeo_solver.hpp
 * @author Sasha [fleming@stce.rwth-aachen.de]
 */
#ifndef _DAEO_SOLVER_HPP
#define _DAEO_SOLVER_HPP

#include <limits>
#include <vector>

#include "boost/numeric/interval.hpp"
#include "Eigen/Dense"
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

    DAEOWrappedFunction<XPRIME> const m_xprime;
    DAEOWrappedFunction<OBJECTIVE> const m_objective;
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

    void solve_daeo(NUMERIC_T const t0, NUMERIC_T const t_end, NUMERIC_T const dt0, NUMERIC_T const x0)
    {
        // get inital yi and y*
        auto minimizer_res = m_optimizer.find_minima_at(m_t, m_x, m_params);
        NUMERIC_T h_star = std::numeric_limits<NUMERIC_T>::max();
        NUMERIC_T y_star;
        size_t i_star;
        for (size_t i = 0; i<minimizer_res.minima_intervals.size(); i++)
        {
            NUMERIC_T h = m_objective.value(m_t, m_x, median(minimizer_res.minima_intervals[i]), m_params);
            if (h < h_star)
            {
                h_star = h;
                y_star = median(minimizer_res.minima_intervals[i]);
                i_star = i;
            }
        }

        // next portion relies on the assumption that two minima of h don't "cross paths" inside of a time step
        // even if they did, would it really matter? since we don't do any implicit function silliness
        // we probably wouldn't even be able to tell if this did happen

        // it may be beneficial to periodically check all of the y_is and see if they're close to each other before and after
        // solving for the values at the next time step.
        // This would trigger a search for minima of h(x, y) again, since we may have "lost" one
    }

    /*
        G and delG are used by newton iteration to find x and y at the next time step.
        There is the condition on x from the trapezoidal rule:
        x_{k+1} = x_{k} + dt/2 * (f(x_{k+1}, y_{k+1}) + f(x_{k}, y_{k}))
        After the zeroth time step (t=t0), we have good initial guesses for the minima y_{k+1}
        (assuming they don't drift "too far")
        The other equations are provided by dh/dyi = 0 at x_{k+1} and y_{k+1}

        from this structure we compute G and delG in blocks, since we have derivative routines available for f=x' and h

    */

    vector<NUMERIC_T> G(NUMERIC_T const t, NUMERIC_T const dt,
                        NUMERIC_T const x0, NUMERIC_T const x1,
                        vector<NUMERIC_T> const &y0, vector<NUMERIC_T> const &y1, vector<NUMERIC_T> &p)
    {
        NUMERIC_T t1 = t + dt;
        vector<NUMERIC_T> result(1 + y0.size());
        result[0] = x0 - x1 + dt / 2 * (m_xprime.value(t1, x1, y1, p) + m_xprime.value(t, x0, y0, p));
        vector<NUMERIC_T> dh1dy(m_objective.ddy(t1, x1, y1, p));
        for (size_t i = 1; i < result.size(); i++)
        {
            result[i] = dh1dy[i - 1];
        }
        return result;
    }

    /**
     * @brief Gradient of the function used for newton iteration.
     * @param[in] t The "next" point in time where we are solving for x and y
     * @param[in] dt The time step size at k-1
     * @param[in] x
     * @param[in] y
     * @param[in] p
     */
    vector<vector<NUMERIC_T>> delG(NUMERIC_T const t, NUMERIC_T const dt,
                                   NUMERIC_T const x, vector<NUMERIC_T> const &y, vector<NUMERIC_T> &p)
    {
        size_t ndims = 1 + y.size();
        vector<vector<NUMERIC_T>> result(ndims, vector<NUMERIC_T>(ndims));
        result[0][0] = -1 + dt / 2 * m_xprime.ddx(t, x, y, p);
        vector<NUMERIC_T> A1(m_objective.d2dxdy(t, x, y));
        vector<NUMERIC_T> A2(m_xprime.ddy(t, x, y, p));
        vector<vector<NUMERIC_T>> B(m_objective.d2dy2());
        for (size_t j = 1; j < ndims; j++)
        {
            result[0][j] = A2[j - 1];
        }
        for (size_t i = 1; i < ndims; i++)
        {
            result[i][0] = A1[i - 1];
            for (size_t j = 0; j < ndims; j++)
            {
                result[i][j] = B[i - 1][j - 1];
            }
        }
        return result;
    }

    std::tuple<NUMERIC_T, vector<NUMERIC_T>> solve_G_is_zero(NUMERIC_T const t, NUMERIC_T const dt,
                                                             NUMERIC_T const x, vector<NUMERIC_T> const &y,
                                                             vector<NUMERIC_T> const &p)
    {
        size_t iter = 0;
        NUMERIC_T x_next = x;
        vector<NUMERIC_T> y_next(y);
        vector<NUMERIC_T> g_temp;
        vector<vector<NUMERIC_T>> delg_temp;

        using lhs_t = Eigen::Matrix<NUMERIC_T, Eigen::Dynamic, Eigen::Dynamic>;
        using rhs_t = Eigen::Matrix<NUMERIC_T, Eigen::Dynamic, 1>;

        while (iter < m_settings.MAX_NEWTON_ITERATIONS)
        {
            g_temp = G(t, dt, x, x_next, y, y_next, p);
            delg_temp = delG(t + dt, dt, x_next, y_next, p);

            // BAD
            // TODO use Eigen everywhere!
            lhs_t lhs(delg_temp.size(), delg_temp.size());
            rhs_t rhs(g_temp.size());
            for (size_t i = 0; i < delg_temp.size(); i++)
            {
                rhs(i) = g_temp[i];
                for (size_t j = 0; j < delg_temp.size(); j++)
                {
                    lhs(i, j) = delg_temp[i][j];
                }
            }
            auto diff = lhs.colPivHouseholderQr().solve(rhs);
            x_next = x_next - diff(0);
            for (size_t i = 1; i < g_temp.size(); i++)
            {
                y_next[i - 1] = y_next[i - 1] - diff(i);
            }
            if (diff.norm() < m_settings.NEWTON_EPS)
            {
                break;
            }

            iter++;
        }
    }
};

#endif