/**
 * @file daeo_solver.hpp
 * @author Sasha [fleming@stce.rwth-aachen.de]
 */
#ifndef _DAEO_SOLVER_HPP
#define _DAEO_SOLVER_HPP

#include <algorithm>
#include <limits>
#include <tuple>
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

    DAEOWrappedFunction<XPRIME> const m_xprime;
    DAEOWrappedFunction<OBJECTIVE> const m_objective;
    LocalOptimaBNBSolver<OBJECTIVE, NUMERIC_T, suggested_solver_policies<NUMERIC_T>> m_optimizer;
    DAEOSolverSettings<NUMERIC_T> m_settings;

public:
    DAEOTrackingSolver(XPRIME const &t_xprime, OBJECTIVE const &t_objective,
                       BNBSolverSettings<NUMERIC_T> &t_opt_settings,
                       DAEOSolverSettings<NUMERIC_T> &t_global_settings)
        : m_xprime(t_xprime), m_objective(t_objective),
          m_optimizer(t_objective, t_opt_settings), m_settings(t_global_settings)
    {
        m_optimizer.set_search_domain({{m_settings.y0_min, m_settings.y0_max}});
    }

    template <typename IT>
    vector<vector<NUMERIC_T>> median_helper(BNBSolverResults<NUMERIC_T, IT> const &b)
    {
        vector<vector<NUMERIC_T>> out;
        for (auto &y_i : b.minima_intervals)
        {
            out.emplace_back(y_i.size());
            std::transform(y_i.begin(), y_i.end(), out.back().begin(),
                           [](auto it)
                           { return median(it); });
        }
        return out;
    }

    template <typename IT>
    std::tuple<NUMERIC_T, vector<NUMERIC_T>, size_t> find_optimum_in_results(BNBSolverResults<NUMERIC_T, IT> const &b,
                                                                             NUMERIC_T const t, NUMERIC_T const x,
                                                                             vector<NUMERIC_T> const &params)
    {
        vector<vector<NUMERIC_T>> y_medians(median_helper(b));
        NUMERIC_T h_star = std::numeric_limits<NUMERIC_T>::max();
        size_t i_star;
        for (size_t i = 0; i < y_medians.size(); i++)
        {
            NUMERIC_T h = m_objective.value(t, x, y_medians[i], params);
            if (h < h_star)
            {
                h_star = h;
                i_star = i;
            }
        }
        return {h_star, y_medians[i_star], i_star};
    }

    void solve_daeo(NUMERIC_T const t0, NUMERIC_T const t_end,
                    NUMERIC_T const dt0, NUMERIC_T const x0,
                    vector<NUMERIC_T> const &params)
    {
        NUMERIC_T t = t0;
        NUMERIC_T x = x0;
        
        NUMERIC_T h_star;
        vector<NUMERIC_T> y_star;
        size_t i_star;
        
        fmt::println("Starting to solve DAEO at t={:.4e} with x={:.4e}", t, x0);
        auto bnb_results_0 = m_optimizer.find_minima_at(t, x, params, true);
        fmt::println("\nBNB optimizer yields candidates for y at {:::.4e}", bnb_results_0.minima_intervals);
        std::tie(h_star, y_star, i_star) = find_optimum_in_results(bnb_results_0, t, x, params);
        fmt::println("Optimum is determined to be h({:.2e}, {:.2e}, {::.4e}, {::.2e}) = {:.4e}\n", t, x, y_star, params, h_star);
        fmt::println("G is {::.4e}", G(t, dt0, x, x, y_star, y_star, params));
        auto dG = delG(t, dt0, x, y_star, params);
        fmt::println("delG is {:::.4e}", dG);
        auto [x_next, y_next] = solve_G_is_zero(t, dt0, x, y_star, params);
        fmt::println("x_next is {:.4e}, y_next is {::.4e}", x_next, y_next);

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
                        vector<NUMERIC_T> const &y0, vector<NUMERIC_T> const &y1, vector<NUMERIC_T> const &p)
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
                                   NUMERIC_T const x, vector<NUMERIC_T> const &y, vector<NUMERIC_T> const &p)
    {
        size_t ndims = 1 + y.size();
        vector<vector<NUMERIC_T>> result(ndims, vector<NUMERIC_T>(ndims));
        result[0][0] = -1 + dt / 2 * m_xprime.ddx(t, x, y, p);
        vector<NUMERIC_T> A1(m_objective.d2dxdy(t, x, y, p));
        vector<NUMERIC_T> A2(m_xprime.ddy(t, x, y, p));
        vector<vector<NUMERIC_T>> B(m_objective.d2dy2(t, x, y, p));
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
        using rhs_t = Eigen::Vector<NUMERIC_T, Eigen::Dynamic>;

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
            // fmt::println("LHS is {:d}x{:d}", lhs.rows(), lhs.cols());
            // fmt::println("RHS is {:d}x{:d}", rhs.rows(), rhs.cols());
            rhs_t diff = lhs.colPivHouseholderQr().solve(rhs);
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

        return {x_next, y_next};
    }
};

#endif