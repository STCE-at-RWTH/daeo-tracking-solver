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

#include "settings.hpp"
#include "local_optima_solver.hpp"
#include "utils/fmt_extensions.hpp"

using boost::numeric::median;
using std::vector;

template <typename XPRIME, typename OBJECTIVE, typename NUMERIC_T, int YDIMS, int NPARAMS>
class DAEOTrackingSolver
{
public:
    using optimizer_t = LocalOptimaBNBSolver<OBJECTIVE, NUMERIC_T, suggested_solver_policies<NUMERIC_T>, YDIMS, NPARAMS>;
    using y_t = optimizer_t::y_t;
    using y_hessian_t = optimizer_t::y_hessian_t;
    using params_t = optimizer_t::params_t;
    using interval_t = optimizer_t::interval_t;

private:
    DAEOWrappedFunction<XPRIME> const m_xprime;
    DAEOWrappedFunction<OBJECTIVE> const m_objective;
    optimizer_t m_optimizer;
    DAEOSolverSettings<NUMERIC_T> m_settings;

public:
    DAEOTrackingSolver(XPRIME const &t_xprime, OBJECTIVE const &t_objective,
                       BNBSolverSettings<NUMERIC_T> &t_opt_settings,
                       DAEOSolverSettings<NUMERIC_T> &t_global_settings)
        : m_xprime(t_xprime), m_objective(t_objective),
          m_optimizer(t_objective, t_opt_settings), m_settings(t_global_settings)
    {
        m_optimizer.set_search_domain(typename optimizer_t::y_interval_t(interval_t{m_settings.y0_min, m_settings.y0_max}));
    }

    // this needs to be somewhere else
    std::tuple<NUMERIC_T, y_t, size_t> find_optimum(vector<y_t> const &y_k,
                                                    NUMERIC_T const t, NUMERIC_T const x,
                                                    params_t const &params)
    {
        NUMERIC_T h_star = std::numeric_limits<NUMERIC_T>::max();
        size_t i_star;
        for (size_t i = 0; i < y_k.size(); i++)
        {
            NUMERIC_T h = m_objective.value(t, x, y_k[i], params);
            if (h < h_star)
            {
                h_star = h;
                i_star = i;
            }
        }
        return {h_star, y_k[i_star], i_star};
    }

    vector<y_t> y_k_medians(typename optimizer_t::results_t const &optres, NUMERIC_T const t,
                            NUMERIC_T const x, params_t const &params)
    {
        vector<y_t> y_medians;
        for (auto &y_i : optres.minima_intervals)
        {
            y_medians.emplace_back();
            y_medians.back().resize(y_i.size());
            std::transform(y_i.begin(), y_i.end(), y_medians.back().begin(),
                           [](auto ival)
                           { return median(ival); });
        }
        return y_medians;
    }

    size_t estimate_steps_without_gopt(vector<y_t> y_k, vector<y_t> dydt_est, NUMERIC_T dt)
    {
        size_t N_est = std::numeric_limits<size_t>::max();
        for(size_t i = 0; i<dydt_est.size(); i++){
            for(size_t j = i+1; j<dydt_est.size(); j++){
                // for every pair of y_ks, see how long before they run into each other
                y_t drift_steps = Eigen::abs(dydt_est[i])
            }
        }
        return N_est
    }

    std::tuple<vector<NUMERIC_T>, vector<NUMERIC_T>> solve_daeo(NUMERIC_T const t0, NUMERIC_T const t_end,
                                                                NUMERIC_T const dt0, NUMERIC_T const x0,
                                                                params_t const &params)
    {
        NUMERIC_T t = t0;
        NUMERIC_T x = x0;
        NUMERIC_T dt = dt0;

        vector<NUMERIC_T> xs({x0});
        vector<NUMERIC_T> ts({t0});

        NUMERIC_T h_star;
        vector<y_t> y_k;
        y_t y_star;
        size_t i_star;

        NUMERIC_T x_next;
        vector<y_t> y_k_next;
        size_t i_star_next;

        // We need to take one time step to estimate dy_k/dt
        fmt::println("Starting to solve DAEO at t={:.4e} with x={:.4e}", t, x0);
        typename optimizer_t::results_t bnb_results_0 = m_optimizer.find_minima_at(t, x, params, true);
        fmt::println("  BNB optimizer yields candidates for y at {:::.4e}", bnb_results_0.minima_intervals);
        y_k = y_k_medians(bnb_results_0, t, x, params);
        std::tie(h_star, y_star, i_star) = find_optimum(y_k, t, x, params);
        fmt::println("  Optimum is determined to be h({:.2e}, {:.2e}, {::.4e}, {::.2e}) = {:.4e}", t, x, y_star, params, h_star);
        fmt::println("    this occurs at index {:d}", i_star);
        std::tie(x_next, y_k_next) = solve_G_is_zero(t, dt0, x, i_star, y_k, params);
        fmt::println("  x_next is {:.4e}, y_next is {:::.4e}", x_next, y_k_next);

        vector<y_t> dydt_est(y_k.size());
        for (size_t i = 0; i < dydt_est.size(); i++)
        {
            dydt_est[i] = (y_k_next[i] - y_k[i]) / dt;
        }
        y_k = y_k_next;
        fmt::println("  estimated dydt is {:::.4e}", dydt_est);
        t += dt;
        x = x_next;
        xs.push_back(x);
        ts.push_back(t);

        // next portion relies on the assumption that two minima of h don't "cross paths" inside of a time step
        // even if they did, would it really matter? since we don't do any implicit function silliness
        // we probably wouldn't even be able to tell if this did happen

        // it may be beneficial to periodically check all of the y_is and see if they're close to each other before and after
        // solving for the values at the next time step.
        // This would trigger a search for minima of h(x, y) again, since we may have "lost" one

        size_t iter = 1;
        size_t iterations_since_search = 0;
        while (t < t_end)
        {
            std::tie(h_star, y_star, i_star_next) = find_optimum(y_k, t, x_next, params);
            if (i_star_next != i_star)
            {
                fmt::println("  **EVENT OCCURRED IN ITERATION {:d}! WE MUST REWIND**", iter);
            }
            // after handling events, we can move on.
            i_star = i_star_next;
            fmt::println("  Optimum is determined to be h({:.2e}, {:.2e}, {::.4e}, {::.2e}) = {:.4e}", t, x, y_star, params, h_star);
            std::tie(x_next, y_k_next) = solve_G_is_zero(t, dt, x, i_star, y_k, params);
            fmt::println("  x_next is {:.4e}, y_next is {:::.4e}", x_next, y_k_next);
            t += dt;
            x = x_next;
            y_k = y_k_next;
            xs.push_back(x);
            ts.push_back(t);

            iter++;
            iterations_since_search++;
        }
        return {ts, xs};
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

    /**
     * @brief Compute G. We are searching for x1, y1 s.t. G(...) = 0.
     * @param[in] t Start of the current time step
     * @param[in] dt Current time step size
     * @param[in] x0 Value of x at the beginning of the time step
     * @param[in] x1 "Guess" value of x at t+dt
     * @param[in] y0 Value of y at the beginning of the time step
     * @param[in] y1 "Guess" value of y at t+dt
     * @param[in] p Parameter vector.
     */
    Eigen::VectorX<NUMERIC_T> G(NUMERIC_T const t, NUMERIC_T const dt,
                                NUMERIC_T const x0, NUMERIC_T const x1,
                                size_t const i_star,
                                vector<y_t> const &y0, vector<y_t> const &y1, params_t const &p)
    {
        NUMERIC_T t1 = t + dt;
        // y0 is each of the possible minima we are considering.
        // if we only consider one, y0 has only one item (this is the case with global optimization every step)
        Eigen::VectorX<NUMERIC_T> result(1 + y0.size() * y0[0].rows());
        result(0) = x0 - x1 + dt / 2 * (m_xprime.value(t1, x1, y1[i_star], p) + m_xprime.value(t, x0, y0[i_star], p));
        for (size_t i = 0; i < y0.size(); i++)
        {
            result(Eigen::seqN(1 + i * y0[0].rows(), y0[0].rows())) = m_objective.grad_y(t1, x1, y1[i], p);
        }
        return result;
    }

    /**
     * @brief Gradient of the function used for newton iteration.
     * @param[in] t_next The "next" point in time where we are solving for x and y
     * @param[in] dt The time step size at k-1
     * @param[in] x
     * @param[in] y_k All possible minima we are considering at the future time point.
     * @param[in] p
     */
    Eigen::MatrixX<NUMERIC_T> delG(NUMERIC_T const t_next, NUMERIC_T const dt,
                                   NUMERIC_T const x, size_t i_star,
                                   vector<y_t> const &y_k, params_t const &p)
    {
        using Eigen::seqN;

        int ysize = y_k[0].rows();
        int ndims = 1 + y_k.size() * ysize;
        Eigen::MatrixX<NUMERIC_T> result(ndims, ndims);
        result(0, 0) = -1 + dt / 2 * m_xprime.grad_x(t_next, x, y_k[i_star], p);
        for (size_t i = 0; i < y_k.size(); i++)
        {
            result(0, seqN(1 + i * ysize, ysize)) = dt / 2 * m_xprime.grad_y(t_next, x, y_k[i], p);
            result(seqN(1 + i * ysize, ysize), 0) = m_objective.d2dxdy(t_next, x, y_k[i], p);
            result(seqN(1 + i * ysize, ysize), seqN(1 + i * ysize, ysize)) = m_objective.hess_y(t_next, x, y_k[i], p);
        }
        return result;
    }

    std::tuple<NUMERIC_T, vector<y_t>> solve_G_is_zero(NUMERIC_T const t, NUMERIC_T const dt,
                                                       NUMERIC_T const x, size_t i_star,
                                                       vector<y_t> const &y_k,
                                                       params_t const &p)
    {
        size_t iter = 0;
        int temp_dims = 1 + y_k.size() * y_k[0].rows();
        NUMERIC_T x_next = x;
        vector<y_t> y_next(y_k);
        Eigen::VectorX<NUMERIC_T> g_temp(temp_dims);
        Eigen::VectorX<NUMERIC_T> diff(temp_dims);
        Eigen::MatrixX<NUMERIC_T> delg_temp(temp_dims, temp_dims);

        while (iter < m_settings.MAX_NEWTON_ITERATIONS)
        {
            g_temp = G(t, dt, x, x_next, i_star, y_k, y_next, p);
            delg_temp = delG(t + dt, dt, x_next, i_star, y_next, p);
            diff = delg_temp.colPivHouseholderQr().solve(g_temp);
            x_next = x_next - diff(0);
            for (size_t i = 0; i < y_k.size(); i++)
            {
                y_next[i] = y_next[i] - diff(Eigen::seqN(1 + i * y_k[0].rows(), y_k[0].rows()));
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