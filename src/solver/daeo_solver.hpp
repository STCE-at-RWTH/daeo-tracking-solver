/**
 * @file daeo_solver.hpp
 * @author Sasha [fleming@stce.rwth-aachen.de]
 */
#ifndef _DAEO_SOLVER_HPP
#define _DAEO_SOLVER_HPP

#include <algorithm>
#include <chrono>
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
    using optimizer_t = BNBLocalOptimizer<OBJECTIVE, NUMERIC_T, suggested_solver_policies<NUMERIC_T>, YDIMS, NPARAMS>;
    using y_t = optimizer_t::y_t;
    using y_hessian_t = optimizer_t::y_hessian_t;
    using params_t = optimizer_t::params_t;
    using interval_t = optimizer_t::interval_t;

    DAEOSolverSettings<NUMERIC_T> settings;

private:
    DAEOWrappedFunction<XPRIME> const m_xprime;
    DAEOWrappedFunction<OBJECTIVE> const m_objective;
    optimizer_t m_optimizer;

public:
    DAEOTrackingSolver(XPRIME const &t_xprime, OBJECTIVE const &t_objective,
                       BNBOptimizerSettings<NUMERIC_T> &t_opt_settings,
                       DAEOSolverSettings<NUMERIC_T> &t_global_settings)
        : settings(t_global_settings), m_xprime(t_xprime), m_objective(t_objective),
          m_optimizer(t_objective, t_opt_settings)

    {
        m_optimizer.set_search_domain(y_t{settings.y0_min}, y_t{settings.y0_max});
    }

    // this needs to be somewhere else
    std::tuple<y_t, size_t> find_optimum(vector<y_t> const &y_k,
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
        return {y_k[i_star], i_star};
    }

    vector<y_t> y_k_medians(typename optimizer_t::results_t const &optres, NUMERIC_T const t,
                            NUMERIC_T const x, params_t const &params)
    {
        vector<y_t> y_medians;
        for (auto &y_i : optres.minima_intervals)
        {
            y_medians.emplace_back(y_i.size());
            y_medians.back() = y_i.unaryExpr([](auto ival)
                                             { return median(ival); });
        }
        return y_medians;
    }

    size_t estimate_steps_without_gopt(NUMERIC_T t, NUMERIC_T dt, NUMERIC_T x,
                                       vector<y_t> y_k, vector<y_t> dydt,
                                       params_t p)
    {
        size_t N_est = std::numeric_limits<size_t>::max();
        vector<NUMERIC_T> dhdt(y_k.size());
        vector<NUMERIC_T> h_k(y_k.size());
        for (size_t i = 0; i < dhdt.size(); i++)
        {
            h_k[i] = m_objective.value(t, x, y_k[i], p);
            // compute total derivative from partial h partial x and partial h partial y
            dhdt[i] = m_objective.grad_x(t, x, y_k[i], p) * m_xprime.value(t, x, y_k[i], p) +
                      m_objective.grad_y(t, x, y_k[i], p).dot(dydt[i]);
        }
        for (size_t i = 0; i < dhdt.size(); i++)
        {
            for (size_t j = i + 1; j < dhdt.size(); j++)
            {
                // compare each pair of minima
                NUMERIC_T dist = abs(h_k[i] - h_k[j]);              // distance between
                NUMERIC_T rate = abs(dhdt[i]) + abs(dhdt[j]);       // maximum rate of decrease
                size_t n = static_cast<size_t>((dist / rate / dt)); // we want to truncate here!
                if (n < N_est)
                {
                    N_est = n;
                }
            }
        }
        return N_est;
    }

    std::tuple<vector<NUMERIC_T>, vector<NUMERIC_T>> solve_daeo(NUMERIC_T const t0, NUMERIC_T const t_end,
                                                                NUMERIC_T const dt0, NUMERIC_T const x0,
                                                                params_t const &params, std::string tag = "")
    {
        using clock = std::chrono::high_resolution_clock;
        NUMERIC_T t = t0, dt = dt0;
        NUMERIC_T x_next, x = x0;
        NUMERIC_T h_star;

        vector<y_t> y_k, y_k_next;
        y_t y_star;
        size_t i_star, i_star_next;

        vector<NUMERIC_T> xs({x0});
        vector<NUMERIC_T> ts({t0});

        DAEOSolverLogger logger(tag);
        if (settings.LOGGING_ENABLED)
        {
            logger.log_computation_begin(clock::now(), 0, t, dt, x);
        }

        // We need to take one time step to estimate dy_k/dt
        fmt::println("Starting to solve DAEO at t={:.4e} with x={:.4e}", t, x0);
        typename optimizer_t::results_t opt_res = m_optimizer.find_minima_at(t, x, params, false);

        fmt::println("  BNB optimizer yields candidates for y at {:::.4e}", opt_res.minima_intervals);
        y_k = y_k_medians(opt_res, t, x, params);
        std::tie(y_star, i_star) = find_optimum(y_k, t, x, params);
        if (settings.LOGGING_ENABLED)
        {
            logger.log_global_optimization(clock::now(), 0, t, x, y_k, i_star);
        }

        // next portion relies on the assumption that two minima of h don't "cross paths" inside of a time step
        // even if they did, would it really matter? since we don't do any implicit function silliness
        // we probably wouldn't even be able to tell if this did happen
        // it may be beneficial to periodically check all of the y_is and see if they're close to each other before and after
        // solving for the values at the next time step.
        // This would trigger a search for minima of h(x, y) again, since we may have "lost" one

        size_t iter = 0;
        size_t iterations_since_search = 0;
        while (t < t_end)
        {
            fmt::println("{:d} Optimum at h({:.2e}, {:.2e}, {::.4e}, {::.2e}) = {:.4e}",
                         iter, t, x, y_star, params, h_star);
            fmt::println("    this occurs at index {:d}", i_star);
            std::tie(x_next, y_k_next) = solve_G_is_zero(t, dt0, x, i_star, y_k, params);
            fmt::println("  x_next is {:.4e}, y_next is {:::.4e}", x_next, y_k_next);
            vector<y_t> dydt(estimate_dydt(dt, y_k, y_k_next));
            fmt::println("  estimated dydt is {:::.4e}", dydt);
            fmt::println("  number of steps to optimum change is guessed to be {:d}",
                         estimate_steps_without_gopt(t, dt, x, y_k_next, dydt, params));
            std::tie(y_star, i_star_next) = find_optimum(y_k_next, t + dt, x_next, params);
            fmt::println("  event fn. value is {:.6e}", event_function(t + dt, x_next, i_star, i_star_next, y_k, params));
            if ((i_star_next != i_star) &&
                (event_function(t + dt, x_next, i_star, i_star_next, y_k, params) >= std::numeric_limits<NUMERIC_T>::epsilon()))
            {
                fmt::println("  **EVENT OCCURRED IN ITERATION {:d}! SOLVE EVENT FUNCTION HERE**", iter);
                fmt::println("  **REWINDING TO t={:6e}**", t);
                auto [t_event, x_event, i_star_event, y_event] = locate_event(t, dt, x, i_star, y_k, params);
                dydt = estimate_dydt(dt, y_k, y_event);
                std::tie(std::ignore, i_star) = find_optimum(y_event, t_event, x_event, params);
                fmt::println("  Located Event at t={:.6e}, x={:.4e}, i*={:d}", t_event, x_event, i_star);
                if (settings.LOGGING_ENABLED)
                {
                    logger.log_event_correction(clock::now(), iter, t, t_event - t, x, x_event - x, y_k, dydt, i_star);
                }
                t = t_event;
                y_k = y_event;
                x = x_event;
                i_star = i_star_event;
            }
            else
            {
                // we don't need to handle events, we can move on.
                if (settings.LOGGING_ENABLED)
                {
                    logger.log_time_step(clock::now(), iter, t, dt, x, x_next - x, y_k, dydt, i_star);
                }
                i_star = i_star_next;
                t += dt;
                x = x_next;
                y_k = y_k_next;
            }
            xs.push_back(x);
            ts.push_back(t);

            iter++;
            iterations_since_search++;
        }
        if (settings.LOGGING_ENABLED)
        {
            logger.log_computation_end(clock::now(), iter, t, x, y_k, i_star);
        }
        return {ts, xs};
    }

    vector<y_t> estimate_dydt(NUMERIC_T dt, vector<y_t> const &y, vector<y_t> const &y_next)
    {
        vector<y_t> dydt(y.size());
        for (size_t i = 0; i < y.size(); i++)
        {
            dydt[i] = (y_next[i] - y[i]) / dt;
        }
        return dydt;
    }

    /**
     * @brief locate a global optimum change event in the interval [t, t+dt]
     * @param[in] t
     * @param[in] dt
     * @param[in] x Value of x at time t
     * @param[in] y_k List of local optima y_k at time t
     * @param[in] params
     */
    std::tuple<NUMERIC_T, NUMERIC_T, size_t, vector<y_t>> locate_event(NUMERIC_T const t, NUMERIC_T const dt,
                                                                       NUMERIC_T const x, size_t const i_star,
                                                                       vector<y_t> const &y_k, params_t const &params)
    {
        // first, locate the event.
        NUMERIC_T t_split = t;
        NUMERIC_T dt_split = dt / 2;
        NUMERIC_T x_split;
        vector<y_t> y_k_split;
        y_t y_star_split;
        size_t i_star_split;

        while (dt_split > settings.TOL_T)
        {
            // possible optimization here: reduce size of G by only choosing two yk
            std::tie(x_split, y_k_split) = solve_G_is_zero(t_split, dt_split, x, i_star, y_k, params);
            std::tie(y_star_split, i_star_split) = find_optimum(y_k_split, t_split + dt_split, x_split, params);
            NUMERIC_T H_split = event_function(t_split + dt_split, x_split, i_star, i_star_split, y_k_split, params);
            fmt::println("  t = {:.6e}, dt={:.4e}, x={:.4e}, H={:.6e}, i={:d}",
                         t_split, dt_split, x_split, H_split, i_star_split);
            if (fabs(H_split) <= std::numeric_limits<NUMERIC_T>::epsilon())
            {
                // exactly located event
                t_split += dt_split;
                break;
            }
            else if (H_split < 0)
            {
                // t+dt_split < t_event
                t_split += dt_split;
                dt_split = dt_split / 2;
            }
            else
            {
                // t+dt_split > t_event
                dt_split = dt_split / 2;
            }
        }
        return {t_split, x_split, i_star_split, y_k_split};
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
        int ydims = y0[0].rows();
        // y0 is each of the possible minima we are considering.
        // if we only consider one, y0 has only one item (this is the case with global optimization every step)
        Eigen::VectorX<NUMERIC_T> result(1 + y0.size() * y0[0].rows());
        result(0) = x0 - x1 + dt / 2 * (m_xprime.value(t1, x1, y1[i_star], p) + m_xprime.value(t, x0, y0[i_star], p));
        for (size_t i = 0; i < y0.size(); i++)
        {
            result(Eigen::seqN(1 + i * ydims, ydims)) = m_objective.grad_y(t1, x1, y1[i], p);
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

    /**
     * @brief find @c x_next and @c y_k_next such that G(...) = 0 at t+dt.
     */
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

        while (iter < settings.MAX_NEWTON_ITERATIONS)
        {
            g_temp = G(t, dt, x, x_next, i_star, y_k, y_next, p);
            delg_temp = delG(t + dt, dt, x_next, i_star, y_next, p);
            diff = delg_temp.colPivHouseholderQr().solve(g_temp);
            x_next = x_next - diff(0);
            for (size_t i = 0; i < y_k.size(); i++)
            {
                y_next[i] = y_next[i] - diff(Eigen::seqN(1 + i * y_k[0].rows(), y_k[0].rows()));
            }
            if (diff.norm() < settings.NEWTON_EPS)
            {
                break;
            }
            iter++;
        }

        return {x_next, y_next};
    }

    /*
        H and dHdt are used to find the precise event location by newton iteration.
    */

    /**
     * @brief Event function between optima @c i1 (current optimum at (t, x))
     *   and the candidate next optimum @c i2 at (t, x)
     * @param[in] t
     * @param[in] x
     * @param[in] i1 Index of the first local optimum in @c y_k
     * @param[in] i2 Index of the second local optimum in @c y_k
     * @param[in] y_k List of all local optima @c y_k
     * @param[in] p Parameter vector.
     */
    NUMERIC_T event_function(NUMERIC_T const t, NUMERIC_T const x,
                             size_t i1, size_t i2,
                             vector<y_t> const &y_k, params_t const &p)
    {
        return m_objective.value(t, x, y_k[i1], p) - m_objective.value(t, x, y_k[i2], p);
    }
};

#endif