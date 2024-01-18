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
#include "fmt/format.h"
#include "fmt/ranges.h"

#include "settings.hpp"
#include "local_optima_solver.hpp"
#include "utils/fmt_extensions.hpp"
#include "utils/ntuple.hpp"

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

    struct DAEOSolutionState
    {
        NUMERIC_T t;
        NUMERIC_T x;
        size_t i_star;
        vector<y_t> y;

        DAEOSolutionState() = default;

        DAEOSolutionState(NUMERIC_T t_t, NUMERIC_T t_x, typename optimizer_t::results_t gopt_results, params_t p) : t{t_t}, x{t_x}
        {
            using boost::numeric::median;
            for (auto &y_i : gopt_results.minima_intervals)
            {
                y.emplace_back(y_i.size());
                y.back() = y_i.unaryExpr([](auto ival)
                                         { return median(ival); });
            }
        }

        DAEOSolutionState(DAEOSolutionState const &other) : t{other.t}, x{other.x}, i_star{other.i_star}, y(other.y){};

        friend void swap(DAEOSolutionState &a, DAEOSolutionState &b)
        {
            using std::swap;
            swap(a.t, b.t);
            swap(a.x, b.x);
            swap(a.i_star, b.i_star);
            swap(a.y, b.y);
        }

        DAEOSolutionState(DAEOSolutionState &&other) noexcept : DAEOSolutionState()
        {
            swap(*this, other);
        };

        DAEOSolutionState &operator=(DAEOSolutionState other)
        {
            swap(*this, other);
            return *this;
        }

        /**
         * @brief Return the number of local optima present at the this point in time.
         */
        inline size_t n_local_optima() const { return y.size(); }

        // Eigen should help with compile time optimization for this.
        // TODO make if constexpr(Eigen::Dynamic) if necessary.

        inline int ydims() const { return y[0].rows(); }

        // what the heck am I doing
        /**
         * @brief Get a (const) reference to the global optimum y*
         */
        inline y_t &y_star() { return y[i_star]; }
        inline const y_t &y_star() const { return y[i_star]; }
    };

    DAEOTrackingSolver(XPRIME const &t_xprime, OBJECTIVE const &t_objective,
                       BNBOptimizerSettings<NUMERIC_T> &t_opt_settings,
                       DAEOSolverSettings<NUMERIC_T> &t_global_settings)
        : m_xprime(t_xprime), m_objective(t_objective), settings(t_global_settings),
          m_optimizer(t_objective, y_t{settings.y0_min}, y_t{settings.y0_max}, t_opt_settings) {}

private:
    DAEOWrappedFunction<XPRIME> const m_xprime;
    DAEOWrappedFunction<OBJECTIVE> const m_objective;
    DAEOSolverSettings<NUMERIC_T> settings;
    optimizer_t m_optimizer;

    /**
     * @brief Update the index of the global optimum.
     */
    void update_optimum(DAEOSolutionState s, const params_t &p)
    {
        NUMERIC_T h_star = std::numeric_limits<NUMERIC_T>::max();
        for (size_t i = 0; i < s.n_local_optima(); i++)
        {
            NUMERIC_T h = m_objective.value(s.t, s.x, s.y[i], p);
            if (h < h_star)
            {
                h_star = h;
                s.i_star = i;
            }
        }
    }

    vector<y_t> compute_dy(vector<y_t> const &y, vector<y_t> const &y_next)
    {
        vector<y_t> dy(y.size());
        for (size_t i = 0; i < y.size(); i++)
        {
            dy[i] = (y_next[i] - y[i]);
        }
        return dy;
    }

    size_t estimate_steps_without_gopt(NUMERIC_T t, NUMERIC_T dt, NUMERIC_T x,
                                       vector<y_t> y_k, vector<y_t> dy,
                                       params_t p)
    {
        size_t N_est = std::numeric_limits<size_t>::max();
        Eigen::VectorX<NUMERIC_T> dhdt(y_k.size());
        Eigen::VectorX<NUMERIC_T> h_k(y_k.size());
        for (size_t i = 0; i < y_k.size(); i++)
        {
            h_k(i) = m_objective.value(t, x, y_k[i], p);
            // compute total derivative from partial h partial x and partial h partial y
            dhdt(i) = m_objective.grad_x(t, x, y_k[i], p) * m_xprime.value(t, x, y_k[i], p) +
                      m_objective.grad_y(t, x, y_k[i], p).dot(dy[i]) / dt;
        }
        for (size_t i = 0; i < y_k.size(); i++)
        {
            for (size_t j = i + 1; j < y_k.rows(); j++)
            {
                // compare each pair of minima
                NUMERIC_T dist = abs(h_k(i) - h_k(j));              // distance between
                NUMERIC_T rate = abs(dhdt(i)) + abs(dhdt(j));       // maximum rate of decrease
                size_t n = static_cast<size_t>((dist / rate / dt)); // we want to truncate here!
                if (n < N_est)
                {
                    N_est = n;
                }
            }
        }
        return N_est;
    }

public:
    ntuple<2, NUMERIC_T> solve_daeo(NUMERIC_T t, NUMERIC_T t_end,
                                    NUMERIC_T dt, NUMERIC_T x0,
                                    params_t const &params, std::string tag = "")
    {
        using clock = std::chrono::high_resolution_clock;
        DAEOSolverLogger logger(tag);
        if (settings.LOGGING_ENABLED)
        {
            logger.log_computation_begin(clock::now(), 0, t, dt, x0);
        }
        fmt::println("Starting to solve DAEO at t={:.4e} with x={:.4e}", t, x0);
        typename optimizer_t::results_t opt_res = m_optimizer.find_minima_at(t, x0, params, false);
        fmt::println("  BNB optimizer yields candidates for y at {:::.4e}", opt_res.minima_intervals);
        DAEOSolutionState current(t, x0, opt_res, params);
        update_optimum(current, params);
        DAEOSolutionState next;
        vector<y_t> dy;
        if (settings.LOGGING_ENABLED)
        {
            logger.log_global_optimization(clock::now(), 0, current.t, current.x, current.y, current.i_star);
        }

        // next portion relies on the assumption that two minima of h don't "cross paths" inside of a time step
        // even if they did, would it really matter? since we don't do any implicit function silliness
        // we probably wouldn't even be able to tell if this did happen
        // it may be beneficial to periodically check all of the y_is and see if they're close to each other before and after
        // solving for the values at the next time step.
        // This would trigger a search for minima of h(x, y) again, since we may have "lost" one

        size_t iter = 0;
        size_t iterations_since_search = 0;
        while (current.t < t_end)
        {
            next = integrate_daeo(current, dt, params);
            update_optimum(next, params);
            // dydt = estimate_dydt(dt, y_k, y_k_next);
            if (settings.EVENT_DETECTION_AND_CORRECTION && next.i_star != current.i_star)
            {
                fmt::println("**EVENT OCCURRED IN ITERATION {:d}! SOLVE EVENT FUNCTION HERE**", iter);
                fmt::println("**REWINDING TO t={:6e}**", t);
                DAEOSolutionState event = locate_and_integrate_to_event(current, next, params);
                fmt::println("  Event occurs at t={:.6e}, x={:.6e}", event.t, event.x);
                NUMERIC_T dt_event = event.t - current.t;
                dy = compute_dy(current.y, event.y);
                if (settings.LOGGING_ENABLED)
                {
                    logger.log_event_correction(clock::now(), iter, event.t, dt_event, event.x, event.x - current.x, event.y, dy, event.i_star);
                }
                // complete time step from event back to the grid.
                NUMERIC_T dt_grid = dt - dt_event;
                next = integrate_daeo(event, dt_grid, params);
                update_optimum(next, params);
                dy = compute_dy(event.y, next.y);
                if (settings.LOGGING_ENABLED)
                {
                    logger.log_time_step(clock::now(), iter, next.t, dt_grid, next.x, next.x - event.x, next.y, dy, next.i_star);
                }
            }
            else
            {
                dy = compute_dy(current.y, next.y);
                // we don't need to handle events, we can move on.
                if (settings.LOGGING_ENABLED)
                {
                    logger.log_time_step(clock::now(), iter, next.t, dt, next.x, next.x - current.x, next.y, dy, next.i_star);
                }
            }

            current = next;

            iter++;
            iterations_since_search++;
        }
        if (settings.LOGGING_ENABLED)
        {
            logger.log_computation_end(clock::now(), iter, current.t, current.x, current.y, current.i_star);
        }
        return {current.t, current.x};
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
     * @param[in] start Value of the solution trajectory at the beginning of a time step.
     * @param[in] end Value (potentially a guess) of the solution trajectory at the end of a time step.
     * @param[in] dt Current time step size.
     * @param[in] p Parameter vector.
     */
    Eigen::VectorX<NUMERIC_T> G(DAEOSolutionState const &start, DAEOSolutionState const &guess, NUMERIC_T const dt, params_t const &p)
    {
        int ydims = start.ydims();
        // fix i_star (assume no event in this part of the program)
        size_t i_star = start.i_star;
        Eigen::VectorX<NUMERIC_T> result(1 + start.n_local_optima() * ydims);
        result(0) = start.x - guess.x + dt / 2 * (m_xprime.value(start.t, start.x, start.y[i_star], p) + m_xprime.value(guess.t, guess.x, guess.y[i_star], p));
        for (size_t i = 0; i < start.n_local_optima(); i++)
        {
            result(Eigen::seqN(1 + i * ydims, ydims)) = m_objective.grad_y(guess.t, guess.x, guess.y[i], p);
        }
        return result;
    }

    /**
     * @brief Gradient of the function used for newton iteration.
     * @param[in] guess The guessed value of the solution trajectory.
     * @param[in] dt The time step size from the beginning of the time step to where @c guess is evaluated.
     * @param[in] p Parameter vector.
     */
    Eigen::MatrixX<NUMERIC_T> delG(DAEOSolutionState const &guess, NUMERIC_T const dt, params_t const &p)
    {
        using Eigen::seqN;
        int ydims = guess.ydims();
        int ndims = 1 + guess.n_local_optima() * ydims;
        Eigen::MatrixX<NUMERIC_T> result(ndims, ndims);
        result(0, 0) = -1 + dt / 2 * m_xprime.grad_x(guess.t, guess.x, guess.y[guess.i_star], p);
        for (size_t i = 0; i < guess.n_local_optima(); i++)
        {
            result(0, seqN(1 + i * ydims, ydims)) = dt / 2 * m_xprime.grad_y(guess.t, guess.x, guess.y[i], p);
            result(seqN(1 + i * ydims, ydims), 0) = m_objective.d2dxdy(guess.t, guess.x, guess.y[i], p);
            result(seqN(1 + i * ydims, ydims), seqN(1 + i * ydims, ydims)) = m_objective.hess_y(guess.t, guess.x, guess.y[i], p);
        }
        return result;
    }

    /**
     * @brief Integrate the differential part of the DAEO from time @c t to @c t+dt
     * @param[in] start The value of the solution trajectory at @c t=t0
     * @param[in] p Parameter vector.
     * @return The value of solution trajectory at @c t=t+dt.
     * @details Integrates the ODE using the trapezoidal rule. Additionally solves ∂h/∂y_k = 0
     * simultaenously. Uses Newton's method.
     */
    DAEOSolutionState integrate_daeo(DAEOSolutionState const &start, NUMERIC_T dt, params_t const &p)
    {
        // copy into our guess and advance time
        DAEOSolutionState next(start);
        next.t += dt;
        Eigen::VectorX<NUMERIC_T> g, delg, diff;
        size_t iter = 0;
        while (iter < settings.MAX_NEWTON_ITERATIONS)
        {
            g = G(start, next, dt, p);
            delg = delG(next, dt, p);
            diff = delg.colPivHouseholderQr().solve(g);
            next.x = next.x - diff(0);
            for (size_t i = 0; i < start.n_local_optima(); i++)
            {
                next.y[i] = next.y[i] - diff(Eigen::seqN(1 + i * start.ydims(), start.ydims()));
            }
            if (diff.norm() < settings.NEWTON_EPS)
            {
                break;
            }
            iter++;
        }
        return next;
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
    inline NUMERIC_T H(DAEOSolutionState const &state, size_t const i1, size_t const i2, params_t const &p)
    {
        return m_objective.value(state.t, state.x, state.y[i1], p) - m_objective.value(state.t, state.x, state.y[i2], p);
    }

    /**
     * @brief
     */
    inline NUMERIC_T dHdx(DAEOSolutionState const &state, size_t const i1, size_t const i2, params_t const &p)
    {
        return m_objective.grad_x(state.t, state.x, state.y[i1], p) - m_objective.grad_x(state.t, state.x, state.y[i2], p);
    }

    /**
     * @brief Locate and correct an event that happens between @c start and @c end
     * @param[in] start
     * @param[in] end
     * @param[in] p
     * @return Value of solution at event.
     */
    DAEOSolutionState locate_and_integrate_to_event(DAEOSolutionState const &start, DAEOSolutionState const &end, params_t const &p)
    {
        NUMERIC_T H_value, dHdt_value, dt_guess;
        DAEOSolutionState guess;
        dt_guess = (end.t - start.t) / 2;
        size_t iter = 0;
        while (iter < settings.MAX_NEWTON_ITERATIONS)
        {
            // integrate to t_guess
            guess = integrate_daeo(start, dt_guess, p);
            update_optimum(guess, p);
            H_value = H(guess, start.i_star, end.i_star, p);
            if (fabs(H_value) < settings.NEWTON_EPS)
            {
                break;
            }
            dHdt_value = dHdx(guess, start.i_star, end.i_star, p) * m_xprime.value(guess.t, guess.x, guess.y_star(), p);
            // newton iteration.
            guess.t -= H_value / dHdt_value;
            dt_guess = guess.t - start.t;
            iter++;
        }
        return guess;
    }
};

#endif