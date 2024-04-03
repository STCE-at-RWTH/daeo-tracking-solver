/**
 * @file daeo_solver.hpp
 * @author Sasha [fleming@stce.rwth-aachen.de]
 * @brief Implementation of solver for Differential-Algebriac Equations with
 * Embedded Optimization Criteria.
 */
#ifndef _DAEO_SOLVER_HPP
#define _DAEO_SOLVER_HPP

#include <algorithm>
#include <chrono>
#include <limits>
#include <tuple>
#include <vector>

#include "Eigen/Dense"
#include "boost/numeric/interval.hpp"
#include "fmt/format.h"
#include "fmt/ranges.h"

#include "local_optima_solver.hpp"
#include "settings.hpp"
#include "utils/fmt_extensions.hpp"

using std::vector;

template <typename NUMERIC_T, int YDIMS> struct DAEOSolutionState {
  NUMERIC_T t;
  NUMERIC_T x;
  size_t i_star;
  vector<Eigen::Vector<NUMERIC_T, YDIMS>> y;

  /**
   * @brief Return the number of local optima present at the this point in time.
   */
  inline size_t n_local_optima() const { return y.size(); }

  // Eigen should help with compile time optimization for this.
  // TODO make if constexpr(Eigen::Dynamic) if necessary.

  /**
   * @brief Return the number of dimensions of each local optimizer.
   */
  inline int ydims() const { return y[0].rows(); }

  // what the heck am I doing
  /**
   * @brief Get a (const) reference to the global optimum y*
   */
  inline Eigen::Vector<NUMERIC_T, YDIMS> &y_star() { return y[i_star]; }
  inline const Eigen::Vector<NUMERIC_T, YDIMS> &y_star() const {
    return y[i_star];
  }
};

template <typename XPRIME, typename OBJECTIVE, typename NUMERIC_T, int YDIMS,
          int NPARAMS>
class DAEOTrackingSolver {
public:
  using optimizer_t =
      BNBLocalOptimizer<OBJECTIVE, NUMERIC_T,
                        suggested_solver_policies<NUMERIC_T>, YDIMS, NPARAMS>;
  using y_t = Eigen::Vector<NUMERIC_T, YDIMS>;
  using y_hessian_t = Eigen::Matrix<NUMERIC_T, YDIMS, YDIMS>;
  using params_t = Eigen::Vector<NUMERIC_T, NPARAMS>;
  using interval_t = typename optimizer_t::interval_t;
  using solution_state_t = DAEOSolutionState<NUMERIC_T, YDIMS>;

  DAEOTrackingSolver(XPRIME const &t_xprime, OBJECTIVE const &t_objective,
                     BNBOptimizerSettings<NUMERIC_T> &t_opt_settings,
                     DAEOSolverSettings<NUMERIC_T> &t_global_settings)
      : m_xprime(t_xprime), m_objective(t_objective),
        settings(t_global_settings),
        m_optimizer(t_objective, y_t{settings.y0_min}, y_t{settings.y0_max},
                    t_opt_settings) {}

  vector<solution_state_t> solve_daeo(NUMERIC_T t, NUMERIC_T t_end,
                                      NUMERIC_T dt, NUMERIC_T x0,
                                      params_t const &params,
                                      std::string tag = "") {
    using clock = std::chrono::high_resolution_clock;
    vector<solution_state_t> solution_trajectory;
    DAEOSolverLogger logger(tag);
    if (settings.LOGGING_ENABLED) {
      logger.log_computation_begin(clock::now(), 0, t, dt, x0);
    }
    fmt::println("Starting to solve DAEO at t={:.4e}, x0={:.4e}, and dt={:.4e}",
                 t, x0, dt);
    typename optimizer_t::results_t opt_res = m_optimizer.find_minima_at(
        t, x0, params, settings.ONLY_GLOBAL_OPTIMIZATION);
    fmt::println("  BNB optimizer yields candidates for y at {:::.4e}",
                 opt_res.minima_intervals);
    solution_state_t current, next;
    current = solution_state_from_optimizer_results(t, x0, opt_res, params);
    vector<y_t> dy;
    if (settings.LOGGING_ENABLED) {
      logger.log_global_optimization(clock::now(), 0, current.t, current.x,
                                     current.y, current.i_star);
    }

    // next portion relies on the assumption that two minimizers of h don't
    // "cross paths" inside of a time step even if they did, would it really
    // matter? since we don't do any implicit function silliness we probably
    // wouldn't even be able to tell if this did happen it may be beneficial to
    // periodically check all of the y_is and see if they're close to each other
    // before and after solving for the values at the next time step. This would
    // trigger a search for minima of h(x, y) again, since we may have "lost"
    // one

    size_t iter = 0, iterations_since_search = 0;
    bool event_found;
    while (current.t < t_end) {
      solution_trajectory.push_back(current);
      // we have not found an event in this time step (yet).
      event_found = false;
      next = integrate_daeo(current, dt, params);
      if (iterations_since_search == settings.SEARCH_FREQUENCY) {
        opt_res = m_optimizer.find_minima_at(next.t, next.x, params,
                                             settings.ONLY_GLOBAL_OPTIMIZATION);
        solution_state_t from_opt = solution_state_from_optimizer_results(
            next.t, next.x, opt_res, params);
        if (settings.LOGGING_ENABLED) {
          logger.log_global_optimization(clock::now(), iter, from_opt.t,
                                         from_opt.x, from_opt.y,
                                         from_opt.i_star);
        }
        // check if we need to rewind multiple time steps
        fmt::println("  Checking identity of new optima at t={:.2e}",
                     from_opt.t);
        fmt::println("  BNB optimizer yields candidates for y at {:::.4e}",
                     from_opt.y);
        if (from_opt.n_local_optima() <= current.n_local_optima()) {
          correct_optimizer_permutation(from_opt, current);
          fmt::println("  Optimizer may have vanished.");
          fmt::println("  Reordered optima to match {:::.4e}", from_opt.y);
          fmt::println("               new order is {:::.4e}", current.y);
        } else {
          correct_optimizer_permutation(current, from_opt);
          fmt::println("  Optimizer emerged.");
          fmt::println("  Reordered optima to match {:::.4e}", current.y);
          fmt::println("               new order is {:::.4e}", from_opt.y);
        }

        if (opt_res.minima_intervals.size() == 0) {
          fmt::println("*** SCREAMING CRYING VOMITING ON THE FLOOR ***");
          return solution_trajectory;
        }
        // check for event and correct any error that may have accumulated
        // from the local optimizer tracking
        event_found =
            (next.y_star() - from_opt.y_star()).norm() > settings.EVENT_EPS;
        if (event_found)
          fmt::println("  event found by gopt");
        next = std::move(from_opt);
        iterations_since_search = 0;
      } else {
        // checking for events with i_star isn't correct.
        // z.B we pick up an optimizer at the _beginning_ of next.y
        // during a global optimizer run
        // so we have to use a more expensive check
        // or avoid this simple check when gopt has run
        update_optimizer(next, params);
        event_found = next.i_star != current.i_star;
      }

      if (settings.EVENT_DETECTION_AND_CORRECTION && event_found) {
        fmt::println("  Detected global optimzer switch between (t={:.6e}, "
                     "y={::.4e}) and (t={:.6e}, y={::.4e})",
                     current.t, current.y_star(), next.t, next.y_star());
        fmt::println("    Jump size is {:.6e}",
                     (current.y_star() - next.y_star()).norm());
        fmt::println("    Integrating to event from t={:.6e}", current.t);
        // locate the event and take a time step to it
        solution_state_t event =
            locate_and_integrate_to_event_bisect(current, next, params);
        fmt::println("    Event at t={:.6e}, x={:.4e}", event.t, event.x);
        // assign new global optimizer index
        fmt::println("    Old optimizers are {:::.4e}, i_star is {:d}",
                     current.y, current.i_star);
        event.i_star = next.i_star;
        fmt::println("    New optimizers are {:::.4e}, i_star is {:d}", event.y,
                     event.i_star);
        NUMERIC_T dt_event = event.t - current.t;
        if (settings.LOGGING_ENABLED) {
          dy = compute_dy(current.y, event.y);
          logger.log_event_correction(clock::now(), iter, event.t, dt_event,
                                      event.x, event.x - current.x, event.y, dy,
                                      event.i_star);
        }
        // complete time step from event back to the grid.
        NUMERIC_T dt_grid = dt - dt_event;
        fmt::println("    Integrating away from event at t={:.6e}, x={:.4e} "
                     "with step size {:.6e}",
                     event.t, event.x, dt_grid);
        fmt::println("    End time of step is t={:.6e}", event.t + dt_grid);
        next = integrate_daeo(event, dt_grid, params);
      }
      // we don't need to handle events, we can move on.
      if (settings.LOGGING_ENABLED) {
        dy = compute_dy(current.y, next.y);
        logger.log_time_step(clock::now(), iter, next.t, dt, next.x,
                             next.x - current.x, next.y, dy, next.i_star);
      }
      current = std::move(next);
      iter++;
      iterations_since_search++;
    }
    if (settings.LOGGING_ENABLED) {
      logger.log_computation_end(clock::now(), iter, current.t, current.x,
                                 current.y, current.i_star);
    }
    return solution_trajectory;
  }

private:
  DAEOWrappedFunction<XPRIME> const m_xprime;
  DAEOWrappedFunction<OBJECTIVE> const m_objective;
  DAEOSolverSettings<NUMERIC_T> settings;
  optimizer_t m_optimizer;

  /**
   * @brief Create a valid solution state from the results of the global
   * optimizer at (t, x).
   */
  solution_state_t solution_state_from_optimizer_results(
      NUMERIC_T const t, NUMERIC_T const x,
      typename optimizer_t::results_t gopt_results, params_t p) {
    using boost::numeric::median;
    vector<y_t> y;
    for (auto &y_i : gopt_results.minima_intervals) {
      y.emplace_back(y_i.size());
      y.back() = y_i.unaryExpr([](auto ival) { return median(ival); });
    }
    solution_state_t ss{t, x, 0, y};
    update_optimizer(ss, p);
    return ss;
  }

  /**
   * @brief Update the index of the global optimizer @c y★ in a solution state
   * @c s given a parameter vector @c p .
   * @param[inout] s The solution state state in which to update the optimizer
   * index.
   * @param[in] p The parameter vector to pass through to the objective
   * function.
   */
  void update_optimizer(solution_state_t &s, const params_t &p) {
    NUMERIC_T h_star = std::numeric_limits<NUMERIC_T>::max();
    for (size_t i = 0; i < s.n_local_optima(); i++) {
      NUMERIC_T h = m_objective.value(s.t, s.x, s.y[i], p);
      if (h < h_star) {
        h_star = h;
        s.i_star = i;
      }
    }
  }

  vector<y_t> compute_dy(vector<y_t> const &y, vector<y_t> const &y_next) {
    vector<y_t> dy(y.size());
    for (size_t i = 0; i < y.size(); i++) {
      dy[i] = (y_next[i] - y[i]);
    }
    return dy;
  }

  void correct_optimizer_permutation(solution_state_t const &fewer_optimizers,
                                     solution_state_t &more_optimizers) {
    vector<size_t> permutation(fewer_optimizers.n_local_optima());
    for (size_t i = 0; i < permutation.size(); i++) {
      for (size_t j = 0; j < more_optimizers.n_local_optima(); j++) {
        // we reuse event eps here since it represents "acceptable accumulated
        // integration error"
        if ((more_optimizers.y[j] - fewer_optimizers.y[i]).norm() <
            settings.EVENT_EPS) {
          permutation[i] = j;
        }
      }
    }
    for (size_t i = 0; i < permutation.size(); i++) {
      std::swap(more_optimizers.y[i], more_optimizers.y[permutation[i]]);
      if (i == more_optimizers.i_star) {
        more_optimizers.i_star = permutation[i];
      }
    }
  }

  /*
      G and delG are used by newton iteration to find x and y at the next time
     step. There is the condition on x from the trapezoidal rule: x_{k+1} =
     x_{k} + dt/2 * (f(x_{k+1}, y_{k+1}) + f(x_{k}, y_{k})) After the zeroth
     time step (t=t0), we have good initial guesses for the minima y_{k+1}
      (assuming they don't drift "too far")
      The other equations are provided by dh/dyi = 0 at x_{k+1} and y_{k+1}

      from this structure we compute G and delG in blocks, since we have
     derivative routines available for f=x' and h
  */

  /**
   * @brief Compute G. We are searching for x1, y1 s.t. G(...) = 0.
   * @param[in] start Value of the solution trajectory at the beginning of a
   * time step.
   * @param[in] end Value (potentially a guess) of the solution trajectory at
   * the end of a time step.
   * @param[in] dt Current time step size.
   * @param[in] p Parameter vector.
   */
  Eigen::VectorX<NUMERIC_T>
  trapezoidal_integrator(solution_state_t const &start,
                         solution_state_t const &guess, NUMERIC_T const dt,
                         params_t const &p) {
    int ydims = start.ydims();
    // fix i_star (assume no event in this part of the program)
    size_t i_star = start.i_star;
    Eigen::VectorX<NUMERIC_T> result(1 + start.n_local_optima() * ydims);
    result(0) = start.x - guess.x +
                dt / 2 *
                    (m_xprime.value(start.t, start.x, start.y[i_star], p) +
                     m_xprime.value(guess.t, guess.x, guess.y[i_star], p));
    for (size_t i = 0; i < start.n_local_optima(); i++) {
      result(Eigen::seqN(1 + i * ydims, ydims)) =
          m_objective.grad_y(guess.t, guess.x, guess.y[i], p);
    }
    return result;
  }

  /**
   * @brief Gradient of the function used for newton iteration.
   * @param[in] guess The guessed value of the solution trajectory.
   * @param[in] dt The time step size from the beginning of the time step to
   * where @c guess is evaluated.
   * @param[in] p Parameter vector.
   */
  Eigen::MatrixX<NUMERIC_T>
  trapezoidal_integrator_jacobian(solution_state_t const &guess,
                                  NUMERIC_T const dt, params_t const &p) {
    using Eigen::seqN;
    int ydims = guess.ydims();
    int ndims = 1 + guess.n_local_optima() * ydims;
    Eigen::MatrixX<NUMERIC_T> result(ndims, ndims);
    result(0, 0) =
        -1 +
        dt / 2 * m_xprime.grad_x(guess.t, guess.x, guess.y[guess.i_star], p);
    for (size_t i = 0; i < guess.n_local_optima(); i++) {
      result(0, seqN(1 + i * ydims, ydims)) =
          dt / 2 * m_xprime.grad_y(guess.t, guess.x, guess.y[i], p);
      result(seqN(1 + i * ydims, ydims), 0) =
          m_objective.d2dxdy(guess.t, guess.x, guess.y[i], p);
      result(seqN(1 + i * ydims, ydims), seqN(1 + i * ydims, ydims)) =
          m_objective.hess_y(guess.t, guess.x, guess.y[i], p);
    }
    return result;
  }

  /**
   * @brief Integrate the differential part of the DAEO from time @c t to @c
   * t+dt
   * @param[in] start The value of the solution at @c t=t0
   * @param[in] dt The size of the time step to integrate.
   * @param[in] p Parameter vector.
   * @return The value of solution at @c t=t+dt.
   * @details Integrates the ODE using the trapezoidal rule. Additionally solves
   * ∂h/∂y_k = 0 simultaenously using Newton's method.
   */
  solution_state_t integrate_daeo(solution_state_t const &start, NUMERIC_T dt,
                                  params_t const &p) {
    // copy into our guess and advance time
    solution_state_t next(start);
    next.t += dt;
    Eigen::VectorX<NUMERIC_T> G, diff;
    Eigen::MatrixX<NUMERIC_T> jacG;
    size_t iter = 0;
    while (iter < settings.MAX_NEWTON_ITERATIONS) {
      G = trapezoidal_integrator(start, next, dt, p);
      jacG = trapezoidal_integrator_jacobian(next, dt, p);
      diff = jacG.colPivHouseholderQr().solve(G);
      next.x = next.x - diff(0);
      for (size_t i = 0; i < start.n_local_optima(); i++) {
        next.y[i] =
            next.y[i] - diff(Eigen::seqN(1 + i * start.ydims(), start.ydims()));
      }
      if (diff.norm() < settings.NEWTON_EPS) {
        break;
      }
      iter++;
    }
    return next;
  }

  /*
      H and dHdt are used to find the precise event location by newton
     iteration.
  */

  /**
   * @brief Event function between optima @c y1 (current optimum at (t, x))
   *   and the candidate next optimum @c y2 at (t, x)
   * @param[in] t
   * @param[in] x
   * @param[in] y1 Global optimizer from the beginning of the time step,
   * integrated to @c t
   * @param[in] y2 Global optimizer at the end of the time step, integrated to
   * @c t
   * @param[in] p Parameter vector.
   */
  inline NUMERIC_T event_function(NUMERIC_T t, NUMERIC_T x, y_t const &y1,
                                  y_t const &y2, params_t const &p) {
    return m_objective.value(t, x, y1, p) - m_objective.value(t, x, y2, p);
  }

  /**
   * @brief @b TOTAL derivative of @c H w.r.t. @c x
   */
  inline NUMERIC_T event_function_ddx(NUMERIC_T t, NUMERIC_T x, y_t const &y1,
                                      y_t const &y2, params_t const &p) {
    return m_objective.grad_x(t, x, y1, p) - m_objective.grad_x(t, x, y2, p);
  }

  /**
   * @brief Locate and correct an event that happens between @c start and @c
   * end. We assume that no new optimizers emerge inside this time step!
   * @param[in] start
   * @param[in] end
   * @param[in] p
   * @return Value of solution at event.
   */
  solution_state_t
  locate_and_integrate_to_event_newton(solution_state_t const &start,
                                       solution_state_t const &end,
                                       params_t const &p) {
    NUMERIC_T H, dHdt, dt_guess;
    solution_state_t guess;
    dt_guess = (end.t - start.t) / 2;
    size_t iter = 0;
    bool escaped = false;
    while (iter < settings.MAX_NEWTON_ITERATIONS) {
      // integrate to t_guess
      guess = integrate_daeo(start, dt_guess, p);
      // evaluate event function
      H = event_function(guess.t, guess.x, guess.y[start.i_star],
                         guess.y[end.i_star], p);
      if (fabs(H) < settings.NEWTON_EPS) {
        break;
      }
      // only scream once
      if (!escaped && (start.t > guess.t || guess.t > end.t)) {
        fmt::println("  Escaped bounds on event locator!!");
        escaped = true;
        // breaking may save us here.
        break;
      }
      dHdt = (event_function_ddx(guess.t, guess.x, guess.y[start.i_star],
                                 guess.y[end.i_star], p) *
              m_xprime.value(guess.t, guess.x, guess.y_star(), p));
      // dHdt += partial h partial t at y1 and y2... maybe not necessary.
      // newton iteration.
      guess.t -= H / dHdt;
      dt_guess = guess.t - start.t;
      iter++;
    }
    return guess;
  }

  /**
   * @brief Locate and correct an event that happens between @c start and @c end
   * using Bisection We assume that no new optimizers emerge inside this time
   * step!
   * @param[in] start
   * @param[in] end
   * @param[in] p
   * @return Value of solution at event.
   */
  solution_state_t
  locate_and_integrate_to_event_bisect(solution_state_t const &start,
                                       solution_state_t const &end,
                                       params_t const &p) {
    solution_state_t guess;
    NUMERIC_T delta = (end.t - start.t) / 2;
    NUMERIC_T dt_guess = delta;
    NUMERIC_T H;
    size_t iter = 0;
    while (iter <
           settings.MAX_NEWTON_ITERATIONS) // maybe we need to set this higher
    {
      guess = integrate_daeo(start, dt_guess, p);
      H = event_function(guess.t, guess.x, guess.y[start.i_star],
                         guess.y[end.i_star], p);
      delta = delta / 2;
      if (H < settings.NEWTON_EPS) {
        break;
      } else if (H < 0) {
        dt_guess += delta;
      } else {
        dt_guess -= delta;
      }
      iter++;
    }
    return guess;
  }
};

#endif