/**
 * @file local_optima_solver.hpp
 * @author Sasha [fleming@stce.rwth-aachen.de]
 * @brief Implementation of a branch and bound optimizer to find regions
 * containing local optima of f(x; p).
 */
#ifndef _LOCAL_OPTIMA_BNB_HPP // header guard
#define _LOCAL_OPTIMA_BNB_HPP

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <queue>
#include <vector>

#include "Eigen/Dense" // must include Eigen BEFORE dco/c++
#include "boost/numeric/interval.hpp"
#include "boost/numeric/interval/utility_fwd.hpp"
#include "dco.hpp"
#include "fmt/format.h"
#include "fmt/ranges.h"

#include "logging.hpp"
#include "objective.hpp"
#include "settings.hpp"
#include "utils/fmt_extensions.hpp"
#include "utils/sylvesters_criterion.hpp"

using std::vector;

/**
 * @struct BNBSolverResults
 * @brief Bookkeeping for global information the solver needs to keep track of.
 */
template <typename NUMERIC_T, typename INTERVAL_T, int YDIMS>
struct BNBSolverResults {
  /**
   * @brief An upper bound for all local minima.
   */
  NUMERIC_T optima_upper_bound = std::numeric_limits<NUMERIC_T>::max();

  vector<Eigen::Vector<INTERVAL_T, YDIMS>> minima_intervals;
};

/**
 * @class BNBLocalOptimizer
 * @brief Finds regions containing local minima of h(t, x, y; p)
 * @tparam OBJECTIVE_T Type of the objective function.
 * @tparam NUMERIC_T Type of the parameters to h(t, x, y; p).
 * @tparam YDIMS Size of the search space y
 * @tparam NPARAMS Size of the parameter vector p
 */
template <typename OBJECTIVE_T, typename NUMERIC_T = double,
          typename POLICIES = suggested_solver_policies<double>,
          int YDIMS = Eigen::Dynamic, int NPARAMS = Eigen::Dynamic>
class BNBLocalOptimizer {

public:
  using interval_t = boost::numeric::interval<NUMERIC_T, POLICIES>;
  using results_t = BNBSolverResults<NUMERIC_T, interval_t, YDIMS>;
  /**
   * @brief Eigen::Vector type of the search arguments y
   */
  using y_t = Eigen::Vector<NUMERIC_T, YDIMS>;

  /**
   * @brief Eigen::Vector type of the parameter vector p
   */
  using params_t = Eigen::Vector<NUMERIC_T, NPARAMS>;

  /**
   * @brief Eigen::Vector type of intervals for the search arguments y
   */
  using y_interval_t = Eigen::Vector<interval_t, YDIMS>;

  /**
   * @brief Eigen::Matrix type of the Hessian of h(...) w.r.t y
   */
  using y_hessian_t = Eigen::Matrix<NUMERIC_T, YDIMS, YDIMS>;

  /**
   * @brief Eigen::Matrix type of the Hessian of h(...) w.r.t y as an interval
   * type
   */
  using y_hessian_interval_t = Eigen::Matrix<interval_t, YDIMS, YDIMS>;

private:
  /**
   * @brief The objective function h(t, x, y; p) of which to find the minima.
   */
  DAEOWrappedFunction<OBJECTIVE_T> const m_objective;
  BNBOptimizerSettings<NUMERIC_T> const settings;

  /**
   * @brief The domain of @c objective to search for local optima.
   */
  y_interval_t initial_search_domain;

public:
  /**
   * @brief Initialize the solver with an objective function and settings.
   */
  BNBLocalOptimizer(OBJECTIVE_T const &t_objective, y_t const &t_LL,
                    y_t const &t_UR,
                    BNBOptimizerSettings<NUMERIC_T> const &t_settings)
      : m_objective{t_objective}, settings{t_settings},
        initial_search_domain(t_LL.rows()) {
    for (int i = 0; i < initial_search_domain.rows(); i++) {
      initial_search_domain(i) = interval_t(t_LL(i), t_UR(i));
    }
  }

  BNBLocalOptimizer(OBJECTIVE_T const &t_objective,
                    y_interval_t const &t_domain,
                    BNBOptimizerSettings<NUMERIC_T> const &t_settings)
      : m_objective{t_objective}, settings{t_settings}, initial_search_domain{
                                                            t_domain} {}

  /**
   * @brief Find minima in @c y of @c h(t,x,y;p) using the set search domain.
   * @param[in] t
   * @param[in] x
   * @param[in] params
   * @param[in] only_global Should only the global maximum be found?
   * @returns Solver results struct.
   */
  results_t find_minima_at(NUMERIC_T t, NUMERIC_T x, params_t const &params,
                           bool const only_global) {
    std::queue<y_interval_t> workq;
    workq.push(initial_search_domain);
    size_t i = 0;
    BNBOptimizerLogger logger("opt_log");
    auto comp_start = std::chrono::high_resolution_clock::now();
    if (settings.LOGGING_ENABLED) {
      logger.log_computation_begin(comp_start, i, workq.front());
    }
    results_t sresults;
    while (!workq.empty() && i < settings.MAXITER) {
      y_interval_t y_i(workq.front());
      workq.pop();
      // std::queue has no push_range on my GCC
      for (auto &y_bisected : process_interval(i, t, x, y_i, params, sresults,
                                               only_global, logger)) {
        workq.push(std::move(y_bisected));
      }
      i++;
    }

    // one last check for the global.
    // am I saving any work with this? TBD.
    if (only_global) {
      vector<y_interval_t> res;
      NUMERIC_T h_max = std::numeric_limits<NUMERIC_T>::max();
      interval_t h;
      size_t i_star = 0;
      for (size_t i = 0; i < sresults.minima_intervals.size(); i++) {
        h = m_objective.value(t, x, sresults.minima_intervals[i], params);
        if (h.upper() < h_max) {
          h_max = h.upper();
          i_star = i;
        }
      }
      res.push_back(sresults.minima_intervals[i_star]);
      sresults.minima_intervals = std::move(res);
    }

    auto comp_end = std::chrono::high_resolution_clock::now();
    if (settings.LOGGING_ENABLED) {
      logger.log_computation_end(comp_end, i, sresults.minima_intervals.size());
    }
    return sresults;
  }

private:
  /**
   * @brief Process an interval `y` and try to refine it via B&B.
   * @param[in] tasknum
   * @param[in] t
   * @param[in] x
   * @param[in] y
   * @param[in] params
   * @param[inout] sresults reference to the global optimizer status
   * @param[in] only_global only find the global optimum?
   * @param[in] logger
   * @returns Vector of intervals to be added to the work queue.
   * @details
   */
  vector<y_interval_t> process_interval(size_t tasknum, NUMERIC_T t,
                                        NUMERIC_T x, y_interval_t const &y_i,
                                        params_t const &params,
                                        results_t &sresults, bool only_global,
                                        BNBOptimizerLogger &logger) {
    using clock = std::chrono::high_resolution_clock;
    size_t result_code = 0;
    if (settings.LOGGING_ENABLED) {
      logger.log_task_begin(clock::now(), tasknum, y_i);
    }

    // convergence test
    vector<bool> dims_converged(y_i.rows(), true);
    bool allconverged = true;
    for (int k = 0; k < y_i.rows(); k++) {
      dims_converged[k] = (boost::numeric::width(y_i(k)) <= settings.TOL_Y ||
                           (y_i(k).lower() == 0 && y_i(k).upper() == 0));
      allconverged = allconverged && dims_converged[k];
    }
    if (allconverged) {
      result_code |= CONVERGENCE_TEST_PASS;
    }

    // value test
    interval_t h(m_objective.value(t, x, y_i, params));
    // fails if the lower end of the interval is larger than global upper bound
    // we update the global upper bound if the upper end of the interval is less
    // than the global optimum bound we only mark failures here, and we could
    // bail at this point, if we wished.
    // TODO bail early and avoid derivative tests
    if (sresults.optima_upper_bound < h.lower()) {
      result_code |= VALUE_TEST_FAIL;
    } else if (h.upper() < sresults.optima_upper_bound) {
      sresults.optima_upper_bound = h.upper();
    }

    // first derivative test
    // fails if it is not possible for the gradient to be zero inside the
    // interval y
    // TODO bail early and avoid hessian test
    y_interval_t dhdy(m_objective.grad_y(t, x, y_i, params));
    bool grad_pass = std::all_of(dhdy.begin(), dhdy.end(), [](interval_t ival) {
      return boost::numeric::zero_in(ival);
    });

    if (grad_pass) {
      result_code |= GRADIENT_TEST_PASS;
    } else {
      result_code |= GRADIENT_TEST_FAIL;
    }

    // second derivative test
    y_hessian_interval_t d2hdy2(m_objective.hess_y(t, x, y_i, params));
    result_code |= hessian_test(d2hdy2);

    // take actions based upon test results.
    vector<y_interval_t> candidate_intervals;
    logger.log_all_tests(clock::now(), tasknum, result_code, y_i, h, dhdy,
                         d2hdy2.rowwise(), dims_converged);
    if (only_global && (result_code & VALUE_TEST_FAIL)) {
      // value test fail + we only want to locate the global optimum
      // do nothing :)
    } else if ((result_code & GRADIENT_TEST_FAIL) |
               (result_code & HESSIAN_NEGATIVE_DEFINITE)) {
      // gradient test OR hessian test failed
      // do nothing :)
    } else if (result_code & HESSIAN_POSITIVE_DEFINITE) {
      // hessian test passed!
      // hessian is SPD -> h(x) on interval is convex
      y_interval_t y_res =
          narrow_via_bisection(t, x, y_i, params, dims_converged);
      // need to perform value test again with narrowed interval.
      interval_t h_res = m_objective.value(t, x, y_res, params);
      if (only_global && sresults.optima_upper_bound < h.lower()) {
        result_code |= VALUE_TEST_FAIL;
      } else {
        sresults.optima_upper_bound = h_res.upper();
        sresults.minima_intervals.push_back(y_res);
      }
      result_code |= CONVERGENCE_TEST_PASS;
    } else if (result_code & CONVERGENCE_TEST_PASS) {
      fmt::println("  Strange behavior in task {:d}, result code is {:d}.",
                   tasknum, result_code);
      fmt::println(
          "  Gradient test FAIL, hessian test INCONCLUSIVE, convergence PASS");
      fmt::println("  DISCARDING INTERVAL");
    } else {
      // second derivative test is inconclusive...
      // interval contains a change of sign in the gradient, but it is not
      // locally convex. therefore, we choose to bisect the interval and
      // continue the search.
      candidate_intervals = bisect_interval(t, x, y_i, params, dims_converged);
    }
    if (settings.LOGGING_ENABLED) {
      logger.log_task_complete(clock::now(), tasknum, y_i, result_code);
    }
    // return any generated work
    return candidate_intervals;
  }

  /**
   * @brief check the hessian for positive- or negative-definite-ness.
   * @param[in] d2hdy2
   */
  inline OptimizerTestCode hessian_test(y_hessian_interval_t const &d2hdy2) {
    if (is_positive_definite(d2hdy2)) {
      return HESSIAN_POSITIVE_DEFINITE;
    } else if (is_negative_definite(d2hdy2)) {
      return HESSIAN_NEGATIVE_DEFINITE;
    }
    return HESSIAN_MAYBE_INDEFINITE;
  }

  bool zero_in_or_absolutely_near(interval_t y, NUMERIC_T tol) {
    return boost::numeric::zero_in(y) ||
           (fabs(y.lower()) < tol && fabs(y.upper()) < tol);
  }

  /**
   * @brief Bisects the n-dimensional range @c x in each dimension that is not
   * flagged in @c dims_converged. Additionally performs a gradient check at the
   * bisection point and discards the LEFT interval if a bisection happened
   * exactly on the optimizer point.
   * @returns @c vector of n-dimensional intervals post-split
   */
  vector<y_interval_t> bisect_interval(NUMERIC_T t, NUMERIC_T x,
                                       y_interval_t const &y, params_t const &p,
                                       vector<bool> const &dims_converged) {
    vector<y_interval_t> res;
    res.emplace_back(y.rows());
    res[0] = y;
    for (int i = 0; i < y.rows(); i++) {
      if (dims_converged[i]) {
        continue;
      }
      size_t result_size = res.size();
      NUMERIC_T split_point = boost::numeric::median(y(i));
      for (size_t j = 0; j < result_size; j++) {
        y_interval_t splitting_plane = res[j];
        splitting_plane(i).assign(split_point, split_point);
        y_interval_t grad = m_objective.grad_y(t, x, splitting_plane, p);
        if (zero_in_or_absolutely_near(grad(i), settings.TOL_Y)) {
          // capture the splitting plane in an interval smaller than TOL
          NUMERIC_T cut_L = (split_point + res[j](i).lower()) / 2;
          NUMERIC_T cut_R = (split_point + res[j](i).upper()) / 2;
          res.emplace_back(y.rows());
          res.back() = res[j];
          res.back()(i).assign(cut_L, cut_R);
          // add the right side to the end of the result vector
          res.emplace_back(y.rows());
          res.back() = res[j];
          res.back()(i).assign(cut_R, y(i).upper());
          // update res[j] to be the left side
          res[j](i).assign(res[j](i).lower(), cut_L);
        } else {
          // add the right side to the end of the result vector
          res.emplace_back(y.rows());
          res.back() = res[j];
          res.back()(i).assign(split_point, y(i).upper());
          // update res[j] to be the left side
          res[j](i).assign(res[j](i).lower(), split_point);
        }
      }
    }
    return res;
  }

  /**
   * @brief Shrink an interval via bisection.
   */
  y_interval_t narrow_via_bisection(NUMERIC_T t, NUMERIC_T x,
                                    y_interval_t const &y_in,
                                    params_t const &params,
                                    vector<bool> &dimsconverged) {
    y_interval_t y(y_in);
    y_t y_m(y.rows());
    y_t gradient_y_m(y.rows());

    size_t iteration = 0;
    while (iteration < settings.MAX_REFINE_ITER) {
      if (std::all_of(dimsconverged.begin(), dimsconverged.end(),
                      [](bool b) -> bool { return b; })) {
        break;
      }
      y_m = y.unaryExpr([](auto ival) { return boost::numeric::median(ival); });
      gradient_y_m = m_objective.grad_y(t, x, y_m, params);
      for (int i = 0; i < y.size(); i++) {
        if (!dimsconverged[i]) {
          // increasing at midpoint -> minimum is to the left of midpoint
          if (gradient_y_m(i) > 0) {
            y(i) = interval_t(y[i].lower(), y_m[i]);
          } else {
            y(i) = interval_t(y_m[i], y[i].upper());
          }
        }
        dimsconverged[i] = boost::numeric::width(y(i)) <= settings.TOL_Y;
      }
      iteration++;
    }
    return y;
  }
};

#endif // header guard