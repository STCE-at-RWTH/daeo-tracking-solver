/**
 * @file local_optima_solver.hpp
 * @author Sasha [fleming@stce.rwth-aachen.de]
 * @brief Implementation of a branch and bound solver to find regions containing local optima of f(x; p).
 */
#ifndef _LOCAL_OPTIMA_BNB_HPP // header guard
#define _LOCAL_OPTIMA_BNB_HPP

#include <algorithm>
#include <chrono>
#include <limits>
#include <queue>
#include <vector>

#include "boost/numeric/interval.hpp"
#include "Eigen/Dense" // must include Eigen BEFORE dco/c++
#include "dco.hpp"
#include "fmt/format.h"
#include "fmt/ranges.h"

#include "objective.hpp"
#include "logging.hpp"
#include "settings.hpp"
#include "utils/fmt_extensions.hpp"
#include "utils/sylvesters_criterion.hpp"

using boost::numeric::median;
using boost::numeric::width;
using std::vector;

/**
 * @struct BNBSolverResults
 * @brief Bookkeeping for global information the solver needs to keep track of.
 */
template <typename NUMERIC_T, typename INTERVAL_T, int YDIMS>
struct BNBSolverResults
{
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
template <typename OBJECTIVE_T,
          typename NUMERIC_T = double,
          typename POLICIES = suggested_solver_policies<double>,
          int YDIMS = Eigen::Dynamic,
          int NPARAMS = Eigen::Dynamic>
class BNBLocalOptimizer
{

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
     * @brief Eigen::Matrix type of the Hessian of h(...) w.r.t y as an interval type
     */
    using y_hessian_interval_t = Eigen::Matrix<interval_t, YDIMS, YDIMS>;

private:
    /**
     * @brief The objective function h(t, x, y; p) of which to find the minima.
     */
    DAEOWrappedFunction<OBJECTIVE_T> const objective;
    BNBOptimizerSettings<NUMERIC_T> const settings;

    /**
     * @brief The domain of @c objective to search for local optima.
     */
    y_interval_t initial_search_domain;

public:
    /**
     * @brief Initialize the solver with an objective function and settings.
     */
    BNBLocalOptimizer(OBJECTIVE_T const &t_objective, y_t const &t_LL, y_t const &t_UR,
                      BNBOptimizerSettings<NUMERIC_T> const &t_settings)
        : objective{t_objective}, settings{t_settings}, initial_search_domain(t_LL.rows())
    {
        for (int i = 0; i < initial_search_domain.rows(); i++)
        {
            initial_search_domain(i) = interval_t(t_LL(i), t_LL(i));
        }
    }

    BNBLocalOptimizer(OBJECTIVE_T const &t_objective, y_interval_t const &t_domain,
                      BNBOptimizerSettings<NUMERIC_T> const &t_settings)
        : objective{t_objective}, settings{t_settings}, initial_search_domain{t_domain} {}

    /**
     * @brief Find minima in @c y of @c h(t,x,y;p) using the set search domain.
     * Clears internal work queue.
     * @param[in] t
     * @param[in] x
     * @param[in] params
     * @param[in] only_global Should only the global maximum be found?
     * @returns Solver results struct.
     */
    results_t find_minima_at(NUMERIC_T t, NUMERIC_T x, params_t const &params, bool const only_global)
    {
        std::queue<y_interval_t> workq;
        workq.push(initial_search_domain);
        size_t i = 0;
        BNBOptimizerLogger logger("opt_log");
        auto comp_start = std::chrono::high_resolution_clock::now();
        if (settings.LOGGING_ENABLED)
        {
            logger.log_computation_begin(comp_start, i, workq.front());
        }
        results_t sresults;
        while (!workq.empty() && i < settings.MAXITER)
        {
            y_interval_t y_i(workq.front());
            workq.pop();
            process_interval(i, t, x, y_i, params, sresults, only_global, logger);
            i++;
        }
        auto comp_end = std::chrono::high_resolution_clock::now();
        if (settings.LOGGING_ENABLED)
        {
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
     * @details
     */
    void process_interval(size_t tasknum, NUMERIC_T t, NUMERIC_T x,
                          y_interval_t const &y_i, params_t const &params,
                          results_t &sresults, bool only_global,
                          BNBOptimizerLogger &logger)
    {
        using clock = std::chrono::high_resolution_clock;
        size_t result_code = 0;
        if (settings.LOGGING_ENABLED)
        {
            logger.log_task_begin(clock::now(), tasknum, y_i);
        }

        // convergence test
        vector<bool> dims_converged(y_i.rows(), true);
        bool allconverged = true;
        for (int k = 0; k < y_i.rows(); k++)
        {
            dims_converged[k] = (width(y_i(k)) <= settings.TOL_X || (y_i(k).lower() == 0 && y_i(k).upper() == 0));
            allconverged = allconverged && dims_converged[k];
        }
        result_code = result_code | (allconverged ? CONVERGENCE_TEST_PASS : 0);

        // value test
        interval_t h(objective.value(t, x, y_i, params));
        if (sresults.optima_upper_bound <= h.lower())
        { // if lower end of interval larger than possible lower bound...
            result_code = result_code | VALUE_TEST_FAIL;
        }
        else
        {
            result_code = result_code | VALUE_TEST_PASS;
            sresults.optima_upper_bound = h.upper();
        }

        // first derivative test
        y_interval_t dhdy(objective.grad_y(t, x, y_i, params));
        bool grad_pass = std::all_of(dhdy.begin(), dhdy.end(), [](interval_t ival)
                                     { return boost::numeric::zero_in(ival); });
        result_code = result_code | (grad_pass ? GRADIENT_TEST_PASS : GRADIENT_TEST_FAIL);

        // second derivative test
        y_hessian_interval_t d2hdy2(objective.hess_y(t, x, y_i, params));
        result_code = result_code | hessian_test(d2hdy2);

        // take actions based upon test results.
        logger.log_all_tests(clock::now(), tasknum, result_code, y_i, h, dhdy, d2hdy2.rowwise(), dims_converged);
        if ((result_code & VALUE_TEST_FAIL) && only_global)
        {
            // value test fail
            // do nothing :)
        }
        else if ((result_code & GRADIENT_TEST_FAIL) | (result_code & HESSIAN_NEGATIVE_DEFINITE))
        {
            // gradient test OR hessian test failed
            // do nothing :)
        }
        else if (result_code & HESSIAN_POSITIVE_DEFINITE)
        {
            // hessian test passed!
            // hessian is SPD -> h(x) on interval is convex
            y_interval_t y_res = narrow_via_bisection(t, x, y_i, params, dims_converged);
            // need to perform value test again with narrowed interval.
            interval_t h_res = objective.value(t, x, y_res, params);
            if (only_global && h_res.lower() > sresults.optima_upper_bound)
            {
                result_code = result_code | VALUE_TEST_FAIL;
            }
            else
            {
                sresults.optima_upper_bound = h_res.upper();
                sresults.minima_intervals.push_back(y_res);
            }
            result_code = result_code | CONVERGENCE_TEST_PASS;
        }
        else if (result_code & CONVERGENCE_TEST_PASS)
        {
            fmt::print(std::cout, "  Strange behavior in task {:d}, result code is {:d}.\n", tasknum, result_code);
            fmt::print(std::cout, "  Gradient test FAIL, hessian test INCONCLUSIVE, convergence PASS\n");
            fmt::print(std::cout, "  DISCARDING INTERVAL\n");
        }
        else
        {
            // second derivative test is inconclusive...
            // interval contains a change of sign in the gradient, but it is not locally convex.
            // therefore, we choose to bisect the interval and continue the search.
            auto ivals = bisect_interval(y_i, dims_converged);
            for (auto &ival : ivals)
            {
                workq.push(ival);
            }
        }
        if (settings.LOGGING_ENABLED)
        {
            logger.log_task_complete(clock::now(), tasknum, y_i, result_code);
        }
    }

    /**
     * @brief check the hessian for positive- or negative-definite-ness.
     * @param[in] d2hdy2
     */
    inline OptimizerTestCode hessian_test(y_hessian_interval_t const &d2hdy2)
    {
        if (is_positive_definite(d2hdy2))
        {
            return HESSIAN_POSITIVE_DEFINITE;
        }
        else if (is_negative_definite(d2hdy2))
        {
            return HESSIAN_NEGATIVE_DEFINITE;
        }
        return HESSIAN_MAYBE_INDEFINITE;
    }

    /**
     * @brief Bisects the n-dimensional range @c x in each dimension that is not flagged in @c dims_converged
     * @returns @c vector of n-dimensional intervals post-split
     */
    vector<y_interval_t> bisect_interval(y_interval_t const &y, vector<bool> const &dims_converged)
    {
        vector<y_interval_t> res;
        res.emplace_back(y.rows());
        res[0] = y;
        if (!dims_converged[0])
        {
            NUMERIC_T m = median(res[0](0));
            res.emplace_back(y.rows());
            res.back() = y;
            res[0](0).assign(res[0](0).lower(), m);
            res.back()(0).assign(m, res.back()(0).upper());
        }
        for (int i = 1; i < y.rows(); i++)
        {
            if (dims_converged[i])
            {
                continue;
            }
            size_t result_size = res.size();
            NUMERIC_T m = median(res[0](i));
            for (size_t j = 0; j < result_size; j++)
            {
                res.emplace_back(y.rows());
                res.back() = y;
                res[j](i).assign(res[j](i).lower(), m);
                res.back()(i).assign(m, res.back()(i).upper());
            }
        }
        // fmt::print("Split interval {::.2f} into {:::.2f}\n", y, res);
        return res;
    }

    /**
     * @brief Shrink an interval via bisection.
     */
    y_interval_t narrow_via_bisection(NUMERIC_T t, NUMERIC_T x, y_interval_t const &y_in,
                                      params_t const &params, vector<bool> &dimsconverged)
    {
        y_interval_t y(y_in);
        y_t y_m(y.rows());
        y_t gradient_y_m(y.rows());

        size_t iteration = 0;
        while (iteration < settings.MAX_REFINE_ITER)
        {
            if (std::all_of(dimsconverged.begin(), dimsconverged.end(),
                            [](bool b) -> bool
                            { return b; }))
            {
                break;
            }

            for (int i = 0; i < y.rows(); i++)
            {
                y_m(i) = median(y(i));
            }
            // objective_gradient(t, x, y_m, params, temp, midgrad);
            gradient_y_m = objective.grad_y(t, x, y_m, params);
            for (int i = 0; i < y.size(); i++)
            {
                if (!dimsconverged[i])
                {
                    // increasing at midpoint -> minimum is to the left of midpoint
                    if (gradient_y_m(i) > 0)
                    {
                        y(i) = interval_t(y[i].lower(), y_m[i]);
                    }
                    else
                    {
                        y(i) = interval_t(y_m[i], y[i].upper());
                    }
                }
                dimsconverged[i] = width(y(i)) <= settings.TOL_X;
            }
            iteration++;
        }
        return y;
    }
};

#endif // header guard