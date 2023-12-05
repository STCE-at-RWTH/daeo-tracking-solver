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
 * @class LocalOptimaBNBSolver
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
class LocalOptimaBNBSolver
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

    /**
     * @brief The objective function h(t, x, y; p) of which to find the minima.
     */
    DAEOWrappedFunction<OBJECTIVE_T> const m_objective;

    /**
     * @brief
     */
    BNBSolverSettings<NUMERIC_T> const m_settings;

    /**
     * @brief The prefix name for the solver log file
     */
    std::string m_log_name;

private:
    /**
     * @brief Queue of intervals in the domain of @c m_objective to search for local optima.
     */
    std::queue<y_interval_t> m_workq;

public:
    /**
     * @brief Initialize the solver with an objective function and settings.
     */
    LocalOptimaBNBSolver(OBJECTIVE_T const &t_objective,
                         BNBSolverSettings<NUMERIC_T> const &t_settings)
        : m_objective{t_objective}, m_settings{t_settings}, m_log_name{"bnb_log"} {}

    void set_search_domain(y_interval_t y)
    {
        m_workq.push(y);
    }

    /**
     * @brief Find minima in @c y of @c h(t,x,y;p) using the set search domain.
     * @param[in] t
     * @param[in] x
     * @param[in] params
     * @param[in] logging Should a log file be written? Default @c false
     * @returns Solver results struct.
     */
    results_t find_minima_at(NUMERIC_T t,
                             NUMERIC_T x,
                             params_t const &params,
                             bool logging = false)
    {
        if (m_workq.empty())
        {
            results_t r;
            return r;
        }

        size_t i = 0;
        BNBSolverLogger logger(m_workq.front().size(), params.size(), std::string(DATA_OUTPUT_DIR) + "/" + m_log_name);
        auto comp_start = std::chrono::high_resolution_clock::now();
        logger.log_computation_begin(i, comp_start, m_workq.front());

        results_t sresults;
        while (!m_workq.empty() && i < m_settings.MAXITER)
        {
            //std::cout << "Iteration: " << i++ << std::endl;
            y_interval_t y_i(m_workq.front());
            m_workq.pop();
            process_interval(i, t, x, y_i, params, sresults, logger, logging);
            //std::cout << "there are still " << m_workq.size() << " things to do" << std::endl;
        }

        auto comp_end = std::chrono::high_resolution_clock::now();
        logger.log_computation_end(i, comp_end, sresults.minima_intervals.size());
        m_workq = {};

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
     * @param[inout] sresults reference to the global solver status
     * @param[in] logger
     * @param[in] logging
     * @details
     */
    void process_interval(size_t tasknum,
                          NUMERIC_T t,
                          NUMERIC_T x,
                          y_interval_t const &y_i,
                          params_t const &params,
                          results_t &sresults,
                          BNBSolverLogger &logger,
                          bool logging)
    {
        using clock = std::chrono::high_resolution_clock;
        size_t result_code = 0;
        if (logging)
        {
            logger.log_task_begin(tasknum, clock::now(), y_i);
        }

        vector<bool> dims_converged(y_i.rows(), true);
        bool allconverged = true;
        for (int k = 0; k < y_i.rows(); k++)
        {
            dims_converged[k] = (width(y_i(k)) <= m_settings.TOL_X || (y_i(k).lower() == 0 && y_i(k).upper() == 0));
            allconverged = allconverged && dims_converged[k];
        }
        result_code = result_code | (allconverged ? CONVERGENCE_TEST_PASS : 0);

        interval_t h(m_objective.value(t, x, y_i, params));
        y_interval_t dhdy(m_objective.grad_y(t, x, y_i, params));
        bool grad_pass = std::all_of(dhdy.begin(), dhdy.end(), [](interval_t ival)
                                     { return boost::numeric::zero_in(ival); });
        result_code = result_code | (grad_pass ? GRADIENT_TEST_PASS : GRADIENT_TEST_FAIL);

        y_hessian_interval_t d2hdy2(m_objective.hess_y(t, x, y_i, params));
        if (is_positive_definite(d2hdy2))
        {
            result_code = result_code | HESSIAN_POSITIVE_DEFINITE;
        }
        else if (is_negative_definite(d2hdy2))
        {
            result_code = result_code | HESSIAN_NEGATIVE_DEFINITE;
        }
        else
        {
            result_code = result_code | HESSIAN_MAYBE_INDEFINITE;
        }

        // logger.log_all_tests(tasknum, clock::now(), result_code, y_i, h, dhdy, d2hdy2, dims_converged);
        if ((result_code & GRADIENT_TEST_FAIL) | (result_code & HESSIAN_NEGATIVE_DEFINITE))
        {
            // gradient test OR hessian test failed
            if (logging)
            {
                logger.log_task_complete(tasknum, clock::now(), y_i, result_code);
            }
        }
        else if (result_code & HESSIAN_POSITIVE_DEFINITE)
        {
            // hessian test passed!
            // hessian is SPD -> h(x) on interval is convex
            auto y_res = narrow_via_bisection(t, x, y_i, params, dims_converged);
            if (logging)
            {
                logger.log_task_complete(tasknum, clock::now(), y_i, CONVERGENCE_TEST_PASS | result_code);
            }
            sresults.minima_intervals.push_back(y_res);
        }
        else if (result_code & CONVERGENCE_TEST_PASS)
        {
            fmt::print(std::cout, "  Strange behavior in task {:d}, result code is {:d}.\n", tasknum, result_code);
            fmt::print(std::cout, "  Gradient test FAIL, hessian test INCONCLUSIVE, convergence PASS\n");
            fmt::print(std::cout, "  DISCARDING INTERVAL\n");
            if (logging)
            {
                logger.log_task_complete(tasknum, clock::now(), y_i, result_code);
            }
        }
        else
        {
            // second derivative test is inconclusive...
            // interval contains a change of sign in the gradient, but it is not locally convex.
            // therefore, we choose to bisect the interval and continue the search.
            auto ivals = bisect_interval(y_i, dims_converged);
            for (auto &ival : ivals)
            {
                m_workq.push(ival);
            }
            if (logging)
            {
                logger.log_task_complete(tasknum, clock::now(), y_i, result_code);
            }
        }
    }

    /**
     * @brief Cuts the n-dimensional range @c x in each dimension that is not flagged in @c dims_converged
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
        //fmt::print("Split interval {::.2f} into {:::.2f}\n", y, res);
        return res;
    }

    /**
     * @brief Shrink an interval via bisection.
     */
    y_interval_t narrow_via_bisection(NUMERIC_T t, NUMERIC_T x, y_interval_t const &y_in,
                                      params_t const &params, vector<bool> dimsconverged)
    {
        y_interval_t y(y_in);
        y_t y_m(y.rows());
        y_t gradient_y_m(y.rows());

        size_t iteration = 0;
        while (iteration < m_settings.MAX_REFINE_ITER)
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
            gradient_y_m = m_objective.grad_y(t, x, y_m, params);
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
                dimsconverged[i] = width(y(i)) <= m_settings.TOL_X;
            }
            iteration++;
        }
        return y;
    }
};

#endif // header guard