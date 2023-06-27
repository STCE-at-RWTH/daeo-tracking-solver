/**
 * @file local_optima_bnb.hpp
 * @author Sasha [fleming@stce.rwth-aachen.de]
 * @brief Implementation of a branch and bound solver to find regions containing local optima of f(x; p).
 */
#ifndef _LOCAL_OPTIMA_BNB_HPP // header guard
#define _LOCAL_OPTIMA_BNB_HPP

#include <algorithm>
#include <limits>
#include <queue>
#include <vector>

#include "dco.hpp"
#include "boost/numeric/interval.hpp"

#include "bnb_settings.hpp"
#include "utils.hpp"

using boost::numeric::median;
using boost::numeric::width;
using std::vector;

/**
 * @struct BNBSolverResults
 * @brief Bookkeeping for global information the solver needs to keep track of.
 */
template <typename NUMERIC_T, typename INTERVAL_T>
struct BNBSolverResults
{
    NUMERIC_T optima_supremum = std::numeric_limits<NUMERIC_T>::max();
    vector<vector<INTERVAL_T>> minima_intervals;
};

/**
 * @class LocalOptimaBNBSolver
 * @brief Finds regions containing local minima of h(x;p)
 * @tparam OBJECTIVE_T Type of the objective function.
 * @tparam NUMERIC_T Type of the parameters to f(x; p).
 * @tparam INTERVAL_T Interval type (created from NUMERIC_T).
 */
template <typename OBJECTIVE_T, typename NUMERIC_T, typename POLICIES>
class LocalOptimaBNBSolver
{

public:
    typedef boost::numeric::interval<NUMERIC_T, POLICIES> interval_t;
    typedef BNBSolverResults<NUMERIC_T, interval_t> results_t;

    /**
     * @brief The objective function h(x; p) of which to find the minima.
     */
    OBJECTIVE_T const &m_objective;

    /**
     * @brief
     */
    BNBSolverSettings<NUMERIC_T> const m_settings;

    /**
     * @brief Initialize the solver with an objective function and settings.
     */
    LocalOptimaBNBSolver(OBJECTIVE_T const &t_objective,
                         BNBSolverSettings<NUMERIC_T> const &t_settings)
        : m_objective{t_objective}, m_settings{t_settings} {}

    /**
     * @brief
     */
    // BNBSolverLogger<NUMERIC_T> const m_logger;

    results_t find_minima(vector<interval_t> domain, vector<NUMERIC_T> const &params)
    {
        size_t i = 0;

        std::cout << "doing the thing" << std::endl;

        std::queue<vector<interval_t>> workq;
        workq.push(domain);

        results_t sresults;

        while (!workq.empty() && i < m_settings.MAXITER)
        {
            i++;

            vector<interval_t> item(workq.front());
            workq.pop();
            auto created_intervals = process_interval(item, params, sresults, workq);
            std::cout << "did a thing and made " << created_intervals << " new things." << std::endl;
        }

        return sresults;
    }

private:
    /**
     * @brief Process an interval `x` and try to refine it via B&B.
     * @param[in] x
     * @param[in] params
     * @param[in] sresults reference to the global solver status
     * @param[in] workq reference to the work queue
     * @return Any new intervals created by the refining process
     *
     * @details
     */
    size_t process_interval(vector<interval_t> const &x,
                            vector<NUMERIC_T> const &params,
                            results_t &sresults, std::queue<vector<interval_t>> &workq)
    {
        std::cout << "processing interval " << print_vector(x).str() << std::endl;
        vector<bool> dims_converged(x.size(), true);
        bool allconverged = true;
        for (size_t i = 0; i < x.size(); i++)
        {
            dims_converged[i] = (width(x[i]) <= m_settings.TOL_X || (x[i].lower() == 0 && x[i].upper() == 0));
            allconverged = allconverged && dims_converged[i];
        }

        std::cout << "  allconverged=" << allconverged << " dims are " << print_vector(dims_converged).str() << std::endl;

        // have convergence in x
        if (allconverged)
        {
            sresults.minima_intervals.push_back(x);
            return 0;
        }

        bool grad_pass = false;
        bool hess_pass = false;

        interval_t h;
        vector<interval_t> dhdx(x.size());
        objective_gradient(x, params, h, dhdx);
        grad_pass = std::all_of(dhdx.begin(), dhdx.end(), [](interval_t ival)
                                { return boost::numeric::zero_in(ival); });
        std::cout << "  h(x) is " << h << std::endl;
        std::cout << "  gradient test passed: " << grad_pass << " and gradient is " << print_vector(dhdx).str() << std::endl;
        // interval cannot contain a local minimum, since the derivative doesn't change sign.
        // therefore we discard it.
        if (!grad_pass)
        {
            std::cout << "  discarding interval " << print_vector(x).str() << std::endl;
            return 0;
        }

        vector<vector<interval_t>> d2hdx2(x.size(), vector<interval_t>(x.size()));
        objective_hessian(x, params, h, d2hdx2);
        hess_pass = sylvesters_criterion(d2hdx2);

        std::cout << "  hessian test pass: " << hess_pass << std::endl;

        // interval contains a change of sign in the gradient, but it is not locally convex.
        // therefore, we choose to bisect the interval and continue the search.
        if (grad_pass && !hess_pass)
        {
            auto items = bisect_interval(x, dims_converged);
            for (auto &item : items)
            {
                workq.push(item);
            }
            return items.size();
        }

        if (grad_pass && hess_pass)
        {
            std::cout << "  Need to narrow interval " << print_vector(x).str() << ", perhaps via gradient descent.";
        }

        // interval contains a change of sign in the gradient AND is locally convex
        // therefore, we decide to narrow the interval via some search method
        return 0;
    }

    /**
     * @brief Cuts the n-dimensional range @c x in each dimension that is not flagged in @c dims_converged
     * @returns @c vector of n-dimensional intervals post-split
     */
    vector<vector<interval_t>> bisect_interval(vector<interval_t> const &x, vector<bool> const &dims_converged)
    {
        vector<vector<interval_t>> res;
        if (dims_converged[0])
        {
            res.emplace_back(1, interval_t(x[0]));
        }
        else
        {
            res.emplace_back(1, interval_t(x[0].lower(), median(x[0])));
            res.emplace_back(1, interval_t(median(x[0]), x[0].upper()));
        }
        for (size_t i = 1; i < x.size(); i++)
        {
            size_t n = res.size();
            for (size_t j = 0; j < n; j++)
            {
                if (dims_converged[i])
                {
                    res[j].emplace_back(x[i]);
                }
                else
                {
                    vector<interval_t> temp(res[j]);
                    res[j].emplace_back(x[i].lower(), median(x[i]));
                    temp.emplace_back(median(x[i]), x[i].upper());
                    res.push_back(temp);
                }
            }
        }
        return res;
    }

    /**
     * @brief Computes the gradient of the objective function using @c dco/c++ adjoint mode.
     * @param[in] x
     * @param[in] params
     * @param[inout] h
     * @param[inout] dhdx
     *
     * @details
     */
    void objective_gradient(vector<interval_t> const &x,
                            vector<NUMERIC_T> const &params,
                            interval_t &h,
                            vector<interval_t> &dhdx)
    {
        // define dco types and get a pointer to the tape
        // unsure how to use ga1sm to expand this to multithreaded programs
        using dco_mode_t = dco::ga1s<interval_t>;
        using active_t = dco_mode_t::type;
        dco::smart_tape_ptr_t<dco_mode_t> tape;
        tape->reset();
        // create active variables and allocate active output variables
        vector<active_t> xa(x.size());
        for (size_t i = 0; i < x.size(); i++)
        {
            xa[i] = x[i];
        }
        active_t ha;
        tape->register_variable(xa.begin(), xa.end());
        // write and interpret the tape
        ha = m_objective(x, params);
        dco::derivative(ha) = 1;
        tape->interpret_adjoint();
        // copy values from active variables to output variables
        h = dco::value(ha);
        for (size_t i = 0; i < x.size(); i++)
        {
            dhdx[i] = dco::derivative(xa[i]);
        }
    }

    /**
     * @brief Computes the hessian of the objective function using dco/c++ tangent mode over adjoint mode.
     * @param[in] x
     * @param[in] params
     * @param[inout] h
     * @param[inout] d2hdx2
     *
     * @details
     */
    void objective_hessian(vector<interval_t> const &x,
                           vector<NUMERIC_T> const &params,
                           interval_t &h,
                           vector<vector<interval_t>> &d2hdx2)
    {
        using dco_tangent_t = dco::gt1s<interval_t>::type;
        using dco_mode_t = dco::ga1s<dco_tangent_t>;
        using active_t = dco_mode_t::type;
        dco::smart_tape_ptr_t<dco_mode_t> tape;
        const size_t ndims = x.size();
        active_t ha;
        vector<active_t> xa(ndims);
        dco::passive_value(xa) = x;
        tape->register_variable(xa.begin(), xa.end());
        auto start_position = tape->get_position();

        for (size_t hrow = 0; hrow < ndims; hrow++)
        {
            dco::derivative(dco::value(xa[hrow])) = 1; // wiggle x[hcol]
            ha = m_objective(xa, params);
            dco::value(dco::derivative(ha)) = 1;                  // set sensitivity to wobbles in h to 1
            tape->interpret_adjoint_and_reset_to(start_position); // interpret and rewind the tape
            for (size_t hcol = 0; hcol < ndims; hcol++)
            {
                d2hdx2[hrow][hcol] = dco::derivative(dco::derivative(xa[hcol]));
                // reset any accumulated values
                dco::derivative(dco::derivative(xa[hcol])) = 0;
                dco::value(dco::derivative(xa[hcol])) = 0;
            }
            dco::derivative(dco::value(xa[hrow])) = 0; // no longer wiggling x[hcol]
        }
        h = dco::passive_value(ha);
    }
};

#endif // header guard