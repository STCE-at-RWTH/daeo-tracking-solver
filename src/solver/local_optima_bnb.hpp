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
using std::vector;

#include "dco.hpp"
#include "boost/numeric/interval.hpp"

#include "bnb_settings.hpp"
#include "utils.hpp"

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
    using INTERVAL_T = boost::numeric::interval<NUMERIC_T, POLICIES>;
    /**
     * @brief The objective function h(x; p) of which to find the minima.
     */
    OBJECTIVE_T const &m_objective;

    /**
     * @brief
     */
    BNBSolverSettings<NUMERIC_T> const m_settings;

    /**
     * @brief
     */
    BNBSolverLogger<NUMERIC_T> const m_logger;

    void find_minima(vector<INTERVAL_T> domain, vector<NUMERIC_T> const &params)
    {
        size_t i = 0;

        std::queue<vector<INTERVAL_T>> workq;
        workq.push(domain);

        solverstatus status;

        while (!workq.empty() && i < settings.MAXITER)
        {
            i++;
            auto item = workq.pop();
            auto created_intervals = process_interval(item, params, status, workq);
        }
    }

private:
    /**
     * @struct solverstatus
     * @brief Bookkeeping for global information the solver needs to keep track of.
     */
    struct solverstatus
    {
        NUMERIC_T optima_supremum = std::numeric_limits<NUMERIC_T>::max();
        vector<INTERVAL_T> minima_intervals;
    };

    /**
     * @brief Process an interval `x` and try to refine it via B&B.
     * @param[in] x
     * @param[in] params
     * @return Any new intervals created by the refining process
     *
     * @details
     */
    size_t process_interval(vector<INTERVAL_T> const &x,
                            vector<NUMERIC_T> const &params,
                            solverstatus &sstatus, std::queue<vector<INTERVAL_T>> &workq)
    {
        vector<bool> dims_converged(x.size(), true);
        bool allconverged = true;
        for (size_t i = 0; i < x.size(); i++)
        {
            dims_converged[i] = (x[i].width() <= settings.TOL_X || (x[i].lower() == 0 && x[i].upper == 0));
            allconverged = allconverged && dims_converged[i];
        }
        // have convergence in x
        if (allconverged)
        {
            sstatus.minima_intervals.push_back(x);
            return 0;
        }

        bool val_pass = true; // not sure how to do value testing.
        bool grad_pass = false;
        bool hess_pass = false;

        INTERVAL_T h;
        vector<INTERVAL_T> dhdx(x.size());
        objective_gradient(x, params, h, dhdx);
        grad_pass = std::all_of(dhdx.begin(), dhdx.end(), [](INTERVAL_T ival)
                                { return boost::numeric::interval::zero_in(ival); });
        // interval cannot contain a local minimum, since the derivative doesn't change sign.
        // therefore we discard it.
        if (!grad_pass)
        {
            return 0;
        }

        vector<vector<INTERVAL_T>> d2hdx2(x.size() vector<INTERVAL_T>(x.size()));
        objective_hessian(x, params, h, d2hdx2);
        hess_pass = sylvesters_criterion(d2hdx2);

        // interval contains a change of sign in the gradient, but it is not locally convex.
        // therefore, we choose to bisect the interval and continue the search.
        if (grad_pass && !hess_pass)
        {

            if (any_active)
            {
                size_t n = 0;
                for (auto &item : bisect_interval(x, dims_converged))
                {
                    n++;
                    workq.push(item);
                }
                return n;
            }
        }

        // interval contains a change of sign in the gradient AND is locally convex
    }

    vector<vector<INTERVAL_T>> bisect_interval(vector<INTERVAL_T> const &x, vector<bool> const &dims_active)
    {
        vector<vector<INTERVAL_T>> res(2, vector<INTERVAL_T>(x.size()));
        for (size_t i = 0; i < x.size(); i++)
        {
            if (!dims_active[i])
            {
                
            }
            else
            {
            }
        }
        return res;
    }

    /**
     * @brief Computes the gradient of the objective function using dco/c++ adjoint mode.
     * @param[in] x
     * @param[in] params
     * @param[inout] h
     * @param[inout] dhdx
     *
     * @details
     */
    void objective_gradient(vector<INTERVAL_T> const &x,
                            vector<NUMERIC_T> const &params,
                            INTERVAL_T &h,
                            vector<INTERVAL_T> &dhdx)
    {
        // define dco types and get a pointer to the tape
        // unsure how to use ga1sm to expand this to multithreaded programs
        using dco_mode_t = dco::ga1s<INTERVAL_T>;
        using active_t = dco_mode_t::type;
        dco::smart_tape_ptr_t<dco_mode_t> tape;
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
    void objective_hessian(vector<INTERVAL_T> const &x,
                           vector<NUMERIC_T> const &params,
                           INTERVAL_T &h,
                           vector<vector<INTERVAL_T>> &d2hdx2)
    {
        using dco_tangent_t = dco::gt1s<INTERVAL_T>::type;
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
        h = dco::passive_value(ha)
    }
};

#endif // header guard