/**
 * @file local_optima_bnb.hpp
 * @author Sasha [fleming@stce.rwth-aachen.de]
 * @brief Implementation of a branch and bound solver to find regions containing local optima of f(x; p).
 */
#ifndef _LOCAL_OPTIMA_BNB_HPP // header guard
#define _LOCAL_OPTIMA_BNB_HPP

#include <bitset>
#include <queue>
#include <vector>
using std::vector;

#include "dco.hpp"
#include "boost/numeric/interval.hpp"

#include "bnb_settings.hpp"

/**
 * @class LocalOptimaBNBSolver
 * @brief Finds regions containing local minima of h(x;p)
 * @tparam OBJECTIVE_T Type of the objective function.
 * @tparam NUMERIC_T Type of the parameters to f(x; p).
 * @tparam INTERVAL_T Interval type (created from NUMERIC_T).
 */
template <typename OBJECTIVE_T, typename NUMERIC_T, typename INTERVAL_T>
class LocalOptimaBNBSolver
{

public:
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

        while (!workq.empty() && i < settings.MAXITER)
        {
            i++;
            for (auto &item : process_interval(workq.pop(), params))
            {
                workq.push(item);
            }
        }
    }

private:
    /**
     * @brief Process an interval `x` and try to refine it via B&B.
     * @param[in] x
     * @return Any new intervals created by the refining process
     *
     * @details
     */
    vector<vector<INTERVAL_T>> process_interval(vector<INTERVAL_T> const &x, vector<NUMERIC_T> const &params)
    {

        vector<vector<INTERVAL_T>> newitems;

        vector<bool> dims_active(x.size(), true);
        for (size_t i = 0; i < x.size(); i++)
        {
            dims_active[i] = (x[i].width() <= settings.TOL_X || (x[i].lower() == 0 && x[i].upper == 0));
        }

        bool val_pass = false;
        bool grad_pass = false;
        bool hess_pass = false;

        INTERVAL_T h;
        vector<INTERVAL_T> dhdx(x.size());
        vector<vector<INTERVAL_T>> d2hdx2(x.size() vector<INTERVAL_T>(x.size()));

        // condition for splitting
        bool any_active = !std::none_of(dims_active.begin(), dims_active.end(),
                                        [](bool v)
                                        { return v; });
        if (any_active)
        {
            for (auto &item : bisect_interval(x, dims_active))
            {
                newitems.push_back(item);
            }
        }

        return newitems;
    }

    vector<vector<INTERVAL_T>> bisect_interval(vector<INTERVAL_T> const &x, vector<bool> const &dims_active)
    {
        vector<vector<INTERVAL_T>> res(2, vector<INTERVAL_T>(x.size()));
        for (size_t i = 0; i < x.size(); i++)
        {
            if (dims_active[i])
            {
            }
            else
            {
            }
        }
        return res;
    }

    /**
     *
     */
    void value_test(bitset<3> &test_status, INTERVAL_T &h, vector<NUMERIC_T> params)
    {
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
     * @brief Computes the hessian of the objective function using dco/c++ adjoint mode over tangent mode.
     * @param[in] x
     * @param[in] params
     * @param[inout] h
     * @param[inout] d2hdx2
     *
     * @details
     */
    void objective_hessian(vector<INTERVAL_T> const &x,
                           vector<NUMERIC_T> const &params,
                           vector<vector<INTERVAL_T>> &d2hdx2)
    {
        using dco_tangent_t = dco::gt1s<T>::type;
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
            dco::value(dco::derivative(ha)) = 1; // set sensitivity to wobbles in y to 1
            tape->interpret_adjoint_and_reset_to(start_position);
            for (size_t hcol = 0; hcol < ndims; hcol++)
            {
                d2hdx2[hrow][hcol] = dco::derivative(dco::derivative(xa[hcol]));
                // reset any accumulated values
                dco::derivative(dco::derivative(xa[hcol])) = 0;
                dco::value(dco::derivative(xa[hcol])) = 0;
            }
            dco::derivative(dco::value(xa[hrow])) = 0; // no longer wiggling x[hcol]
        }
    }
};

#endif // header guard