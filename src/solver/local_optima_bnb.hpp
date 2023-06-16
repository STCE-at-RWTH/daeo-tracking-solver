/**
 * @file local_optima_bnb.hpp
 * @author Sasha [fleming@stce.rwth-aachen.de]
 * @brief Implementation of a branch and bound solver to find regions containing local optima of f(x; p).
 */
#ifndef _LOCAL_OPTIMA_BNB_HPP // header guard
#define _LOCAL_OPTIMA_BNB_HPP

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
    BNBSolverSettings<NUMERIC_T> const &m_settings;

    void find_minima(vector<INTERVAL_T> domain, vector<NUMERIC_T> params)
    {
        size_t i = 0;
        std::queue<vector<INTERVAL_T>> workq;
        workq.push(domain);

        while (!workq.empty() && i < settings.MAXITER)
        {
            i++;
            for (auto &item : refine_interval(workq.pop()))
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
    vector<vector<INTERVAL_T>> refine_interval(vector<INTERVAL_T> const &x)
    {
        vector<vector<INTERVAL_T>> newitems;

        vector<bool> dims_converged(x.size(), true);
        for (size_t i = 0; i < x.size(); i++)
        {
            dims_converged[i] = !(x[i].width() <= settings.TOL_X || (x[i].lower() == 0 && x[i].upper == 0));
        }

        bool value_test = false;
        bool gradient_test = false;
        bool hessian_test = false;
        INTERVAL_T h;
        vector<INTERVAL_T> dhdx(x.size(), INTERVAL_T::whole());

        return newitems;
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
        // since OMP task-based programs
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

        // set up the tape
        tape->reset();
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
                           INTERVAL_T &h,
                           vector<vector<INTERVAL_T>> &d2hdx2)
    {
    }
};

#endif // header guard