/**
 * @file daeo_solver.hpp
 * @author Sasha [fleming@stce.rwth-aachen.de]
 * @brief Wrapper classes for functions of the form f(t, x, y, p).
 */
#ifndef _FUNCTION_WRAPPER_HPP
#define _FUNCTION_WRAPPER_HPP

#include <concepts>
#include <tuple>
#include <vector>

#include "boost/numeric/interval.hpp"
#include "Eigen/Dense" // include eigen before dco
#include "dco.hpp"

/**
 * @brief Wraps a function of the form f(t, x, y, p) for use with the optimizer and solver.
 * Assumes that the return type of f is a scalar and matches the type of scalar y.
 */
template <typename FN>
class DAEOWrappedFunction
{

    // This return type is clumsy. Need to figure out a way to express
    // "promote this to a dco type, otherwise promote to an interval, otherwise return a passive numerical value"

    /**
     * @brief The function to wrap and augment with partial derivative routines.
     */
    FN const m_fn;

public:
    DAEOWrappedFunction(FN const &t_fn) : m_fn{t_fn} {};

    /**
     * @brief Evaluate @c m_fn at the provided arguments.
     */
    template <typename NUMERIC_T, typename XT, typename YT, int YDIMS, int PDIMS>
    YT value(NUMERIC_T const t, XT const x,
             Eigen::Vector<YT, YDIMS> const &y,
             Eigen::Vector<NUMERIC_T, PDIMS> const &p) const
    {
        return m_fn(t, x, y, p);
    }

    template <typename NUMERIC_T, typename Y_ACTIVE_T, int YDIMS, int PDIMS>
    Eigen::Vector<Y_ACTIVE_T, YDIMS> grad_y(NUMERIC_T const t, NUMERIC_T const x,
                                            Eigen::Vector<Y_ACTIVE_T, YDIMS> const &y,
                                            Eigen::Vector<NUMERIC_T, PDIMS> const &p) const
    {
        // define dco types and get a pointer to the tape
        // unsure how to use ga1sm to expand this to multithreaded programs
        using dco_mode_t = dco::ga1s<Y_ACTIVE_T>;
        using active_t = dco_mode_t::type;
        dco::smart_tape_ptr_t<dco_mode_t> tape;
        tape->reset();
        // define active inputs
        // if size of y is not known at compile time, we need to set the size explicitly
        // otherwise the Vector<...>(nrows) constructor is a no-op
        Eigen::Vector<active_t, YDIMS> y_active(y.rows());
        // no vector assignment routines are available for eigen+dco
        for (int i = 0; i < y.rows(); i++)
        {
            dco::value(y_active(i)) = y(i); // eigen accesses done with operator()
            tape->register_variable(y_active(i));
        }
        // and active outputs
        active_t h_active = m_fn(t, x, y_active, p);
        tape->register_output_variable(h_active);
        dco::derivative(h_active) = 1;
        tape->interpret_adjoint();
        // harvest derivative
        Eigen::Vector<Y_ACTIVE_T, YDIMS> dhdy(y.rows());
        for (int i = 0; i < y.rows(); i++)
        {
            dhdy(i) = dco::derivative(y_active(i));
        }
        return dhdy;
    }

    template <typename NUMERIC_T, typename X_ACTIVE_T, int YDIMS, int PDIMS>
    X_ACTIVE_T grad_x(NUMERIC_T const t, X_ACTIVE_T const x,
                      Eigen::Vector<NUMERIC_T, YDIMS> const &y,
                      Eigen::Vector<NUMERIC_T, PDIMS> const &p) const
    {
        using dco_mode_t = dco::gt1s<X_ACTIVE_T>;
        using active_t = dco_mode_t::type;
        active_t x_active;
        active_t h_active;
        dco::value(x_active) = x;
        dco::derivative(x_active) = 1;
        h_active = m_fn(t, x_active, y, p);
        return dco::derivative(h_active);
    }

    template <typename NUMERIC_T, typename Y_ACTIVE_T, int YDIMS, int PDIMS>
    Eigen::Matrix<Y_ACTIVE_T, YDIMS, YDIMS> hess_y(NUMERIC_T const t, NUMERIC_T const x,
                                                   Eigen::Vector<Y_ACTIVE_T, YDIMS> const &y,
                                                   Eigen::Vector<NUMERIC_T, PDIMS> const &p) const
    {
        using dco_tangent_t = dco::gt1s<Y_ACTIVE_T>::type;
        using dco_mode_t = dco::ga1s<dco_tangent_t>;
        using active_t = dco_mode_t::type;
        dco::smart_tape_ptr_t<dco_mode_t> tape;
        active_t h_active;
        Eigen::Vector<active_t, YDIMS> y_active(y.rows());
        for (int i = 0; i < y.rows(); i++)
        {
            dco::passive_value(y_active(i)) = y(i);
            tape->register_variable(y_active(i));
        }
        auto start_position = tape->get_position();
        // Hessian of a scalar function is a symmetric square matrix (provided second derivative symmetry holds)
        Eigen::Matrix<Y_ACTIVE_T, YDIMS, YDIMS> d2hdy2(y.rows(), y.rows());
        // these loops go row-by-row, which is slow. easy performance gains are to be had.
        for (int hrow = 0; hrow < y.rows(); hrow++)
        {
            dco::derivative(dco::value(y_active(hrow))) = 1; // wiggle y[hrow]
            h_active = m_fn(t, x, y_active, p);              // compute h
            dco::value(dco::derivative(h_active)) = 1;       // set sensitivity to wobbles in h to 1
            // dhdy[hrow] = dco::derivative(dco::value(h_active));   // wobble in h generated by wiggling y[hrow]
            tape->interpret_adjoint_and_reset_to(start_position); // interpret and rewind the tape
            for (int hcol = 0; hcol < y.rows(); hcol++)
            {
                d2hdy2(hrow, hcol) = dco::derivative(dco::derivative(y_active[hcol]));
                // reset any accumulated values
                dco::derivative(dco::derivative(y_active(hcol))) = 0;
                dco::value(dco::derivative(y_active(hcol))) = 0;
            }
            dco::derivative(dco::value(y_active(hrow))) = 0; // no longer wiggling y[hrow]
        }
        return d2hdy2;
    }

    template <typename NUMERIC_T, typename XY_ACTIVE_T, int YDIMS, int PDIMS>
    Eigen::Vector<XY_ACTIVE_T, YDIMS> d2dxdy(NUMERIC_T const t, XY_ACTIVE_T const x,
                                             Eigen::Vector<XY_ACTIVE_T, YDIMS> const &y,
                                             Eigen::Vector<NUMERIC_T, PDIMS> const &p) const
    {
        using dco_tangent_t = dco::gt1s<XY_ACTIVE_T>::type;
        using dco_mode_t = dco::ga1s<dco_tangent_t>;
        using active_t = dco_mode_t::type;
        dco::smart_tape_ptr_t<dco_mode_t> tape;
        active_t h_active;
        Eigen::Vector<active_t, YDIMS> y_active(y.rows());
        for (int i = 0; i < y.rows(); i++)
        {
            dco::passive_value(y_active(i)) = y(i);
            tape->register_variable(y_active(i));
        }
        active_t x_active;
        dco::passive_value(x_active) = x;
        tape->register_variable(x_active);
        dco::derivative(dco::value(x_active)) = 1; // wiggle x
        h_active = m_fn(t, x_active, y_active, p); // compute h
        dco::value(dco::derivative(h_active)) = 1; // sensitivity to h is 1
        tape->interpret_adjoint();
        // If YDIMS is known at compile time, constructor is a no-op.
        Eigen::Vector<XY_ACTIVE_T, YDIMS> ddxddy(y.rows());
        // harvest derivative
        for (int i = 0; i < ddxddy.rows(); i++)
        {
            ddxddy(i) = dco::derivative(dco::derivative(y_active(i))); // harvest d2dxdy
        }
        return ddxddy;
    }
};

#endif