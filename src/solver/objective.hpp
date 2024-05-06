/**
 * @file daeo_solver.hpp
 * @author Sasha [fleming@stce.rwth-aachen.de]
 * @brief Wrapper classes for functions of the form f(t, x, y, p).
 */
#ifndef _FUNCTION_WRAPPER_HPP
#define _FUNCTION_WRAPPER_HPP

#include <vector>

// It's recommended to include eigen/boost before DCO
#include "Eigen/Dense"
#include "boost/numeric/interval.hpp"
#include "dco.hpp"

#include "utils/ntuple.hpp"

/**
 * @brief Wraps a function of the form f(t, x, y, p) for use with the optimizer
 * and solver. Assumes that the return type of f is a scalar and matches the
 * type of scalar y.
 */
template <typename FN> class DAEOWrappedFunction {

  mutable size_t n_h_evaluations = 0;
  mutable size_t n_dy_evaluations = 0;
  mutable size_t n_d2y_evaluations = 0;
  mutable size_t n_dx_evaluations = 0;
  mutable size_t n_d2xy_evaluations = 0;

  // This return type is clumsy. Need to figure out a way to express
  // "promote this to a dco type, otherwise promote to an interval, otherwise
  // return a passive numerical value"

  /**
   * @brief The function to wrap and augment with partial derivative routines.
   */
  FN const wrapped_fn;

public:
  DAEOWrappedFunction(FN const &t_fn) : wrapped_fn{t_fn} {};

  /**
   * @brief Evaluate @c m_fn at the provided arguments.
   * @returns Value of @c m_fn .
   */
  template <typename NUMERIC_T, typename XT, typename YT, int YDIMS, int PDIMS>
  auto value(NUMERIC_T const t, XT const x, Eigen::Vector<YT, YDIMS> const &y,
             Eigen::Vector<NUMERIC_T, PDIMS> const &p) const
      -> decltype(wrapped_fn(t, x, y, p)) {
    n_h_evaluations += 1;
    return wrapped_fn(t, x, y, p);
  }

  template <typename NUMERIC_T, typename Y_ACTIVE_T, int YDIMS, int PDIMS>
  auto grad_y(NUMERIC_T const t, NUMERIC_T const x,
              Eigen::Vector<Y_ACTIVE_T, YDIMS> const &y,
              Eigen::Vector<NUMERIC_T, PDIMS> const &p) const
      -> Eigen::Vector<decltype(wrapped_fn(t, x, y, p)), YDIMS> {
    n_dy_evaluations += 1;
    // define dco types and get a pointer to the tape
    // unsure how to use ga1sm to expand this to multithreaded programs
    using dco_mode_t = dco::ga1s<Y_ACTIVE_T>;
    using active_t = typename dco_mode_t::type;
    dco::smart_tape_ptr_t<dco_mode_t> tape;
    tape->reset();
    // define active inputs
    // if size of y is not known at compile time, we need to set the size
    // explicitly otherwise the Vector<...>(nrows) constructor is a no-op
    Eigen::Vector<active_t, YDIMS> y_active(y.rows());
    // no vector assignment routines are available for eigen+dco (until dco
    // base 4.2)
    for (int i = 0; i < y.rows(); i++) {
      dco::value(y_active(i)) = y(i); // eigen accesses done with operator()
      tape->register_variable(y_active(i));
    }
    // and active outputs
    active_t h_active = wrapped_fn(t, x, y_active, p);
    tape->register_output_variable(h_active);
    dco::derivative(h_active) = 1;
    tape->interpret_adjoint();
    // harvest derivative
    Eigen::Vector<Y_ACTIVE_T, YDIMS> dhdy(y.rows());
    for (int i = 0; i < y.rows(); i++) {
      dhdy(i) = dco::derivative(y_active(i));
    }
    return dhdy;
  }

  template <typename NUMERIC_T, typename X_ACTIVE_T, int YDIMS, int PDIMS>
  X_ACTIVE_T grad_x(NUMERIC_T const t, X_ACTIVE_T const x,
                    Eigen::Vector<NUMERIC_T, YDIMS> const &y,
                    Eigen::Vector<NUMERIC_T, PDIMS> const &p) const {
    n_dx_evaluations += 1;
    using dco_mode_t = dco::gt1s<X_ACTIVE_T>;
    using active_t = typename dco_mode_t::type;
    active_t x_active;
    active_t h_active;
    dco::value(x_active) = x;
    dco::derivative(x_active) = 1;
    h_active = wrapped_fn(t, x_active, y, p);
    return dco::derivative(h_active);
  }

  template <typename NUMERIC_T, typename Y_ACTIVE_T, int YDIMS, int PDIMS>
  Eigen::Matrix<Y_ACTIVE_T, YDIMS, YDIMS>
  hess_y(NUMERIC_T const t, NUMERIC_T const x,
         Eigen::Vector<Y_ACTIVE_T, YDIMS> const &y,
         Eigen::Vector<NUMERIC_T, PDIMS> const &p) const {
    n_d2y_evaluations += 1;
    using dco_tangent_t = typename dco::gt1s<Y_ACTIVE_T>::type;
    using dco_mode_t = dco::ga1s<dco_tangent_t>;
    using active_t = typename dco_mode_t::type;
    dco::smart_tape_ptr_t<dco_mode_t> tape;
    active_t h_active;
    Eigen::Vector<active_t, YDIMS> y_active(y.rows());
    for (int i = 0; i < y.rows(); i++) {
      dco::passive_value(y_active(i)) = y(i);
      tape->register_variable(y_active(i));
    }
    auto start_position = tape->get_position();
    // Hessian of a scalar function is a symmetric square matrix (provided
    // second derivative symmetry holds)
    Eigen::Matrix<Y_ACTIVE_T, YDIMS, YDIMS> d2hdy2(y.rows(), y.rows());
    // these loops go row-by-row, which is slow. easy performance gains are to
    // be had.
    for (int hrow = 0; hrow < y.rows(); hrow++) {
      dco::derivative(dco::value(y_active(hrow))) = 1; // wiggle y[hrow]
      h_active = wrapped_fn(t, x, y_active, p);        // compute h
      // set sensitivity to wobbles in h to 1
      dco::value(dco::derivative(h_active)) = 1;
      tape->interpret_adjoint_and_reset_to(start_position);
      for (int hcol = 0; hcol < y.rows(); hcol++) {
        d2hdy2(hrow, hcol) = dco::derivative(dco::derivative(y_active[hcol]));
        // reset any accumulated values
        dco::derivative(dco::derivative(y_active(hcol))) = 0;
        dco::value(dco::derivative(y_active(hcol))) = 0;
      }
      // no longer wiggling y[hrow]
      dco::derivative(dco::value(y_active(hrow))) = 0;
    }
    return d2hdy2;
  }

  template <typename NUMERIC_T, typename XY_ACTIVE_T, int YDIMS, int PDIMS>
  Eigen::Vector<XY_ACTIVE_T, YDIMS>
  d2dxdy(NUMERIC_T const t, XY_ACTIVE_T const x,
         Eigen::Vector<XY_ACTIVE_T, YDIMS> const &y,
         Eigen::Vector<NUMERIC_T, PDIMS> const &p) const {
    n_d2xy_evaluations += 1;
    using dco_tangent_t = typename dco::gt1s<XY_ACTIVE_T>::type;
    using dco_mode_t = dco::ga1s<dco_tangent_t>;
    using active_t = typename dco_mode_t::type;
    dco::smart_tape_ptr_t<dco_mode_t> tape;
    active_t h_active;
    Eigen::Vector<active_t, YDIMS> y_active(y.rows());
    for (int i = 0; i < y.rows(); i++) {
      dco::passive_value(y_active(i)) = y(i);
      tape->register_variable(y_active(i));
    }
    active_t x_active;
    dco::passive_value(x_active) = x;
    tape->register_variable(x_active);
    dco::derivative(dco::value(x_active)) = 1;       // wiggle x
    h_active = wrapped_fn(t, x_active, y_active, p); // compute h
    dco::value(dco::derivative(h_active)) = 1;       // sensitivity to h is 1
    tape->interpret_adjoint();
    // If YDIMS is known at compile time, constructor is a no-op.
    Eigen::Vector<XY_ACTIVE_T, YDIMS> ddxddy(y.rows());
    // harvest derivative
    for (int i = 0; i < ddxddy.rows(); i++) {
      ddxddy(i) =
          dco::derivative(dco::derivative(y_active(i))); // harvest d2dxdy
    }
    return ddxddy;
  }

  /**
   * @brief Returns, as a tuple of 5 elements, the number of calls to each of
   * the provided drivers.
   */
  ntuple<5, size_t> statistics() const {
    return {n_h_evaluations, n_dy_evaluations, n_dx_evaluations,
            n_d2y_evaluations, n_d2xy_evaluations};
  }
};

#endif