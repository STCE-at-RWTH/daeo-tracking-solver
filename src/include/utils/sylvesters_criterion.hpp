#ifndef _SYLVESTERS_CRITERION_HPP
#define _SYLVESTERS_CRITERION_HPP

#include "boost/numeric/interval.hpp"
#include "Eigen/Dense"

/**
 * @brief Check if argument is less than or equal to zero.
 */
template <typename T>
inline bool nonpositive(T arg) { return arg <= 0; }

/**
 * @brief Check if the lower end of the interval is less than or equal to zero.
 */
template <typename T, typename P>
inline bool nonpositive(boost::numeric::interval<T, P> arg) { return arg.lower() <= 0; }

/**
 * @brief Check if the argument is greater than or equal to zero
 */
template <typename T>
inline bool nonnegative(T arg) { return arg >= 0; }

/**
 * @brief Check if the upper end of the interval is greater than or equal to zero
 */
template <typename T, typename P>
inline bool nonnegative(boost::numeric::interval<T, P> arg) { return arg.upper() >= 0; }

template <typename T, int NDIMS>
bool is_positive_definite(Eigen::Matrix<T, NDIMS, NDIMS> const &A)
{
    if constexpr (NDIMS == Eigen::Dynamic)
    {
        if (A.rows() != A.cols())
        {
            return false;
        }
    } // non-square matrices cannot be positive definite
    auto d = A.determinant();
    for (int n = 0; n < A.rows(); n++)
    {
        Eigen::MatrixX<T> nth_principal_submatrix = A.block(0, 0, n + 1, n + 1);
        if (nonpositive(nth_principal_submatrix.determinant()))
        {
            return false;
        }
    }
    return true;
}

template <typename T, int NDIMS>
bool is_negative_definite(Eigen::Matrix<T, NDIMS, NDIMS> const &A)
{
    if constexpr (NDIMS == Eigen::Dynamic)
    {
        if (A.rows() != A.cols())
        {
            return false;
        }
    } // non-square matrices cannot be positive definite

    for (int n = 0; n < A.rows(); n++)
    {
        Eigen::MatrixX<T> nth_principal_submatrix(n + 1, n + 1);
        nth_principal_submatrix = A.block(0, 0, n + 1, n + 1);
        if (nonnegative(nth_principal_submatrix.determinant()))
        {
            return false;
        }
    }
    return true;
}

#endif