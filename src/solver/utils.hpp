#ifndef _BNB_UTILS_HPP
#define _BNB_UTILS_HPP

#include <iostream>
#include <vector>
using std::vector;

#include "boost/numeric/interval.hpp"
#include "fmt/format.h"

/**
 * @brief Get submatrix of square matrix A formed by removing columns p and q
 */
template <typename T>
vector<vector<T>> submatrix(vector<vector<T>> const &A, const size_t p, const size_t q)
{
    vector<vector<T>> Apq(A.size() - 1, vector<T>(A.size() - 1));
    size_t m = 0;
    size_t n = 0;
    // iterate over A and Apq but skip indexing Apq if i == p or j == q
    for (size_t i = 0; i < A.size(); i++)
    {
        if (i == p)
        {
            continue;
        }
        for (size_t j = 0; j < A.size(); j++)
        {
            if (j == q)
            {
                continue;
            }
            Apq[m][n] = A[i][j];
            n++;
        }
        m++;
    }
    return Apq;
}

/**
 * @brief Compute the determinant of a square matrix A expressed as std::vector of vectors
 * @param A Square matrix.
 * @return The determinant.
 */
template <typename T>
T determinant(vector<vector<T>> A)
{
    // assert that A has size and is square
    assert(A.size() > 0 && A.size() == A[0].size());
    T det;
    // write out A.size() = 1, 2, 3 explicitly to save on the creation of 2x2 and 3x3 minors
    if (A.size() == 1)
    {
        det = A[0][0];
    }
    else if (A.size() == 2)
    {
        det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
    }
    else if (A.size() == 3)
    {
        det += A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]);
        det -= A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]);
        det += A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
    }
    else
    {
        // det A = \sum_{i=1}^N -1^(i-1) * det (i,1 submatrix of A)
        det = 0;
        // even-numbered minors
        for (size_t i = 0; i < A.size(); i += 2)
        {
            det += determinant(submatrix(A, 0, i));
        }
        // odd-numbered minors
        for (size_t i = 1; i < A.size(); i += 2)
        {
            det -= determinant(submatrix(A, 0, i));
        }
    }
    return det;
}

/**
 * @brief Check if argument is less than or equal to zero.
 */
template <typename T>
inline bool nonpositive(T arg)
{
    return arg <= 0;
}

/**
 * @brief Check if the lower end of the interval is less than or equal to zero.
 */
template <typename T, typename P>
inline bool nonpositive(boost::numeric::interval<T, P> arg)
{
    return arg.lower() <= 0;
}

template <typename T>
inline bool nonnegative(T arg){
    return arg >= 0;
}

template <typename T, typename P>
inline bool nonnegative(boost::numeric::interval<T, P> arg){
    return arg.upper() >= 0;
}

/**
 * @brief Use Sylvester's criterion to check for postive-definiteness of a Hermitian matrix
 * @param A Hermitian matrix A
 * @return The positive definite-ness of the matrix.
 */
template <typename T>
bool sylvesters_criterion(vector<vector<T>> const &A)
{
    for (size_t n = 1; n <= A.size(); n++)
    {
        vector<vector<T>> nth_principal_minor(n, vector<T>(n));
        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 0; j < n; j++)
            {
                nth_principal_minor[i][j] = A[i][j];
            }
        }
        auto det = determinant(nth_principal_minor);
        if (nonpositive(det))
        {
            return false;
        }
    }
    return true;
}

template <typename T>
std::stringstream print_vector(vector<T> const &arg)
{
    std::stringstream out;
    out << "[";
    for (size_t i = 0; i < arg.size(); i++)
    {
        out << arg[i];
        if (i < arg.size() - 1)
        {
            out << ", ";
        };
    }
    out << "]";
    return out;
}
#endif