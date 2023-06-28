#include <cmath>
#include <iostream>
#include <vector>
using std::vector;

#include "boost/numeric/interval.hpp"
using boost::numeric::square;

#include "dco.hpp"

template <typename T>
using boost_interval_transc_t = boost::numeric::interval<
    T, boost::numeric::interval_lib::policies<
           boost::numeric::interval_lib::save_state<
               boost::numeric::interval_lib::rounded_transc_std<T>>,
           boost::numeric::interval_lib::checking_base<T>>>;
using double_ival = boost_interval_transc_t<double>;

template <typename T>
std::stringstream print_vector(vector<T> &arg)
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

/**
 * @brief Get the pq-minor of square matrix A
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
    if (A.size() == 1)
    {
        det = A[0][0];
    }
    else if (A.size() == 2)
    {
        det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
    }
    else
    {
        // det A = \sum_{i=1}^N -1^(i-1) * det (i,1 minor of A)
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
 * @brief Use Sylvester's criterion to check for postive-definiteness of a Hermitian matrix
 * @param A Hermitian matrix A
 * @return The positive definite-ness of the matrix.
 */
template <typename T>
bool sylvesters_criterion(vector<vector<T>> const &A)
{
    for (size_t n = 1; n <= A.size(); n++)
    {
        vector<vector<T>> nth_minor(n, vector<T>(n));
        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 0; j < n; j++)
            {
                nth_minor[i][j] = A[i][j];
            }
        }
        if (determinant(nth_minor) <= 0)
        {
            return false;
        }
    }
    return true;
}

template <typename AT, typename PT>
AT f1(vector<AT> const &x, PT const &params)
{
    return pow(x[0] - params[0], 2) - params[1] * pow(x[1], 2);
}

template <typename T, typename FP_T>
void gradient(vector<T> const &x,
              T &y,
              vector<T> &dfdx, vector<FP_T> const &params)
{
    using dco_mode_t = dco::ga1s<T>;
    using active_t = dco_mode_t::type;
    dco::smart_tape_ptr_t<dco_mode_t> tape;

    // create active variables
    vector<active_t> xa(x.size());
    active_t ya;
    for (size_t i = 0; i < x.size(); i++)
    {
        dco::value(xa[i]) = x[i];
    }
    tape->reset();
    tape->register_variable(xa.begin(), xa.end());
    ya = f1(xa, params);
    y = dco::value(ya);
    dco::derivative(ya) = 1;
    tape->interpret_adjoint();
    for (size_t i = 0; i < xa.size(); i++)
    {
        dfdx[i] = dco::derivative(xa[i]);
    }
}

template <typename T, typename FP_T>
void hessian_via_grad(vector<T> const &x,
                      vector<vector<T>> &ddxdfdx,
                      vector<FP_T> const &params)
{
    using dco_mode_t = dco::gt1s<T>;
    using active_t = dco_mode_t::type;

    const size_t ndims = x.size();
    vector<active_t> xa(ndims);
    for (size_t i = 0; i < ndims; i++)
    {
        dco::value(xa[i]) = x[i];
        dco::derivative(xa[i]) = 0;
    }
    for (size_t hcol = 0; hcol < ndims; hcol++)
    {
        dco::derivative(xa[hcol]) = 1;
        active_t y;
        vector<active_t> dfdx(ndims);
        gradient(xa, y, dfdx, params);

        for (size_t hrow = 0; hrow < ndims; hrow++)
        {
            ddxdfdx[hrow][hcol] = dco::derivative(dfdx[hrow]);
        }

        dco::derivative(xa[hcol] = 0);
    }
}

template <typename T, typename FP_T>
void hessian(vector<T> const &x,
             vector<vector<T>> &df2dx2,
             vector<FP_T> const &params)
{
    using dco_tangent_t = dco::gt1s<T>::type;
    using dco_mode_t = dco::ga1s<dco_tangent_t>;
    using active_t = dco_mode_t::type;

    dco::smart_tape_ptr_t<dco_mode_t> tape;

    const size_t ndims = x.size();
    active_t y_active;
    vector<active_t> x_active(ndims);
    dco::passive_value(x_active) = x;
    tape->register_variable(x_active.begin(), x_active.end());
    auto start_position = tape->get_position();

    for (size_t hrow = 0; hrow < ndims; hrow++)
    {
        dco::derivative(dco::value(x_active[hrow])) = 1; // wiggle x[hcol]
        y_active = f1(x_active, params);
        dco::value(dco::derivative(y_active)) = 1; // set sensitivity to wobbles in y to 1
        tape->interpret_adjoint_and_reset_to(start_position);
        for (size_t hcol = 0; hcol < ndims; hcol++)
        {
            df2dx2[hrow][hcol] = dco::derivative(dco::derivative(x_active[hcol]));
            // reset any accumulated values
            dco::derivative(dco::derivative(x_active[hcol])) = 0;
            dco::value(dco::derivative(x_active[hcol])) = 0;
        }
        dco::derivative(dco::value(x_active[hrow])) = 0; // no longer wiggling x[hcol]
    }
}

int main(int argc, char *argv[])
{
    vector<double_ival> x({double_ival::hull(-5, 2),
                           double_ival::hull(-4, 1)});
    vector<double> p({1.0, -2.0});
    double stepsize = 0.1;

    double_ival t{x[0].lower(), x[0].upper()};

    double_ival y;
    vector<double_ival> dfdx(x.size());
    vector<double> dfdx_left(x.size());
    vector<double> dfdx_right(x.size());
    std::cout << "doing" << std::endl;
    std::cout << "Initial interval guess is " << print_vector(x).str() << std::endl
              << "Step size is " << stepsize << std::endl;
    for (size_t step_idx = 0; step_idx < 15; step_idx++)
    {
        double y_left;
        double y_right;

        vector<double> x_left(x.size());
        vector<double> x_right(x.size());
        for (size_t i = 0; i < x.size(); i++)
        {
            x_left[i] = x[i].lower();
            x_right[i] = x[i].upper();
        }
        gradient(x, y, dfdx, p);
        gradient(x_left, y_left, dfdx_left, p);
        gradient(x_right, y_right, dfdx_right, p);
        std::cout << "step #" << step_idx << std::endl
                  << "y=          " << y << std::endl
                  << "dydx=       " << print_vector(dfdx).str() << std::endl
                  << "dydx (l)=   " << print_vector(dfdx_left).str() << std::endl
                  << "dydx (r)=   " << print_vector(dfdx_right).str() << std::endl;

        // gradient test
        bool fail = false;
        for (double_ival &ddxi : dfdx)
        {
            if (ddxi.lower() > 0 || ddxi.upper() < 0)
            {

                fail = true;
                break;
            }
        }
        if (fail)
        {
            std::cout << "GRADIENT TEST FAILED" << std::endl;
            break;
        }

        std::cout << "GRADIENT TEST PASSED" << std::endl;
        // hessian test
        vector<vector<double_ival>> d2fdx2(x.size(), vector<double_ival>(x.size()));
        hessian(x, d2fdx2, p);
        std::cout << "HESSIAN TIME" << std::endl;
        for (size_t i = 0; i < d2fdx2.size(); i++)
        {
            std::cout << print_vector(d2fdx2[i]).str() << std::endl;
        }
        double_ival det = determinant(d2fdx2);
        bool test = sylvesters_criterion(d2fdx2);
        std::cout << "HESSIAN DETERMINANT IS " << det << std::endl;
        if (!test)
        {
            std::cout << "HESSIAN TEST FOR CONVEXITY FAILED" << std::endl;
            break;
        }
        // adjust interval bounds
        double l, dl;
        double r, dr;
        for (size_t i = 0; i < dfdx.size(); i++)
        {
            dl = -stepsize * dfdx_left[i];
            dr = -stepsize * dfdx_right[i];
            l = x[i].lower() + dl;
            r = x[i].upper() + dr;
            // std::cout << "dl[" << i << "] = " << dl << std::endl
            //           << "dr[" << i << "] = " << dr << std::endl;
            if (l < r)
            {
                x[i].assign(l, r);
            }
            else
            {
                x[i].assign(r, l);
            }
        }

        std::cout << "x=" << print_vector(x).str() << std::endl
                  << std::endl;
    }

    std::cout << "done" << std::endl;
}