#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>
using std::vector;

#include "boost/numeric/interval.hpp"
using boost::numeric::square;

#include "dco.hpp"
#include "fmt/format.h"
#include "fmt/chrono.h"
#include "fmt/ranges.h"

#include "utils/io.hpp"
#include "utils/matrices.hpp"

#include "fmt_extensions/interval.hpp"

template <typename T>
using boost_interval_transc_t = boost::numeric::interval<
    T, boost::numeric::interval_lib::policies<
           boost::numeric::interval_lib::save_state<
               boost::numeric::interval_lib::rounded_transc_std<T>>,
           boost::numeric::interval_lib::checking_base<T>>>;
using double_ival = boost_interval_transc_t<double>;

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

void dosolverproto()
{
    vector<double_ival> x({double_ival::hull(-5, 2),
                           double_ival::hull(-4, 1)});
    vector<double> p({1.0, -2.0});
    double stepsize = 0.1;

    double_ival t{x[0].lower(), x[0].upper()};

    std::cout << fmt::format("using Format: {:.2e}\n", t);
    std::cout << t << std::endl;

    double_ival y;
    vector<double_ival> dfdx(x.size());
    vector<double> dfdx_left(x.size());
    vector<double> dfdx_right(x.size());
    std::cout << "doing" << std::endl;
    std::cout << "Initial interval guess is " << print_vector(x) << std::endl
              << "Step size is " << stepsize << std::endl;
    for (size_t step_idx = 0; step_idx < 2; step_idx++)
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
                  << "dydx=       " << print_vector(dfdx) << std::endl
                  << "dydx (l)=   " << print_vector(dfdx_left) << std::endl
                  << "dydx (r)=   " << print_vector(dfdx_right) << std::endl;

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
            std::cout << print_vector(d2fdx2[i]) << std::endl;
        }
        double_ival det = determinant(d2fdx2);
        bool test = is_positive_definite(d2fdx2);
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

        std::cout << "x=" << print_vector(x) << std::endl
                  << std::endl;
    }

    std::cout << "done" << std::endl;
}

int main(int argc, char *argv[])
{
    using std::chrono::system_clock;
    using std::chrono::time_point;
    using time_point_t = time_point<system_clock>;

    time_point_t ttime = system_clock::now();
    fmt::println("{}", ttime);

    vector<double_ival> tvec{{-1, 2}, {3, 4.5}};

    auto telm = tvec[0];

    double_ival ival_1(2., 4.);
    double val_1 = 3.0;
    std::common_type<double_ival, double>::type val_2;
    val_2 = ival_1 - val_1; 

    constexpr bool canfmtival = fmt::is_formattable<double_ival, char>::value;
    std::cout << std::boolalpha << canfmtival << "\n";

    fmt::println("{::.4e}", tvec);
    return 0;
}