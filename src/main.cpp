#include <cmath>
#include <iostream>
#include <numbers>
#include <vector>

#include "boost/numeric/interval.hpp"
#include "solver/settings.hpp"
#include "solver/local_optima_solver.hpp"

using std::vector;
using namespace std::numbers;

template <typename T>
using boost_interval_transc_t = boost::numeric::interval<
    T, suggested_solver_policies<T>>;
using double_ival = boost_interval_transc_t<double>;

int main(int argc, char **argv)
{
    /* Implement the scenario described in 
    *  Numerical simulation of differential-algebraic equations
    *    with embedded global optimization criteria [Deussen, Hüser, Naumann]
    *
    */
    double x0 = 1.;
    auto dx = [](const auto &t, const auto &x, const auto &y, const auto &p) -> auto
    {
        return -(p[0] + y)*x;
    };

    auto h = [](const auto &t, const auto &x, const auto& y, const auto &p) -> auto
    {
        return pow(p[1] - pow(y, 2), 2)-(x-p[2])*sin(y*p[3]);
    };

    BNBSolverSettings<double> settings;
    settings.TOL_X = 1.0e-4;
    settings.TOL_Y = 1.0e-4;
    settings.MAXITER = 10000;
    settings.MAX_REFINE_ITER = 20;

    using solver_t = LocalOptimaBNBSolver<decltype(h),
                                          double,
                                          suggested_solver_policies<double>>;

    vector<solver_t::interval_t> y0{
        {-8.0, 12.0}
    };

    vector<double> p{
        2.0,
        1.0,
        0.5,
        pi/2
    };

    solver_t solver(h, settings);
    auto results = solver.find_minima(y0, p, true);
    fmt::print("Found {} minima:\n", results.minima_intervals.size());
    for (auto &ival : results.minima_intervals)
    {
        fmt::print("f({::.4e}, {::.2e}) = {:.4e}\n", ival, params1, obj1(ival, p));
    }
    return 0;
}