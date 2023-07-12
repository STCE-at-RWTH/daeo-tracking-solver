#include <cmath>
#include <iostream>
#include <numbers>
#include <vector>

#include "boost/numeric/interval.hpp"
#include "solver/settings.hpp"
#include "solver/local_optima_solver.hpp"

using std::vector;

template <typename T>
using boost_interval_transc_t = boost::numeric::interval<
    T, suggested_solver_policies<T>>;
using double_ival = boost_interval_transc_t<double>;

int main(int argc, char **argv)
{

    auto obj1 = [](const auto &x, const auto &p) -> auto
    {
        return pow(x[0] - p[0], 2) + sin(4.0 * x[0]);
    };

    auto obj2 = [](const auto& x, const auto &p) -> auto
    {
        using std::numbers::pi;
        return sin(pi*x[0]*p[0]/pi)*cos(pi*x[1]*p[1]);
    };



    BNBSolverSettings<double> settings;
    settings.TOL_X = 1.0e-4;
    settings.TOL_Y = 1.0e-4;
    settings.MAXITER = 10000;
    settings.MAX_REFINE_ITER = 20;

    using solver_t = LocalOptimaBNBSolver<decltype(obj2),
                                          double,
                                          suggested_solver_policies<double>>;

    vector<solver_t::interval_t> x1{
        {-8.0, 12.0}
    };
    vector<double> params1{
        1.0,
    };

    solver_t solver(obj2, settings);
    vector<solver_t::interval_t> x2{
        {-2, 2}, {-2, 2}
    };
    vector<double> params2{
        1.0, 1.0
    };
    auto results = solver.find_minima(x2, params2, true);
    fmt::print("Found {} minima:\n", results.minima_intervals.size());
    for (auto &ival : results.minima_intervals)
    {
        fmt::print("f({::.4e}, {::.2e}) = {:.4e}\n", ival, params2, obj2(ival, params2));
    }
    return 0;
}