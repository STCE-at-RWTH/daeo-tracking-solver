#include <cmath>
#include <iostream>
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

    auto objective = [](const auto &x, const auto &p) -> auto
    {
        return pow(x[0] - p[0], 2) + sin(4.0 * x[0]);
    };

    BNBSolverSettings<double> settings;
    settings.TOL_X = 1.0e-2;
    settings.TOL_Y = 1.0e-4;
    settings.MAXITER = 60;

    using solver_t = LocalOptimaBNBSolver<decltype(objective),
                                          double,
                                          suggested_solver_policies<double>>;

    solver_t solver(objective, settings);
    vector<solver_t::interval_t> x0{
        {-8.0, 12.0},
    };
    vector<double> params{
        1.0,
    };
    auto results = solver.find_minima(x0, params);
    std::cout << "Found " << results.minima_intervals.size() << " minima: " << std::endl;
    for (auto &ival : results.minima_intervals)
    {
        std::cout << print_vector(ival).str() << std::endl;
    }
    return 0;
}