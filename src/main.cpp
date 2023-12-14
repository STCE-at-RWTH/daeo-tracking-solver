#include <cmath>
#include <iostream>
#include <numbers>
#include <vector>

#include "boost/numeric/interval.hpp"

#include "solver/daeo_solver.hpp"
#include "solver/local_optima_solver.hpp"
#include "solver/objective.hpp"
#include "solver/settings.hpp"
#include "solver/logging.hpp"

using std::vector;
using namespace std::numbers;

template <typename T>
using boost_interval_transc_t = boost::numeric::interval<
    T, suggested_solver_policies<T>>;
using double_ival = boost_interval_transc_t<double>;

constexpr int NUM_Y_DIMS = 1;
constexpr int NUM_PARAMS = 4;

int main(int argc, char **argv)
{
    /* Implement the scenario described in
     *  Numerical simulation of differential-algebraic equations
     *    with embedded global optimization criteria [Deussen, Hüser, Naumann]
     *
     */
    double x0{1.0};
    auto f = [](const auto t, const auto x, const auto &y, const auto &p) -> auto
    {
        return -(p(0) + y(0)) * x;
    };

    auto f_griewank = [](const auto t, const auto x, const auto &y, const auto &p) -> auto
    {
        return y;
    };

    auto h = [](const auto t, const auto x, const auto &y, const auto &p) -> auto
    {
        return pow(p(1) - pow(y(0), 2), 2) - (x - p(2)) * sin(y(0) * p(3));
    };

    auto h_griewank = [](const auto t, const auto x, const auto &y, const auto &p) -> auto
    {
        return pow(x - y, 2) + sin(p(1) * y);
    };

    BNBSolverSettings<double> optimizer_settings;
    optimizer_settings.TOL_X = 1.0e-6;
    optimizer_settings.TOL_Y = 1.0e-6;
    optimizer_settings.MAXITER = 1000;
    optimizer_settings.MAX_REFINE_ITER = 20;

    DAEOSolverSettings<double> solver_settings;
    solver_settings.TOL_T = 1.0e-8;
    solver_settings.y0_min = -8.0;
    solver_settings.y0_max = 12.0;

    using optimizer_t = LocalOptimaBNBSolver<decltype(h),
                                             double,
                                             suggested_solver_policies<double>,
                                             NUM_Y_DIMS, NUM_PARAMS>;
    using solver_t = DAEOTrackingSolver<decltype(f), decltype(h), double, NUM_Y_DIMS, NUM_PARAMS>;

    optimizer_t::y_t yll {-8.0};
    optimizer_t::y_t yur {12.0};
    optimizer_t::params_t p(2.0, 1.0, 0.5, pi / 2);

    optimizer_t optimizer(h, optimizer_settings);
    optimizer.set_search_domain(yll, yur);
    auto results = optimizer.find_minima_at(0, x0, p, true, true);
    fmt::println("double epsilon is {:.6e}", std::numeric_limits<double>::epsilon());
    fmt::print("Found {} minima:\n", results.minima_intervals.size());
    for (auto &y_argmin : results.minima_intervals)
    {
        fmt::print("f(0, {:.4e}, {::.4e}, {::.2e}) = {:.4e}\n", x0, y_argmin, p, h(0, x0, y_argmin, p));
    }
    //solver_t solver(f, h, optimizer_settings, solver_settings);
    //auto [t, x_path] = solver.solve_daeo(0.0, 1.0, 0.01, 1.0, p);

    return 0;
}