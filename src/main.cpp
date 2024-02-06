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
using boost_interval_transc_t =
    boost::numeric::interval<T, suggested_solver_policies<T>>;
using double_ival = boost_interval_transc_t<double>;

constexpr int NUM_Y_DIMS = 1;
constexpr int NUM_PARAMS = 4;

template <typename T>
void run_simple_example(DAEOSolverSettings<T> solver_s, BNBOptimizerSettings<T> optimizer_s)
{
    auto f = [](const auto t, const auto x, const auto &y, const auto &p) -> auto
    {
        return -(p(0) + y(0)) * x;
    };

    auto h = [](const auto t, const auto x, const auto &y, const auto &p) -> auto
    {
        return pow(p(1) - pow(y(0), 2), 2) - (x - p(2)) * sin(y(0) * p(3));
    };

    // using optimizer_t = BNBLocalOptimizer<decltype(h), double, suggested_solver_policies<double>, NUM_Y_DIMS, NUM_PARAMS>;
    using solver_t = DAEOTrackingSolver<decltype(f), decltype(h), double, NUM_Y_DIMS, NUM_PARAMS>;
    typename solver_t::params_t p(2.0, 1.0, 0.5, pi / 2);

    for (int i = 0; i < 7; i++)
    {
        double dt = pow(10.0, -i);
        solver_t solver(f, h, optimizer_s, solver_s);
        solver.solve_daeo(0, 1, dt, 1.0, p, fmt::format("simple_example_10_minus_{:d}", i));
    }

    solver_s.EVENT_DETECTION_AND_CORRECTION = false;
    for (int i = 0; i < 7; i++)
    {
        double dt = pow(10.0, -i);
        solver_t solver(f, h, optimizer_s, solver_s);
        solver.solve_daeo(0, 1, dt, 1.0, p, fmt::format("simple_example_10_minus_{:d}_noevents", i));
    }
}

template <typename T>
void run_griewank_example(DAEOSolverSettings<T> &solver_s, BNBOptimizerSettings<T> &optimizer_s)
{
    auto f = [](const auto t, const auto x, const auto &y, const auto &p) -> auto
    {
        return y(0);
    };

    auto h = [](const auto t, const auto x, const auto &y, const auto &p) -> auto
    {
        return pow(x - y(0), 2) + sin(p(0) * y(0));
    };

    // using optimizer_t = BNBLocalOptimizer<decltype(h), T, suggested_solver_policies<T>, NUM_Y_DIMS, NUM_PARAMS>;
    using solver_t = DAEOTrackingSolver<decltype(f), decltype(h), double, 1, 1>;
    typename solver_t::params_t p{5.0};

    solver_t solver(f, h, optimizer_s, solver_s);

    solver.solve_daeo(0., 1.5, 0.00005, 1.0, p, "griewank_example");
}

void run_simple_example_perf_study()
{
    auto f = [](const auto t, const auto x, const auto &y, const auto &p) -> auto
    {
        return -(p(0) + y(0)) * x;
    };

    auto h = [](const auto t, const auto x, const auto &y, const auto &p) -> auto
    {
        return pow(p(1) - pow(y(0), 2), 2) - (x - p(2)) * sin(y(0) * p(3));
    };

    // using optimizer_t = BNBLocalOptimizer<decltype(h), double, suggested_solver_policies<double>, NUM_Y_DIMS, NUM_PARAMS>;
    using solver_t = DAEOTrackingSolver<decltype(f), decltype(h), double, NUM_Y_DIMS, NUM_PARAMS>;
    typename solver_t::params_t p(2.0, 1.0, 0.5, pi / 2);

    DAEOSolverSettings<double> solver_s;
    solver_s.NEWTON_EPS = 1.0e-8;
    solver_s.EVENT_EPS = 5.0e-6;
    solver_s.y0_min = -6.0;
    solver_s.y0_max = 6.0;

    BNBOptimizerSettings<double> opt_s;
    opt_s.TOL_X = 1.0e-6;
    opt_s.TOL_Y = 1.0e-8;
    opt_s.MAXITER = 1000;
    opt_s.MAX_REFINE_ITER = 20;
    opt_s.LOGGING_ENABLED = false;

    /**
     * TRACK LOCAL, NEVER REOPTIMIZE
     */
    solver_s.SEARCH_FREQUENCY = std::numeric_limits<size_t>::max();
    solver_s.EVENT_DETECTION_AND_CORRECTION = true;
    solver_s.ONLY_GLOBAL_OPTIMIZATION = false;

    for (int i = 1; i < 7; i++)
    {
        double dt = pow(10.0, -i);
        solver_t solver(f, h, opt_s, solver_s);
        solver.solve_daeo(0, 1, dt, 1.0, p, fmt::format("se_tracking_10_minus{:d}", i));
    }

    /**
     * TRACK ONLY GLOBAL
    */

    for (int i = 1; i<7; i++){
        double dt = pow(10.0, -i);
    }
}

int main(int argc, char **argv)
{
    /* Implement the scenario described in
     *  Numerical simulation of differential-algebraic equations
     *    with embedded global optimization criteria [Deussen, Hüser, Naumann]
     *
     */

    fmt::println("double epsilon is {:.6e}", std::numeric_limits<double>::epsilon());

    BNBOptimizerSettings<double> optimizer_settings;
    optimizer_settings.TOL_X = 1.0e-6;
    optimizer_settings.TOL_Y = 1.0e-8;
    optimizer_settings.MAXITER = 1000;
    optimizer_settings.MAX_REFINE_ITER = 20;
    optimizer_settings.LOGGING_ENABLED = false;

    DAEOSolverSettings<double> simple_solver_settings;
    simple_solver_settings.NEWTON_EPS = 1.0e-8;
    simple_solver_settings.EVENT_EPS = 5.0e-6;
    simple_solver_settings.y0_min = -6.0;
    simple_solver_settings.y0_max = 6.0;
    simple_solver_settings.SEARCH_FREQUENCY = 1000000000000000;

    DAEOSolverSettings<double> griewank_solver_settings = simple_solver_settings;
    griewank_solver_settings.SEARCH_FREQUENCY = 50;
    griewank_solver_settings.y0_min = 0.0;

    run_simple_example(simple_solver_settings, optimizer_settings);
    // run_griewank_example(griewank_solver_settings, optimizer_settings);
}
