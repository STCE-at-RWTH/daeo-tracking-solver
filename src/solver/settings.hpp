/**
 * @file settings.hpp
 * @author Sasha [fleming@stce.rwth-aachen.de]
 * @brief Defines the configuration and logging options for @ref local_optima_bnp.hpp and @ref daeo_solver.hpp
 */

#ifndef _SOLVER_SETTINGS_HPP
#define _SOLVER_SETTINGS_HPP

#include <chrono>
#include <fstream>
#include <string>

#include "boost/numeric/interval.hpp"
#include "fmt/core.h"
#include "fmt/chrono.h"
#include "fmt/ostream.h"
#include "fmt/ranges.h"

#include "fmt_extensions/interval.hpp"
#include "utils/io.hpp"

using boost::numeric::interval;
using std::vector;

template <typename T>
using suggested_solver_policies = boost::numeric::interval_lib::policies<
    boost::numeric::interval_lib::save_state<
        boost::numeric::interval_lib::rounded_transc_std<T>>,
    boost::numeric::interval_lib::checking_base<T>>;

template <typename NUMERIC_T>
struct BNBSolverSettings
{
    NUMERIC_T TOL_X;
    NUMERIC_T TOL_Y;

    std::size_t MAXITER = 10000;
    std::size_t MAX_REFINE_ITER = 4;
};

template <typename NUMERIC_T>
struct DAEOSolverSettings
{
    NUMERIC_T TOL_T;
    NUMERIC_T dt;

    NUMERIC_T y0_min;
    NUMERIC_T y0_max;

    size_t SEARCH_FREQUENCY = 1;
    size_t MAX_NEWTON_ITERATIONS = 5;
    size_t NEWTON_EPS;
};

enum BNBEventCodes {
    COMPUTATION_BEGIN,
    COMPUTATION_COMPLETE,
    TASK_BEGIN,
    TASK_COMPLETE,
    CONVERGENCE_TEST,
    GRADIENT_TEST,
    HESSIAN_TEST,
    ALL_TESTS
};

auto format_as(BNBEventCodes evc) { return fmt::underlying(evc); }

enum TestResultCode
{
    CONVERGENCE_TEST_PASS = 1,
    GRADIENT_TEST_FAIL = 2,
    GRADIENT_TEST_PASS = 4,
    HESSIAN_NEGATIVE_DEFINITE = 8,
    HESSIAN_MAYBE_INDEFINITE = 16,
    HESSIAN_POSITIVE_DEFINITE = 32
};

auto format_as(TestResultCode evc) { return fmt::underlying(evc); }

constexpr char BNB_LOG_COLUMN_NAMES[]{"TASKNUM\tTSTAMP\tEVENTID\tEXTRACODE\tX\tH\tDHDX\tD2HDX2\tCONVERGENCE"};
constexpr char LOG_TNUM_TSTAMP[]{"{:d}\t{:%S}\t"}; //{:%Y-%d-%m %H:%M:%S},"
constexpr char LOG_EID_EXTRA[]{"{:d}\t{:d}\t"};
constexpr char LOG_NUMERIC_VAL[]{"{:.8e}\t"};
constexpr char LOG_VECTOR_NUMERIC_VALS[]{"{::.8e}\t"};
constexpr char LOG_MATRIX_NUMERIC_VALS[]{"{:::.8e}\t"};

using sys_time_point_t = std::chrono::time_point<std::chrono::system_clock>;
class BNBSolverLogger
{
    size_t m_dims;
    size_t m_params;
    size_t m_threadcount;

    sys_time_point_t m_logging_start;

    vector<std::ofstream> outs;

public:
    BNBSolverLogger(size_t t_dims, size_t t_params, std::string const &filename)
        : m_dims{t_dims}, m_params{t_params}, m_threadcount{1}
    {
        outs.emplace_back(fmt::format("{}_thread_0.csv", filename));
        outs[0] << BNB_LOG_COLUMN_NAMES << "\n";
    };

    BNBSolverLogger(size_t t_dims, size_t t_params, size_t t_threads, std::string const &filename)
        : m_dims{t_dims}, m_params{t_params}, m_threadcount{t_threads}
    {
        for (size_t i = 0; i < m_threadcount; i++)
        {
            outs.emplace_back(fmt::format("{}_thread_{:d}.csv", filename, i));
            outs[i] << BNB_LOG_COLUMN_NAMES << "\n";
        }
    };

    ~BNBSolverLogger()
    {
        for (size_t i = 0; i < m_threadcount; i++)
        {
            outs[i].close();
        }
    }

    template <typename T>
    void log_computation_begin(size_t tasknum, sys_time_point_t time, vector<T> const &domain, size_t threadid = 0)
    {
        m_logging_start = time;
        fmt::print(outs[threadid], LOG_TNUM_TSTAMP, tasknum, time - m_logging_start);
        fmt::print(outs[threadid], LOG_EID_EXTRA, 0, 0);
        fmt::print(outs[threadid], LOG_VECTOR_NUMERIC_VALS, domain);
        fmt::print(outs[threadid], "None\tNone\tNone\tNone\n");
    }

    void log_computation_end(size_t tasknum, sys_time_point_t time, size_t n_results, size_t threadid = 0)
    {
        fmt::print(outs[threadid], LOG_TNUM_TSTAMP, tasknum, time - m_logging_start);
        fmt::print(outs[threadid], LOG_EID_EXTRA, COMPUTATION_COMPLETE, n_results);
        fmt::print(outs[threadid], "None\t");
        fmt::print(outs[threadid], "None\tNone\tNone\tNone\n");
    }

    template <typename T>
    void log_task_begin(size_t tasknum, sys_time_point_t time, vector<T> const &ival, size_t threadid = 0)
    {
        fmt::print(outs[threadid], LOG_TNUM_TSTAMP, tasknum, time - m_logging_start);
        fmt::print(outs[threadid], LOG_EID_EXTRA, TASK_BEGIN, 0);
        fmt::print(outs[threadid], LOG_VECTOR_NUMERIC_VALS, ival);
        fmt::print(outs[threadid], "None\tNone\tNone\tNone\n");
    }

    template <typename T>
    void log_task_complete(size_t tasknum, sys_time_point_t time, vector<T> const &ival, size_t reason, size_t threadid = 0)
    {
        fmt::print(outs[threadid], LOG_TNUM_TSTAMP, tasknum, time - m_logging_start);
        fmt::print(outs[threadid], LOG_EID_EXTRA, TASK_COMPLETE, reason);
        fmt::print(outs[threadid], LOG_VECTOR_NUMERIC_VALS, ival);
        fmt::print(outs[threadid], "None\tNone\tNone\tNone\n");
    }

    void log_convergence_test(size_t tasknum, sys_time_point_t time, vector<bool> const &convergence, size_t threadid = 0)
    {
        fmt::print(outs[threadid], LOG_TNUM_TSTAMP, tasknum, time - m_logging_start);
        fmt::print(outs[threadid], LOG_EID_EXTRA, CONVERGENCE_TEST, 0);
        fmt::print(outs[threadid], "None\tNone\tNone\tNone\t{::d}\n", convergence);
    }

    template <typename T>
    void log_gradient_test(size_t tasknum, sys_time_point_t time,
                           vector<T> const &x, T const &h,
                           vector<T> const &dhdx, size_t threadid = 0)
    {
        fmt::print(outs[threadid], LOG_TNUM_TSTAMP, tasknum, time - m_logging_start);
        fmt::print(outs[threadid], LOG_EID_EXTRA, GRADIENT_TEST, 0);
        fmt::print(outs[threadid], LOG_VECTOR_NUMERIC_VALS, x);
        fmt::print(outs[threadid], LOG_NUMERIC_VAL, h);
        fmt::print(outs[threadid], LOG_VECTOR_NUMERIC_VALS, dhdx);
        fmt::print(outs[threadid], "None\tNone\n");
    }

    template <typename T>
    void log_hessian_test(size_t tasknum, sys_time_point_t time,
                          TestResultCode testres,
                          vector<T> const &x, T const &h,
                          vector<T> const &dhdx, vector<vector<T>> &ddhdxx,
                          size_t threadid = 0)
    {
        fmt::print(outs[threadid], LOG_TNUM_TSTAMP, tasknum, time - m_logging_start);
        fmt::print(outs[threadid], LOG_EID_EXTRA, HESSIAN_TEST, testres);
        fmt::print(outs[threadid], LOG_VECTOR_NUMERIC_VALS, x);
        fmt::print(outs[threadid], LOG_NUMERIC_VAL, h);
        fmt::print(outs[threadid], LOG_VECTOR_NUMERIC_VALS, dhdx);
        fmt::print(outs[threadid], LOG_MATRIX_NUMERIC_VALS, ddhdxx);
        fmt::print(outs[threadid], "None\n");
    }

    template <typename T>
    void log_all_tests(size_t tasknum, sys_time_point_t time,
                       size_t combined_results,
                       vector<T> const &x, T const &h,
                       vector<T> const &dhdx, vector<vector<T>> &ddhdxx,
                       vector<bool> const &convergence,
                       size_t threadid = 0)
    {
        fmt::print(outs[threadid], LOG_TNUM_TSTAMP, tasknum, time - m_logging_start);
        fmt::print(outs[threadid], LOG_EID_EXTRA, ALL_TESTS, combined_results);
        fmt::print(outs[threadid], LOG_VECTOR_NUMERIC_VALS, x);
        fmt::print(outs[threadid], LOG_NUMERIC_VAL, h);
        fmt::print(outs[threadid], LOG_VECTOR_NUMERIC_VALS, dhdx);
        fmt::print(outs[threadid], LOG_MATRIX_NUMERIC_VALS, ddhdxx);
        fmt::print(outs[threadid], "{::d}\n", convergence);
    }
};

#endif