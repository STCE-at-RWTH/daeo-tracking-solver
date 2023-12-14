/**
 *
 */
#ifndef _SOLVER_LOGGING_HPP
#define _SOLVER_LOGGING_HPP

#include <chrono>
#include <fstream>
#include <ranges>
#include <string>
#include <vector>

#include "fmt/core.h"
#include "fmt/chrono.h"
#include "fmt/ostream.h"
#include "fmt/ranges.h"

#include "utils/fmt_extensions.hpp"

using std::vector;

enum BNBEventCodes
{
    COMPUTATION_BEGIN,
    COMPUTATION_COMPLETE,
    TASK_BEGIN,
    TASK_COMPLETE,
    VALUE_TEST,
    CONVERGENCE_TEST,
    GRADIENT_TEST,
    HESSIAN_TEST,
    ALL_TESTS
};

auto format_as(BNBEventCodes evc) { return fmt::underlying(evc); }

enum TestResultCode
{
    CONVERGENCE_TEST_PASS = 1,
    VALUE_TEST_FAIL = 2,
    VALUE_TEST_PASS = 4,
    GRADIENT_TEST_FAIL = 8,
    GRADIENT_TEST_PASS = 16,
    HESSIAN_NEGATIVE_DEFINITE = 32,
    HESSIAN_MAYBE_INDEFINITE = 64,
    HESSIAN_POSITIVE_DEFINITE = 128
};

auto format_as(TestResultCode evc) { return fmt::underlying(evc); }

constexpr char BNB_LOG_COLUMN_NAMES[]{"TASKNUM\tTSTAMP\tEVENTID\tEXTRACODE\tX\tH\tDHDX\tD2HDX2\tCONVERGENCE"};
constexpr char LOG_TNUM_TSTAMP[]{"{:d}\t{:%S}\t"};
constexpr char LOG_EID_EXTRA[]{"{:d}\t{:d}\t"};
constexpr char LOG_NUMERIC_VAL[]{"{:.8e}\t"};
constexpr char LOG_ITERABLE_NUMERIC_VALS[]{"{::.8e}\t"};
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
        outs.emplace_back(fmt::format("{}_thread_0.tsv", filename));
        outs[0] << BNB_LOG_COLUMN_NAMES << "\n";
    };

    BNBSolverLogger(size_t t_dims, size_t t_params, size_t t_threads, std::string const &filename)
        : m_dims{t_dims}, m_params{t_params}, m_threadcount{t_threads}
    {
        for (size_t i = 0; i < m_threadcount; i++)
        {
            outs.emplace_back(fmt::format("{}_thread_{:d}.tsv", filename, i));
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

    template <std::ranges::range Y>
    void log_computation_begin(size_t tasknum, sys_time_point_t time, Y const &domain, size_t threadid = 0)
    {
        m_logging_start = time;
        fmt::print(outs[threadid], LOG_TNUM_TSTAMP, tasknum, time - m_logging_start);
        fmt::print(outs[threadid], LOG_EID_EXTRA, 0, 0);
        fmt::print(outs[threadid], LOG_ITERABLE_NUMERIC_VALS, domain);
        fmt::print(outs[threadid], "None\tNone\tNone\tNone\n");
    }

    void log_computation_end(size_t tasknum, sys_time_point_t time, size_t n_results, size_t threadid = 0)
    {
        fmt::print(outs[threadid], LOG_TNUM_TSTAMP, tasknum, time - m_logging_start);
        fmt::print(outs[threadid], LOG_EID_EXTRA, COMPUTATION_COMPLETE, n_results);
        fmt::print(outs[threadid], "None\t");
        fmt::print(outs[threadid], "None\tNone\tNone\tNone\n");
    }

    template <std::ranges::range Y>
    void log_task_begin(size_t tasknum, sys_time_point_t time, Y const &y, size_t threadid = 0)
    {
        fmt::print(outs[threadid], LOG_TNUM_TSTAMP, tasknum, time - m_logging_start);
        fmt::print(outs[threadid], LOG_EID_EXTRA, TASK_BEGIN, 0);
        fmt::print(outs[threadid], LOG_ITERABLE_NUMERIC_VALS, y);
        fmt::print(outs[threadid], "None\tNone\tNone\tNone\n");
    }

    template <std::ranges::range Y>
    void log_task_complete(size_t tasknum, sys_time_point_t time, Y const &y, size_t reason, size_t threadid = 0)
    {
        fmt::print(outs[threadid], LOG_TNUM_TSTAMP, tasknum, time - m_logging_start);
        fmt::print(outs[threadid], LOG_EID_EXTRA, TASK_COMPLETE, reason);
        fmt::print(outs[threadid], LOG_ITERABLE_NUMERIC_VALS, y);
        fmt::print(outs[threadid], "None\tNone\tNone\tNone\n");
    }

    void log_convergence_test(size_t tasknum, sys_time_point_t time, vector<bool> const &convergence, size_t threadid = 0)
    {
        fmt::print(outs[threadid], LOG_TNUM_TSTAMP, tasknum, time - m_logging_start);
        fmt::print(outs[threadid], LOG_EID_EXTRA, CONVERGENCE_TEST, 0);
        fmt::print(outs[threadid], "None\tNone\tNone\tNone\t{::d}\n", convergence);
    }

    template <typename T, std::ranges::range Y, std::ranges::range DHDY>
    void log_gradient_test(size_t tasknum, sys_time_point_t time,
                           Y const &y, T const &h,
                           DHDY const &dhdy, size_t threadid = 0)
    {
        fmt::print(outs[threadid], LOG_TNUM_TSTAMP, tasknum, time - m_logging_start);
        fmt::print(outs[threadid], LOG_EID_EXTRA, GRADIENT_TEST, 0);
        fmt::print(outs[threadid], LOG_ITERABLE_NUMERIC_VALS, y);
        fmt::print(outs[threadid], LOG_NUMERIC_VAL, h);
        fmt::print(outs[threadid], LOG_ITERABLE_NUMERIC_VALS, dhdy);
        fmt::print(outs[threadid], "None\tNone\n");
    }

    /**
     * Log the result of the Hessian test.
    */
    template <typename T, std::ranges::range Y, std::ranges::range DHDY, std::ranges::range DDHDDY_ROWS>
    void log_hessian_test(size_t tasknum, sys_time_point_t time,
                          TestResultCode testres,
                          Y const &y, T const &h,
                          DHDY const &dhdy, DDHDDY_ROWS &d2hdy2,
                          size_t threadid = 0)
    {
        fmt::print(outs[threadid], LOG_TNUM_TSTAMP, tasknum, time - m_logging_start);
        fmt::print(outs[threadid], LOG_EID_EXTRA, HESSIAN_TEST, testres);
        fmt::print(outs[threadid], LOG_ITERABLE_NUMERIC_VALS, y);
        fmt::print(outs[threadid], LOG_NUMERIC_VAL, h);
        fmt::print(outs[threadid], LOG_ITERABLE_NUMERIC_VALS, dhdy);
        fmt::print(outs[threadid], LOG_MATRIX_NUMERIC_VALS, d2hdy2);
        fmt::print(outs[threadid], "None\n");
    }

    /**
     * @brief Log results from all tests. Pass the Hessian as an iterator of rows via (...).rowwise().
    */
    template <typename T, std::ranges::range Y, std::ranges::range DHDY, std::ranges::range DDHDYY_ROWS>
    void log_all_tests(size_t tasknum, sys_time_point_t time,
                       size_t combined_results,
                       Y const &y, T const &h,
                       DHDY const &dhdy, DDHDYY_ROWS const &d2hdy2,
                       vector<bool> const &convergence,
                       size_t threadid = 0)
    {
        fmt::print(outs[threadid], LOG_TNUM_TSTAMP, tasknum, time - m_logging_start);
        fmt::print(outs[threadid], LOG_EID_EXTRA, ALL_TESTS, combined_results);
        fmt::print(outs[threadid], LOG_ITERABLE_NUMERIC_VALS, y);
        fmt::print(outs[threadid], LOG_NUMERIC_VAL, h);
        fmt::print(outs[threadid], LOG_ITERABLE_NUMERIC_VALS, dhdy);
        fmt::print(outs[threadid], LOG_MATRIX_NUMERIC_VALS, d2hdy2);
        fmt::print(outs[threadid], "{::d}\n", convergence);
    }
};
#endif