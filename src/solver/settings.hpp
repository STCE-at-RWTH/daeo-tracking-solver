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

#include "fmt_extensions/interval.hpp"

using std::vector;
using namespace std::chrono;

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

enum BNBEventCodes
{
    COMPUTATION_BEGIN,
    COMPUTATION_COMPLETE,
    TASK_BEGIN,
    TASK_COMPLETE,
    GRADIENT_TEST,
    HESSIAN_TEST,
};

auto format_as(BNBEventCodes evc) { return fmt::underlying(evc); }

enum TestResultCode
{
    CONVERGENCE_TEST_PASS = 1,
    GRADIENT_TEST_FAIL = 2,
    HESSIAN_NEGATIVE_DEFINITE = 4,
    HESSIAN_MAYBE_INDEFINITE = 8,
    HESSIAN_POSITIVE_DEFINITE = 16
};

auto format_as(TestResultCode evc) { return fmt::underlying(evc); }


#define BNB_LOG_COLUMN_NAMES "TASKNUM,TSTAMP,EVENTID,EXTRACODE,X,H,DHDX,D2HDX2,CONVERGENCE"
#define LOG_TNUM_TSTAMP "{0:d},{1}"
//%Y-%d-%m %H:%M:%S},"
#define LOG_EID_EXTRA "{},{:d},"
#define LOG_NUMERIC_VAL "{:+ .6e}"

class BNBSolverLogger
{
    size_t m_dims;
    size_t m_params;
    size_t m_threadcount;

    vector<std::ofstream> outs;

public:
    BNBSolverLogger(size_t t_dims, size_t t_params, std::string const &file)
        : m_dims{t_dims}, m_params{t_params}, m_threadcount{1}
    {
        outs.emplace_back(fmt::format("{}_thread_0.log", file));
        outs[1] << BNB_LOG_COLUMN_NAMES << "\n";
    };

    BNBSolverLogger(size_t t_dims, size_t t_params, size_t t_threads, std::string const &file)
        : m_dims{t_dims}, m_params{t_params}, m_threadcount{t_threads}
    {
        for (size_t i = 0; i < m_threadcount; i++)
        {
            outs.emplace_back(fmt::format("{}_thread_{:d}.log", file, i));
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

    template <typename TIME, typename T>
    void log_computation_begin(size_t tasknum, TIME time, vector<T> const &domain, size_t threadid = 0)
    {
        fmt::print(outs[threadid], LOG_TNUM_TSTAMP, tasknum, time);
        fmt::print(outs[threadid], LOG_EID_EXTRA, 0, 0);
        fmt::print(outs[threadid], "[");
        bool first = true;
        for (auto const &val : domain)
        {
            if (first)
                first = false;
            else
                fmt::print(outs[threadid], ",");
            fmt::print(outs[threadid], LOG_NUMERIC_VAL, val);
        }
        fmt::print(outs[threadid], "]");
        fmt::print(outs[threadid], "None,None,None,None");
    }

    template <typename TIME, typename T>
    void log_computation_end(size_t tasknum, TIME time, vector<T> const &domain, size_t n_results, size_t threadid = 0)
    {
        fmt::print(outs[threadid], LOG_TNUM_TSTAMP, tasknum, time);
        fmt::print(outs[threadid], LOG_EID_EXTRA, 1, n_results);
        fmt::print(outs[threadid], "[");
        bool first = true;
        for (auto const &val : domain)
        {
            if (first)
                first = false;
            else
                fmt::print(outs[threadid], ",");
            fmt::print(outs[threadid], LOG_NUMERIC_VAL, val);
        }
        fmt::print(outs[threadid], "]");
        fmt::print(outs[threadid], "None,None,None,None");
    }

    template <typename TIME, typename T>
    void log_task_begin(size_t tasknum, TIME time, vector<T> const &ival, size_t threadid = 0)
    {
        fmt::print(outs[threadid], LOG_TNUM_TSTAMP, tasknum, time);
        fmt::print(outs[threadid], LOG_EID_EXTRA, 2, 0);
        fmt::print(outs[threadid], "[");
        bool first = true;
        for (auto const &xi : ival)
        {
            if (first)
                first = false;
            else
                fmt::print(outs[threadid], ",");
            fmt::print(outs[threadid], LOG_NUMERIC_VAL, xi);
        }
        fmt::print(outs[threadid], "]");
        fmt::print(outs[threadid], "None,None,None,None");
    }

    template <typename TIME, typename T>
    void log_task_complete(size_t tasknum, TIME time, vector<T> const &ival, size_t reason, size_t threadid = 0)
    {
        fmt::print(outs[threadid], LOG_TNUM_TSTAMP, tasknum, time);
        fmt::print(outs[threadid], LOG_EID_EXTRA, 3, reason);
        fmt::print(outs[threadid], "[");
        bool first = true;
        for (auto const &xi : ival)
        {
            if (first)
                first = false;
            else
                fmt::print(outs[threadid], ",");
            fmt::print(outs[threadid], LOG_NUMERIC_VAL, xi);
        }
        fmt::print(outs[threadid], "]");
        fmt::print(outs[threadid], "None,None,None,None");
    }

    template <typename TIME>
    void log_convergence_test(size_t tasknum, TIME time, vector<bool> convergence)
    {
    }

    template <typename TIME>
    void log_gradient_test(size_t tasknum, TIME computation_start, TIME computation_end)
    {
    }
};

#endif