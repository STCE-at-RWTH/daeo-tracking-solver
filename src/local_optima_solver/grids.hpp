#ifndef _DAEO_SOLVER_GRIDS_HPP
#define _DAEO_SOLVER_GRIDS_HPP

#include <cmath>

template <typename T>
struct UniformGrid1D
{
    const T m_start;
    const T m_end;
    const T m_stepsize;
    const size_t m_N;

    UniformGrid1D(T const &start, T const &end,
                  T const &stepsize, size_t const &N) : m_start{start}, m_end{end}, m_stepsize{stepsize} m_N{N} {}
    UniformGrid1D(T const &start, T const &end, size_t const &N)
    {
        T dx = (end - start) / N;
        UniformGrid1D(start, end, dx, N);
    }

    UniformGrid1D(T const &start, size_t const &N, T const &stepsize)
    {
        T end = start + N * stepsize;
        UniformGrid1D(start, end, stepsize, N);
    }

    UniformGrid1D(T const &start, T const &end, T const &stepsize)
    {
        size_t N = ceil((end - start) / stepsize);
        UniformGrid1D(start, N * stepsize, stepsize, N);
    }
};

#endif