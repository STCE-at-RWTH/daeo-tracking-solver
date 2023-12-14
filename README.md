# DAEO Solver with Optima Tracking

Solve equations of the form 
$$
    x(0, x) = x_0\\
    x'(t, x, y) = f(t, x, y)\\
    y(t) ∈ \argmin_y h(x, y)
$$
by tracking all possible local optima $y_i$ of the objective function $h$.

# Build Guide
- NAG's `dco/c++` CMake scripts cannot locate `dco` base. In this case, you'll need to make sure `dco.hpp` and `libdcoc.a` are in your build path by default.
- This project uses our fork of Boost's `interval`, which updates some parts of the library to C++11 and fixes some bugs caused by use with Eigen 3.4. It is provided as a submodule here.
- Output logs are, by default, placed in `data/out`, which is created when CMake configures.