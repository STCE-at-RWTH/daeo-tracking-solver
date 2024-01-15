# DAEO Solver with Optima Tracking

Solve equations of the form 
$$
    x(0, x) = x_0\\
    x'(t, x, y) = f(t, x, y)\\
    y(t) ∈ \argmin_y h(x, y)
$$
by tracking all possible local optima $y_i$ of the objective function $h$.

# Build Guide
- NAG's `dco/c++` CMake scripts cannot locate `dco/c++` installations with the new product code. In this case, you'll need to make sure `dco.hpp` and `libdcoc.a` are in your build path by default.
- This project uses our fork of Boost's `interval`, which updates some parts of the library to C++11 and fixes some bugs caused by use with Eigen 3.4. It is provided as a submodule here.
- Output logs are, by default, placed in `data/out`, which is created when CMake configures.

# To-Do List for AD2024
- [ ] Investigate the use of the implicit function theorem to find $\partial_x y^k$.
- [ ] Handle the emergence and disappearance of local optima.
- [ ] Implement a (justifiable) heuristic for re-running global optimization.
- [ ] Implement a solver that only uses global optimization for use in performance comparisons.
    - [ ] Test at higher compiler optimization levels.
    - [ ] Make sure `dco/c++` has full `interval` compatibility.