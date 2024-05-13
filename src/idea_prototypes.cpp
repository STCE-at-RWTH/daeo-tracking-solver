#include <chrono>
#include <cmath>
#include <iostream>
#include <type_traits>
#include <vector>
using std::vector;

#include "boost/numeric/interval.hpp"
using boost::numeric::square;

#include "Eigen/Dense"
#include "dco.hpp"
#include "fmt/chrono.h"
#include "fmt/format.h"
#include "fmt/ranges.h"
#include "utils/propagate_dynamic.hpp"
#include "utils/sylvesters_criterion.hpp"

template <typename T>
using boost_interval_transc_t = boost::numeric::interval<
    T, boost::numeric::interval_lib::policies<
           boost::numeric::interval_lib::save_state<
               boost::numeric::interval_lib::rounded_transc_std<T>>,
           boost::numeric::interval_lib::checking_base<T>>>;
using double_ival = boost_interval_transc_t<double>;

int main(int argc, char *argv[]) {
  Eigen::VectorXd v1;
  v1 = Eigen::VectorXd::Random(4);
  Eigen::VectorXd v2(4);
  v2 << 1.0, 2.0, 3.0, 4.0;
  Eigen::VectorXd v3 = v2.binaryExpr(
      v1, [](auto y, auto z) -> auto{ return y; });
  fmt::println("{::.4e}", v3);
  Eigen::MatrixXd v4;
  v4 = Eigen::MatrixXd::Random(10, 10);
  fmt::println("{:.4e}", bad_determinant(v4));
  fmt::println("{:d}", 4 & 1);
  fmt::println("{:d}", propagate_dynamic<1, 2>::value);
  return 0;
}