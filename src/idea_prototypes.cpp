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
#include "fmt/format.h"
#include "fmt/chrono.h"
#include "fmt/ranges.h"
#include "utils/sylvesters_criterion.hpp"

template <typename T>
using boost_interval_transc_t = boost::numeric::interval<
    T, boost::numeric::interval_lib::policies<
           boost::numeric::interval_lib::save_state<
               boost::numeric::interval_lib::rounded_transc_std<T>>,
           boost::numeric::interval_lib::checking_base<T>>>;
using double_ival = boost_interval_transc_t<double>;

int main(int argc, char *argv[])
{
    Eigen::MatrixXd v1;
    v1 = Eigen::MatrixXd::Random(13, 13);
    std::cout << v1 << "\n\n";
    std::cout << v1(drop_idx{1, 4}, Eigen::all) << std::endl;
    std::cout << bad_determinant(v1) << std::endl;

    return 0;
}