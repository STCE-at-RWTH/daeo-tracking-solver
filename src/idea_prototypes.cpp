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

#include "fmt_extensions/interval.hpp"
#include "utils/propagate_dynamic.hpp"

template <typename T>
using boost_interval_transc_t = boost::numeric::interval<
    T, boost::numeric::interval_lib::policies<
           boost::numeric::interval_lib::save_state<
               boost::numeric::interval_lib::rounded_transc_std<T>>,
           boost::numeric::interval_lib::checking_base<T>>>;
using double_ival = boost_interval_transc_t<double>;


int main(int argc, char *argv[])
{
    using std::chrono::system_clock;
    using std::chrono::time_point;
    using time_point_t = time_point<system_clock>;

    time_point_t ttime = system_clock::now();
    fmt::println("{}", ttime);

    vector<double_ival> tvec{{-1, 2}, {3, 4.5}};
    Eigen::Vector<double_ival, 2> tvec2(double_ival{-1, 2},double_ival{3.5, 4.5});

    auto telm = tvec[0];

    double_ival ival_1(2., 4.);
    double val_1 = 3.0;
    std::common_type<double_ival, double>::type val_2;
    val_2 = ival_1 - val_1; 

    constexpr bool canfmtival = fmt::is_formattable<double_ival, char>::value;
    std::cout << std::boolalpha << canfmtival << "\n";

    fmt::println("{::.4e}", tvec2);

    fmt::println("value is {:d}", propagate_dynamic<-1, 2>::value);
    return 0;
}