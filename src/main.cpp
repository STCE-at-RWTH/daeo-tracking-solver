#include <cmath>
#include <iostream>
#include <vector>
using std::vector;

#include "boost/numeric/interval.hpp"

template <typename T>
using boost_interval_transc_t = boost::numeric::interval<
    T, boost::numeric::interval_lib::policies<
           boost::numeric::interval_lib::save_state<
               boost::numeric::interval_lib::rounded_transc_std<T>>,
           boost::numeric::interval_lib::checking_base<T>>>;
using double_ival = boost_interval_transc_t<double>;

int main(int argc, char **argv)
{
    
    return 0;
}