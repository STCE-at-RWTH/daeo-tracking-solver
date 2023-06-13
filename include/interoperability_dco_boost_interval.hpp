/**
 * @file interoperability_dco_boost_interval.hpp
 * @author Jens Deussen [deussen@stce.rwth-aachen.de]
 * @brief Interoperability between DCO/c++ and boost::numeric::interval.
 *
 * @deprecated Appears to be included in DCO 4.1-alpha-2!
 */

#pragma once

#define DCO_STD_COMPATIBILITY
#define DCO_AUTO_SUPPORT

#include "boost/numeric/interval.hpp"
#include "boost/numeric/interval/io.hpp"

#include "dco.hpp"
#include <type_traits>

// dco_type < boost_interval >
// streaming operator required for debug log in ga1s

namespace boost
{
    namespace numeric
    {
        template <class T, class Policies>
        inline
            typename std::enable_if<!std::is_same<T, int>::value, interval<T, Policies>>::type
            operator*(const int &x, const interval<T, Policies> &y)
        {
            return static_cast<T>(x) * y;
        }

        template <class T, class Policies>
        inline interval<T, Policies> pow(const interval<T, Policies> &x, const interval<T, Policies> &pwr)
        {
            assert(pwr.lower() == pwr.upper());
            if (pwr.lower() == 2)
                return square(x);
            return pow(x, pwr.lower());
        }

        template <class T, class Policies>
        inline interval<T, Policies> round(const interval<T, Policies> &x)
        {
            return interval<T, Policies>(floor(x.lower() + 0.5), floor(x.upper() + 0.5));
        }

        namespace interval_lib
        {
            template <typename T, typename D>
            struct rounding_control<dco::internal::active_type<T, D>> : detail::c99_rounding_control
            {
                using type = dco::internal::active_type<T, D>;
                static type force_rounding(type const &r)
                {
                    type r_ = r;
                    return r_;
                }
                template <class U>
                static U to_int(const U &r) { return rint(dco::passive_value(r)); }
            };

            // constants used in transc.hpp (e.g. for cos) match int definition if not defined explicitly for basetype
            namespace constants
            {
#define BOOST_SPECIALIZE_CONSTANTS_FP(TYPE)                                                     \
    template <>                                                                                 \
    inline TYPE pi_lower<TYPE>() { return pi_lower<dco::mode<TYPE>::passive_t>(); }             \
    template <>                                                                                 \
    inline TYPE pi_upper<TYPE>() { return pi_upper<dco::mode<TYPE>::passive_t>(); }             \
    template <>                                                                                 \
    inline TYPE pi_half_lower<TYPE>() { return pi_half_lower<dco::mode<TYPE>::passive_t>(); }   \
    template <>                                                                                 \
    inline TYPE pi_half_upper<TYPE>() { return pi_half_upper<dco::mode<TYPE>::passive_t>(); }   \
    template <>                                                                                 \
    inline TYPE pi_twice_lower<TYPE>() { return pi_twice_lower<dco::mode<TYPE>::passive_t>(); } \
    template <>                                                                                 \
    inline TYPE pi_twice_upper<TYPE>() { return pi_twice_upper<dco::mode<TYPE>::passive_t>(); }

                BOOST_SPECIALIZE_CONSTANTS_FP(dco::gt1s<float>::type)
                BOOST_SPECIALIZE_CONSTANTS_FP(dco::gt1s<double>::type)
                BOOST_SPECIALIZE_CONSTANTS_FP(dco::ga1s<float>::type)
                BOOST_SPECIALIZE_CONSTANTS_FP(dco::ga1s<double>::type)
                BOOST_SPECIALIZE_CONSTANTS_FP(dco::ga1sm<float>::type)
                BOOST_SPECIALIZE_CONSTANTS_FP(dco::ga1sm<double>::type)
                BOOST_SPECIALIZE_CONSTANTS_FP(dco::gt1s<dco::gt1s<float>::type>::type)
                BOOST_SPECIALIZE_CONSTANTS_FP(dco::gt1s<dco::gt1s<double>::type>::type)
                BOOST_SPECIALIZE_CONSTANTS_FP(dco::ga1s<dco::gt1s<float>::type>::type)
                BOOST_SPECIALIZE_CONSTANTS_FP(dco::ga1s<dco::gt1s<double>::type>::type)
                BOOST_SPECIALIZE_CONSTANTS_FP(dco::gt1s<dco::ga1s<float>::type>::type)
                BOOST_SPECIALIZE_CONSTANTS_FP(dco::gt1s<dco::ga1s<double>::type>::type)
                BOOST_SPECIALIZE_CONSTANTS_FP(dco::ga1sm<dco::gt1s<float>::type>::type)
                BOOST_SPECIALIZE_CONSTANTS_FP(dco::ga1sm<dco::gt1s<double>::type>::type)
                BOOST_SPECIALIZE_CONSTANTS_FP(dco::gt1s<dco::ga1sm<float>::type>::type)
                BOOST_SPECIALIZE_CONSTANTS_FP(dco::gt1s<dco::ga1sm<double>::type>::type)

#undef BOOST_SPECIALIZE_CONSTANTS_FP
            }
        }
    }
}

// adjoint mode checks if partial derivatives are zero
namespace dco
{
    namespace folding
    {
        template <typename BOOST_BASE_TYPE, typename ROUNDING_POLICY>
        struct is_zero_trait<boost::numeric::interval<BOOST_BASE_TYPE, ROUNDING_POLICY>>
        {
            static bool get(const boost::numeric::interval<BOOST_BASE_TYPE, ROUNDING_POLICY> &x)
            {
                return boost::numeric::interval_lib::cereq(x, BOOST_BASE_TYPE(0));
            }
        };
    }
}
