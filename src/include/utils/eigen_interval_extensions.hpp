/**
 * @author Sasha [fleming@stce.rwth-aachen.de]
 */

#ifndef _INTERVAL_EIGEN_TRAITS_HPP
#define _INTERVAL_EIGEN_TRAITS_HPP

#include "Eigen/Core"
#include "boost/numeric/interval.hpp"

namespace Eigen
{
    template <typename T, typename POLICIES>
    struct NumTraits<boost::numeric::interval<T, POLICIES>> : NumTraits<T>
    {

        enum
        {
            IsComplex = 0,
            IsInteger = 0,                     // specialization here if we want intervals of integers
            IsSigned = NumTraits<T>::IsSigned, // does this make sense? intervals of uints or ufloats don't seem useful.
            RequireInitialization = 1,         // intervals need initialization (probably?)
            ReadCost = 2,                      // could also be reasonably set to 1, even though there are two values
            AddCost = 2,                       // two numbers
            MulCost = 2                        // two numbers
        };

        // Intervals cannot be made out of complex numbers due to total ordering requirement
        using Real = boost::numeric::interval<T, POLICIES>;
        using Literal = T;
        // If we end up doing lots of integer interval work, this should be specialized
        using NonInteger = boost::numeric::interval<T, POLICIES>;
        // Intervals aren't intermediates in lazy expressions. Just copy them.
        using Nested = boost::numeric::interval<T, POLICIES>;

        // "highest" returns [max, max] st. highest > everything else
        static inline Real highest()
        {
            return Real(NumTraits<T>::highest());
        }

        // "lowest" returns [min, min] st. lowest < everything else
        static inline Real lowest()
        {
            return Real(NumTraits<T>::lowest());
        }
    };

}; // Eigen

namespace boost::numeric
{
    // intervals need to have some extra stuff tacked on for Eigen
    // namely: abs2, conj, real, imag
    // abs is already defined

    template <typename T, typename POLICIES>
    inline interval<T, POLICIES> abs2(const interval<T, POLICIES> &arg)
    {
        // norm(x)^2 = x^2
        return square(arg);
    };

    template <typename T, typename POLICIES>
    inline const interval<T, POLICIES> &conj(const interval<T, POLICIES> &arg)
    {
        // complex conjugate of a real interval is... itself.
        return arg;
    }

    template <typename T, typename POLICIES>
    inline const interval<T, POLICIES> &real(const interval<T, POLICIES> &arg)
    {
        // real part of a real interval is... itself.
        return arg;
    }

    template <typename T, typename POLICIES>
    inline interval<T, POLICIES> imag(const interval<T, POLICIES> &arg)
    {
        // imaginary part of a real interval is... zero.
        // empty interval constructor returns singleton zero
        return interval<T, POLICIES>();
    }

}; // boost::numeric
#endif