#ifndef _FMT_EXTENSIONS_BOOST_INTERVAL_HPP
#define _FMT_EXTENSIONS_BOOST_INTERVAL_HPP

#include "boost/numeric/interval.hpp"

#include "fmt/core.h"
#include "fmt/format.h"

using boost::numeric::interval;

template <typename T, typename P>
struct fmt::formatter<interval<T, P>> : public fmt::formatter<T>
{
    template <typename FormatContext>
    auto format(interval<T, P> const &ival, FormatContext &ctx) const
    {
        auto &&out = ctx.out();
        format_to(out, "[");
        fmt::formatter<T>::format(ival.lower(), ctx);
        format_to(out, ", ");
        fmt::formatter<T>::format(ival.upper(), ctx);
        return format_to(out, "]");
    }
};

#endif