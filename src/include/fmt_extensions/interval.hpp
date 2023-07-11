#ifndef _FMT_EXTENSIONS_BOOST_INTERVAL_HPP
#define _FMT_EXTENSIONS_BOOST_INTERVAL_HPP

#include "boost/numeric/interval.hpp"

#include "fmt/core.h"
#include "fmt/format.h"

template <typename T, typename P>
struct fmt::formatter<boost::numeric::interval<T, P>> : public fmt::formatter<T>
{
    using ival_t = boost::numeric::interval<T, P>;

    template <typename FormatContext>
    auto format(ival_t &ival, FormatContext &ctx)
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