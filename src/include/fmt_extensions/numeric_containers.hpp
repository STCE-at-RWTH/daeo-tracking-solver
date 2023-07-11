#ifndef _FMT_EXTENSIONS_NUMERIC_CONTAINERS_HPP
#define _FMT_EXTENSIONS_NUMERIC_CONTAINERS_HPP

#include <vector>

#include "fmt/core.h"
#include "fmt/format.h"

template <typename T>
struct fmt::formatter<std::vector<T>> : fmt::formatter<T>
{
    template <typename FormatContext>
    auto format(std::vector<T> &vec, FormatContext &ctx)
    {
        auto &&out = ctx.out();
        format_to(out, "[");
        bool first = true;
        for (auto &el : vec)
        {
            if (first)
            {
                first = false;
            }
            else
            {
                format_to(out, ", ");
            }
            fmt::formatter<T>::format(el, ctx);
        }
    }
    return format_to(out, "]");
};

#endif