#ifndef _UTILS_IO_HPP
#define _UTILS_IO_HPP

#include <sstream>
#include <string>
#include <vector>

#include "fmt/format.h"
#include "fmt/ostream.h"

using std::vector;

template <typename T>
std::string print_vector(vector<T> const &arg)
{
    std::stringstream out;
    out << "[";
    for (size_t i = 0; i < arg.size(); i++)
    {
        out << arg[i];
        if (i < arg.size() - 1)
        {
            out << ", ";
        };
    }
    out << "]";
    return out.str();
}


#endif