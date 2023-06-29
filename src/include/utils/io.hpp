#ifndef _UTILS_IO_HPP
#define _UTILS_IO_HPP

#include <iostream>
#include <vector>

using std::vector;

template <typename T>
std::stringstream print_vector(vector<T> const &arg)
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
    return out;
}


#endif