#include <iostream>
#include <tuple>

#include "ntuple.hpp"
#include "dco.hpp"

int main(int argc, char *argv[])
{
    ntuple<3, int> t = {1, 2, 3};
    std::cout << std::get<2>(t) << std::endl;
}