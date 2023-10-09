
#pragma once

#include <random>
#include <vector>


namespace noise {


class Dirichlet {
public:
    static std::vector<double> Sample(double alpha, int size);
};


}
