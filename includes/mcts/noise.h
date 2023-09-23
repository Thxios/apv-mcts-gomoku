
#pragma once

#include <vector>
#include <random>


namespace noise {

class Dirichlet {
public:
    static std::vector<double> Sample(double alpha, int size);
    static void Seed(int seed);
    
private:
    static std::random_device rd;
    static std::mt19937 gen;
};

}
