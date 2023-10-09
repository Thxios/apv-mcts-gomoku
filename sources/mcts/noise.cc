
#include "mcts/noise.h"


namespace noise {


thread_local std::mt19937 gen(std::random_device{}());


std::vector<double> Dirichlet::Sample(double alpha, int size) {
    std::gamma_distribution<double> g(alpha);
    std::vector<double> prob(size);
    double sum = 0;

    for (int i = 0; i < size; i++) {
        prob[i] = g(gen);
        sum += prob[i];
    }
    for (int i = 0; i < size; i++)
        prob[i] /= sum;

    return prob;
}


}

