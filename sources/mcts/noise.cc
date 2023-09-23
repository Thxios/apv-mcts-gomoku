
#include "mcts/noise.h"



namespace noise {

std::random_device Dirichlet::rd;
std::mt19937 Dirichlet::gen(Dirichlet::rd());

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

void Dirichlet::Seed(int seed) {
    gen = std::mt19937(seed);
}

}

