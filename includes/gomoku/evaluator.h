
#pragma once

#include <vector>
#include <utility>
#include <torch/script.h>
#include "mcts/state.h"
#include "mcts/evaluator.h"
#include "gomoku/board.h"



namespace gomoku {

using mcts::Reward;
using mcts::Action;
using mcts::Prob;
using mcts::Evaluation;


struct Input {
    Input() = default;
    Input(Input&& other) = default;
    Input& operator=(Input&& other) = default;

    torch::Tensor state;
    torch::Tensor turn;
    torch::Tensor mask;
};

using Output = std::pair<float, torch::Tensor>;


class GomokuEvaluator {
public:
    GomokuEvaluator(torch::jit::script::Module&& model_);
    GomokuEvaluator(torch::jit::script::Module&& model_, torch::Device device_);
    GomokuEvaluator(GomokuEvaluator&& other) = default;
    
    std::vector<Output> EvaluateBatch(std::vector<Input>& inputs);

    static Input Preprocess(const Board& board);
    static Evaluation Postprocess(Output&& output, const Board& board);

private:
    torch::jit::script::Module model;
    torch::Device device;
};


}

