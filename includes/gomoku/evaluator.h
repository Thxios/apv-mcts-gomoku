
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
    torch::Tensor state;
    torch::Tensor turn;
    torch::Tensor mask;
};

using Output = std::pair<float, torch::Tensor>;


class GomokuEvaluator {
public:
    explicit GomokuEvaluator(torch::jit::script::Module&& model_);
    ~GomokuEvaluator() = default;
    
    std::vector<Output> EvaluateBatch(std::vector<Input>& inputs);
    Input Preprocess(const Board& board) const;
    Evaluation Postprocess(Output& output, const Board& board) const;

private:
    torch::jit::script::Module model;
    torch::Device device;
};


}

