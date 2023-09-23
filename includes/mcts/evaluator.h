
#pragma once

#include <vector>
#include <utility>
#include "mcts/state.h"


namespace mcts {

using Evaluation = std::pair<Reward, std::vector<std::pair<Action, Prob>>>;


class EvaluatorBase {
public:
    virtual ~EvaluatorBase() = default;
    
    virtual Evaluation Evaluate(const StateBase* state) = 0;
};

}

