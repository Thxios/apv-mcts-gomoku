
#pragma once

#include <iostream>
#include <string>
#include "mcts/tree.h"
#include "gomoku/board.h"
#include "gomoku/eval_queue.h"


namespace gomoku {
namespace selfplay {


using mcts::MCTS;


struct SelplayConfig {
    int compute_budget = 400;
    double dirichlet_alpha = 0.03;
    double noise_eps = 0.25;
};


int SingleSelfplay(
    MCTS::Config tree_config,
    SelplayConfig config,
    EvaluationQueue& evaluator,
    const std::string& save_path = "",
    std::ostream& out = std::cout
);

void ShowTopActions(std::vector<MCTS::ActionInfo>& infos, 
                    int k = 5, 
                    std::ostream& out = std::cout);


}
}
