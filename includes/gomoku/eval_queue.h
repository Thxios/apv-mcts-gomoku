
#pragma once

#include <queue>
#include <thread>
#include <future>
#include <mutex>
#include <utility>
#include <condition_variable>
#include "mcts/evaluator.h"
#include "gomoku/evaluator.h"



namespace gomoku {

using mcts::Evaluation;


class EvaluationQueue : public mcts::EvaluatorBase {
public:
    EvaluationQueue(GomokuEvaluator &evaluator_);
    virtual ~EvaluationQueue();
    
    virtual Evaluation Evaluate(const mcts::StateBase* state);

private:
    void EvaluationThread();

    GomokuEvaluator& evaluator;
    std::thread eval_thread;

    bool running;
    std::queue<std::pair<Input, std::promise<Output>*>> q;
    std::mutex m_q;
    std::condition_variable cv_q;
};


}

