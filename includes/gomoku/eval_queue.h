
#pragma once

#include <vector>
#include <memory>
#include <queue>
#include <thread>
#include <future>
#include <mutex>
#include <utility>
#include <condition_variable>
#include <unordered_set>
#include "mcts/evaluator.h"
#include "gomoku/evaluator.h"



namespace gomoku {

using mcts::Evaluation;


class EvaluationQueue : public mcts::EvaluatorBase {
public:
    using Evaluator = std::unique_ptr<GomokuEvaluator>;

public:
    EvaluationQueue(Evaluator evaluator);
    EvaluationQueue(std::vector<Evaluator> evaluators);
    EvaluationQueue(EvaluationQueue&& other) = delete;
    virtual ~EvaluationQueue();
    
    virtual Evaluation Evaluate(const mcts::StateBase* state);

private:
    // void EvaluationThread();
    void EvaluationThread(Evaluator evaluator);

    // GomokuEvaluator evaluatos;
    // std::thread eval_thread;
    // std::vector<GomokuEvaluator> evaluators;
    std::vector<std::thread> eval_threads;

    bool running;
    std::queue<std::pair<Input, std::promise<Output>*>> q;
    std::mutex m_q;
    std::condition_variable cv_q;

    // std::unordered_set<std::size_t> hashes;
    // std::mutex m_s;
};


}

