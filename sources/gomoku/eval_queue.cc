
#include <vector>
#include <iostream>
#include "gomoku/eval_queue.h"


namespace gomoku {


EvaluationQueue::EvaluationQueue(GomokuEvaluator&& evaluator_)
: evaluator(evaluator_) {
    running = true;
    eval_thread = std::thread(&EvaluationQueue::EvaluationThread, this);
}


EvaluationQueue::~EvaluationQueue() {
    // std::cout << "EvaluationQueue::~EvaluationQueue()" << std::endl;
    {
        std::unique_lock<std::mutex> lock(m_q);
        running = false;
    }
    cv_q.notify_one();
    eval_thread.join();
}


Evaluation EvaluationQueue::Evaluate(const mcts::StateBase* state) {
    const Board& board = dynamic_cast<const Board&>(*state);
    // std::size_t hash = board.Hash();
    // {
    //     std::unique_lock<std::mutex> lock(m_s);
    //     hashes.insert(hash);
    // }
    Input input = evaluator.Preprocess(board);

    std::promise<Output> p;
    std::future<Output> f(p.get_future());
    {
        std::unique_lock<std::mutex> lock(m_q);
        q.push({std::move(input), &p});
    }
    cv_q.notify_one();

    Output output = f.get();
    Evaluation evaluation = evaluator.Postprocess(std::move(output), board);
    return evaluation;
}


void EvaluationQueue::EvaluationThread() {
    // printf("eval thread started");
    while (true) {
        std::vector<Input> inputs;
        std::vector<std::promise<Output>*> promises;
        {
            std::unique_lock<std::mutex> lock(m_q);
            cv_q.wait(lock, [&] {
                return !running || !q.empty();
            });
            if (!running)
                break;
            while (!q.empty()) {
                inputs.push_back(std::move(q.front().first));
                promises.push_back(q.front().second);
                q.pop();
            }
        }
        if (!inputs.empty()) {
            std::vector<Output> results = evaluator.EvaluateBatch(inputs);
            for (int i = 0; i < results.size(); i++) {
                promises[i]->set_value(std::move(results[i]));
            }
        }
    }
    // printf("eval thread stop");
}


}



