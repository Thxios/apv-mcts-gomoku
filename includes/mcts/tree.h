
#pragma once

#include <vector>
#include <utility>
#include <memory>
#include <functional>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include "mcts/state.h"
#include "mcts/evaluator.h"


namespace mcts {

class MCTS {
private:
    class Node {
    public:
        Node(Prob p_, Node* parent=nullptr);
        ~Node();

        void Expand(const std::vector<std::pair<Action, Prob>>& prob_distribution);
        std::pair<Action, Node*> Select(double p_uct) const;
        void Update(Reward z);
        void ApplyVirtualLoss(int vloss);
        void RevertVirtualLoss(int vloss);
        double UCT(double p_uct) const;
        double Q() const;
        Action BestAction() const;
        Action BestAction(const std::function<double(const Node*)>& comp) const;
        std::pair<Action, Node*> DetachAction(Action action);

        inline Node* Parent() const {return parent;}
        inline int N() const {return n.load();}
        inline bool IsRoot() const {return !parent;}
        inline bool IsLeaf() const {return is_leaf.load();}
        
        Prob p;
        std::vector<std::pair<Action, Node*>> children;

    private:
        Node* parent;
        std::atomic<int> n = 0;
        std::atomic<double> w = 0;
        std::atomic<double> n_sqrt = 0;
        std::atomic<bool> expanding = false;
        std::atomic<bool> is_leaf = true;
    };

public:
    struct Config {
        size_t n_threads = 4;
        int virtual_loss = 3;
        double p_uct = 5;
    };

    struct ActionInfo {
        Action action;
        Prob p;
        int n;
        double q, uct;

        ActionInfo(Action action_, Prob p_, int n_, double q_, double uct_)
            : action(action_), p(p_), n(n_), q(q_), uct(uct_) {}
    };

public:
    MCTS(const StateBase& init_state, EvaluatorBase& evaluator);
    MCTS(const StateBase& init_state, EvaluatorBase& evaluator, Config conf);
    ~MCTS();

    void Search(int times);
    void Play(Action action);
    void Reset(const StateBase& init_state);
    void ApplyRootNoise(double alpha, double eps);
    Action GetBestAction() const;
    std::vector<ActionInfo> GetActionInfos() const;

    const Config config;

private:
    void StartThreads();
    void StopThreads();
    void SearchThreadJob(int t_idx);
    void SingleSearch(StateBase* search_state, int t_idx);
    void ExpandRoot();

    Node* root;
    std::unique_ptr<StateBase> state;
    EvaluatorBase& evaluator;

    std::vector<std::thread> threads;
    int counter, done;
    bool running;
    std::mutex m;
    std::condition_variable cv_search;
    std::condition_variable cv_wait;
};

}




