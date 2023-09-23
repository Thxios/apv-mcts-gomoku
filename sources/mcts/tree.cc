
// #include <iostream>
#include "mcts/tree.h"
#include "mcts/noise.h"


namespace mcts {


MCTS::MCTS(const StateBase& init_state, EvaluatorBase& evaluator)
: MCTS::MCTS(init_state, evaluator, Config()) {}


MCTS::MCTS
(const StateBase& init_state, EvaluatorBase& evaluator_, MCTS::Config conf)
: config(conf), state(init_state.GetCopy()), evaluator(evaluator_) {
    root = new Node(1);
    ExpandRoot();
    StartThreads();
}

void MCTS::Reset(const StateBase& init_state) {
    state = init_state.GetCopy();
    delete root;
    root = new Node(1);
    ExpandRoot();
}


MCTS::~MCTS() {
    StopThreads();
    delete root;
}

void MCTS::StartThreads() {
    counter = 0;
    done = 0;
    running = true;
    for (int i = 0; i < config.n_threads; i++) {
        threads.emplace_back(&MCTS::SearchThreadJob, this, i);
    }
}

void MCTS::StopThreads() {
    {
        std::unique_lock<std::mutex> lock(m);
        running = false;
    }
    cv_search.notify_all();
    for (std::thread& thread: threads) {
        thread.join();
    }
}

void MCTS::Search(int times) {
    {
        std::unique_lock<std::mutex> lock(m);
        counter = times;
        done = times;
    }
    cv_search.notify_all();
    {
        std::unique_lock<std::mutex> lock(m);
        cv_wait.wait(lock, [&] {
            return done == 0;
        });
    }
    // std::cout << "root N: " << root->N() << ", Q: " << root->Q() << std::endl;
}


void MCTS::SearchThreadJob(int t_idx) {
    // printf("%d search thread started\n", t_idx);
    while (true) {
        {
            std::unique_lock<std::mutex> lock(m);
            cv_search.wait(lock, [&] {
                return !running || (counter > 0);
            });
            if (!running)
                break;
            // printf("%d: counter %d\n", t_idx, counter);
            counter--;
        }

        std::unique_ptr<StateBase> state_copy = state->GetCopy();
        SingleSearch(state_copy.get(), t_idx);

        {
            std::unique_lock<std::mutex> lock(m);
            done--;
            if (done == 0)
                cv_wait.notify_one();
        }
    }
    // printf("%d search thread terminated\n", t_idx);
}


void MCTS::SingleSearch(StateBase* search_state, int t_idx) {
    // select
    Node* cur = root;
    while (!cur->IsLeaf()) {
        auto [action, child] = cur->Select(config.p_uct);
        search_state->Play(action);
        cur = child;
        cur->ApplyVirtualLoss(config.virtual_loss);
    }

    // evaluate & expand
    Reward z;
    if (search_state->Terminated()) {
        z = search_state->TerminalReward();
    }
    else {
        Evaluation output = evaluator.Evaluate(search_state);
        z = output.first;
        cur->Expand(output.second);
    }

    // backup
    while (!cur->IsRoot()) {
        cur->RevertVirtualLoss(config.virtual_loss);
        cur->Update(z);
        cur = cur->Parent();
        z = -z;
    }
    cur->Update(z);
}


void MCTS::ExpandRoot() {
    if (!state->Terminated() && root->IsLeaf()) {
        Evaluation output = evaluator.Evaluate(state.get());
        root->Expand(output.second);
    }
}


void MCTS::ApplyRootNoise(double alpha, double eps) {
    ExpandRoot();
    if (!root->children.empty()) {
        std::vector<Prob> noise 
            = noise::Dirichlet::Sample(alpha, root->children.size());
        for (int i = 0; i < root->children.size(); i++) {
            root->children[i].second->p 
                = (1 - eps) * root->children[i].second->p + eps * noise[i];
        }
    }
}


void MCTS::Play(Action action) {
    state->Play(action);

    MCTS::Node* next_root = root->DetachAction(action).second;
    if (!next_root) {
        next_root = new MCTS::Node(1);
    }
    delete root;
    root = next_root;
}


Action MCTS::GetBestAction() const {
    return root->BestAction();
}


std::vector<MCTS::ActionInfo> MCTS::GetActionInfos() const {
    std::vector<MCTS::ActionInfo> ret;
    for (const auto& [action, node]: root->children) {
        ret.emplace_back(
            action,
            node->p,
            node->N(),
            node->Q(),
            node->UCT(config.p_uct)
        );
    }
    return ret;
}


}

