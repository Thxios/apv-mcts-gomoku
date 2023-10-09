
#include <cmath>
#include <algorithm>
#include "mcts/tree.h"

namespace mcts {


MCTS::Node::Node(Prob p_, Node* parent_): p(p_), parent(parent_) {}


MCTS::Node::~Node() {
    for (auto& [_, child]: children) {
        delete child;
    }
}


void MCTS::Node::Expand
(const std::vector<std::pair<Action, Prob>>& prob_distribution) {
    if (expanding.exchange(true))
        return;
    for (auto& [action, prob]: prob_distribution) {
        children.emplace_back(action, new Node(prob, this));
    }
    is_leaf.store(false);
}


std::pair<Action, MCTS::Node*> MCTS::Node::Select(double p_uct) const {
    std::vector<double> ucts;
    std::transform(
        children.begin(), children.end(), 
        std::back_inserter(ucts),
        [&](const std::pair<Action, MCTS::Node*>& action_node) {
            return action_node.second->UCT(p_uct);
        }
    );
    int max_idx = std::max_element(ucts.begin(), ucts.end()) - ucts.begin();
    return children[max_idx];
}


void MCTS::Node::Update(Reward z) {
    n.fetch_add(1);
    w.fetch_add(z);
}


void MCTS::Node::ApplyVirtualLoss(int vloss) {
    n.fetch_add(vloss);
    w.fetch_sub(vloss);
}


void MCTS::Node::RevertVirtualLoss(int vloss) {
    n.fetch_sub(vloss);
    w.fetch_add(vloss);
}


double MCTS::Node::UCT(double p_uct) const {
    int n_ = n.load(), pn_ = parent->n.load();
    double w_ = w.load();
    double u = p * p_uct * std::sqrt(pn_) / (double)(1 + n_);
    return (n_ ? (w_ / n_) : 0) + u;
}


double MCTS::Node::Q() const {
    int n_ = n.load();
    return (n_ ? (w.load() / n_) : 0);
}


Action MCTS::Node::BestAction() const {
    return BestAction(
        [](const Node* node) -> double {
            return node->N();
        }
    );
}


Action MCTS::Node::BestAction
(const std::function<double(const Node*)>& comp) const {
    if (children.empty())
        return -1;
    std::vector<double> comps;
    std::transform(
        children.begin(), children.end(), 
        std::back_inserter(comps),
        [&](const std::pair<Action, Node*>& action_node) {
            return comp(action_node.second);
        }
    );
    int max_idx = std::max_element(comps.begin(), comps.end()) - comps.begin();
    return children[max_idx].first;
}


std::pair<Action, MCTS::Node*> MCTS::Node::DetachAction(Action action) {
    auto iter = std::find_if(
        children.begin(), children.end(),
        [action](const std::pair<Action, MCTS::Node*>& action_node) {
            return action_node.first == action;
        }
    );
    if (iter == children.end()) {
        return {action, nullptr};
    }
    std::pair<Action, MCTS::Node*> ret = *iter;
    ret.second->parent = nullptr;
    children.erase(iter);
    return ret;
}

}


