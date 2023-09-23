
#pragma once

#include <memory>


namespace mcts {


using Action = int;
using Prob = double;
using Reward = double;


class StateBase {
public:
    virtual ~StateBase() = default;

    virtual std::unique_ptr<StateBase> GetCopy() const = 0;
    virtual void Play(Action action) = 0;
    virtual bool Terminated() const = 0;
    virtual Reward TerminalReward() const = 0;
};


}
