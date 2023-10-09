
#pragma once

#include <iostream>
#include <filesystem>
#include <indicators/block_progress_bar.hpp>
#include "mcts/tree.h"


namespace gomoku {
namespace selfeval {


namespace pb = indicators;
using mcts::MCTS;


struct AgentConfig {
    MCTS::Config mcts_cfg;
    size_t compute_budget = 1000;
    std::filesystem::path model_path;

    friend std::ostream& operator<<(std::ostream& out, const AgentConfig& cfg);
};


class Server {
public:
    struct Config {
        AgentConfig agent1, agent2;
        std::filesystem::path out_dir;
        size_t max_games;

        friend std::ostream& operator<<(std::ostream& out, const Config& cfg);
    };

public:
    Server(const Config& cfg);
    ~Server() = default;

    void Run();
    void LoadEvauator();

    const Config config;
    
private:


};



}
}

