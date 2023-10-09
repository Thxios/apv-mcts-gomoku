
#pragma once

#include <iostream>
#include <random>
#include <thread>
#include <vector>
#include <mutex>
#include <atomic>
#include <memory>
#include <filesystem>
#include <boost/program_options.hpp>
#include <indicators/block_progress_bar.hpp>
#include <indicators/dynamic_progress.hpp>
#include <indicators/indeterminate_progress_bar.hpp>
#include "mcts/tree.h"
#include "gomoku/eval_queue.h"


namespace gomoku {
namespace selfplay {


namespace pb = indicators;
using mcts::MCTS;


struct SelfplayConfig {
    size_t compute_budget = 1000;
    size_t sample_steps = 15;
    size_t noise_steps = 3;
    double noise_alpha = 0.03;
    double noise_eps = 0.25;
};


class Server {
public:
    struct Config {
        MCTS::Config mcts_cfg;
        SelfplayConfig sp_cfg;
        std::filesystem::path model_path;
        std::filesystem::path out_dir;
        size_t starting_index;
        size_t max_games;
        size_t n_workers;

        friend std::ostream& operator<<(std::ostream& out, const Config& cfg);
    };

public:
    Server(const Config& cfg);
    ~Server() = default;

    void Run();
    void LoadEvauator();

    const Config config;

private:
    void ThreadJob(int pbar_idx);
    int SingleSelfplay(int game_idx, int pbar_idx);
    mcts::Action SelectMove(
        const std::vector<MCTS::ActionInfo>& infos, int turn) const;
    
    std::unique_ptr<EvaluationQueue> evaluator;
    std::atomic<int> game_idx;

    std::unique_ptr<pb::BlockProgressBar> master_pbar;
    std::vector<std::unique_ptr<pb::IndeterminateProgressBar>> bars;
    pb::DynamicProgress<pb::IndeterminateProgressBar> pbar;
    std::mutex m_master;
    int total, done;

    std::filesystem::path out_state_dir, out_txt_dir;
};


boost::program_options::options_description 
GetSelfplayConfig(Server::Config& cfg);


}
}
