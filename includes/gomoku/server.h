
#pragma once

#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <mutex>
#include <atomic>
#include <memory>
#include <filesystem>
#include <indicators/block_progress_bar.hpp>
#include <indicators/dynamic_progress.hpp>
#include <indicators/indeterminate_progress_bar.hpp>
#include "mcts/tree.h"
#include "gomoku/eval_queue.h"


namespace gomoku {
namespace selfplay {


namespace pb = indicators;


struct SelplayConfig {
    size_t compute_budget = 400;
    double dirichlet_alpha = 0.03;
    double noise_eps = 0.25;
};


class Server {
public:
    struct Config {
        mcts::MCTS::Config mcts_cfg;
        gomoku::selfplay::SelplayConfig sp_cfg;
        std::filesystem::path model_path;
        std::filesystem::path out_dir;
        size_t starting_index;
        size_t max_games;
        size_t n_workers;

        friend std::ostream& operator<<(std::ostream& out, const Config& cfg) {
            std::cout << "model path: " << cfg.model_path << "\n";
            std::cout << "output dir: " << cfg.out_dir << "\n";
            std::cout << "logging start index: " << cfg.starting_index << "\n";
            std::cout << "max games: " << cfg.max_games << "\n";
            std::cout << "num workers: " << cfg.n_workers << "\n";
            std::cout << "mcts num threads: " << cfg.mcts_cfg.n_threads << "\n";
            std::cout << "mcts virtual loss: " << cfg.mcts_cfg.virtual_loss << "\n";
            std::cout << "mcts p_uct: " << cfg.mcts_cfg.p_uct << "\n";
            std::cout << "selfplay compute budget: " << cfg.sp_cfg.compute_budget << "\n";
            std::cout << "selfplay noise epsilon: " << cfg.sp_cfg.noise_eps << "\n";
            std::cout << "selfplay noise alpha: " << cfg.sp_cfg.dirichlet_alpha;
            return out;
        }
    };

public:
    Server(const Config& cfg);
    ~Server() = default;

    void Run();
    void LoadEvauator();

    const Config config;

private:
    void ThreadJob(int pbar_idx);
    int SingleSelfplay(
        int game_idx,
        int pbar_idx
    );
    
    std::unique_ptr<EvaluationQueue> evaluator;
    std::vector<std::unique_ptr<pb::IndeterminateProgressBar>> bars;
    pb::DynamicProgress<pb::IndeterminateProgressBar> pbar;
    std::atomic<int> game_idx;
    std::mutex m_master;
    std::unique_ptr<pb::BlockProgressBar> master_pbar;
    int total, done;

    std::filesystem::path out_state_dir, out_txt_dir;
};




}
}
