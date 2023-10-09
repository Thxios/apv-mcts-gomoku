
#include <chrono>
#include <algorithm>
#include <torch/torch.h>
#include <fmt/format.h>
#include "gomoku/server.h"
#include "gomoku/board.h"
#include "gomoku/logger.h"



namespace gomoku {
namespace selfplay {


thread_local std::mt19937 gen(std::random_device{}());


Server::Server(const Server::Config& cfg): config(cfg) {
    out_state_dir = config.out_dir / "state";
    out_txt_dir = config.out_dir / "txt";
    std::filesystem::create_directories(out_state_dir);
    std::filesystem::create_directories(out_txt_dir);
}


void Server::LoadEvauator() {
    std::cout << "===== Load Evaluator =====" << std::endl;
    
    torch::jit::script::Module model(torch::jit::load(config.model_path));
    evaluator = std::make_unique<EvaluationQueue>(
        GomokuEvaluator(std::move(model)));

    std::cout << "===== Evaluator Loaded =====" << std::endl;
}

void Server::ThreadJob(int pbar_idx) {
    int g_idx;
    while ((g_idx = game_idx.fetch_add(1)) < config.max_games) {
        SingleSelfplay(g_idx, pbar_idx);
        {
            std::unique_lock<std::mutex> lock(m_master);
            done++;
            master_pbar->set_option(pb::option::PrefixText(
                fmt::format("[{:>4}/{:>4}]", done, total)));
            master_pbar->tick();
        }
    }
    pbar[pbar_idx].set_option(pb::option::PostfixText("Done")); 
    pbar[pbar_idx].mark_as_completed();
}


void Server::Run() {
    std::cout << "===== Running Selfplay =====" << std::endl;
    // pb::show_console_cursor(false);
    pbar.set_option(pb::option::HideBarWhenComplete(false));

    for (int i = 0; i < config.n_workers; i++) {
        std::unique_ptr<pb::IndeterminateProgressBar> bar_ptr(
            std::make_unique<pb::IndeterminateProgressBar>(
                pb::option::BarWidth(0),
                pb::option::PrefixText(fmt::format("Worker {:02d}:", i)),
                pb::option::Start(""),
                pb::option::End("")
            )
        );
        pbar.push_back(*bar_ptr);
        bars.emplace_back(std::move(bar_ptr));
    }
    pbar.print_progress();

    total = config.max_games - config.starting_index;
    done = 0;
    // master_pbar = std::make_unique<pb::ProgressBar>(
    master_pbar = std::make_unique<pb::BlockProgressBar>(
        pb::option::PrefixText(fmt::format("[{:>4}/{:>4}]", done, total)),
        pb::option::BarWidth{25},
        pb::option::Start{"["},
        pb::option::End{"]"},
        pb::option::ShowElapsedTime{true},
        pb::option::ShowRemainingTime{true},
        pb::option::MaxProgress(total)
    );
    master_pbar->print_progress();

    game_idx.store(config.starting_index);
    std::vector<std::thread> workers;
    for (int i = 0; i < config.n_workers; i++) {
        workers.emplace_back(&Server::ThreadJob, this, i);
    }
    for (auto& worker: workers) {
        worker.join();
    }
    // pb::show_console_cursor(true);
    std::cout << std::endl;
    std::cout << "===== Selfplay Completed =====" << std::endl;
}


int Server::SingleSelfplay(int game_idx, int pbar_idx) {
    std::ofstream out(
        out_txt_dir / fmt::format("{:04d}.txt", game_idx), std::ios::out);

    std::vector<Action> actions;
    std::vector<std::vector<int>> counts;
    SelplayConfig cfg = config.sp_cfg;

    Board board;
    MCTS tree(board, *evaluator, config.mcts_cfg);

    std::chrono::system_clock::time_point st, ed, total_st, total_ed;
    std::vector<MCTS::ActionInfo> action_infos;
    int game_len = 0;

    total_st = std::chrono::system_clock::now();
    while (!board.Terminated()) {
        pbar[pbar_idx].set_option(pb::option::PostfixText(
            fmt::format("Game {} - Turn {}", game_idx, game_len))); 
        pbar[pbar_idx].print_progress();

        if (game_len < cfg.noise_steps) {
            tree.ApplyRootNoise(cfg.noise_alpha, cfg.noise_eps);
        }
        st = std::chrono::system_clock::now();
        if (game_len > 0) {
            tree.Search(cfg.compute_budget);
        }
        ed = std::chrono::system_clock::now();

        action_infos = tree.GetActionInfos();
        // Action best = tree.GetBestAction();
        Action move = SelectMove(action_infos, game_len);

        board.Play(move);
        out << board << '\n';
        out << fmt::format("action: {:>3}, search time: {:.4f} sec\n",
            Coord2String(Action2Coord(move)), 
            std::chrono::duration<double>(ed - st).count());
        ShowTopActions(action_infos, 5, out);
        out << std::endl;
        tree.Play(move);
        tree.Reset(board);
        game_len++;

        actions.push_back(move);
        std::vector<int> single_counts(SIZE * SIZE, 0);
        for (const MCTS::ActionInfo& info: action_infos) {
            single_counts[info.action] = info.n;
        }
        counts.emplace_back(std::move(single_counts));
    }

    total_ed = std::chrono::system_clock::now();
    pbar[pbar_idx].set_option(pb::option::PostfixText(
        fmt::format("Game {} - Ended", game_idx))); 

    Board::State result = board.GetState();
    out << fmt::format("{}, game len: {}, total {:.1f} sec", 
        Board::state2str(result), game_len,
        std::chrono::duration<double>(total_ed - total_st).count())
         << std::endl;

    std::filesystem::path state_save_path
        = out_state_dir / fmt::format("{:04d}.bin", game_idx);
    gomoku::logger::Log selfplay_log(game_len, result, actions, counts);
    try {
        selfplay_log.Save(state_save_path);
    }
    catch (std::exception& e) {
        out << "error saving log to " << state_save_path 
            << ": " << e.what() << std::endl;
        return 1;
    }
    return 0;
}


mcts::Action Server::SelectMove(
    const std::vector<MCTS::ActionInfo>& infos, int turn) const {
    
    std::vector<int> weights(infos.size(), 0);
    if (turn < config.sp_cfg.sample_steps) {
        for (int i = 0; i < infos.size(); i++) {
            weights[i] = infos[i].n;
        }
    }
    else {
        int n_max = std::max_element(
            infos.begin(), infos.end(), 
            [](const MCTS::ActionInfo& a, const MCTS::ActionInfo& b) {
                return a.n < b.n;
            })->n;
        for (int i = 0; i < infos.size(); i++) {
            if (infos[i].n == n_max)
                weights[i] = 1;
        }
    }
    std::discrete_distribution<int> sampler(weights.begin(), weights.end());
    return infos[sampler(gen)].action;
}


std::ostream& operator<<(std::ostream& out, const Server::Config& cfg) {
    std::cout << "model path: " << cfg.model_path << "\n";
    std::cout << "output dir: " << cfg.out_dir << "\n";
    std::cout << "logging start index: " << cfg.starting_index << "\n";
    std::cout << "max games: " << cfg.max_games << "\n";
    std::cout << "num workers: " << cfg.n_workers << "\n";
    std::cout << "mcts num threads: " << cfg.mcts_cfg.n_threads << "\n";
    std::cout << "mcts virtual loss: " << cfg.mcts_cfg.virtual_loss << "\n";
    std::cout << "mcts p_uct: " << cfg.mcts_cfg.p_uct << "\n";
    std::cout << "selfplay compute budget: " << cfg.sp_cfg.compute_budget << "\n";
    std::cout << "selfplay sample steps: " << cfg.sp_cfg.sample_steps << "\n";
    std::cout << "selfplay noise steps: " << cfg.sp_cfg.noise_steps << "\n";
    std::cout << "selfplay noise epsilon: " << cfg.sp_cfg.noise_eps << "\n";
    std::cout << "selfplay noise alpha: " << cfg.sp_cfg.noise_alpha;
    return out;
}

    
} // namespace selfplay
} // namespace gomoku

