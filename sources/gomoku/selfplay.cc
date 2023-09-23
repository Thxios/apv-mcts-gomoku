
#include <utility>
#include <exception>
#include <chrono>
#include <algorithm>
#include <fmt/format.h>
#include "gomoku/selfplay.h"
#include "gomoku/logger.h"



namespace gomoku {
namespace selfplay {


int SingleSelfplay(
    MCTS::Config tree_config,
    SelplayConfig config,
    EvaluationQueue& evaluator,
    const std::string& save_path,
    std::ostream& out
) {
    std::vector<Action> actions;
    std::vector<std::vector<int>> counts;

    Board board;
    MCTS tree(board, evaluator, tree_config);

    std::chrono::system_clock::time_point st, ed;
    std::vector<MCTS::ActionInfo> action_infos;
    int game_len = 0;

    while (!board.Terminated()) {
        game_len++;
        tree.Reset(board);
        tree.ApplyRootNoise(config.dirichlet_alpha, config.noise_eps);

        st = std::chrono::system_clock::now();
        tree.Search(config.compute_budget);
        ed = std::chrono::system_clock::now();

        action_infos = tree.GetActionInfos();
        Action best = tree.GetBestAction();

        board.Play(best);
        out << board << '\n';
        out << fmt::format("action: {:>3}, search time: {:.4f} sec\n",
            Coord2String(Action2Coord(best)), 
            std::chrono::duration<double>(ed - st).count());
        ShowTopActions(action_infos, 5, out);
        out << std::endl;
        tree.Play(best);

        actions.push_back(best);
        std::vector<int> single_counts(SIZE * SIZE, 0);
        for (const MCTS::ActionInfo& info: action_infos) {
            single_counts[info.action] = info.n;
        }
        counts.emplace_back(std::move(single_counts));
    }
    Board::State result = board.GetState();
    
    switch (result) {
    case Board::State::ONGOING:
        out << "ONGOING";
        break;
    case Board::State::BLACK_WIN:
        out << "BLACK(X) WIN";
        break;
    case Board::State::WHITE_WIN:
        out << "WHITE(O) WIN";
        break;
    case Board::State::DRAW:
        out << "DRAW";
        break;
    }
    out << fmt::format(", game len: {:>3}", game_len);
    out << std::endl;

    if (!save_path.empty()) {
        logger::Log selfplay_log(game_len, result, actions, counts);
        try {
            selfplay_log.Save(save_path);
        }
        catch (std::exception& e) {
            out << "error saving log to " << save_path 
                << ": " << e.what() << std::endl;
            std::cerr << "error saving log to " << save_path 
                << ": " << e.what() << std::endl;
            return 1;
        }
    }
    return 0;
}


void ShowTopActions(
    std::vector<mcts::MCTS::ActionInfo>& infos, int k, std::ostream& out) {

    std::sort(
        infos.begin(), infos.end(), 
        [](const mcts::MCTS::ActionInfo& a, const mcts::MCTS::ActionInfo& b) {
            return ((a.n == b.n) ? (a.p > b.p) : (a.n > b.n));
        }
    );

    if (k <= 0 || k > infos.size())
        k = infos.size();
    for (int i = 0; i < k; i++) {
        mcts::MCTS::ActionInfo info = infos[i];
        out << fmt::format(
            "{:3} : N={:3}, P={:.4f}, Q={: .4f}, UCT={: .4f}",
            Coord2String(Action2Coord(info.action)),
            info.n, info.p, info.q, info.uct) << std::endl;
    }
}

    
} // namespace selfplay
} // namespace gomoku

