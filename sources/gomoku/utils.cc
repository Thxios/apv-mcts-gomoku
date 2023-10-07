
#include <fmt/format.h>
#include "gomoku/coord.h"
#include "gomoku/board.h"

namespace gomoku {


std::ostream& operator<<(std::ostream& out, Coord pos) {
    out << "(" << pos.r << ", " << pos.c << ")";
    return out;
}

std::string Coord2String(Coord pos) {
    return Coord2String(pos.r, pos.c);
}

std::string Coord2String(int r, int c) {
    return fmt::format("{}{}", (char)(c + 'a'), r + 1);
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

}

