
#include <string>
#include <iomanip>
#include <exception>
#include <fmt/format.h>
#include <boost/functional/hash.hpp>
#include "gomoku/board.h"


namespace gomoku {

Board::Board() {
    Reset();
}


// Board::Board(const Board& b): turn(b.turn), state(b.state), 
// turn_elapsed(b.turn_elapsed), last_action(b.last_action) {
//     for (int r = 0; r < SIZE; r++) {
//         for (int c = 0; c < SIZE; c++) {
//             board[0][r][c] = b.board[0][r][c];
//             board[1][r][c] = b.board[1][r][c];
//             board[2][r][c] = b.board[2][r][c];
//         }
//     }
// }


std::unique_ptr<mcts::StateBase> Board::GetCopy() const {
    return std::unique_ptr<StateBase>(std::make_unique<Board>(*this));
}


bool Board::Terminated() const {
    return state != State::ONGOING;
}


mcts::Reward Board::TerminalReward() const {
    // switch (state) {
    // case BLACK_WIN:
    //     return 1.;
    // case WHITE_WIN:
    //     return -1.;
    // case ONGOING:
    //     return 0.;
    // }
    // return 0.;
    if (state == State::BLACK_WIN || state == State::WHITE_WIN)
        return 1.;
    else
        return 0.;
}


void Board::Reset() {
    state = State::ONGOING;
    turn = BLACK;
    turn_elapsed = 0;
    last_action = -1;
    for (int r = 0; r < SIZE; r++) {
        for (int c = 0; c < SIZE; c++) {
            board[EMPTY][r][c] = 1;
            board[BLACK][r][c] = 0;
            board[WHITE][r][c] = 0;
        }
    }
}


std::size_t Board::Hash() const {
    std::size_t seed = 0;
    boost::hash_combine<std::size_t>(seed, black_hsum);
    boost::hash_combine<std::size_t>(seed, white_hsum);
    return seed;
}


void Board::Play(mcts::Action action) {
    Coord pos = Action2Coord(action);
    if (Terminated())
        throw std::runtime_error("game already ended");
    if (action < 0 || action >= SIZE * SIZE)
        throw std::runtime_error(fmt::format("action {} is out of range", action));
    if (GetColor(action) != EMPTY)
        throw std::runtime_error(fmt::format("action {} is not empty", action));

    std::size_t seed = 0;
    boost::hash_combine<int>(seed, action);

    board[EMPTY][pos.r][pos.c] = 0;
    board[turn][pos.r][pos.c] = 1;

    if (turn == BLACK) {
        turn = WHITE;
        black_hsum += seed;
    }
    else if (turn == WHITE) {
        turn = BLACK;
        white_hsum += seed;
    }
    turn_elapsed++;

    last_action = action;
    state = CheckState(action);
}


void Board::Play(Coord pos) {
    if (!Inside(pos))
        throw std::runtime_error(
            fmt::format("({}, {}) is out of board", pos.r, pos.c));
    Play(Coord2Action(pos));
}


void Board::Play(int r, int c) {
    Coord pos(r, c);
    if (!Inside(pos))
        throw std::runtime_error(
            fmt::format("({}, {}) is out of board", r, c));
    Play(Coord2Action(r, c));
}


Board::State Board::CheckState(mcts::Action action) const {
    if (action < 0)
        return State::ONGOING;

    Coord pos = Action2Coord(action);
    Color color = GetColor(pos);

    for (Coord delta: DELTA) {
        if (FiveInRow(color, pos, delta))
            return ((color == BLACK) ? State::BLACK_WIN : State::WHITE_WIN);
    }
    
    if (turn_elapsed < SIZE * SIZE)
        return State::ONGOING;
    else
        return State::DRAW;
}

bool Board::FiveInRow(Color color, Coord pos, Coord delta) const {
    if (GetColor(pos) != color)
        return false;

    Coord cur;
    int n_left = 0, n_right = 1;

    cur = pos - delta;
    while (Inside(cur) && board[color][cur.r][cur.c]) {
        cur -= delta;
        n_left++;
    }
    cur = pos + delta;
    while (Inside(cur) && board[color][cur.r][cur.c]) {
        cur += delta;
        n_right++;
    }
    return (n_left + n_right) == 5;
}


std::ostream& operator<<(std::ostream& out, Board& b) {
    Coord last_pos;
    if (b.last_action != -1)
        last_pos = Action2Coord(b.last_action);
    
    out << "    ";
    for (int c = 0; c < SIZE; c++)
        out << (char)(c + 'a') << ' ';
    out << '\n';
    for (int r = 0; r < SIZE; r++) {
        out << std::setw(3) << (r + 1);
        out << ((r == last_pos.r && last_pos.c == 0) ? '[' : ' ');
        for (int c = 0; c < SIZE; c++) {
            out << (".XO"[b.GetColor(r, c)]);
            if (r == last_pos.r && c == last_pos.c - 1)
                out << '[';
            else if (r == last_pos.r && c == last_pos.c)
                out << ']';
            else
                out << ' ';
        }
        out << '\n';
    }

    out << "TURN: " << b.turn_elapsed << " " << (".XO"[b.turn]) 
        << " - " << Board::state2str(b.state);
    return out;
}


}
