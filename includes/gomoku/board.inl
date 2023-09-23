
#include "gomoku/board.h"


namespace gomoku {

inline Color Board::GetColor(Coord pos) const {
    return ((board[EMPTY][pos.r][pos.c]) ? 
        EMPTY : (board[BLACK][pos.r][pos.c] ? BLACK : WHITE));
}

inline Color Board::GetColor(mcts::Action action) const {
    return GetColor(Action2Coord(action));
}

inline Color Board::GetColor(int r, int c) const {
    return ((board[EMPTY][r][c]) ? 
        EMPTY : (board[BLACK][r][c] ? BLACK : WHITE));
}

inline const int8_t* Board::GetDataPtr() const {
    return (int8_t*)board;
}

inline Color Board::GetTurn() const {
    return turn;
}

inline int Board::GetTurnElapsed() const {
    return turn_elapsed;
}

inline Board::State Board::GetState() const {
    return state;
}


inline mcts::Action Coord2Action(Coord coord) {
    return coord.r * SIZE + coord.c;
}

inline mcts::Action Coord2Action(int r, int c) {
    return r * SIZE + c;
}

inline Coord Action2Coord(mcts::Action action) {
    return Coord(action / SIZE, action % SIZE);
}


}

