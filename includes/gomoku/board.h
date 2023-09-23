
#pragma once

#include <cstdint>
#include <memory>
#include <iostream>
#include <string>
#include "mcts/state.h"
#include "gomoku/gomoku.h"
#include "gomoku/coord.h"


namespace gomoku {

class Board : public mcts::StateBase {
public:
    enum class State {
        ONGOING,
        BLACK_WIN,
        WHITE_WIN,
        DRAW
    };

public:
    Board();
    Board(const Board& board) = default;
    Board(Board&& board) = default;
    Board& operator=(const Board& board) = default;
    Board& operator=(Board&& board) = default;
    virtual ~Board() = default;

    virtual std::unique_ptr<mcts::StateBase> GetCopy() const;
    virtual void Play(mcts::Action action);
    virtual bool Terminated() const;
    virtual mcts::Reward TerminalReward() const;

    void Play(Coord pos);
    void Play(int r, int c);
    
    inline const int8_t* GetDataPtr() const;
    inline Color GetTurn() const;
    inline int GetTurnElapsed() const;
    inline State GetState() const;

    friend std::ostream& operator<<(std::ostream& out, Board& board);
    
    const static int DEPTH = 3;

private:
    void Reset();
    State CheckState(mcts::Action action) const;
    bool FiveInRow(Color color, Coord pos, Coord delta) const;

    inline Color GetColor(Coord pos) const;
    inline Color GetColor(mcts::Action action) const;
    inline Color GetColor(int r, int c) const;

    int8_t board[DEPTH][SIZE][SIZE];
    Color turn;
    State state;
    int turn_elapsed;
    mcts::Action last_action;
};


inline mcts::Action Coord2Action(Coord coord);
inline mcts::Action Coord2Action(int r, int c);
inline Coord Action2Coord(mcts::Action action);


}

#include "board.inl"
