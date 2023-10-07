
#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include "gomoku/board.h"


namespace gomoku {
namespace logger {


class Log {
    struct Header {
        int32_t size = SIZE;
        int32_t depth = Board::DEPTH;
        int32_t len = 0;
        int32_t result = 0;
    };

public:
    Log() = default;
    Log(int game_len, 
        Board::State result,
        const std::vector<mcts::Action>& actions, 
        const std::vector<std::vector<int>>& counts);
    Log(const Log& log);
    Log(Log&& log);
    Log& operator=(const Log& log);
    Log& operator=(Log&& log);
    ~Log();

    void Save(const std::string& path) const;

private:
    Header header;
    int flat = SIZE * SIZE;
    int32_t *actions_ptr = nullptr;
    int32_t *counts_ptr = nullptr;
};


}
}
