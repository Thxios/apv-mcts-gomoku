
#include <fstream>
#include <sstream>
#include <exception>
#include <cstring>
#include "gomoku/logger.h"


namespace gomoku {
namespace logger {


Log::Log(const Log& log): header(log.header) {
    flat = header.size * header.size;
    actions_ptr = new int32_t[header.len];
    counts_ptr = new int32_t[header.len * flat];
    std::memcpy(actions_ptr, log.actions_ptr, sizeof(int32_t) * header.len);
    std::memcpy(counts_ptr, log.counts_ptr, sizeof(int32_t) * header.len * flat);
}


Log::Log(Log&& log): header(log.header) {
    flat = header.size * header.size;
    actions_ptr = log.actions_ptr;
    counts_ptr = log.counts_ptr;
    log.header = Log::Header();
    log.actions_ptr = nullptr;
    log.counts_ptr = nullptr;
}


Log& Log::operator=(const Log& log) {
    header = log.header;
    flat = header.size * header.size;
    actions_ptr = new int32_t[header.len];
    counts_ptr = new int32_t[header.len * flat];
    std::memcpy(actions_ptr, log.actions_ptr, sizeof(int32_t) * header.len);
    std::memcpy(counts_ptr, log.counts_ptr, sizeof(int32_t) * header.len * flat);
    return *this;
}


Log& Log::operator=(Log&& log) {
    header = log.header;
    flat = header.size * header.size;
    actions_ptr = log.actions_ptr;
    counts_ptr = log.counts_ptr;
    log.header = Log::Header();
    log.flat = log.header.size * log.header.size;
    log.actions_ptr = nullptr;
    log.counts_ptr = nullptr;
    return *this;
}


Log::~Log() {
    delete[] actions_ptr;
    delete[] counts_ptr;
}


Log::Log(
    int game_len, 
    Board::State result,
    const std::vector<mcts::Action>& actions, 
    const std::vector<std::vector<int>>& counts
) {
    header.len = game_len;
    if (result == Board::State::BLACK_WIN)
        header.result = 1;
    else if (result == Board::State::WHITE_WIN)
        header.result = 2;
    else if (result == Board::State::DRAW)
        header.result = 3;
    else
        header.result = 0;
    
    flat = header.size * header.size;
    actions_ptr = new int32_t[header.len];
    counts_ptr = new int32_t[header.len * flat];
    int offset = 0;
    for (int i = 0; i < game_len; i++, offset += flat) {
        actions_ptr[i] = actions[i];
        for (int action = 0; action < flat; action++) {
            counts_ptr[action + offset] = counts[i][action];
        }
    }
}


void Log::Save(const std::string& path) const {
    std::ofstream out(path, std::ios::out | std::ios::binary);
    if (!out.is_open())
        throw std::runtime_error("log save to " + path + ": cannot open");

    out.write((char*)(&header), sizeof(header));
    out.write((char*)actions_ptr, sizeof(int32_t) * header.len);
    out.write((char*)counts_ptr, sizeof(int32_t) * header.len * flat);

    out.close();
}


}
}
