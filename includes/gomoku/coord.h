
#pragma once

#include <iostream>
#include <string>
#include "gomoku/gomoku.h"


namespace gomoku {

class Coord {
public:
    Coord(): r(-1), c(-1) {}
    Coord(int r_, int c_): r(r_), c(c_) {}
    int r, c;

    inline Coord operator+(const Coord& other) const;
    inline Coord operator-(const Coord& other) const;
    inline Coord operator*(int mult) const;
    inline Coord operator-() const;
    inline void operator+=(const Coord& other);
    inline void operator-=(const Coord& other);

    friend std::ostream& operator<<(std::ostream& out, Coord pos);
};


const Coord DELTA[4] = {
    Coord(-1, -1),
    Coord(-1, 0),
    Coord(-1, 1),
    Coord(0, 1),
};

inline bool Inside(Coord pos);
inline bool Inside(int r, int c);

std::string Coord2String(Coord pos);
std::string Coord2String(int r, int c);

}

#include "coord.inl"
