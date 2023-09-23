
#include "gomoku/coord.h"


namespace gomoku {

inline Coord Coord::operator+(const Coord& other) const {
    return Coord(r + other.r, c + other.c);
}

inline Coord Coord::operator-(const Coord& other) const {
    return Coord(r - other.r, c - other.c);
}

inline Coord Coord::operator*(int mult) const {
    return Coord(mult * r, mult * c);
}

inline Coord Coord::operator-() const {
    return Coord(-r, -c);
}

inline void Coord::operator+=(const Coord& other) {
    r += other.r;
    c += other.c;
}

inline void Coord::operator-=(const Coord& other) {
    r -= other.r;
    c -= other.c;
}

inline bool Inside(Coord pos) {
    return (0 <= pos.r && pos.r < SIZE) && (0 <= pos.c && pos.c < SIZE);
}

inline bool Inside(int r, int c) {
    return (0 <= r && r < SIZE) && (0 <= c && c < SIZE);
}

}
