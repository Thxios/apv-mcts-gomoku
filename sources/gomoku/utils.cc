
#include <fmt/format.h>
#include "gomoku/coord.h"


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

}

