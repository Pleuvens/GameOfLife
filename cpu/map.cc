#include <cstdlib>
#include <ctime>
#include <iostream>
#include <ncurses.h>

#include "map.hh"

Map::Map(size_t height, size_t width)
    : _height(height), _width(width),
    _generation(0), _map(std::vector<int> (height * width))
{
    initscr();
}

Map::~Map() {
    endwin();
}

bool Map::IsValidCoord(size_t j, size_t i) {
    return j < _height && i < _width;
}

int Map::NumberOfAliveNeighbours(size_t j, size_t i) {
    int nb = 0;
    for (size_t y = j - 1; y < j + 2 ; y++) {
        for (size_t x = i - 1 ; x < i + 2 ; x++) {
            if (IsValidCoord(y, x) && y != j && x != i) {
                nb += _map[y * _width + x];
            }
        }
    }
    return nb;
}

void Map::Init() {
    std::srand(std::time(nullptr));
    for (size_t i = 0; i < _height * _width; i++) {
        _map[i] = std::rand() / ((RAND_MAX + 1u) / 2); 
    }
}

void Map::Render() {
    if (!_generation) {
        Init();
        _generation++;
        return;
    }
    for (size_t j = 0; j < _height; j++) {
        for (size_t i = 0; i < _width; i++) {
            auto nbAliveNeighbours = NumberOfAliveNeighbours(j, i);
            if (_map[j * _width + i] == ALIVE) {
                if (nbAliveNeighbours != 2 && nbAliveNeighbours != 3)
                    _map[j * _width + i] = DEAD;
            } else {
                if (nbAliveNeighbours == 3)
                    _map[j * _width + i] = ALIVE;
            }
        }
    }
    _generation++;
}

void Map::ASCIIDisplay() {
    wmove(stdscr, 0, 0);
    for (size_t j = 0; j < _height; j++) {
        for (size_t i = 0; i < _width; i++) {
            if (_map[j * _width + i] == ALIVE) {
                waddch(stdscr, '.');
            } else {
                waddch(stdscr, ' ');
            }
        }
        waddch(stdscr, '\n');
    }
    wrefresh(stdscr);
}
