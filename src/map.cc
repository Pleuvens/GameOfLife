#include <cstdlib>
#include <ctime>
#include <iostream>
#include <ncurses.h>
#include <thread>

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

int Map::BasicCPUNumberOfAliveNeighbours(size_t j, size_t i) {
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

void Map::BasicCPUInit() {
    std::srand(std::time(nullptr));
    for (size_t i = 0; i < _height * _width; i++) {
        _map[i] = std::rand() / ((RAND_MAX + 1u) / 2); 
    }
}

void Map::BasicCPURender() {
    if (!_generation) {
        BasicCPUInit();
        _generation++;
        return;
    }
    for (size_t j = 0; j < _height; j++) {
        for (size_t i = 0; i < _width; i++) {
            auto nbAliveNeighbours = BasicCPUNumberOfAliveNeighbours(j, i);
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

void Map::ParallelCPURender() {
    if (!_generation) {
        BasicCPUInit();
        _generation++;
        return;
    }
    auto nb_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> tasks = std::vector<std::thread>(nb_threads);
    for (size_t i = 0; i < nb_threads; i++) {
        tasks[i] = std::thread(&Map::ParallelCPURenderTask, this,
                i * (_height / nb_threads),
                (i + 1) * (_height / nb_threads),
                i * (_width / nb_threads),
                (i + 1) * (_width / nb_threads));
    }
    for (size_t i = 0; i < nb_threads; i++) {
        tasks[i].join();
    }
    _generation++;
}

void Map::ParallelCPURenderTask(size_t ymin, size_t ymax, size_t xmin, size_t xmax) {
    for (size_t j = ymin; j < ymax; j++) {
        for (size_t i = xmin; i < xmax; i++) {
            auto nbAliveNeighbours = BasicCPUNumberOfAliveNeighbours(j, i);
            if (_map[j * _width + i] == ALIVE) {
                if (nbAliveNeighbours != 2 && nbAliveNeighbours != 3)
                    _map[j * _width + i] = DEAD;
            } else {
                if (nbAliveNeighbours == 3)
                    _map[j * _width + i] = ALIVE;
            }
        }
    }
}

void Map::ASCIIDisplay() {
    wmove(stdscr, 0, 0);
    wprintw(stdscr, "Generation %d:\n", _generation);
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
