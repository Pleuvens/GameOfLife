#pragma once

#include <vector>

enum CELL {
    DEAD,
    ALIVE
};

class Map {
public:
    Map(size_t height, size_t width);
    ~Map();

    bool IsValidCoord(size_t j, size_t i);
    int NumberOfAliveNeighbours(size_t j, size_t i);
    void Init(); 
    void ASCIIDisplay();
    void Render();
private:
    size_t _height;
    size_t _width;
    size_t _generation;
    std::vector<int> _map;
};
