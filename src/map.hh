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
    int BasicCPUNumberOfAliveNeighbours(size_t j, size_t i);
    void BasicCPUInit(); 
    void ASCIIDisplay();
    void BasicCPURender();
private:
    size_t _height;
    size_t _width;
    size_t _generation;
    std::vector<int> _map;
};
