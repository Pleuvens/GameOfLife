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

    inline bool IsValidCoord(size_t j, size_t i) { return j < _height && i < _width; }
    int BasicCPUNumberOfAliveNeighbours(size_t j, size_t i);
    void BasicCPUInit(); 
    void ASCIIDisplay();
    void BasicCPURender();

    void ParallelCPURender();
    void ParallelCPURenderTask(size_t ymin, size_t ymax, size_t xmin, size_t xmax);
private:
    size_t _height;
    size_t _width;
    size_t _generation;
    std::vector<int> _map;
};
