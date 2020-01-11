#pragma once

#include <vector>

enum class Cell
{
    dead,
    alive
};

class Map
{
public:
    Map(size_t height, size_t width);

    Map(const Map&) = delete;
    Map& operator=(const Map&) = delete;
    Map(Map&&) = delete;
    Map& operator=(Map&&) = delete;

    ~Map();

    bool is_valid_coord(size_t j, size_t i) const
    {
        return j < height_ && i < width_;
    }

    void ascii_display() const;

    int number_of_alive_neighbours(size_t j, size_t i) const;

    void basic_cpu_compute();
    void parallel_cpu_compute();
    void parallel_cpu_compute_task(size_t ymin, size_t ymax, size_t xmin,
                                   size_t xmax);

private:
    size_t height_;
    size_t width_;
    size_t generation_ = 0;

    std::vector<Cell> map_;
};
