#pragma once

#include <string>
#include <vector>

enum class Cell
{
    dead,
    alive
};

class Map
{
public:
    explicit Map(const std::string& path);
    Map(size_t height, size_t width);

    Map(const Map&) = default;
    Map& operator=(const Map&) = default;
    Map(Map&&) = default;
    Map& operator=(Map&&) = default;

    ~Map();

    bool is_valid_coord(size_t j, size_t i) const
    {
        return j < height_ && i < width_;
    }

    void ascii_display() const;

    int number_of_alive_neighbours(size_t j, size_t i) const;

    std::vector<Cell> compute_task(size_t ymin, size_t ymax);
    void basic_cpu_compute();
    void parallel_cpu_compute();

private:
    size_t height_ = 0;
    size_t width_ = 0;
    size_t generation_ = 0;

    std::vector<Cell> map_;
};
