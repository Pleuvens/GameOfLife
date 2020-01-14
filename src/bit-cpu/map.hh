#pragma once

#include <stdint.h>
#include <string>
#include <vector>

#define WIDTH_ (width_ / 8)
#define BIT8 (1 << 7)

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

    void ascii_display() const;

    int number_of_alive_neighbours(size_t j, size_t i) const;

    std::vector<uint8_t> compute_task(size_t ymin, size_t ymax);
    void basic_cpu_compute();
    void parallel_cpu_compute();

private:
    size_t height_ = 0;
    size_t width_ = 0;
    size_t generation_ = 0;

    std::vector<uint8_t> map_;
};
