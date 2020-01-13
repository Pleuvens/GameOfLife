#pragma once

#include <GLFW/glfw3.h>
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

    Map() = default;
    Map(const Map&) = default;
    Map& operator=(const Map&) = default;
    Map(Map&&) = default;
    Map& operator=(Map&&) = default;

    ~Map() = default;

    void ascii_display() const;
    
    void gl_init();
    void gl_destroy();
    void gl_display();
    inline int window_should_close() const 
    { return window_ ? glfwWindowShouldClose(window_) : 0; }

    int number_of_alive_neighbours(size_t j, size_t i) const;

    std::vector<Cell> compute_task(size_t ymin, size_t ymax);
    void basic_cpu_compute();
    void parallel_cpu_compute();

private:

    void gl_draw_square() const

    size_t height_ = 0;
    size_t width_ = 0;
    size_t generation_ = 0;
    GLFWwindow* window_ = nullptr;

    std::vector<Cell> map_;
};
