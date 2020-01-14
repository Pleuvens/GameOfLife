#include "callbacks.hh"
#include "map.hh"

#include <ctime>
#include <fstream>
#include <ncurses.h>
#include <stdexcept>
#include <string>
#include <tbb/parallel_for.h>
#include <thread>

Map::Map(const std::string& path)
    : height_{16}
    , width_{48}
    , map_(height_ * WIDTH_)
{
    std::ifstream in(path);
    if (!in.good())
        throw std::invalid_argument("file not found");

    std::string line;
    size_t j = 0;
    while (std::getline(in, line))
    {
        if (line[0] == '!')
            continue;

        for (size_t i = 0; i < line.length(); i++)
        {
            switch (line[i])
            {
            case '.':
                map_[j * WIDTH_ + i / 8] &= ~(BIT8 >> i % 8);
                break;
            case 'O':
                map_[j * WIDTH_ + i / 8] |= BIT8 >> i % 8;
                break;
            default:
                throw std::invalid_argument("invalid format");
            }
        }
        ++j;
    }
}

Map::Map(size_t height, size_t width)
    : height_{height}
    , width_{width}
    , map_(height_ * WIDTH_)
{
    std::srand(std::time(nullptr));
    for (size_t i = 0; i < map_.size(); i++)
        map_[i] = std::rand() % 256;
}

int Map::number_of_alive_neighbours(size_t j, size_t i) const
{
    size_t up_j = (j - 1 + height_) % height_;
    size_t down_j = (j + 1) % height_;
    size_t left_i = (i - 1 + width_) % width_;
    size_t right_i = (i + 1) % width_;

    int nb = ((map_[up_j * WIDTH_ + left_i / 8] & BIT8 >> left_i % 8) != 0)
        + ((map_[up_j * WIDTH_ + i / 8] & BIT8 >> i % 8) != 0)
        + ((map_[up_j * WIDTH_ + right_i / 8] & BIT8 >> right_i % 8) != 0)
        + ((map_[j * WIDTH_ + left_i / 8] & BIT8 >> left_i % 8) != 0)
        + ((map_[j * WIDTH_ + right_i / 8] & BIT8 >> right_i % 8) != 0)
        + ((map_[down_j * WIDTH_ + left_i / 8] & BIT8 >> left_i % 8) != 0)
        + ((map_[down_j * WIDTH_ + i / 8] & BIT8 >> i % 8) != 0)
        + ((map_[down_j * WIDTH_ + right_i / 8] & BIT8 >> right_i % 8) != 0);

    return nb;
}

std::vector<uint8_t> Map::compute_task(size_t ymin, size_t ymax)
{
    auto begin = map_.begin() + ymin * WIDTH_;
    auto end = map_.begin() + ymax * WIDTH_;
    std::vector<uint8_t> map{begin, end};

    for (size_t j = ymin; j < ymax; j++)
    {
        for (size_t i = 0; i < width_; i++)
        {
            auto nb_alive_neighbours = number_of_alive_neighbours(j, i);
            if (map_[j * WIDTH_ + i / 8] & BIT8 >> i % 8)
            {
                if (nb_alive_neighbours != 2 && nb_alive_neighbours != 3)
                    map[(j - ymin) * WIDTH_ + i / 8] &= ~(BIT8 >> i % 8);
            }
            else if (nb_alive_neighbours == 3)
                map[(j - ymin) * WIDTH_ + i / 8] |= BIT8 >> i % 8;
        }
    }

    return map;
}

void Map::basic_cpu_compute()
{
    map_ = compute_task(0, height_);
    generation_++;
}

void Map::parallel_cpu_compute()
{
    std::vector<std::vector<uint8_t>> maps{height_};
    for (auto& map : maps)
        map = std::vector<uint8_t>(WIDTH_);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, height_),
                      [&](tbb::blocked_range<size_t> r) {
                          for (size_t i = r.begin(); i < r.end(); ++i)
                              maps[i] = compute_task(i, i + 1);
                      });

    for (size_t i = 0; i < height_; ++i)
        for (size_t j = 0; j < WIDTH_; ++j)
            map_[i * WIDTH_ + j] = maps[i][j];

    generation_++;
}

void Map::gl_init()
{
    glfwSetErrorCallback(error_callback);
    if (!glfwInit())
         exit(1);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    glfwWindowHint(GLFW_FLOATING, GLFW_TRUE);
    glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);
    glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_TRUE);
    window_ = glfwCreateWindow(width_, height_, "Game of Life", 
           glfwGetPrimaryMonitor(), NULL);
    if (!window_)
    {
        glfwTerminate();
        exit(1);
    }
    glfwSetKeyCallback(window_, key_callback);
    glfwMakeContextCurrent(window_);
}

void Map::gl_draw_square(size_t y, size_t x) const
{
    //Make sure our transformations don't affect any other
    //transformations in other code
    glPushMatrix();
    //Translate rectangle to its assigned x and y position
    glTranslatef(x, y, 0.0f);
    glBegin(GL_QUADS);
    glColor3f(1, 1, 1);
    //Draw the four corners of the rectangle
    glVertex2f(0, 0);
    glVertex2f(0, 1);
    glVertex2f(1, 1);
    glVertex2f(1, 0);
    glEnd();
    glPopMatrix();
}

void Map::gl_display()
{
    if (window_ == nullptr)
        gl_init();
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glLoadIdentity();
    glOrtho(0, width_, height_, 0, 0, 1);
    for (size_t y = 0; y < height_; y++)
    {
        for (size_t x = 0; x < width_; x++)
        {
            if (map_[y * WIDTH_ + x / 8] & BIT8 >> x % 8)
                gl_draw_square(y, x);
        } 
    }
    glfwSwapBuffers(window_);
    glfwPollEvents();
}

void Map::gl_destroy()
{
    if (window_)
        glfwDestroyWindow(window_);
    glfwTerminate();
}
