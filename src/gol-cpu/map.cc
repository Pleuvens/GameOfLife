#include "map.hh"

#include <ctime>
#include <fstream>
#include <iostream>
#include <ncurses.h>
#include <stdexcept>
#include <string>
#include <tbb/parallel_for.h>
#include <thread>

void error_callback(int error, const char* description)
{
    std::cerr << "Error " << error << ": " << description << '\n';
}

Map::Map(const std::string& path)
    : height_{16}
    , width_{48}
    , map_(height_ * width_)
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
                map_[j * width_ + i] = Cell::dead;
                break;
            case 'O':
                map_[j * width_ + i] = Cell::alive;
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
    , map_(height * width)
{
    std::srand(std::time(nullptr));
    for (size_t i = 0; i < height_ * width_; i++)
    {
        map_[i] = Cell(std::rand() / ((RAND_MAX + 1u) / 2));
    }
}

Map::~Map()
{
    if (window_)
        glfwDestroyWindow(window_);
    glfwTerminate();
}

int Map::number_of_alive_neighbours(size_t j, size_t i) const
{
    size_t up_j = (j - 1 + height_) % height_;
    size_t down_j = (j + 1) % height_;
    size_t left_i = (i - 1 + width_) % width_;
    size_t right_i = (i + 1) % width_;

    int nb = map_[up_j * width_ + left_i] == Cell::alive;
    nb += map_[up_j * width_ + i] == Cell::alive;
    nb += map_[up_j * width_ + right_i] == Cell::alive;
    nb += map_[j * width_ + left_i] == Cell::alive;
    nb += map_[j * width_ + right_i] == Cell::alive;
    nb += map_[down_j * width_ + left_i] == Cell::alive;
    nb += map_[down_j * width_ + i] == Cell::alive;
    nb += map_[down_j * width_ + right_i] == Cell::alive;

    return nb;
}

std::vector<Cell> Map::compute_task(size_t ymin, size_t ymax)
{
    auto begin = map_.begin() + ymin * width_;
    auto end = map_.begin() + ymax * width_;
    std::vector<Cell> map{begin, end};

    for (size_t j = ymin; j < ymax; j++)
    {
        for (size_t i = 0; i < width_; i++)
        {
            auto nb_alive_neighbours = number_of_alive_neighbours(j, i);
            if (map_[j * width_ + i] == Cell::alive)
            {
                if (nb_alive_neighbours != 2 && nb_alive_neighbours != 3)
                    map[(j - ymin) * width_ + i] = Cell::dead;
            }
            else
            {
                if (nb_alive_neighbours == 3)
                    map[(j - ymin) * width_ + i] = Cell::alive;
            }
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
    std::vector<std::vector<Cell>> maps{height_};
    for (auto& map : maps)
        map = std::vector<Cell>(width_);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, height_),
                      [&](tbb::blocked_range<size_t> r) {
                          for (size_t i = r.begin(); i < r.end(); ++i)
                              maps[i] = compute_task(i, i + 1);
                      });

    for (size_t i = 0; i < height_; ++i)
        for (size_t j = 0; j < width_; ++j)
            map_[i * width_ + j] = maps[i][j];

    generation_++;
}

void Map::ascii_display() const
{
    wmove(stdscr, 0, 0);
    wprintw(stdscr, "Generation %d:\n", generation_);

    for (size_t j = 0; j < height_; j++)
    {
        for (size_t i = 0; i < width_; i++)
        {
            waddch(stdscr, '=');
            waddch(stdscr, '=');
        }
        waddch(stdscr, '\n');

        waddch(stdscr, '|');
        for (size_t i = 0; i < width_; i++)
        {
            if (map_[j * width_ + i] == Cell::alive)
            {
                waddch(stdscr, 'O');
            }
            else
            {
                waddch(stdscr, ' ');
            }
            waddch(stdscr, '|');
        }
        waddch(stdscr, '\n');
    }

    for (size_t i = 0; i < width_; i++)
    {
        waddch(stdscr, '=');
        waddch(stdscr, '=');
    }
    waddch(stdscr, '\n');

    wrefresh(stdscr);
}

void Map::gl_display()
{
    if (window_ == nullptr)
    {
        glfwSetErrorCallback(error_callback);
        if (!glfwInit())
            exit(1);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        glfwWindowHint(GLFW_FLOATING, GLFW_TRUE);
        glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);
        glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_TRUE);
        window_ = glfwCreateWindow(width_, height_, "Game of Life", NULL, NULL);
        if (!window_)
        {
            glfwTerminate();
            exit(1);
        }
        glfwMakeContextCurrent(window_);
    }
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glLoadIdentity();
    glOrtho(0, width_, height_, 0, 0, 1);
    glBegin(GL_LINES);
    for (size_t y = 0; y < height_; y++)
    {
        for (size_t x = 0; x < width_; x++)
        {
            if (map_[y * width_ + x] == Cell::alive)
            {
                glColor3f(1.0f, 1.0f, 1.0f);
                glVertex2f(x, y);
                glVertex2f(x + 1, y);
            }
        } 
    }
    glEnd();
    glfwSwapBuffers(window_);
}
