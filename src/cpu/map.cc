#include "map.hh"

#include <ctime>
#include <fstream>
#include <ncurses.h>
#include <stdexcept>
#include <string>
#include <thread>

Map::Map(const std::string& path)
    : height_{20}
    , width_{50}
    , map_(height_ * width_)
{
    initscr();

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
    initscr();

    std::srand(std::time(nullptr));
    for (size_t i = 0; i < height_ * width_; i++)
    {
        map_[i] = Cell(std::rand() / ((RAND_MAX + 1u) / 2));
    }
}

Map::~Map()
{
    endwin();
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

void Map::compute_task(size_t ymin, size_t ymax, size_t xmin, size_t xmax)
{
    auto map = map_;

    for (size_t j = ymin; j < ymax; j++)
    {
        for (size_t i = xmin; i < xmax; i++)
        {
            auto nb_alive_neighbours = number_of_alive_neighbours(j, i);
            if (map_[j * width_ + i] == Cell::alive)
            {
                if (nb_alive_neighbours != 2 && nb_alive_neighbours != 3)
                    map[j * width_ + i] = Cell::dead;
            }
            else
            {
                if (nb_alive_neighbours == 3)
                    map[j * width_ + i] = Cell::alive;
            }
        }
    }

    for (size_t j = ymin; j < ymax; j++)
        for (size_t i = xmin; i < xmax; i++)
            map_[j * ymax + i] = map[j * ymax + i];

    map_ = map;
}

void Map::basic_cpu_compute()
{
    compute_task(0, height_, 0, width_);
    generation_++;
}

void Map::parallel_cpu_compute()
{
    auto nb_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> tasks{nb_threads};

    for (size_t i = 0; i < nb_threads; i++)
    {
        tasks[i] = std::thread(
            &Map::compute_task, this, i * (height_ / nb_threads),
            (i + 1) * (height_ / nb_threads), i * (width_ / nb_threads),
            (i + 1) * (width_ / nb_threads));
    }

    for (size_t i = 0; i < nb_threads; i++)
    {
        tasks[i].join();
    }

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
