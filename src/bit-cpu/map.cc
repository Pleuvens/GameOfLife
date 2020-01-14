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
    initscr();

    std::srand(std::time(nullptr));
    for (size_t i = 0; i < map_.size(); i++)
        map_[i] = std::rand() % 256;
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
            else
            {
                if (nb_alive_neighbours == 3)
                    map[(j - ymin) * WIDTH_ + i / 8] |= BIT8 >> i % 8;
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
            if (map_[j * WIDTH_ + i / 8] & BIT8 >> i % 8)
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
