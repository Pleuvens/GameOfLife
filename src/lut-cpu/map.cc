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
    , lut_(LUT_SIZE)
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

    parallel_precompute_lut();
}

Map::Map(size_t height, size_t width)
    : height_{height}
    , width_{width}
    , map_(height_ * WIDTH_)
    , lut_(LUT_SIZE)
{
    initscr();

    std::srand(std::time(nullptr));
    for (size_t i = 0; i < map_.size(); i++)
        map_[i] = std::rand() % 256;

    parallel_precompute_lut();
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

void Map::lut_lookup(size_t ymin, size_t ymax)
{
    for (size_t j = ymin; j < ymax; ++j)
    {
        for (size_t i = 0; i < width_; ++i)
        {
            size_t up_j = (j - 1 + height_) % height_;
            size_t down_j = (j + 1) % height_;
            size_t left_i = (i - 1 + width_) % width_;
            size_t right_i = (i + 1) % width_;

            uint8_t up_left = map_[up_j * width_ + left_i];
            uint8_t up = map_[up_j * width_ + i];
            uint8_t up_right = map_[up_j * width_ + right_i];

            uint8_t curr_left = map_[j * width_ + left_i];
            uint8_t curr = map_[j * width_ + i];
            uint8_t curr_right = map_[j * width_ + right_i];

            uint8_t down_left = map_[down_j * width_ + left_i];
            uint8_t down = map_[down_j * width_ + i];
            uint8_t down_right = map_[down_j * width_ + right_i];

            uint32_t up_val = ((up_left << 16) + (up << 8) + up_right) & 0x1ff80;
            uint32_t curr_val = ((curr_left << 16) + (curr << 8) + curr_right) & 0x1ff80;
            uint32_t down_val = ((down_left << 16) + (down << 8) + down_right) & 0x1ff80;

            uint32_t index = (up_val << 20) + (curr_val << 10) + down_val;

            map_[j * width_ + i] = lut_[index];
        }
    }
}

static inline size_t get_state(size_t x, size_t y, size_t key)
{
    size_t index = y * 10 + x;
    return (key >> ((3 * 10 - 1) - index)) & 1;
}

void Map::precompute_lut(size_t index)
{
    uint8_t state = 0;
    for (size_t cell = 0; cell < 8; ++cell)
    {
        size_t nb_alive = 0;
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 3; ++j)
                nb_alive += get_state(i + cell, j, index);

        size_t center = get_state(cell + 1, 1, index);
        nb_alive -= center;

        if (nb_alive == 3 || (nb_alive == 2 && center == 1))
            state |= (1 << (7 - cell));
    }
    lut_[index] = state;
}

void Map::basic_precompute_lut()
{
    for (size_t i = 0; i < LUT_SIZE / 8; ++i)
        precompute_lut(i);
}

void Map::parallel_precompute_lut()
{
    tbb::parallel_for(tbb::blocked_range<size_t>(0, LUT_SIZE),
                      [&](tbb::blocked_range<size_t> r) {
                          for (size_t i = r.begin(); i < r.end(); ++i)
                              precompute_lut(i);
                      });
}


void Map::basic_cpu_compute()
{
    lut_lookup(0, height_);
    generation_++;
}

void Map::parallel_cpu_compute()
{
    tbb::parallel_for(tbb::blocked_range<size_t>(0, height_),
                      [&](tbb::blocked_range<size_t> r) {
                          for (size_t i = r.begin(); i < r.end(); ++i)
                              lut_lookup(i, i + 1);
                      });

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
