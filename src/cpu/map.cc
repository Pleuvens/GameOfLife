#include "map.hh"

#include <ctime>
#include <ncurses.h>
#include <thread>

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
    int nb = 0;

    for (size_t y = j - 1; y < j + 2; y++)
    {
        for (size_t x = i - 1; x < i + 2; x++)
        {
            if (is_valid_coord(y, x) && y != j && x != i)
            {
                if (map_[y * width_ + x] == Cell::alive)
                    ++nb;
            }
        }
    }

    return nb;
}

void Map::basic_cpu_compute()
{
    for (size_t j = 0; j < height_; j++)
    {
        for (size_t i = 0; i < width_; i++)
        {
            auto nb_alive_neighbours = number_of_alive_neighbours(j, i);
            if (map_[j * width_ + i] == Cell::alive)
            {
                if (nb_alive_neighbours != 2 && nb_alive_neighbours != 3)
                    map_[j * width_ + i] = Cell::dead;
            }
            else
            {
                if (nb_alive_neighbours == 3)
                    map_[j * width_ + i] = Cell::alive;
            }
        }
    }

    generation_++;
}

void Map::parallel_cpu_compute()
{
    auto nb_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> tasks{nb_threads};

    for (size_t i = 0; i < nb_threads; i++)
    {
        tasks[i] = std::thread(
            &Map::parallel_cpu_compute_task, this, i * (height_ / nb_threads),
            (i + 1) * (height_ / nb_threads), i * (width_ / nb_threads),
            (i + 1) * (width_ / nb_threads));
    }

    for (size_t i = 0; i < nb_threads; i++)
    {
        tasks[i].join();
    }

    generation_++;
}

void Map::parallel_cpu_compute_task(size_t ymin, size_t ymax, size_t xmin,
                                    size_t xmax)
{
    for (size_t j = ymin; j < ymax; j++)
    {
        for (size_t i = xmin; i < xmax; i++)
        {
            auto nb_alive_neighbours = number_of_alive_neighbours(j, i);
            if (map_[j * width_ + i] == Cell::alive)
            {
                if (nb_alive_neighbours != 2 && nb_alive_neighbours != 3)
                    map_[j * width_ + i] = Cell::dead;
            }
            else
            {
                if (nb_alive_neighbours == 3)
                    map_[j * width_ + i] = Cell::alive;
            }
        }
    }
}

void Map::ascii_display() const
{
    wmove(stdscr, 0, 0);
    wprintw(stdscr, "Generation %d:\n", generation_);

    for (size_t j = 0; j < height_; j++)
    {
        for (size_t i = 0; i < width_; i++)
        {
            if (map_[j * width_ + i] == Cell::alive)
            {
                waddch(stdscr, '.');
            }
            else
            {
                waddch(stdscr, ' ');
            }
        }
        waddch(stdscr, '\n');
    }

    wrefresh(stdscr);
}
