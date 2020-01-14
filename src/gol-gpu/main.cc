#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

#include "gol-gpu.hh"

static void parse_plaintext(const std::string& path, char* buffer, int width,
                            int)
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
                break;
            case 'O':
                buffer[j * width + i] = 1;
                break;
            default:
                throw std::invalid_argument("invalid format");
            }
        }
        ++j;
    }
}

static void init_random_game(char* buffer, int width, int height)
{
    std::srand(std::time(nullptr));
    for (int i = 0; i < height * width; i++)
        buffer[i] = std::rand() / ((RAND_MAX + 1u) / 2);
}

int main(int argc, char* argv[])
{
    constexpr int width = 50;
    constexpr int height = 20;

    std::vector<char> buffer(width * height);
    if (argc == 2)
        parse_plaintext(argv[1], buffer.data(), width, height);
    else if (argc < 2)
        init_random_game(buffer.data(), width, height);
    else
    {
        std::cerr << "Too many arguments\n";
        return 1;
    }

    gol_gpu(buffer.data(), width, height);

    return 0;
}
