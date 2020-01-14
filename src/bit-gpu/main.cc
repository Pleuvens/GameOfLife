#include "gol-gpu.hh"

#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

#define WIDTH (width / 8)
#define BIT8 (1 << 7)

static void parse_plaintext(const std::string& path, uint8_t* buffer,
                            int width, int)
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
                buffer[j * WIDTH + i / 8] &= ~(BIT8 >> i % 8);
                break;
            case 'O':
                buffer[j * WIDTH + i / 8] |= BIT8 >> i % 8;
                break;
            default:
                throw std::invalid_argument("invalid format");
            }
        }
        ++j;
    }
}

static void init_random_game(uint8_t* buffer, int width, int height)
{
    std::srand(std::time(nullptr));
    for (int i = 0; i < height * WIDTH; i++)
        buffer[i] = std::rand() % 256;
}

int main(int argc, char* argv[])
{
    constexpr int width = 50;
    constexpr int height = 20;

    std::vector<uint8_t> buffer(width * height);
    if (argc == 2)
        parse_plaintext(argv[1], buffer.data(), width, height);
    else if (argc < 2)
        init_random_game(buffer.data(), width, height);
    else
    {
        std::cerr << "Too many arguments\n";
        return 1;
    }

    bit_gpu(buffer.data(), width, height);

    return 0;
}
