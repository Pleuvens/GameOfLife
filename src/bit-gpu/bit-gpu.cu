#include <chrono>
#include <fstream>
#include <iostream>
#include <ncurses.h>
#include <thread>
#include <stdint.h>

#define WIDTH (width / 8)
#define BIT8 (1 << 7)

__attribute__((noinline))
void _abortError(const char* msg, const char* fname, int line)
{
    cudaError_t err = cudaGetLastError();
    std::clog << fname << ": " << "line: " << line << ": " << msg << '\n';
    std::clog << "Error " << cudaGetErrorName(err) << ": "
              << cudaGetErrorString(err) << '\n';
    std::exit(1);
}

#define abortError(msg) _abortError(msg, __FUNCTION__, __LINE__)

__global__
void compute_iteration(uint8_t* buffer, uint8_t* out_buffer, size_t pitch,
                       size_t pitch_out, int width, int height)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int left_x = (x - 1 + width) % width;
    int right_x = (x + 1) % width;
    int up_y = (y - 1 + height) % height;
    int down_y = (y + 1) % height;

    int n_alive = buffer[up_y * pitch + left_x / 8] & BIT8 >> left_x % 8
        + buffer[up_y * pitch + x / 8] & BIT8 >> x % 8
        + buffer[up_y * pitch + right_x / 8] & BIT8 >> right_x % 8
        + buffer[y * pitch + left_x / 8] & BIT8 >> left_x % 8
        + buffer[y * pitch + right_x / 8] & BIT8 >> right_x % 8
        + buffer[down_y * pitch + left_x / 8] & BIT8 >> left_x % 8
        + buffer[down_y * pitch + x / 8] & BIT8 >> x % 8
        + buffer[down_y * pitch + right_x / 8] & BIT8 >> right_x % 8;

    if (n_alive == 3 || (buffer[y * pitch + x / 8] && n_alive == 2))
        out_buffer[y * pitch + x / 8] |= BIT8 >> x % 8;
    else
        out_buffer[y * pitch + x / 8] &= ~(BIT8 >> x % 8);
}

void display(uint8_t *dev_buffer, size_t pitch, int width, int height,
             int generation)
{
    auto buf = new uint8_t[WIDTH * height];
    if (cudaMemcpy2D(buf, WIDTH * sizeof(uint8_t), dev_buffer, pitch,
                     WIDTH * sizeof(uint8_t), height, cudaMemcpyDeviceToHost))
        abortError("Fail memcpy device to host");

    wmove(stdscr, 0, 0);
    wprintw(stdscr, "Generation %d:\n", generation);

    for (size_t j = 0; j < height; j++)
    {
        for (size_t i = 0; i < width; i++)
        {
            waddch(stdscr, '=');
            waddch(stdscr, '=');
        }
        waddch(stdscr, '\n');

        waddch(stdscr, '|');
        for (size_t i = 0; i < width; i++)
        {
            if (buf[j * WIDTH + i / 8] & BIT8 >> i % 8)
                waddch(stdscr, 'O');
            else
                waddch(stdscr, ' ');
            waddch(stdscr, '|');
        }
        waddch(stdscr, '\n');
    }

    for (size_t i = 0; i < width; i++)
    {
        waddch(stdscr, '=');
        waddch(stdscr, '=');
    }
    waddch(stdscr, '\n');

    wrefresh(stdscr);
    delete buf;
}

void run_compute_iteration(uint8_t* dev_buffer, uint8_t* out_dev_buffer,
                           size_t pitch, size_t pitch_out, int width,
                           int height, int n_iterations = 1000)
{
    constexpr int block_size = 32;
    int w = std::ceil(1.f * width / block_size);
    int h = std::ceil(1.f * height / block_size);

    dim3 dimGrid(w, h);
    dim3 dimBlock(block_size, block_size);

    for (int i = 0; i < n_iterations; ++i)
    {
        compute_iteration<<<dimGrid, dimBlock>>>(
                dev_buffer, out_dev_buffer, pitch, pitch_out, width, height);
        std::swap(dev_buffer, out_dev_buffer);
        display(dev_buffer, pitch, width, height, i);
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    if (cudaPeekAtLastError())
        abortError("Computation error");
}

void parse_plaintext(const std::string& path, uint8_t *dev_buffer, size_t pitch,
                     int width, int height)
{
    std::ifstream in(path);
    if (!in.good())
        throw std::invalid_argument("file not found");

    auto buf = new uint8_t[WIDTH * height];
    memset(buf, 0, WIDTH * height);
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
                buf[j * WIDTH + i / 8] &= ~(BIT8 >> i % 8);
                break;
            case 'O':
                buf[j * WIDTH + i / 8] |= BIT8 >> i % 8;
                break;
            default:
                throw std::invalid_argument("invalid format");
            }
        }
        ++j;
    }

    if (cudaMemcpy2D(dev_buffer, pitch, buf, WIDTH * sizeof(uint8_t),
                     WIDTH * sizeof(uint8_t), height, cudaMemcpyHostToDevice))
        abortError("Fail memcpy host to device");
    delete buf;
}

void init_random_game(uint8_t *dev_buffer, size_t pitch, int width, int height)
{
    auto buf = new uint8_t[WIDTH * height];

    std::srand(std::time(nullptr));
    for (size_t i = 0; i < height * WIDTH; i++)
        buf[i] = std::rand() % 256;

    if (cudaMemcpy2D(dev_buffer, pitch, buf, WIDTH * sizeof(uint8_t),
                     width * sizeof(uint8_t), height, cudaMemcpyHostToDevice))
        abortError("Fail memcpy host to device");
    delete buf;
}

int main(int argc, char *argv[])
{
    constexpr int width = 50;
    constexpr int height = 20;

    cudaError_t rc = cudaSuccess;

    // Allocate device memory
    uint8_t* dev_buffer;
    uint8_t* out_dev_buffer;
    size_t pitch;
    size_t pitch_out;

    rc = cudaMallocPitch(&dev_buffer, &pitch, WIDTH * sizeof(uint8_t), height);
    if (rc)
        abortError("Fail buffer allocation");

    rc = cudaMemset2D(dev_buffer, pitch, 0, WIDTH, height);
    if (rc)
        abortError("Fail buffer memset");

    rc = cudaMallocPitch(&out_dev_buffer, &pitch_out, WIDTH * sizeof(uint8_t),
                         height);
    if (rc)
        abortError("Fail output buffer allocation");

    if (argc == 2)
        parse_plaintext(argv[1], dev_buffer, pitch, width, height);
    else if (argc < 2)
        init_random_game(dev_buffer, pitch, width, height);
    else
    {
        std::cerr << "Too many arguments\n";
        return 1;
    }

    initscr();
    run_compute_iteration(dev_buffer, out_dev_buffer, pitch, pitch_out, width,
                          height);
    endwin();

    rc = cudaFree(dev_buffer);
    if (rc)
        abortError("Unable to free buffer");

    rc = cudaFree(out_dev_buffer);
    if (rc)
        abortError("Unable to free output buffer");

    return 0;
}
