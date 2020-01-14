#include <iostream>
#include <ncurses.h>
#include <cstdint>
#include <thread>

#define WIDTH (width / 8)
#define BIT8 (1 << 7)

__attribute__((noinline)) void _abortError(const char* msg, const char* fname,
                                           int line)
{
    cudaError_t err = cudaGetLastError();
    std::clog << fname << ": "
              << "line: " << line << ": " << msg << '\n';
    std::clog << "Error " << cudaGetErrorName(err) << ": "
              << cudaGetErrorString(err) << '\n';
    std::exit(1);
}

#define abortError(msg) _abortError(msg, __FUNCTION__, __LINE__)

__global__ void compute_iteration(uint8_t* buffer, uint8_t* out_buffer,
                                  size_t pitch, size_t pitch_out, int width,
                                  int height)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int up_y = (y - 1 + height) % height;
    int down_y = (y + 1) % height;
    for (int real_x = x * 8; real_x < x * 8 + 8; ++real_x)
    {
        int left_x = (real_x - 1 + width) % width;
        int right_x = (real_x + 1) % width;

        int n_alive = ((buffer[up_y * pitch + left_x / 8] & BIT8 >> left_x % 8) != 0)
            + ((buffer[up_y * pitch + real_x / 8] & BIT8 >> real_x % 8) != 0)
            + ((buffer[up_y * pitch + right_x / 8] & BIT8 >> right_x % 8) != 0)
            + ((buffer[y * pitch + left_x / 8] & BIT8 >> left_x % 8) != 0)
            + ((buffer[y * pitch + right_x / 8] & BIT8 >> right_x % 8) != 0)
            + ((buffer[down_y * pitch + left_x / 8] & BIT8 >> left_x % 8) != 0)
            + ((buffer[down_y * pitch + real_x / 8] & BIT8 >> real_x % 8) != 0)
            + ((buffer[down_y * pitch + right_x / 8] & BIT8 >> right_x % 8) != 0);

        if (n_alive == 3
            || (buffer[y * pitch + real_x / 8] && n_alive == 2))
            out_buffer[y * pitch + real_x / 8] |= BIT8 >> real_x % 8;
        else
            out_buffer[y * pitch + real_x / 8] &= ~(BIT8 >> real_x % 8);
    }
}

void display(uint8_t* dev_buffer, size_t pitch, int width, int height,
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
    int w = std::ceil(1.f * WIDTH / block_size);
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

void bit_gpu(uint8_t* buffer, int width, int height, int n_iterations)
{
    cudaError_t rc = cudaSuccess;

    // Allocate device memory
    uint8_t* dev_buffer;
    uint8_t* out_dev_buffer;
    size_t pitch;
    size_t pitch_out;

    rc = cudaMallocPitch(&dev_buffer, &pitch, WIDTH * sizeof(uint8_t), height);
    if (rc)
        abortError("Fail buffer allocation");

    rc = cudaMallocPitch(&out_dev_buffer, &pitch_out, WIDTH * sizeof(uint8_t),
                         height);
    if (rc)
        abortError("Fail output buffer allocation");

    if (cudaMemcpy2D(dev_buffer, pitch, buffer, WIDTH * sizeof(uint8_t),
                     WIDTH * sizeof(uint8_t), height, cudaMemcpyHostToDevice))
        abortError("Fail memcpy host to device");

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
}
