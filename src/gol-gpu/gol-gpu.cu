#include <chrono>
#include <iostream>
#include <ncurses.h>
#include <thread>

__attribute__((noinline))
static void _abortError(const char* msg, const char* fname, int line)
{
    cudaError_t err = cudaGetLastError();
    std::clog << fname << ": "
              << "line: " << line << ": " << msg << '\n';
    std::clog << "Error " << cudaGetErrorName(err) << ": "
              << cudaGetErrorString(err) << '\n';
    std::exit(1);
}

#define abortError(msg) _abortError(msg, __FUNCTION__, __LINE__)

__global__ void compute_iteration(char* buffer, char* out_buffer, size_t pitch,
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
    char n_alive = buffer[up_y * pitch + left_x] + buffer[up_y * pitch + x]
        + buffer[up_y * pitch + right_x] + buffer[y * pitch + left_x]
        + buffer[y * pitch + right_x] + buffer[down_y * pitch + left_x]
        + buffer[down_y * pitch + x] + buffer[down_y * pitch + right_x];

    out_buffer[y * pitch + x] =
        n_alive == 3 || (buffer[y * pitch + x] && n_alive == 2);
}

static void display(char* dev_buffer, size_t pitch, int width, int height,
                    int generation)
{
    auto buf = new char[width * height];
    if (cudaMemcpy2D(buf, width * sizeof(char), dev_buffer, pitch,
                     width * sizeof(char), height, cudaMemcpyDeviceToHost))
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
            if (buf[j * width + i] == 0)
                waddch(stdscr, ' ');
            else
                waddch(stdscr, 'O');
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

static void run_compute_iteration(char* dev_buffer, char* out_dev_buffer,
                                  size_t pitch, size_t pitch_out, int width,
                                  int height, int n_iterations)
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
        //display(dev_buffer, pitch, width, height, i);
        //std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    if (cudaPeekAtLastError())
        abortError("Computation error");
}

void gol_gpu(char* buffer, int width, int height, int n_iterations)
{
    cudaError_t rc = cudaSuccess;

    // Allocate device memory
    char* dev_buffer;
    char* out_dev_buffer;
    size_t pitch;
    size_t pitch_out;

    rc = cudaMallocPitch(&dev_buffer, &pitch, width * sizeof(char), height);
    if (rc)
        abortError("Fail buffer allocation");

    rc = cudaMallocPitch(&out_dev_buffer, &pitch_out, width * sizeof(char),
                         height);
    if (rc)
        abortError("Fail output buffer allocation");

    if (cudaMemcpy2D(dev_buffer, pitch, buffer, width * sizeof(char),
                     width * sizeof(char), height, cudaMemcpyHostToDevice))
        abortError("Fail memcpy host to device");

    initscr();
    run_compute_iteration(dev_buffer, out_dev_buffer, pitch, pitch_out, width,
                          height, n_iterations);
    endwin();

    rc = cudaFree(dev_buffer);
    if (rc)
        abortError("Unable to free buffer");

    rc = cudaFree(out_dev_buffer);
    if (rc)
        abortError("Unable to free output buffer");
}
