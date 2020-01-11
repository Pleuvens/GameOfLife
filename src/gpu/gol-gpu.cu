#include <chrono>
#include <thread>
#include <iostream>
#include <ncurses.h>

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
void compute_iteration(char* buffer, char* out_buffer, size_t pitch,
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
    char n_alive = buffer[up_y * pitch + left_x] + buffer[up_y * pitch + x] +
        buffer[up_y * pitch + right_x] + buffer[y * pitch + left_x] +
        buffer[y * pitch + right_x] + buffer[down_y * pitch + left_x] +
        buffer[down_y * pitch + x] + buffer[down_y * pitch + right_x];

    out_buffer[y * pitch + x] = n_alive == 3 || (buffer[y * pitch + x]
                                                 && n_alive == 2);
}

void display(char *dev_buffer, size_t pitch, int width, int height)
{
    auto map = new char[width * height];
    size_t pitch2 = width * sizeof(char);

    cudaMemcpy2D(map, pitch2, dev_buffer, pitch, width, height,
               cudaMemcpyDeviceToHost);

    wmove(stdscr, 0, 0);
    for (size_t y = 0; y < height; ++y)
    {
        for (size_t x = 0; x < width; ++x)
        {
            if (map[y * width + x])
                waddch(stdscr, '.');
            else
                waddch(stdscr, 'x');
        }
        waddch(stdscr, '\n');
    }
    wrefresh(stdscr);

    delete map;
}

void run_compute_iteration(char* dev_buffer, char* out_dev_buffer,
                           size_t pitch, size_t pitch_out, int width,
                           int height, int n_iterations = 1000)
{
    constexpr int block_size = 32;
    int w = std::ceil(1.f * width / block_size);
    int h = std::ceil(1.f * height / block_size);

    dim3 dimGrid(w, h);
    dim3 dimBlock(block_size, block_size);

    auto map = new char[width * height];
    memset(map, 0, width * height);
    size_t pitch2 = width * sizeof(char);
    // Glider hardcoded
    map[1 * height + 2] = 1;
    map[2 * height + 3] = 1;
    map[3 * height + 1] = 1;
    map[3 * height + 2] = 1;
    map[3 * height + 3] = 1;

    cudaMemcpy2D(dev_buffer, pitch, map, pitch2, width * sizeof(char), height,
                 cudaMemcpyHostToDevice);
    delete map;


    for (int i = 0; i < n_iterations; ++i)
    {
        compute_iteration<<<dimGrid, dimBlock>>>(
                dev_buffer, out_dev_buffer, pitch, pitch_out, width, height);
        std::swap(dev_buffer, out_dev_buffer);
        display(dev_buffer, pitch, width, height);
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    if (cudaPeekAtLastError())
        abortError("Computation error");
}

int main(int argc, char *argv[])
{
    // FIXME: Parse Game of life plaintext format file

    //constexpr int width = 1024;
    //constexpr int height = 768;
    constexpr int width = 10;
    constexpr int height = 10;

    cudaError_t rc = cudaSuccess;

    // Allocate device memory
    char* dev_buffer;
    char* out_dev_buffer;
    size_t pitch;
    size_t pitch_out;

    rc = cudaMallocPitch(&dev_buffer, &pitch, width * sizeof(char), height);
    if (rc)
        abortError("Fail buffer allocation");

    rc = cudaMemset2D(dev_buffer, pitch, 0, width, height);
    if (rc)
        abortError("Fail buffer memset");

    rc = cudaMallocPitch(&out_dev_buffer, &pitch_out, width * sizeof(char),
                         height);
    if (rc)
        abortError("Fail output buffer allocation");

    initscr();

    run_compute_iteration(dev_buffer, out_dev_buffer, pitch, pitch_out, width,
                          height);

    rc = cudaFree(dev_buffer);
    if (rc)
        abortError("Unable to free buffer");

    rc = cudaFree(out_dev_buffer);
    if (rc)
        abortError("Unable to free output buffer");

    endwin();

    return 0;
}
