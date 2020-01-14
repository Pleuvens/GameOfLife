#include "map.hh"

void gol_cpu(int width, int height, int n_iterations)
{
    Map m(height, width);
    for (int i = 0; i < n_iterations; i++)
        m.basic_cpu_compute();
}

void gol_cpu_parallel(int width, int height, int n_iterations)
{
    Map m(height, width);
    for (int i = 0; i < n_iterations; i++)
        m.parallel_cpu_compute();
}
