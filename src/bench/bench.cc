#include <benchmark/benchmark.h>
#include <vector>

#include "gol-cpu.hh"
#include "gol-gpu.hh"

constexpr int width = 4800;
constexpr int height = 3200;
constexpr int n_iterations = 1000;

void BM_Computing_gpu(benchmark::State& st)
{
    std::vector<char> data(height * width);

    for (auto _ : st)
        gol_gpu(data.data(), width, height, n_iterations);

    st.counters["frame_rate"] =
        benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

void BM_Computing_cpu(benchmark::State& st)
{
    std::vector<char> data(height * width);

    for (auto _ : st)
        gol_cpu(width, height, n_iterations);

    st.counters["frame_rate"] =
        benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

void BM_Computing_cpu_parallel(benchmark::State& st)
{
    std::vector<char> data(height * width);

    for (auto _ : st)
        gol_cpu_parallel(width, height, n_iterations);

    st.counters["frame_rate"] =
        benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

BENCHMARK(BM_Computing_gpu)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK(BM_Computing_cpu)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK(BM_Computing_cpu_parallel)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_MAIN();
