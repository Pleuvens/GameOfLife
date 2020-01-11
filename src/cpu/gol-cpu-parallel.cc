#include <chrono>
#include <csignal>
#include <thread>

#include "map.hh"

namespace
{
    bool running = true;
}

void sig_handler(int)
{
    running = false;
}

int main()
{
    signal(SIGINT, sig_handler);

    Map m{20, 20};

    while (running)
    {
        m.parallel_cpu_compute();
        m.ascii_display();

        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    }

    return 0;
}
