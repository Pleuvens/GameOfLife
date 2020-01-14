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

int main(int argc, char* argv[])
{
    signal(SIGINT, sig_handler);

    Map m(20, 20);

    if (argc == 2)
        m = Map(argv[1]);

    while (running)
    {
        m.basic_cpu_compute();
        m.ascii_display();

        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    return 0;
}
