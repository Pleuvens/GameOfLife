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

    Map m;

    if (argc == 2)
    {
        m = Map(argv[1]);
    } else {
        m = Map(720, 1280);
    }

    while (running)
    {
        m.basic_cpu_compute();
        m.gl_display();

        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    return 0;
}
