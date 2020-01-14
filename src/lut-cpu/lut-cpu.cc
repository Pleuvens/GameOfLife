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

    Map m(720, 1280);

    if (argc == 2)
        m = Map(argv[1]);

    while (running && !m.window_should_close())
    {
        m.basic_cpu_compute();
        m.gl_display();
    }

    m.gl_destroy();

    return 0;
}
