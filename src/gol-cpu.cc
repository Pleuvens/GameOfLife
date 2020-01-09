#include <csignal>
#include <thread>
#include <chrono>

#include "map.hh"

bool running = true;

void sigHandler(int signum) {
    (void)signum;
    running = false;
}

int main() {
    signal(SIGINT, sigHandler);
    Map m = Map(20, 20);
    while (running) {
        m.BasicCPURender();
        m.ASCIIDisplay();
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    }
    return 0;
}
