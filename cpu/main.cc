#include <csignal>

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
        m.Render();
        m.ASCIIDisplay();
    }
    return 0;
}
