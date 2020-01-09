CC=g++
CXXFLAGS= -g -Wall -Wextra -Werror -pedantic -std=c++17
LDFLAGS= -lncurses -lpthread
CPUOBJ= map.o

VPATH=src/

cpu: $(CPUOBJ) gol-cpu.o
	$(CC) $(CXXFLAGS) $(CPUOBJ) gol-cpu.o -o gol-cpu $(LDFLAGS)

cpu-parallel: $(CPUOBJ) gol-cpu-parallel.o
	$(CC) $(CXXFLAGS) $(CPUOBJ) gol-cpu-parallel.o -o gol-cpu-parallel $(LDFLAGS)


cpu-clean:
	$(RM) gol-cpu $(CPUOBJ)
