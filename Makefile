CC=g++
CXXFLAGS= -g -Wall -Wextra -Werror -pedantic -std=c++17
LDFLAGS= -lncurses
CPUOBJ= map.o main.o

VPATH=cpu/

cpu: $(CPUOBJ)
	$(CC) $(CXXFLAGS) $(CPUOBJ) -o gol-cpu $(LDFLAGS)

cpu-clean:
	$(RM) gol-cpu $(CPUOBJ)
