NULL=

CC=g++

CXXFLAGS= \
	  -Wall \
	  -Wextra \
	  -Werror \
	  -pedantic \
	  -std=c++17 \
	  -g \
	  -fsanitize=address \
	  $(NULL)

LDFLAGS= \
	 -lncurses \
	 -lpthread \
	 -fsanitize=address \
	 $(NULL)

OBJ_CPU= \
	 map.o \
	 $(NULL)

VPATH=src:src/cpu

.PHONY: cpu-clean clean

all: gol-cpu

gol-cpu: $(OBJ_CPU)

gol-cpu-parallel: $(OBJ_CPU)

cpu-clean:
	$(RM) gol-cpu.o gol-cpu-parallel.o $(OBJ_CPU)
	$(RM) gol-cpu gol-cpu-parallel

clean: cpu-clean
