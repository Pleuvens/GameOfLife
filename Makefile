NULL=

CC=g++
NVCC=nvcc

CXXFLAGS= \
	  -Wall \
	  -Wextra \
	  -Werror \
	  -pedantic \
	  -std=c++17 \
          -O2 \
	  $(NULL)

LDFLAGS= \
	 -lncurses \
	 -lpthread \
	 -ltbb \
	 -fsanitize=address \
	 $(NULL)

OBJ_CPU= \
	 map.o \
	 $(NULL)

OBJ_GPU= \
         gol-gpu.o \
         $(NULL)

VPATH=src:src/cpu:src/gpu

.PHONY: cpu-clean clean

all: gol-cpu gol-cpu-parallel

gol-cpu: $(OBJ_CPU)

gol-cpu-parallel: $(OBJ_CPU)

%.o: %.cu
	$(NVCC) -c -o $@ $^

gol-gpu: $(OBJ_GPU)
	$(NVCC) -o $@ $^ $(LDFLAGS)

cpu-clean:
	$(RM) gol-cpu.o gol-cpu-parallel.o $(OBJ_CPU)
	$(RM) gol-cpu gol-cpu-parallel

gpu-clean:
	$(RM) gol-gpu $(OBJ_GPU)

clean: cpu-clean gpu-clean
