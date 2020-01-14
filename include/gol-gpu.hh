#pragma once

#include <cstdint>

void gol_gpu(char* buffer, int width, int height, int n_iterations = 1000);
void bit_gpu(uint8_t* buffer, int width, int height, int n_iterations = 1000);
