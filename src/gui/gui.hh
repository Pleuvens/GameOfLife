#pragma once

#include <GLFW/glfw3.h>

GLFWwindow* gui_init(size_t height, size_t width);

void gui_display(GLFWwindow *window, char *bufferi, size_t pitch, size_t height,
    size_t width);

int window_should_close(GLFWwindow *window);

void gui_destroy(GLFWwindow *window);
