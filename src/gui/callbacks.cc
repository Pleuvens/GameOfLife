#include "callbacks.hh"

#include <iostream>

void error_callback(int error, const char* description)
{
    std::cerr << "Error " << error << ": " << description << '\n';
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    (void)scancode;
    (void)mods;
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
}
