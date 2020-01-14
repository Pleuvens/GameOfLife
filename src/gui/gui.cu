#include "callbacks.hh"
#include "gui.hh"

#include <iostream>

__attribute__((noinline))
static void _guiAbortError(const char* msg, const char* fname, int line)
{
    cudaError_t err = cudaGetLastError();
    std::clog << fname << ": " << "line: " << line << ": " << msg << '\n';
    std::clog << "Error " << cudaGetErrorName(err) << ": "
              << cudaGetErrorString(err) << '\n';
    std::exit(1);
}

#define guiAbortError(msg) _guiAbortError(msg, __FUNCTION__, __LINE__)

GLFWwindow* gui_init(size_t height, size_t width)
{
    glfwSetErrorCallback(error_callback);
    if (!glfwInit())
         exit(1);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    glfwWindowHint(GLFW_FLOATING, GLFW_TRUE);
    glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);
    glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_TRUE);
    GLFWwindow *window = glfwCreateWindow(width, height, "Game of Life", 
           glfwGetPrimaryMonitor(), NULL);
    if (!window)
    {
        glfwTerminate();
        exit(1);
    }
    glfwSetKeyCallback(window, key_callback);
    glfwMakeContextCurrent(window);
    return window;
}

void gui_draw_square(size_t y, size_t x)
{
    //Make sure our transformations don't affect any other
    //transformations in other code
    glPushMatrix();
    //Translate rectangle to its assigned x and y position
    glTranslatef(x, y, 0.0f);
    glBegin(GL_QUADS);
    glColor3f(1, 1, 1);
    //Draw the four corners of the rectangle
    glVertex2f(0, 0);
    glVertex2f(0, 1);
    glVertex2f(1, 1);
    glVertex2f(1, 0);
    glEnd();
    glPopMatrix();
}

void gui_display(GLFWwindow *window, char *buffer, size_t pitch, size_t height,
        size_t width)
{
    auto buf = new char[width * height];
    if (cudaMemcpy2D(buf, width * sizeof(char), buffer, pitch,
                     width * sizeof(char), height, cudaMemcpyDeviceToHost))
        guiAbortError("Fail memcpy device to host");

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glLoadIdentity();
    glOrtho(0, width, height, 0, 0, 1);
    for (size_t y = 0; y < height; y++)
    {
        for (size_t x = 0; x < width; x++)
        {
            if (buf[y * width + x] == 1)
                gui_draw_square(y, x);
        }
    }
    glfwSwapBuffers(window);
    glfwPollEvents();
    delete buf;
}

int window_should_close(GLFWwindow *window)
{
    return window ? glfwWindowShouldClose(window) : 0;
}

void gui_destroy(GLFWwindow *window)
{
    if (window)
        glfwDestroyWindow(window);
    glfwTerminate();
}
