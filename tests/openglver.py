import glfw
from OpenGL.GL import *

glfw.init()

window = glfw.create_window(500, 500, "OpenGL Ver Check", None, None)

glfw.make_context_current(window)

print("OpenGL version:", glGetString(GL_VERSION).decode())
print("GLSL version:", glGetString(GL_SHADING_LANGUAGE_VERSION).decode())
print("Renderer:", glGetString(GL_RENDERER).decode())
print("Vendor:", glGetString(GL_VENDOR).decode())

glfw.terminate()