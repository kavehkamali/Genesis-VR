# gl_version_test.py
from OpenGL import GL
from OpenGL.GLUT import glutInit, glutCreateWindow, glutInitDisplayMode, glutInitWindowSize, GLUT_DOUBLE, GLUT_RGBA

glutInit([])
glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA)
glutInitWindowSize(100, 100)
glutCreateWindow(b"GL Test")

version = GL.glGetString(GL.GL_VERSION)
print("GL_VERSION:", version.decode() if isinstance(version, bytes) else version)
