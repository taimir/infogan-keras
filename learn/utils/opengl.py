from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.GL.framebufferobjects import *
from PIL import Image
import numpy as np


def setup_buffer(width, height):
    TEXTURE_LEVEL = 0

    glutInit()
    # glutInitWindowPosition(0, 0)
    # glutInitWindowSize(300, 400)
    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE | GLUT_DEPTH)
    # create an invisible window
    glutCreateWindow("skeletons")
    glutHideWindow()

    # tutorial: http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-14-render-to-texture/
    # Create a frame buffer object that will hold the frames with rendered data
    frame_buffer = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer)

    # create a texture - effectively the 2D screen image, which will be the actual rendered result
    renderedImage = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, renderedImage)
    # let it be empty (None data) 300 x 400 2D RGB image of type 256-pixel values
    glTexImage2D(GL_TEXTURE_2D, TEXTURE_LEVEL, GL_RGB,
                 width, height, 0, GL_RGB,
                 GL_UNSIGNED_BYTE, None)

    # details ...
    # magnification filter (zoom in), nearest neighbour
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    # minifying filter (zoom out), nearest neighbour
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

    # we now attach the texture to the frame buffer we created,
    # making OpenGL to render to this image (color rendering)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, renderedImage, TEXTURE_LEVEL)
    glDrawBuffers(1, [GL_COLOR_ATTACHMENT0])

    # assert the framebuffer was constructed correctly
    assert glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE

    # let OpenGL draw over the whole buffer
    glViewport(0, 0, width, height)


def get_current_pixels(width, height):
    buf = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    image = Image.frombytes(mode="RGB", size=(width, height),
                            data=buf)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    return np.array(image)


def render_offscreen(verticies, edges, width, height, field_of_view=90,
                     near_clip_dist=0.1, far_clip_dist=100.0,
                     trans_x=0.0,
                     trans_y=-0.5,
                     trans_z=-5.5):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    gluPerspective(field_of_view, (width/height), near_clip_dist, far_clip_dist)
    glTranslatef(trans_x, trans_y, trans_z)

    glBegin(GL_LINES)
    for edge in edges:
        for vertex_index in edge:
            glVertex3fv(verticies[vertex_index])
    glEnd()
    glFlush()

    arr = get_current_pixels(width, height)
    return arr
