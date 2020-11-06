from OpenGL.GLU import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from math import sqrt, pi, sin, cos, tan
from PIL import Image
from PIL import ImageOps

import os
import numpy as np
import argparse
import json, yaml
import random as rd
import cv2

from generator import startGL, drawCourt, readPose, toRad, toDeg, drawCircle, patternCircle, drawTrack

quadric = gluNewQuadric()
eye = np.zeros(3)
obj = np.zeros(3)
up  = np.zeros(3)
fovy = 0
deg = 0

def rotate(angle):
    global eye, obj, up

    _eye = np.array ([[cos(angle), -sin(angle), 0],
                      [sin(angle),  cos(angle), 0],
                      [0, 0, 1]]) @ eye.reshape(-1,1)

    _up = np.array ([[cos(angle), -sin(angle), 0],
                      [sin(angle),  cos(angle), 0],
                      [0, 0, 1]]) @ up.reshape(-1,1)

    return _eye.reshape(-1), obj, _up.reshape(-1)

def setupCam(pose, f=None):
    global fovy, eye, obj, up
    eye = pose[0]
    obj = pose[1]
    up = pose[2]
    # up = np.array([0,0,1])
    if f == None:
        fovy = fovy
    else:
        fovy = f

def reshape(w, h):
    global fovy
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(fovy, w / h, 0.1, 100000.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def keyboardFunc(c, x, y):
    global eye, obj, up, deg
    # glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    if ord(c.decode('utf-8')) == 27:
        print('exit...')
        os._exit(0)
    elif ord(c.decode('utf-8')) == ord('d') or ord(c.decode('utf-8')) == ord('D'):
        # print("D+")
        deg += toRad(5)
        deg %= 2*pi
        glutPostRedisplay()
    elif ord(c.decode('utf-8')) == ord('a') or ord(c.decode('utf-8')) == ord('A'):
        # print("A-")
        deg -= toRad(5)
        deg %= 2*pi
        glutPostRedisplay()

def sphere(x,y,z):
    global quadric
    glColor3f(0, 1, 1)
    glTranslatef(x,y,z)
    gluSphere(quadric,0.05,32,32)
    glTranslatef(-x,-y,-z)

def drawFunc():
    global eye, obj, up, deg
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    _eye, _obj, _up = rotate(deg)
    if toDeg(deg) > 35 and toDeg(deg) < 144:
        _up = -_up
    print(_eye, _up, sep="\n")
    gluLookAt(_eye[0], _eye[1], _eye[2],  
              _obj[0], _obj[0], _obj[0],  
              _up[0], _up[1], _up[0])

    # Draw badminton court
    drawCourt()

    # Draw pattern
    if cfg['pattern'] == "two_circle":
        # draw pattern for two circle method
        patternCircle(radius=cfg['size'], space=cfg['space'], chance=1.0)
    elif cfg['pattern'] == "chessboard":
        # draw pattern for chessboard method
        patternChess(cube_size=cfg['size'], chess_size=cfg['chess'])

    # Draw track
    start = np.array([-1.5, 2])
    end   = np.array([2, -5.5])
    drawCircle(start, 0.05, [0, 255, 255])
    drawCircle(end, 0.05, [255, 255, 0])
    drawTrack(start, end)

    pred = np.array([[ 1.99745002, -5.49453576,  2.0014967 ],
       [ 1.82350007, -5.12178587,  2.42821695],
       [ 1.64886553, -4.747569  ,  2.81637635],
       [ 1.47251276, -4.36967019,  3.14820552],
       [ 1.29919676, -3.99827877,  3.44273031],
       [ 1.12320515, -3.6211539 ,  3.68706172],
       [ 0.94907774, -3.24802373,  3.89503171],
       [ 0.77255887, -2.86976901,  4.04669892],
       [ 0.59878216, -2.49739033,  4.16221417],
       [ 0.42094726, -2.11631555,  4.22656156],
       [-0.10745599, -0.98402288,  4.15999727],
       [-0.2811105 , -0.61190606,  4.04635707],
       [ 0.24490242, -1.73907661,  4.24939626],
       [ 0.06885927, -1.3618413 ,  4.22859204],
       [-0.45719577, -0.2345805 ,  3.8912387 ],
       [-0.62785622,  0.13112046,  3.69112827],
       [-0.80362013,  0.50775741,  3.44512108],
       [-0.98138612,  0.88868454,  3.15014598],
       [-1.15523681,  1.26122174,  2.81085932],
       [-1.33133548,  1.63857603,  2.43168363],
       [-1.50786414,  2.01685172,  2.00008132]])
    for i in pred:
        sphere(i[0], i[1], i[2])

    print("Deg:", toDeg(deg))
    print()

    glutSwapBuffers()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, help='pose config json file')
    parser.add_argument("--fovy", type=float, default=40, help='fovy of image')
    parser.add_argument("--height", type=int, default=1060, help='height of image')
    parser.add_argument("--width", type=int, default=1920, help='width of image')
    parser.add_argument("--num", type=int, default=1,
                        help='number of samples')
    args = parser.parse_args()

    f = open(args.json)
    cfg = json.load(f)
    f.close()

    focal = args.height / (2*tan(pi*args.fovy/360))
    print("focal", focal)
    print("Generate {} pattern".format(cfg['pattern']))
    pose,_ = next(readPose(cfg, args))
    setupCam(pose, args.fovy)

    startGL(cfg, args)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboardFunc)
    glutDisplayFunc(drawFunc)
    glutMainLoop()