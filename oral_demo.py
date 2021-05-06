import numpy as np
import os
import cv2
import argparse, csv, json
import pandas as pd

from OpenGL.GLU import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from math import sqrt, pi, sin, cos, tan

from h2pose.Hfinder import Hfinder
from h2pose.H2Pose import H2Pose
from trackDetector import trackDetector
from tracker2D import tracker2D
from generator import startGL, readPose, toRad, toDeg, drawCourt, drawNet, drawCircle, patternCircle, drawTrack

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

def reshape(w, h):
    global tf
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(tf.fovy, w / h, 0.1, 100000.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

class Pseudo3d(object):
    """docstring for Pseudo3d"""
    def __init__(self, start_wcs, end_wcs, track2D, H, K, args=None):
        super(Pseudo3d, self).__init__()
        self.args = args
        self.gt = False

        # Opengl param
        self.fovy = 0
        self.eye = np.zeros(3)
        self.obj = np.zeros(3)
        self.up = np.zeros(3)
        self.quadric = gluNewQuadric()

        # Curve param
        self.start_wcs = start_wcs
        self.end_wcs = end_wcs
        self.track2D = track2D

        # Visualizer changable factor
        self._f = 0 # guess focal offset
        self.rad = 0

        # 3D information
        self.H = H
        self.K = K
        self.Rt, self.c2w, self.Cam_wcs, self.Ori_ccs = self.getPose()
        self.P = self.getP()

        # Coord. system changer mtx
        # self.c2w = np.zeros((3,3))

        # OpenGL Init camera pose parameters for gluLookAt() function
        # init_pose, _ = next(readPose(self.cfg, self.args))
        init_pose = (
            np.array([0., -15.5885, 9.]),
            np.array([0., 0., 0.]), 
            np.array([0. , 0.5 , 0.866])
        )
        self.setupCam(init_pose, self.args.fovy)

    def setupCam(self, pose, fovy):
        self.eye = pose[0]
        self.obj = pose[1]
        self.up = pose[2]
        self.fovy = fovy
        # print(self.eye, self.obj, self.up, sep="\n")

    def getGeussK(self):
        return self.K + np.array([[self._f, 0, 0],
                                  [0, self._f, 0],
                                  [0,       0, 0]])

    def getPose(self):
        h2p = H2Pose(self.getGeussK(), self.H)
        return h2p.getRt(), h2p.getC2W(), h2p.getCamera(), h2p.getCenter()

    def getP(self):
        P = self.getGeussK() @ self.Rt
        return P

    def rotate(self, angle):
        rot = np.array([[cos(angle), -sin(angle), 0],
                        [sin(angle),  cos(angle), 0],
                        [0, 0, 1]])
        _eye = rot @ self.eye.reshape(-1,1)

        _up = rot @ self.up.reshape(-1,1)

        return _eye.reshape(-1), self.obj, _up.reshape(-1)

    def updateF(self, silence=False, noise=0):
        self.Rt, self.c2w, self.Cam_wcs, self.Ori_ccs = self.getPose()

        # To evaluate Pose Estimation Noise
        if noise > 0:
            # print(self.c2w)
            if not silence:
                print('@@@@@@@@@@@@@@ Adding noise to Pose @@@@@@@@@@@@@@')
            d = toRad(noise)
            d = (np.random.rand((1))*2-0.5)*d
            Rx = np.array([[1,0,0],[0,cos(d),-sin(d)],[0,sin(d),cos(d)]])
            Ry = np.array([[cos(d),0,sin(d)],[0,1,0],[-sin(d),0,cos(d)]])
            Rz = np.array([[cos(d),-sin(d),0],[sin(d),cos(d),0],[0,0,1]])
            self.c2w = (Rz@Ry@Rx@self.c2w.T).T
            # print(self.c2w)
            # os._exit(0)

        if self.gt:
            if not silence:
                print('@@@@@@@@@@@@@@ Using GT @@@@@@@@@@@@@@')
            self.td = trackDetector(self.getGeussK(), self.H, 
                [0.0, -15.588457268119896, 8.999999999999998], 
                np.array([[1,0,0], 
                    [0,-0.5,0.8660254037844387], 
                    [0,-0.8660254037844387,-0.5]]), silence=silence)
        else:
            if not silence:
                print('@@@@@@@@@@@@@@ Using PD @@@@@@@@@@@@@@')
            self.td = trackDetector(self.getGeussK(), self.H, self.Cam_wcs.reshape(-1), self.c2w, silence=silence)
        self.td.set2Dtrack(self.track2D)
        if self.start_wcs.shape[0] == 2 and self.end_wcs.shape[0] == 2:
            self.td.setShotPoint2d(self.start_wcs, self.end_wcs)
        elif self.start_wcs.shape[0] == 3 and self.end_wcs.shape[0] == 3:
            self.td.setShotPoint3d(self.start_wcs, self.end_wcs)
        else:
            print("wrong anchor format")
            exit(-1)
        self.td.setTrackPlane()
        self.track3D = self.td.get3Dtrack()
        return self.track3D


def keyboardFunc(c, x, y):
    global tf
    if ord(c.decode('utf-8')) == 27:
        print('exit...')
        os._exit(0)
    elif ord(c.decode('utf-8')) == ord('d') or ord(c.decode('utf-8')) == ord('D'):
        tf.rad += toRad(5)
        tf.rad %= 2*pi
        glutPostRedisplay()
    elif ord(c.decode('utf-8')) == ord('a') or ord(c.decode('utf-8')) == ord('A'):
        tf.rad -= toRad(5)
        tf.rad %= 2*pi
        glutPostRedisplay()
    elif ord(c.decode('utf-8')) == ord('w') or ord(c.decode('utf-8')) == ord('w'):
        tf._f += 100
        glutPostRedisplay()
    elif ord(c.decode('utf-8')) == ord('s') or ord(c.decode('utf-8')) == ord('S'):
        tf._f -= 100
        glutPostRedisplay()
    # elif ord(c.decode('utf-8')) == ord(' '):
    #     tf.gt = not tf.gt
    #     glutPostRedisplay()

def sphere(x, y, z, color, size=0.05):
    global tf
    glColor3f(color[0], color[1], color[2])
    glTranslatef(x, y, z)
    gluSphere(tf.quadric, size, 32, 32)
    glTranslatef(-x, -y, -z)

def drawFunc():
    global tf
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    _eye, _obj, _up = tf.rotate(tf.rad)
    gluLookAt(_eye[0], _eye[1], _eye[2],  
              _obj[0], _obj[1], _obj[2],  
              _up[0], _up[1], _up[2])

    # Draw badminton court
    drawCourt()
    drawNet()

    # Draw Pseudo3D track
    pred = tf.updateF(silence=0)
    for i in pred:
        size = 0.05 if tf._f!=0 else 0.05
        if tf.gt:
            sphere(i[0], i[1], i[2], color=[0,1,1], size = size)
        else:
            sphere(i[0], i[1], i[2], color=[1,0,0], size = size)

    # Draw ancher point
    sphere(tf.td.start_wcs[0], tf.td.start_wcs[1], 0, color=[0,0,1], size = size)
    sphere(tf.td.end_wcs[0], tf.td.end_wcs[1], 0, color=[0,0,1], size = size)

    print("Focal length offset:", tf._f, "Deg:", toDeg(tf.rad))
    print()

    glutSwapBuffers()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", type=str, help='csv file output from TrackNetV2')
    parser.add_argument("--fovy", type=float, default=40, help='fovy of visualize window')
    parser.add_argument("--height", type=int, default=1060, help='height of visualize window')
    parser.add_argument("--width", type=int, default=1920, help='width of visualize window')
    args = parser.parse_args()

    # Prepare TrcakNetV2 result
    trackdf = pd.read_csv(args.track)
    hit_idx = trackdf[trackdf['Hit']>0].reset_index()
    print(trackdf)
    print(hit_idx)
    start = hit_idx.loc[[1]]
    end = hit_idx.loc[[2]]
    start_idx = start['Frame'].to_numpy()[0]
    end_idx = end['Frame'].to_numpy()[0]+1
    print('Input:')
    track2D = trackdf[start_idx:end_idx].to_numpy()[:,2:4]
    print(track2D)
    # exit(0)

    # Prepare Homography matrix (image(pixel) -> court(meter))
    # 4 Court corners of CHEN_Long_CHOU_Tien_Chen_Denmark_Open_2019_QuarterFinal recorded in homography_matrix.csv
    cap = cv2.VideoCapture(0)
    ret, img = cap.read()
    cap.release()
    court2D = []
    court3D = [[-3.05, 6.7], [3.05, 6.7], [3.05, -6.7], [-3.05, -6.7]]
    hf = Hfinder(img, court2D=court2D, court3D=court3D)
    Hmtx = hf.getH()
    f = open('oral/Hcam00.json')
    Hmtx = np.array(json.load(f))
    f.close()
    print(Hmtx)

    # Prepare Intrinsic matix of video
    f = open('oral/Kcam00.json')
    Kmtx = np.array(json.load(f)['Kmtx'])
    f.close()
    print(Kmtx)

    # Pseudo3D trajectory transform (2D->3D)
    tf = Pseudo3d(start_wcs=track2D[0] - [0, 10], 
        end_wcs=track2D[-1] - [0, 10], 
        track2D=track2D, 
        args=args, 
        H=Hmtx, 
        K=Kmtx
    )

    # OpenGL visualizer Init
    startGL(args)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboardFunc)
    glutDisplayFunc(drawFunc)
    glutMainLoop()