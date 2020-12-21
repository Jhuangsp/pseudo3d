import numpy as np
import os
import cv2
import glob
import argparse, json

from OpenGL.GLU import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from math import sqrt, pi, sin, cos, tan

from Hfinder import Hfinder
from trackDetector import trackDetector
from tracker2D import tracker2D
from generator import startGL, readPose, toRad, toDeg, drawCourt, drawCircle, patternCircle, drawTrack

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

class Pseudo3d_Synth(object):
    """docstring for Pseudo3d_Synth"""
    def __init__(self, img, path_j, args, H, K):
        super(Pseudo3d_Synth, self).__init__()
        self.img = img

        jsonf = open(path_j)
        self.cfg = json.load(jsonf)
        jsonf.close()
        self.args = args
        self.gt = True

        # Opengl param
        self.imgshape = self.img.shape[:2]
        self.fovy = 0
        self.eye = np.zeros(3)
        self.obj = np.zeros(3)
        self.up = np.zeros(3)
        self.quadric = gluNewQuadric()
        # self.court2D = [[776, 325], [1141, 325], [1318, 929], [601, 929]]
        # self.court2D = []
        # self.court3D = [[-3.05, 6.7], [3.05, 6.7], [3.05, -6.7], [-3.05, -6.7]]

        # Curve param
        self.start_wcs = np.array([-1.5, 2, 0])
        self.end_wcs = np.array([2, -5.5, 0])
        self.track2D = self.getTrack2D()

        # Visualizer changable factor
        self._f = 0 # guess focal offset
        self.rad = 0

        # 3D information
        self.H = H
        self.K = K
        self.Rt = self.getRt()
        self.P = self.getP()

        # Coord. system changer mtx
        self.c2w = np.zeros((3,3))

        # OpenGL Init camera pose parameters for gluLookAt() function
        init_pose, _ = next(readPose(self.cfg, self.args, 1))
        self.setupCam(init_pose, self.args.fovy)

    def setupCam(self, pose, fovy):
        self.eye = pose[0]
        self.obj = pose[1]
        self.up = pose[2]
        self.fovy = fovy
        # print(self.eye, self.obj, self.up, sep="\n")

    def getGeussK(self):
        # K = np.zeros((3,3))
        # K[0,0] = self.K[0,0] + self._f
        # K[1,1] = self.K[1,1] + self._f
        # K[0,2] = self.K[0,2]
        # K[1,2] = self.K[1,2]
        # K[2,2] = 1
        # print(K)
        return self.K + np.array([[self._f, 0, 0],
                                  [0, self._f, 0],
                                  [0,       0, 0]])

    def getRt(self):
        R = np.zeros((3,3))
        t = np.zeros(3)
        Rt = np.zeros((3,4))

        K_inv = np.linalg.inv(self.getGeussK())
        H_inv = np.linalg.inv(self.H) # H_inv: wcs -> ccs
        multiple = K_inv@H_inv[:,0]
        lamda = 1/np.linalg.norm(multiple, ord=None, axis=None, keepdims=False)
        
        R[:,0] = lamda*(K_inv@H_inv[:,0])
        R[:,1] = lamda*(K_inv@H_inv[:,1])
        R[:,2] = np.cross(R[:,0], R[:,1])
        t = np.array(lamda*(K_inv@H_inv[:,2]))

        Rt[:,:3] = R
        Rt[:,3] = t
        return Rt

    def getP(self):
        P = self.getGeussK() @ self.Rt
        return P

    def getTrack2D(self):
        tr2d = tracker2D(self.img)
        return tr2d.getTrack2D()

    def rotate(self, angle):
        rot = np.array([[cos(angle), -sin(angle), 0],
                        [sin(angle),  cos(angle), 0],
                        [0, 0, 1]])
        _eye = rot @ self.eye.reshape(-1,1)

        _up = rot @ self.up.reshape(-1,1)

        return _eye.reshape(-1), self.obj, _up.reshape(-1)

    def updateF(self):
        self.Rt = self.getRt()

        # Get world pose described in CCS
        print("World pose described in CCS")
        cir_position_ccs = (self.Rt @ [[0],[0],[0],[1]])
        cir_pose_i_ccs = (self.Rt @ [[1],[0],[0],[1]]) - cir_position_ccs
        cir_pose_j_ccs = (self.Rt @ [[0],[1],[0],[1]]) - cir_position_ccs
        cir_pose_k_ccs = (self.Rt @ [[0],[0],[1],[1]]) - cir_position_ccs

        # print("center")
        # print(cir_position_ccs.T)
        # print("pose")
        # print(cir_pose_i_ccs.T)
        # print(cir_pose_j_ccs.T)
        # print(cir_pose_k_ccs.T)
        # print()

        # Get camera pose described in WCS
        print("Camera pose described in WCS")
        self.c2w = np.array(
            [cir_pose_i_ccs.reshape(-1),
             cir_pose_j_ccs.reshape(-1),
             cir_pose_k_ccs.reshape(-1)]
        )
        cam_position_wcs = (self.c2w @ -cir_position_ccs)
        cam_pose_i_wcs = (self.c2w @ [[1],[0],[0]])
        cam_pose_j_wcs = (self.c2w @ [[0],[1],[0]])
        cam_pose_k_wcs = (self.c2w @ [[0],[0],[1]])

        # print("center")
        # print(cam_position_wcs.T)
        # print("pose")
        # print(cam_pose_i_wcs.T)
        # print(cam_pose_j_wcs.T)
        # print(cam_pose_k_wcs.T)
        # print()

        # # Get world pose described in Image coord. system
        # print("World pose described in Image coord. system")
        # x_2d = (P @ [[1],[0],[0],[1]])
        # y_2d = (P @ [[0],[1],[0],[1]])
        # z_2d = (P @ [[0],[0],[1],[1]])

        # x_2d = x_2d / x_2d[2]
        # y_2d = y_2d / y_2d[2]
        # z_2d = z_2d / z_2d[2]

        # print(x_2d.T)
        # print(y_2d.T)
        # print(z_2d.T)
        # print("\n")

        if self.gt:
            td = trackDetector(self.getGeussK(), self.H, 
                [0.0, -15.588457268119896, 8.999999999999998], 
                np.array([[1,0,0], 
                    [0,-0.5,0.8660254037844387], 
                    [0,-0.8660254037844387,-0.5]]))
        else:
            td = trackDetector(self.getGeussK(), self.H, cam_position_wcs.T, self.c2w)
        td.set2Dtrack(self.track2D)
        td.setShotPoint3d(self.start_wcs, self.end_wcs)
        td.setTrackPlane()
        self.track3D = td.get3Dtrack()
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
    elif ord(c.decode('utf-8')) == ord(' '):
        tf.gt = not tf.gt
        glutPostRedisplay()

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
    # if toDeg(tf.rad) > 35 and toDeg(tf.rad) < 144:
    #     _up = -_up
    print(_eye, _up, sep="\n")
    gluLookAt(_eye[0], _eye[1], _eye[2],  
              _obj[0], _obj[1], _obj[2],  
              _up[0], _up[1], _up[2])

    # Draw badminton court
    drawCourt()

    # Draw pattern
    if tf.cfg['pattern'] == "two_circle":
        # draw pattern for two circle method
        patternCircle(radius=tf.cfg['size'], space=tf.cfg['space'], chance=1.0)
    elif tf.cfg['pattern'] == "chessboard":
        # draw pattern for chessboard method
        patternChess(cube_size=tf.cfg['size'], chess_size=tf.cfg['chess'])

    # Draw Ground Truth track & ancher point
    drawCircle(tf.start_wcs[:2], 0.05, [0, 255, 255])
    drawCircle(tf.end_wcs[:2], 0.05, [255, 255, 0])
    drawTrack(tf.start_wcs[:2], tf.end_wcs[:2])

    '''
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
    '''
    # Draw Pseudo3D track
    pred = tf.updateF()
    for i in pred:
        size = 0.05 if tf._f!=0 else 0.03
        if tf.gt:
            sphere(i[0], i[1], i[2], color=[0,1,1], size = size)
        else:
            sphere(i[0], i[1], i[2], color=[0,1,0], size = size)

    print("Focal length offset:", tf._f, "Deg:", toDeg(tf.rad))
    print()

    glutSwapBuffers()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, help='pose config json file')
    parser.add_argument("--fovy", type=float, default=40, help='fovy of image')
    parser.add_argument("--height", type=int, default=1060, help='height of image')
    parser.add_argument("--width", type=int, default=1920, help='width of image')
    # parser.add_argument("--num", type=int, default=1,
    #                     help='number of samples')
    args = parser.parse_args()

    img = cv2.imread("../cameraPose/data/track/track_00000000.png")

    # Prepare Homography matrix (image(pixel) -> court(meter))
    court2D = []
    court3D = [[-3.05, 6.7], [3.05, 6.7], [3.05, -6.7], [-3.05, -6.7]]
    hf = Hfinder(img, court2D=court2D, court3D=court3D)
    Hmtx = hf.getH()

    # Prepare Intrinsic matix of video
    video_h = img.shape[0]
    video_w = img.shape[1]
    video_focal_length = video_h / (2*tan(pi*args.fovy/360))
    Kmtx = np.array(
        [[video_focal_length, 0, video_w/2],
         [0, video_focal_length, video_h/2],
         [0,                  0,         1]]
    )

    tf = Pseudo3d_Synth(img=img, 
        path_j=args.json, 
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