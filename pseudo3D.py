import numpy as np
import os
import cv2
import argparse, csv, json

from OpenGL.GLU import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from math import sqrt, pi, sin, cos, tan

from Hfinder import Hfinder
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
        self.Rt = self.getRt()
        self.P = self.getP()

        # Coord. system changer mtx
        self.c2w = np.zeros((3,3))

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
            self.td = trackDetector(self.getGeussK(), self.H, 
                [0.0, -15.588457268119896, 8.999999999999998], 
                np.array([[1,0,0], 
                    [0,-0.5,0.8660254037844387], 
                    [0,-0.8660254037844387,-0.5]]))
        else:
            self.td = trackDetector(self.getGeussK(), self.H, cam_position_wcs.T, self.c2w)
        self.td.set2Dtrack(self.track2D)
        self.td.setShotPoint2d(self.start_wcs, self.end_wcs)
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
    pred = tf.updateF()
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
    shots = []
    shots_start = []
    shots_end = []
    shots_frame = []
    with open(args.track, newline='') as ral_f:
        ral = csv.DictReader(ral_f)
        served = False
        for f in ral:
            if not served:
                if f['Hit'] == 'True':
                    tmp = []
                    tmp.append([float(f['X']), float(f['Y'])])
                    try:
                        shots_start.append([float(f['EndX']), float(f['EndY'])])
                        shots_end.append([float(f['EndX']), float(f['EndY'])])
                    except:
                        shots_start.append([-1,-1])
                        shots_end.append([-1,-1])
                    shots_frame.append(float(f['Frame']))
                    served = True
                else: continue
            else:
                if f['Hit'] == 'True':
                    shots.append(tmp)
                    tmp = []
                    tmp.append([float(f['X']), float(f['Y'])])
                    shots_start.append([float(f['StartX']), float(f['StartY'])])
                    shots_end.append([float(f['EndX']), float(f['EndY'])])
                    shots_frame.append(float(f['Frame']))
                else:
                    if f['Visibility'] == '1':
                        tmp.append([float(f['X']), float(f['Y'])])
                        shots_frame.append(float(f['Frame']))

    # Prepare Homography matrix (image(pixel) -> court(meter))
    # 4 Court corners of CHEN_Long_CHOU_Tien_Chen_Denmark_Open_2019_QuarterFinal recorded in homography_matrix.csv
    # court2D = [[448, 256.6], [818.2, 256.2], [981.2, 646.4], [278.8, 649]] 
    # 4 Court corners of CHEN_Long_CHOU_Tien_Chen_World_Tour_Finals_Group_Stage
    # court2D = [[415.4,358.2], [863.4,358.2], [1018.4,672], [262.8,672]]
    # 4 Court corners of CHEN_Yufei_TAI_Tzu_Ying_Malaysia_Masters_2020_Finals
    # court2D = [[416.2,422.4], [879.6,422.2], [945.4,723.8], [225.8,718]]
    # 4 Court corners of CHEN_Yufei_TAI_Tzu_Ying_World_Tour_Finals_Finals
    # court2D = [[340.8,354.4], [823.2,353.2], [837.6,677.2], [120,675.6]]
    # 4 Court corners of CHOU_Tien_Chen_Anthony_Sinisuka_GINTING_World_Tour_Finals_Group_Stage
    court2D = [[338.4,329.4], [889.8,332.2], [952.4,705.6], [133.8,695.2]]
    court3D = [[-3.05, 6.7], [3.05, 6.7], [3.05, -6.7], [-3.05, -6.7]]
    hf = Hfinder(None, court2D=court2D, court3D=court3D)
    Hmtx = hf.getH()

    # Prepare Intrinsic matix of video
    video_h = 720
    video_w = 1280
    video_focal_length = 2000
    Kmtx = np.array(
        [[video_focal_length, 0, video_w/2],
         [0, video_focal_length, video_h/2],
         [0,                  0,         1]]
    )

    # Pseudo3D trajectory transform (2D->3D)
    now = 5
    tf = Pseudo3d(start_wcs=np.array(shots_start[now]), 
        end_wcs=np.array(shots_end[now]), 
        track2D=np.array(shots[now]), 
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