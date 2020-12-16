import numpy as np
import os
import cv2
import glob
import argparse, csv, json

from OpenGL.GLU import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from math import sqrt, pi, sin, cos, tan

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

class Pseudo3d(object):
    """docstring for Pseudo3d"""
    def __init__(self, start_wcs, end_wcs, track2D, path_j, args, H=None, K=None, f=None):
        super(Pseudo3d, self).__init__()
        jsonf = open(path_j)
        self.cfg = json.load(jsonf)
        jsonf.close()
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

        # Changable factor
        self.f = f # prediction of focal
        self._f = self.f # ground truth of focal
        self.rad = 0
        # Changable affact
        if K == None and self.f != None:
            self.K = self.getK()
        else:
            self.K = K
        self.H = H
        self.Rt = self.getRt()
        self.P = self.getP()


        # Coord. system changer mtx
        self.c2w = np.zeros((3,3))

        pose, _ = next(readPose(self.cfg, self.args))
        self.setupCam(pose, self.args.fovy)

    def setupCam(self, pose, fovy):
        self.eye = pose[0]
        self.obj = pose[1]
        self.up = pose[2]
        self.fovy = fovy
        print(self.eye, self.obj, self.up, sep="\n")

    def setStartWCS(self):
        pass

    def setEndWCS(self):
        pass
    
    def setTrack2D(self):
        pass

    def setK(self):
        pass

    def setH(self):
        pass

    def getK(self):
        K = np.zeros((3,3))
        K[0,0] = self.f
        K[1,1] = self.f
        K[0,2] = 1280/2
        K[1,2] = 720/2
        K[2,2] = 1
        print(K)
        return K

    def getRt(self):
        R = np.zeros((3,3))
        t = np.zeros(3)
        Rt = np.zeros((3,4))

        K_inv = np.linalg.inv(self.K)
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
        P = self.K @ self.Rt
        return P

    def rotate(self, angle):
        rot = np.array([[cos(angle), -sin(angle), 0],
                        [sin(angle),  cos(angle), 0],
                        [0, 0, 1]])
        _eye = rot @ self.eye.reshape(-1,1)

        _up = rot @ self.up.reshape(-1,1)

        return _eye.reshape(-1), self.obj, _up.reshape(-1)

    def updateF(self):
        self.K = self.getK()
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
            td = trackDetector(self.K, self.H, 
                [0.0, -15.588457268119896, 8.999999999999998], 
                np.array([[1,0,0], 
                    [0,-0.5,0.8660254037844387], 
                    [0,-0.8660254037844387,-0.5]]))
        else:
            td = trackDetector(self.K, self.H, cam_position_wcs.T, self.c2w)
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
        tf.f += 100
        glutPostRedisplay()
    elif ord(c.decode('utf-8')) == ord('s') or ord(c.decode('utf-8')) == ord('S'):
        tf.f -= 100
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
    # if toDeg(tf.rad) > 35 and toDeg(tf.rad) < 144:
    #     _up = -_up
    print(_eye, _up, sep="\n")
    gluLookAt(_eye[0], _eye[1], _eye[2],  
              _obj[0], _obj[1], _obj[2],  
              _up[0], _up[1], _up[2])

    # Draw badminton court
    drawCourt()

    # Draw track
    pred = tf.updateF()
    for i in pred:
        size = 0.05 if tf._f!=tf.f else 0.03
        if tf.gt:
            sphere(i[0], i[1], i[2], color=[0,1,1], size = size)
        else:
            sphere(i[0], i[1], i[2], color=[0,1,0], size = size)

    print("Deg:", toDeg(tf.rad))
    print()

    glutSwapBuffers()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, help='pose config json file')
    parser.add_argument("--tracks", type=str, help='track file output from TrackNetV2')
    parser.add_argument("--fovy", type=float, default=40, help='fovy of image')
    parser.add_argument("--height", type=int, default=1060, help='height of image')
    parser.add_argument("--width", type=int, default=1920, help='width of image')
    parser.add_argument("--num", type=int, default=1,
                        help='number of samples')
    args = parser.parse_args()

    rallys = glob.glob(os.path.join(args.tracks, 'set*.csv'))
    rallys.sort()

    shots = []
    shots_land = []
    shots_frame = []
    for rally in rallys:
        with open(rally, newline='') as ral_f:
            ral = csv.DictReader(ral_f)
            served = False
            for f in ral:
                if not served:
                    if f['Hit'] == 'True':
                        tmp = []
                        tmp.append([float(f['X']), float(f['Y'])])
                        shots_land.append([float(f['LandX']), float(f['LandY']), 0.0])
                        shots_frame.append(float(f['Frame']))
                        served = True
                    else: continue
                else:
                    if f['Hit'] == 'True':
                        shots.append(tmp)
                        tmp = []
                        tmp.append([float(f['X']), float(f['Y'])])
                        shots_land.append([float(f['LandX']), float(f['LandY']), 0.0])
                        shots_frame.append(float(f['Frame']))
                    else:
                        if f['Visibility'] == '1':
                            tmp.append([float(f['X']), float(f['Y'])])
                            shots_frame.append(float(f['Frame']))

    print(shots[0])

    Hmtx = np.array([[1.9523045866733588, 0.9801671951476165, -1065.7854887959163], 
                     [0.006811070912976109, 8.546177245698969, -1833.8532474305387], 
                     [-1.0178084275277081e-05, 0.005529519693682916, 1.0]])

    now = 2
    tf = Pseudo3d(start_wcs=np.array(shots_land[now]), end_wcs=np.array(shots_land[now+1]), track2D=np.array(shots[now]), path_j=args.json, args=args, H=Hmtx, K=None, f=600)

    startGL(tf.cfg, tf.args)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboardFunc)
    glutDisplayFunc(drawFunc)
    glutMainLoop()