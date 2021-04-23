import numpy as np
import cv2
import os, argparse
import pprint, json

from OpenGL.GLU import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from math import sqrt, pi, sin, cos, tan

from h2pose.H2Pose import H2Pose
from h2pose.Hfinder import Hfinder
from tracker2D import tracker2D
from generator import startGL, readPose, toRad, toDeg, drawCourt, drawCircle, patternCircle, drawTrack

pp = pprint.PrettyPrinter(indent=4)

class multiCamTriang(object):
    """docstring for multiCamTriang"""
    def __init__(self, track2Ds, poses, eye, Ks, silence=False):
        super(multiCamTriang, self).__init__()
        self.track2Ds = track2Ds             # shape:(num_cam, num_frame, xy(2)) 2D track from TrackNetV2
        self.poses = poses                   # shape:(num_cam, c2w(3, 3)) transform matrix from ccs to wcs
        self.eye = eye                       # shape:(num_cam, 1, xyz(3)) camera position in wcs
        self.Ks = Ks                         # shape:(num_cam, K(3,3)) intrinsic matrix
        self.f = (Ks[:,0,0] + Ks[:,1,1]) / 2 # shape:(num_cam) focal length
        self.p = Ks[:,0:2,2]                 # shape:(num_cam, xy(2)) principal point
        self.silence = silence

        self.num_cam, self.num_frame, _ = self.track2Ds.shape
        self.backProject()
        self.getApprox3D()

    def backProject(self):
        # Back project the 2d points of all frames to the 3d ray in world coordinate system

        # Shift origin to principal point
        self.track2Ds_ccs = self.track2Ds - self.p[:,None,:]

        # Back project 2D track to the CCS
        self.track2Ds_ccs = self.track2Ds_ccs / self.f[:,None,None]
        track_d = np.ones((self.num_cam, self.num_frame, 1))
        self.track2Ds_ccs = np.concatenate((self.track2Ds_ccs, track_d), axis=2)
        if not self.silence:
            print("2D in CCS")
            pp.pprint(self.track2Ds_ccs)

        # 2D track described in WCS
        self.track2D_wcs = self.poses @ np.transpose(self.track2Ds_ccs, (0,2,1)) # shape:(num_cam, 3, num_frame)
        self.track2D_wcs = np.transpose(self.track2D_wcs, (0,2,1)) # shape:(num_cam, num_frame, 3)
        self.track2D_wcs = self.track2D_wcs / np.linalg.norm(self.track2D_wcs, axis=2)[:,:,None]
        if not self.silence:
            print("2D in WCS")
            pp.pprint(self.track2D_wcs)

    def getApprox3D(self):
        # Calculate the approximate solution of the ball postition by the least square method
        # n-lines intersection == 2n-planes intersection

        planeA = np.copy(self.track2D_wcs)
        planeA[:,:,0] = 0
        planeA[:,:,1] = -self.track2D_wcs[:,:,2]
        planeA[:,:,2] = self.track2D_wcs[:,:,1]

        # check norm == 0
        planeA_tmp = np.copy(self.track2D_wcs)
        planeA_tmp[:,:,0] = -self.track2D_wcs[:,:,2]
        planeA_tmp[:,:,1] = 0
        planeA_tmp[:,:,2] = self.track2D_wcs[:,:,0]
        mask = np.linalg.norm(planeA, axis=2)==0
        planeA[mask] = planeA_tmp[mask]

        # # check norm == 0
        # planeA_tmp = np.copy(self.track2D_wcs)
        # planeA_tmp[:,:,0] = -self.track2D_wcs[:,:,1]
        # planeA_tmp[:,:,1] = self.track2D_wcs[:,:,0]
        # planeA_tmp[:,:,2] = 0
        # mask = np.linalg.norm(planeA, axis=2)==0
        # planeA[mask] = planeA_tmp[mask]

        planeB = np.cross(self.track2D_wcs, planeA)

        Amtx = np.concatenate((planeA, planeB), axis=0) # shape:(2num_cam, num_frame, 3)
        b = np.concatenate((self.eye*planeA, self.eye*planeB), axis=0).sum(-1)[:,:,None] # shape:(2num_cam, num_frame, 1)

        Amtx = np.transpose(Amtx, (1,0,2)) # shape:(num_frame, 2num_cam, 3)
        b = np.transpose(b, (1,0,2)) # shape:(num_frame, 2num_cam, 1)

        left = np.transpose(Amtx, (0,2,1)) @ Amtx # shape:(num_frame, 3, 3)
        right = np.transpose(Amtx, (0,2,1)) @ b # shape:(num_frame, 3, 1)

        self.track3D = np.linalg.pinv(left) @ right # shape:(num_frame, 3, 1)
        self.track3D = self.track3D.reshape(-1,3)
        # print(self.track3D)
        '''
        [[-1.52680479,  2.06202114,  2.04934252],
        [-1.34739384,  1.67499119,  2.49134506],
        [-1.16905542,  1.29068153,  2.88204759],
        [-0.9924781,   0.91104616,  3.22891526],
        [-0.81207975,  0.5231891,   3.53126069],
        [-0.63397124,  0.14283066,  3.78343654],
        [-0.46045112, -0.23837983,  3.99178557],
        [-0.28176591, -0.62200495,  4.15098372],
        [-0.10533186, -1.00410534,  4.26820641],
        [ 0.07375752, -1.38452603,  4.337159  ],
        [ 0.25253153, -1.77248105,  4.36025826],
        [ 0.43151949, -2.15530492,  4.33638467],
        [ 0.61271096, -2.53986812,  4.26825079],
        [ 0.78954231, -2.91919479,  4.14984721],
        [ 0.96996572, -3.30069187,  3.9908667 ],
        [ 1.14671596, -3.68684962,  3.78039425],
        [ 1.32658648, -4.07093151,  3.52737657],
        [ 1.5030584,  -4.45345251,  3.22569203],
        [ 1.68413826, -4.8356843,   2.88023936],
        [ 1.86211489, -5.22080128,  2.48178244],
        [ 2.0396313,  -5.60385936,  2.04277793]]
        '''

if __name__ == '__main__':

    Kmtx = np.array([[1456.16303, 0, 1920/2],
                     [0, 1456.16303, 1060/2],
                     [0,          0,      1]])

    # # Prepare Homography matrix (image(pixel) -> court(meter))
    # poses = []
    # eye = []
    # Ks = []
    # track2Ds = []
    # img_list = ["synthetic_track/stereo/camera1/track_00000000.png",
    #             "synthetic_track/stereo/camera2/track_00000000.png"]
    # for name in img_list:
    #     # Read image
    #     print(name)
    #     img = cv2.imread(name)
        
    #     # Get Pose
    #     pf = poseFinder(img, Kmtx, pad=[0,0,0,0], downScale=False)

    #     # Append data
    #     eye.append(pf.getCamera().T)
    #     poses.append(pf.getC2W())
    #     Ks.append(Kmtx)

    #     # Find Track
    #     # If you using real data please remove following lines
    #     # and create your own track2Ds
    #     tr2d = tracker2D(img)
    #     if "camera2" in name:
    #         track2Ds.append(sorted(tr2d.getTrack2D(), key=lambda x:x[0], reverse=True))
    #     else:
    #         track2Ds.append(sorted(tr2d.getTrack2D(), key=lambda x:x[0]))

    # # Start triangulation
    # mct = multiCamTriang(
    #     track2Ds=np.array(track2Ds), 
    #     poses=np.array(poses), 
    #     eye=np.array(eye), 
    #     Ks=np.array(Ks)
    # )

    # ========================================================

    # H matrix
    f = open(os.path.join("synthetic_track/stereo_thesis/camera1", 'Hmtx.json'))
    Hmtx_cam1 = np.array(json.load(f))
    f.close()

    # H matrix
    f = open(os.path.join("synthetic_track/stereo_thesis/camera2", 'Hmtx.json'))
    Hmtx_cam2 = np.array(json.load(f))
    f.close()

    # 2d data
    track2ds_cam1 = np.load(os.path.join("synthetic_track/stereo_thesis/camera1", 'data2d.npy')) # (1000, 15, 2)

    # 2d data
    track2ds_cam2 = np.load(os.path.join("synthetic_track/stereo_thesis/camera2", 'data2d.npy')) # (1000, 15, 2)

    # get C2W
    h2p1 = H2Pose(Kmtx, Hmtx_cam1)
    h2p2 = H2Pose(Kmtx, Hmtx_cam2)
    print(h2p1.getC2W())
    print(h2p1.getCamera().T)
    print(h2p2.getC2W())
    print(h2p2.getCamera().T)

    # Start triangulation
    for track2d_cam1, track2d_cam2 in zip(track2ds_cam1, track2ds_cam2):
        mct = multiCamTriang(
            track2Ds=np.array([track2d_cam1, track2d_cam2]), 
            poses=np.array([h2p1.getC2W(), h2p2.getC2W()]), 
            eye=np.array([h2p1.getCamera().T, h2p2.getCamera().T]),
            Ks=np.array([Kmtx, Kmtx])
        )
        print(mct.track3D)
        break