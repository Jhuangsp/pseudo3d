import numpy as np
import cv2
import os, csv, glob, yaml
import argparse
import pprint

from tracker2D import tracker2D

pp = pprint.PrettyPrinter(indent=4)

def ccs2wcs(points):
    '''
    points = np.array([[p1_x, p1_y, p1_z],
                       [p1_x, p1_y, p1_z],
                       [p1_x, p1_y, p1_z],
                       ...
                       [pn_x, pn_y, pn_z],])

    return: n converted 3d coordinatation
    '''
    try:
        if points.shape == (3,):
            points = points.reshape(3,1)
        assert len(points.shape) == 2, 'The shape of points should be (n, 3)'
        assert points.shape[1] == 3, 'The shape of points should be (n, 3)'
        return (np.array([[1,0,0], [0,-0.5,0.8660254037844387], [0,-0.8660254037844387,-0.5]]) @ points.T).T
    except:
        return None


class trackDetector(object):
    """docstring for trackDetector"""
    def __init__(self, intrinsic, H, camCenter, c2w, silence=False):
        super(trackDetector, self).__init__()
        self.silence = silence
        self.f = (intrinsic[0,0] + intrinsic[1,1]) / 2
        self.p = (intrinsic[0,2], intrinsic[1,2])
        # src_pts = [(-3.05, 6.7), (3.05, 6.7), (3.05, -6.7), (-3.05, -6.7)]
        self.H = H
        self.o_wcs = (177.5, 480.0, 1)
        # self.campos = campos
        self.camera_wcs = camCenter
        self.c2w = c2w

    def set2Dtrack(self, track2D):
        # Shift origin to principal point
        self.track2D_ccs = track2D - self.p

        # Back project 2D track to the CCS
        self.track2D_ccs = self.track2D_ccs / self.f
        len_t = len(track2D)
        track_d = np.ones((len_t, 1))
        self.track2D_ccs = np.append(self.track2D_ccs, track_d, axis=1)

        # Unified unit (cm -> m)
        # self.track2D_ccs /= 100

        # 2D track described in WCS
        # self.track2D_wcs = self.campos.ccs2wcs(self.track2D_ccs)
        if not self.silence:
            print("2D in CCS")
            pp.pprint(self.track2D_ccs)
        # self.track2D_wcs = ccs2wcs(self.track2D_ccs)
        self.track2D_wcs = (self.c2w @ self.track2D_ccs.T).T
        if not self.silence:
            print("2D in WCS")
            pp.pprint(self.track2D_wcs)

    def setShotPoint2d(self, strat, end):
        # Transform image point to WCS

        self.start_wcs = (self.H @ np.array([[strat[0]],[strat[1]],[1]])).reshape(-1)
        self.end_wcs   = (self.H @ np.array([[end[0]],[end[1]],[1]])).reshape(-1)
        self.start_wcs = (self.start_wcs / self.start_wcs[2])#  - self.o_wcs
        self.end_wcs   = (self.end_wcs / self.end_wcs[2])#  - self.o_wcs
        self.start_wcs[2] = 0
        self.end_wcs[2] = 0

        # Align X-Y axis
        # self.start_wcs = self.start_wcs * [1,-1, 1]
        # self.end_wcs   = self.end_wcs * [1,-1, 1]

        # Unified unit (2cm -> m)
        # self.start_wcs /= 50
        # self.end_wcs   /= 50
        if not self.silence:
            print('Set Start point (wcs)', self.start_wcs)
            print('Set End point (wcs)', self.end_wcs)

    def setShotPoint3d(self, strat, end):
        self.start_wcs = strat
        self.end_wcs   = end
        if not self.silence:
            print('Set Start point (wcs)', self.start_wcs)
            print('Set End point (wcs)', self.end_wcs)

    def setTrackPlane(self):
        # Calculate the Normal vector of the Track Plane
        direction = self.end_wcs - self.start_wcs
        if not self.silence:
            print('Shooting direction (wcs)',direction)
        self.N = np.array([direction[1], -direction[0], 0])
        self.N /= np.linalg.norm(self.N)
        if not self.silence:
            print("N of track plane", self.N)

    def get3Dtrack(self):
        self.molecular = ((self.start_wcs - self.camera_wcs) * self.N).sum()
        self.decimal = (self.track2D_wcs * self.N).sum(axis=1)

        self.t = (self.molecular/self.decimal).reshape(-1,1)
        if not self.silence:
            print("Scale factor from 2D to 3D")
            print(self.t)
        self.track3D_wcs = self.track2D_wcs * self.t + self.camera_wcs
        if not self.silence:
            print("3D in WCS")
            pp.pprint(self.track3D_wcs)
        return self.track3D_wcs


if __name__ == '__main__':
    import CameraPose as cp
    parser = argparse.ArgumentParser(description='Camera Pose Estimation')
    parser.add_argument('--infolder', '-i', type=str, required=True,
                        help='path to input image files')
    parser.add_argument('--outfolder', '-o', type=str, required=True,
                        help='path to store output files')
    parser.add_argument('--exten', type=str, default='png',
                        help='input files extension')

    args = parser.parse_args()
    print(args)

    infolder = args.infolder
    outfolder = args.outfolder
    images = glob.glob(os.path.join(infolder, 'calib_*.' + args.exten))
    tracks = glob.glob(os.path.join(infolder, 'track_*.' + args.exten))
    labels = glob.glob(os.path.join(infolder, 'labels', '*.yml'))
    for image, track, label in zip(images, tracks, labels):
        name = image.split('.')
        name.pop(-1)
        name = '.'.join(name)
        print('------------ Starting', name, '------------')
        img = cv2.imread(image)
        with open(label, 'r') as fy:
            data = yaml.load(fy, Loader=yaml.FullLoader)
        assert data['pattern'] == 'two_circle', 'The label pattern must be \'two_circle\' not {}'.format(data['pattern'])
        cmtx_new = np.zeros((3,3))
        cmtx_new[0,0] = data['intrinsic']['focal']
        cmtx_new[1,1] = data['intrinsic']['focal']
        cmtx_new[2,2] = 1
        cmtx_new[0,2] = data['intrinsic']['w'] // 2
        cmtx_new[1,2] = data['intrinsic']['h'] // 2
        name = outfolder + os.sep + name.split(os.sep)[-1]

        campos = cp.CameraPose(img, cmtx_new, None, radius=data['radius'], saveImgs=name)
        cam_r = campos.getCameraPose()

        tk2D = tracker2D(track)
        track2D = tk2D.getTrack2D()
        print("2D in ICS")
        print(track2D)
        start = np.array([-1.5, 2, 0])
        end   = np.array([2, -5.5, 0])
        # start = np.where(np.all(tra==[0,255,255], axis=-1))
        # start = np.array([start[0].sum()/len(start[0]), start[1].sum()/len(start[1])])
        # end   = np.where(np.all(tra==[255,255,0], axis=-1))
        # end   = np.array([end[0].sum()/len(end[0]), end[1].sum()/len(end[1])])
        # os._exit(0)

        # Homography matrix that transform image coord to bird-eye coord
        H = np.array([[1.937212133548434, 0.9723841192392153, -1056.6377626494277], 
                      [-0.0025994988668978703, 8.481073794245475, -1808.4793238629488], 
                      [-1.732999358826142e-05, 0.0054985503952042665, 1.0]])
        td = trackDetector(cmtx_new, H, 
            [0.0, -15.588457268119896, 8.999999999999998], 
            np.array([[1,0,0], 
                [0,-0.5,0.8660254037844387], 
                [0,-0.8660254037844387,-0.5]]))
        td.set2Dtrack(track2D)
        td.setShotPoint3d(start, end)
        td.setTrackPlane()
        td.get3Dtrack()