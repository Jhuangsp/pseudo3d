import numpy as np
import cv2
import os
import glob
import argparse
import yaml

import EllipsesDetection as ed

class CameraPose(object):
    """docstring for CameraPose"""
    def __init__(self, img, intrinsic, distortion, radius, saveImgs=None):
        super(CameraPose, self).__init__()

        self.intrinsic = intrinsic
        self.distortion = distortion
        self.ellipses_detector = ed.EllipsesDetection(img, intrinsic, distortion, radius=radius, saveImgs=saveImgs)
        if len(self.ellipses_detector.ellipses_) >= 2 and self.ellipses_detector.success:
            # fine tune using plane fitting
            self.centers = self.ellipses_detector.getCircleCenters()
            print(self.centers)
            X, normal = self.fitPlane(np.array(self.centers))
            print(X, normal)
            center = self.centers[0]
            circle_on_x = self.centers[1]
            center[2] = (center[:2]*normal[:2]).sum()+X[2]
            circle_on_x[2] = (circle_on_x[:2]*normal[:2]).sum()+X[2]

            # Support Plane Coordinate System (described in the camera coordinate system)
            print('Circle centers:\n', center)
            self.O = center
            self.i = (circle_on_x - center) / np.linalg.norm(circle_on_x - center)
            # self.k = (self.ellipses_detector.getCircleNormals()[0] + self.ellipses_detector.getCircleNormals()[1]) / 2
            self.k = normal if np.dot(normal, self.ellipses_detector.getCircleNormals()[0]) >= 0 else -normal
            self.k = self.k / np.linalg.norm(self.k)
            print('Circle normal:\n', self.k)
            
            inner = np.dot(self.i, self.k)
            self.i = self.i - inner*self.k
            self.i = self.i / np.linalg.norm(self.i)

            self.j = np.cross(self.k, self.i)
            self.j = self.j / np.linalg.norm(self.j)
            
            # Camera Coordinate System (described in the world coordinate system)
            self.Optical_center = np.array([self.i, self.j, self.k]) @ (np.array([0,0,0]) - self.O).reshape(3,1)
            self.i_c = np.array([self.i, self.j, self.k]) @ np.array([1,0,0]).reshape(3,1)
            self.j_c = np.array([self.i, self.j, self.k]) @ np.array([0,1,0]).reshape(3,1)
            self.k_c = np.array([self.i, self.j, self.k]) @ np.array([0,0,1]).reshape(3,1)

            self.Optical_center = self.Optical_center.reshape(3)
            self.i_c = self.i_c.reshape(3)
            self.j_c = self.j_c.reshape(3)
            self.k_c = self.k_c.reshape(3)

    def fitPlane(self, points):
        # fitting plane function ax + by + c = z
        A = np.zeros(points.shape)
        b = np.zeros((points.shape[0],1))
        A[:,:2] = points[:,:2]
        A[:,2] = 1
        b[:,0] = points[:,2]

        # AX = b, AtAX = Atb
        AtA = A.T @ A
        Atb = A.T @ b
        try:
            X = np.linalg.inv(AtA) @ Atb
        except:
            X = np.linalg.pinv(AtA) @ Atb

        # general form ax + by -z + c = 0
        X = X.reshape(-1)
        normal = np.array([X[0], X[1], -1])
        return X, normal

    def getCameraPosition(self):
        try:
            return self.Optical_center
        except:
            return None

    def getCameraPose(self):
        try:
            return (self.i_c, self.j_c, self.k_c)
        except:
            return None

    def getCirclePosition(self):
        try:
            return self.O
        except:
            return None

    def getCirclePose(self):
        try:
            return (self.i, self.j, self.k)
        except:
            return None

    def ccs2wcs(self, points):
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
            return (np.array([self.i, self.j, self.k]) @ points.T).T
        except:
            return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Camera Pose Estimation')
    parser.add_argument('--infolder', '-i', type=str, required=True,
                        help='path to input image files')
    parser.add_argument('--outfolder', '-o', type=str, required=True,
                        help='path to store output files')
    parser.add_argument('--exten', type=str, default='png',
                        help='input files extension')

    args = parser.parse_args()
    print(args)
    # os._exit(0)

    infolder = args.infolder
    outfolder = args.outfolder
    images = glob.glob(os.path.join(infolder, '*.' + args.exten))
    labels = glob.glob(os.path.join(infolder, 'labels', '*.yml'))
    for image, label in zip(images[:32], labels[:32]):
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
        
        camPoss = CameraPose(img, cmtx_new, None, radius=data['radius'], saveImgs=name)

        cam_position_spcs = camPoss.getCameraPosition()
        cam_pose_spcs = camPoss.getCameraPose()
        cir_position_ccs = camPoss.getCirclePosition()
        cir_pose_ccs = camPoss.getCirclePose()

        if not os.path.isdir(name):
            os.makedirs(name)
        fs = cv2.FileStorage(name + "/pose.yml", cv2.FILE_STORAGE_WRITE)

        if type(cam_position_spcs) == type(None) or type(cam_pose_spcs) == type(None) or type(cir_position_ccs) == type(None) or type(cir_pose_ccs) == type(None):
            fs.write('success', 0)
            print()
            print('Not enough detected circles .... skip')
        else:
            fs.write('success', 1)
            print()
            print('Camera position in SPCS:\n', cam_position_spcs)
            print('Camera pose:')
            print(cam_pose_spcs[0])
            print(cam_pose_spcs[1])
            print(cam_pose_spcs[2])
            print()
            print('Circle position in CCS:\n', cir_position_ccs)
            print('Circle pose:')
            print(cir_pose_ccs[0])
            print(cir_pose_ccs[1])
            print(cir_pose_ccs[2])

            fs.write('circle_center', np.array(camPoss.centers))
            fs.write('cam_position_spcs', cam_position_spcs)
            fs.write('cam_pose_i_spcs', cam_pose_spcs[0])
            fs.write('cam_pose_j_spcs', cam_pose_spcs[1])
            fs.write('cam_pose_k_spcs', cam_pose_spcs[2])
            fs.write('cir_position_ccs', cir_position_ccs)
            fs.write('cir_pose_i_ccs', cir_pose_ccs[0])
            fs.write('cir_pose_j_ccs', cir_pose_ccs[1])
            fs.write('cir_pose_k_ccs', cir_pose_ccs[2])

        fs.release()

# if __name__ == '__main__':
#     img = cv2.imread('Left.png')

#     cv_file = cv2.FileStorage("Left_cmx_dis.xml", cv2.FILE_STORAGE_READ)
#     cmtx = cv_file.getNode("intrinsic").mat()
#     dist = cv_file.getNode("distortion").mat()
#     print("Intrinsic Matrix \n", cmtx)
#     print("Distortion Coefficient \n", dist)
#     cv_file.release()
#     cmtx_new = cmtx.copy()
#     map1, map2 = cv2.initUndistortRectifyMap(cmtx, dist, np.eye(3), cmtx_new, (img.shape[1], img.shape[0]), cv2.CV_16SC2)
#     img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
#     camPoss = CameraPose(img, cmtx_new, dist)

#     print('Camera position:\n', camPoss.getCameraPosition())
#     print('Camera pose:')
#     print(camPoss.getCameraPose()[0])
#     print(camPoss.getCameraPose()[1])
#     print(camPoss.getCameraPose()[2])
#     print('Test transform:')
#     print('before:\n', np.array([[1,0,0],
#                                  [0,1,0],
#                                  [0,0,1],
#                                  [1,1,1]]))
#     print('after:\n', camPoss.world2camera(np.array([[1,0,0],
#                                  [0,1,0],
#                                  [0,0,1],
#                                  [1,1,1]])))