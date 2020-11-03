import cv2
import numpy as np
from math import pi, sin, cos, sqrt
import time
import os
import glob

from Ellipse import Ellipse


class EllipsesDetection():
    """docstring for EllipsesDetection"""

    def __init__(self, img, intrinsic, distortion, radius=1, saveImgs=None):
        super(EllipsesDetection, self).__init__()

        self.imgC = img
        self.saveImgs = saveImgs
        self.intrinsic = intrinsic
        self.distortion = distortion
        self.radius = radius
        self.ellipses_ = []
        self.ori_h, self.ori_w, _ = img.shape
        self.success = False

        # open output dir
        if self.saveImgs != None:
            self.draw = np.zeros((img.shape[0]*2, img.shape[1]*2, 3), 'uint8')
            if not os.path.isdir(self.saveImgs):
                os.makedirs(self.saveImgs)
            else:
                for f in glob.glob(os.path.join(self.saveImgs, '*')):
                    os.remove(f)

        self.findEllipses()
        self.estimatePose()

    def findEllipses(self):
        t1 = time.time()
        self.edge_detection()
        t2 = time.time()
        print('Done Edge detection {} (sec)'.format(t2 - t1))
        self.contour_detection()
        t1 = time.time()
        print('Done Contour detection {} (sec)'.format(t1 - t2))
        self.hough_ellipses()
        t2 = time.time()

        if self.ellipses_[0].info == "first" and self.ellipses_[1].info == "second":
            self.success = True
        elif self.ellipses_[0].info == "second" and self.ellipses_[1].info == "first":
            self.ellipses_[0], self.ellipses_[1] = self.ellipses_[1], self.ellipses_[0]
            self.success = True
        else:
            for e in self.ellipses_:
                print(e.info)
            print("order weird")
            self.success = False

        print('Done Ellipses detection {} (sec)'.format(t2 - t1))

    def edge_detection(self):
        gray = cv2.cvtColor(self.imgC, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0).astype(int)
        self.edges = cv2.Canny(gray.astype('uint8'), 85, 85*3, apertureSize=3)

    def contour_detection(self):
        self.contours, self.hierarchy = cv2.findContours(self.edges,
                                                         cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    def saveVisual(self, cnt, center, c_int, dectected, name):

        def drawCenter(img, cnt, c_int, color):
            cv2.drawContours(img, [cnt], -1, color, 1)
            cv2.circle(img, c_int, 5, (0, 0, 255), 1)
            return img

        # saving visualized image
        tmp_img = np.copy(self.imgC)
        tmp_cnt = np.zeros_like(tmp_img)

        # draw contour
        tmp_img = drawCenter(tmp_img, cnt, c_int, (0, 0, 255))
        tmp_cnt = drawCenter(tmp_cnt, cnt, c_int, (255, 255, 255))
        self.draw[:self.ori_h, :self.ori_w] = tmp_img
        self.draw[self.ori_h:, :self.ori_w] = tmp_cnt

        # draw max distance map
        self.draw[:self.ori_h, self.ori_w:] = cv2.cvtColor(
            ((self.max_distance/np.max(self.max_distance))*255).astype('uint8'), cv2.COLOR_GRAY2BGR)

        # draw detected ellipse
        if dectected:
            tmp_img = np.copy(self.imgC)
            tmp_img = cv2.ellipse(
                tmp_img, (center, (self.a*2, self.winner[0]*2), self.winner[1]), (0, 255, 0), 1)
            tmp_cnt = cv2.ellipse(
                tmp_cnt, (center, (self.a*2, self.winner[0]*2), self.winner[1]), (0, 255, 0), 1)
            self.draw[self.ori_h:, :self.ori_w] = tmp_cnt
            self.draw[self.ori_h:, self.ori_w:] = tmp_img
        else:
            self.draw[self.ori_h:, self.ori_w:] = [255, 255, 255]

        # draw hough space
        b, degree = np.where(self.hough_space > self.top_votes - len(cnt)/10)
        self.hough_space = ((self.hough_space/self.hough_space.max()) * 255).astype('uint8') # normalize
        self.hough_space = cv2.cvtColor(self.hough_space, cv2.COLOR_GRAY2BGR) # to color
        self.hough_space[b, degree] = [0, 0, 255] # highlight winners
        self.hough_space[int(self.winner[0]), int(self.winner[1])] = [0, 255, 0] # highlight average winner
        self.hough_space = cv2.resize(
            self.hough_space.astype('uint8'), 
            (self.hough_space.shape[1]*6, self.hough_space.shape[0]*6), 
            interpolation=cv2.INTER_NEAREST)
        self.draw[-self.hough_space.shape[0]:, self.ori_w:self.ori_w +
                  self.hough_space.shape[1]] = self.hough_space
        cv2.imwrite(os.path.join(self.saveImgs, name), self.draw)

        self.draw = np.zeros((self.ori_h*2, self.ori_w*2, 3), 'uint8')
        print()
        pass

    def hough_ellipses(self):

        def sameEllipses(e1, e2):
            same = True
            same = same and e1.lengthOfAxis[0] - e2.lengthOfAxis[0] < 10
            same = same and e1.lengthOfAxis[1] - e2.lengthOfAxis[1] < 10
            same = same and sqrt(
                (e1.center[0]-e2.center[0])**2+(e1.center[1]-e2.center[1])**2) < 10
            large = max(e1.theta, e2.theta)
            small = min(e1.theta, e2.theta)
            same = same and min(large - small, (small + pi) - large) < 5
            return same

        def getPartial(data, center, size, shape):
            rx = [max(center[0]-size, 0), min(center[0]+size, shape[0])]
            ry = [max(center[1]-size, 0), min(center[1]+size, shape[1])]
            data_p = np.copy(data[ry[0]:ry[1], rx[0]:rx[1]])
            return rx, ry, data_p

        def maxDistance(src, dst):
            distance_patch = np.sqrt((src[0]-cnt[:,0])**2 + (src[1]-cnt[:,1])**2)
            max_distance_patch = distance_patch.max(axis=2)
            return max_distance_patch

        def vote(shape, center, voters):
            hough_space = np.zeros(shape, int)
            a = shape[0] - 1

            # for each point in contour votes for ellipse angle and minor axis
            for v in voters:
                for DEGREE in range(shape[1]):  # 0~180 degree
                    RAD = DEGREE * pi / 180
                    XX = ((v[0]-center[0])*cos(RAD) +
                          (v[1]-center[1])*sin(RAD))**2 / (a**2)
                    YY = (-(v[0]-center[0])*sin(RAD) +
                          (v[1]-center[1])*cos(RAD))**2
                    B = sqrt(abs(YY/(1-XX)))+1
                    if B > 0 and B <= a:
                        B = round(B) % int(a+1)
                        hough_space[B, DEGREE] = hough_space[B, DEGREE]+1

            # get the top votes (select group of elects rather than just one)
            top_votes = hough_space.max()
            b, degree = np.where(hough_space > top_votes - len(cnt)/10)
            top_votes = hough_space[b, degree]
            top_votes = top_votes.sum()/len(top_votes)

            # get average b
            avg_b = b.sum()/len(b)

            # get average theta
            avg_w = degree.sum()/len(degree)
            # deal with candidates that near to 0 and 180 degree
            if avg_w - degree[0] > 20:
                degree_t = np.copy(degree)
                degree_t[degree_t < 90] = degree_t[degree_t < 90] + 180
                avg_w = degree_t.sum()/len(degree_t)
                avg_w = avg_w % 180

            return (avg_b, avg_w), top_votes, hough_space
        
        def buildEllipse(winner, center, top_votes, num_voters, major_axis, imgSize):
            minor_axis = winner[0]
            theta = winner[1]
            theshold = 0.1
            ratio = 4
            # votes must high enough  and the major/minor rate cant be too much
            if top_votes <= num_voters*theshold or major_axis/minor_axis > ratio:  # 0.15 / 2.5
                print('[No result] {}/{} < {}, major/minor {} > {}'.format(
                        top_votes, num_voters, theshold, 
                        major_axis/minor_axis, ratio))
                e = None
                success = False
            else:
                print('[Detected] minor = {}, theta={}'.format(minor_axis, theta*pi/180))
                e = Ellipse((major_axis, minor_axis), center, theta, imgSize)
                success = True
            return success, e

        for i, cnt in enumerate(self.contours):

            # Only pick long enough contour
            # if len(cnt) > (self.ori_h+self.ori_w) / 100 and len(cnt) < (self.ori_h+self.ori_w) * 6 / 7:
            if len(cnt) > 12 and len(cnt) < (self.ori_h+self.ori_w) * 1 / 4:
                print('contours #{}'.format(i))

                # calculate likely center of contour (used to construct partial searching field)
                cnt = cnt.reshape(-1, 2)
                x_g, y_g = np.round(cnt.sum(0)/len(cnt)).astype(int)

                # setup center and a with geometric constrain
                x = np.arange(self.ori_w)
                y = np.arange(self.ori_h)
                xv, yv = np.meshgrid(x, y)
                self.max_distance = np.zeros((self.ori_h, self.ori_w), int)
                max_st = time.time()

                patch_size = 100
                # rx, ry define the partial searching field
                # crop partial searching field from full searching field
                rx, ry, xv_p = getPartial(xv, (x_g, y_g), patch_size, (self.ori_w, self.ori_h))
                rx, ry, yv_p = getPartial(yv, (x_g, y_g), patch_size, (self.ori_w, self.ori_h))

                # creat searching map for each point of contour
                xv_p = np.expand_dims(xv_p, axis=2).repeat(len(cnt), axis=2)
                yv_p = np.expand_dims(yv_p, axis=2).repeat(len(cnt), axis=2)

                # calculate distance from all points in image to each each point of contour.
                # then, for each point in image, find the max distance to the contour
                max_distance_patch = maxDistance((xv_p, yv_p), cnt)

                # find the smallest values of the max distance as major axis of ellipse
                self.a = np.min(max_distance_patch)
                # where the smallest values is can be the candidates of center of the ellipse
                center_ph, center_pw = np.where(max_distance_patch == self.a)
                center_w = xv_p[center_ph, center_pw, 0]
                center_h = yv_p[center_ph, center_pw, 0]
                self.max_distance[ry[0]:ry[1], rx[0]:rx[1]] = max_distance_patch

                # average all centers
                center = (center_w.sum()/len(center_w), center_h.sum()/len(center_h))
                c_int = (int(round(center[0])), int(round(center[1])))

                print("Max:", time.time() - max_st)
                print("center h & w: \n", np.array([center_h, center_w]))
                print("center x & y: \n", center, "a:", self.a)

                # hough transform and voting
                hough_st = time.time()
                # shape: 0-a (minor axis <= major axis) and 0-180 degree
                self.winner, self.top_votes, self.hough_space = vote((int(self.a)+1,180), center, cnt)
                avg_b, avg_w = self.winner
                print("Hough:", time.time() - hough_st)

                # build ellipse
                success, e = buildEllipse(self.winner, center, self.top_votes, len(cnt), self.a, (self.ori_w, self.ori_h))

                if success:
                    # check duplicate
                    # put inner/outter contour into one single contour
                    for inx_e, e1 in enumerate(self.ellipses_):
                        if sameEllipses(e1, e):
                            if e1.lengthOfAxis <= e.lengthOfAxis:
                                self.ellipses_.pop(inx_e)
                                e.info = e1.info
                                self.ellipses_.insert(inx_e, e)
                            break
                    else:
                        is_black = (self.imgC[c_int[1],c_int[0]] == [0, 0, 0]).all()
                        is_white = (self.imgC[c_int[1],c_int[0]] == [255, 255, 255]).all()
                        is_red   = (self.imgC[c_int[1],c_int[0]] == [0, 0, 255]).all()
                        is_green = (self.imgC[c_int[1],c_int[0]] == [0, 255, 0]).all()
                        if not is_black and not is_white:  # test only
                            if is_red:
                                e.info = "first"
                                self.ellipses_.insert(0, e)
                            elif is_green:
                                e.info = "second"
                                self.ellipses_.insert(0, e)
                            else:
                                self.ellipses_.append(e)

                # Visualize the detection
                if self.saveImgs != None:
                    self.saveVisual(cnt, center, c_int, success, 'draw_{}.png'.format(i))

    def estimatePose(self):
        for ellipses in self.ellipses_:
            ellipses.cone.set(ellipses.boxEllipses)
            ellipses.cone.pose(self.intrinsic, self.radius)
            ellipses.cone.normal2Rotation(0)
            ellipses.cone.normal2Rotation(1)
            print(ellipses.cone.R[0])
            print(ellipses.cone.R[1])
            print(ellipses.cone.normals[0])
            print(ellipses.cone.normals[1])
            print('')

        if len(self.ellipses_) == 2:
            inner = self.ellipses_[0].cone.normals @ self.ellipses_[1].cone.normals.T
            print("Normal vector matching: \n", inner)
            maxidx = np.unravel_index(inner.argmax(), inner.shape)
            self.ellipses_[0].cone.sol = maxidx[0]
            self.ellipses_[1].cone.sol = maxidx[1]
            print("Ellipses {}: {}".format(0, self.ellipses_[0].cone.sol))
            print("Ellipses {}: {}".format(1, self.ellipses_[1].cone.sol))

    def getCircleCenters(self):
        return [e.cone.translations[e.cone.sol] for e in self.ellipses_]

    def getCircleNormals(self):
        return [e.cone.normals[e.cone.sol] for e in self.ellipses_]

    def getCircleRotations(self):
        return [e.cone.R[e.cone.sol] for e in self.ellipses_]


if __name__ == '__main__':
    # img = cv2.imread('e1.jpg')
    # img = cv2.resize(img, (img.shape[1]//4, img.shape[0]//4))
    img = cv2.imread('Left.png')

    cv_file = cv2.FileStorage("Left_cmx_dis.xml", cv2.FILE_STORAGE_READ)
    cmtx = cv_file.getNode("intrinsic").mat()
    dist = cv_file.getNode("distortion").mat()
    print("Intrinsic Matrix \n", cmtx)
    print("Distortion Coefficient \n", dist)
    cv_file.release()
    cmtx_new = cmtx.copy()
    map1, map2 = cv2.initUndistortRectifyMap(cmtx, dist, np.eye(
        3), cmtx_new, (img.shape[1], img.shape[0]), cv2.CV_16SC2)
    img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT)
    detectionNode = EllipsesDetection(img, cmtx_new, dist, radius=5, saveImgs='result3')
