import numpy as np
import os
import cv2
import glob

corner_x = 5
corner_y = 9
objp = np.zeros((corner_x*corner_y,3), np.float32)
objp[:,:2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1,2) * 0.028

objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
img_size = (frame.shape[1], frame.shape[0])
count = 0
while True:
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (corner_x,corner_y), None)
        if ret:
            cv2.drawChessboardCorners(gray, (corner_x,corner_y), corners, ret)
        cv2.imshow('HI', gray)
        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == 32:
            cv2.imwrite('{}.png'.format(count), frame)
            objpoints.append(objp)
            imgpoints.append(corners)
            count += 1
            print('got {} image'.format(count))
np.save('objpoints.npy', objpoints)
np.save('imgpoints.npy', imgpoints)
objpoints = np.load('objpoints.npy')
imgpoints = np.load('imgpoints.npy')
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
print(mtx)
print(dist)

# img = cv2.imread('11.png')
# newcameramtx = mtx.copy()
# mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, np.eye(3), newcameramtx, img_size, 5)
# print(newcameramtx)
# img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
# cv2.imwrite('11_new.png', img)
