import numpy as np
import cv2
import os, csv, glob, yaml
from sklearn.cluster import MeanShift

class tracker2D(object):
    """docstring for tracker2D"""
    def __init__(self, img):
        super(tracker2D, self).__init__()
        self.img = img

        track2D = np.where(np.all(self.img==[255,0,255], axis=-1))
        track2D = np.moveaxis(np.array((track2D[1], track2D[0])), 0, -1)
        self.cluster = MeanShift(bandwidth=2).fit(track2D)

    def getTrack2D(self):
        return self.cluster.cluster_centers_

