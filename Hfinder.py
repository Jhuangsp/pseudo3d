import cv2
import numpy as np


class Hfinder(object):
    """docstring for Hfinder"""
    def __init__(self, img, court2D=[]):
        super(Hfinder, self).__init__()
        self.img = img
        self.court3D = [[-3.05, 6.7], [3.05, 6.7], [3.05, -6.7], [-3.05, -6.7]]
        self.court2D = court2D
        self.H = np.zeros((3,3))
        self.calculateH(self.img)

    def getH(self):
        return self.H        

    def mouseEvent(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            self.court2D.append([x, y])

    def calculateH(self, img):
        if len(self.court2D) == 0:
            cv2.namedWindow("Please pick 4 point of court")
            cv2.setMouseCallback("Please pick 4 point of court", self.mouseEvent)
            while True:
                cv2.imshow("Please pick 4 point of court", img)
                key = cv2.waitKey(1)
                if key == 27:
                    cv2.destroyAllWindows()
                    break

        self.court2D = np.array(self.court2D)
        self.court3D = np.array(self.court3D)
        self.H, status = cv2.findHomography(self.court2D, self.court3D)
