import cv2
import numpy as np

class Hfinder(object):
    """docstring for Hfinder"""
    def __init__(self, img=None, court2D=[], court3D=[[-3.05, 6.7], [3.05, 6.7], [3.05, -6.7], [-3.05, -6.7]]):
        super(Hfinder, self).__init__()
        self.img = cv2.copyMakeBorder(img, 50,50,50,50, cv2.BORDER_CONSTANT, value=[0,0,0]) # padding
        self.court2D = court2D
        self.court3D = court3D
        self.H = np.zeros((3,3))
        self.calculateH(self.img)

    def getH(self):
        return self.H        

    def mouseEvent(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            if len(self.court2D) < 4:
                self.court2D.append([x, y])
            else:
                idx = np.linalg.norm(np.array(self.court2D) - np.array([[x, y]]), axis=1).argmin()
                self.court2D[idx] = [x, y]

    def calculateH(self, img):
        if len(self.court2D) == 0:
            cv2.namedWindow("Please pick 4 point of court")
            cv2.setMouseCallback("Please pick 4 point of court", self.mouseEvent)
            while True:
                show = np.copy(img)
                for c in self.court2D:
                    cv2.circle(show, (c[0], c[1]), 3, (38, 28, 235), -1)
                if len(self.court2D) > 1:
                    cv2.drawContours(show, [np.array(self.court2D)], 0, (38, 28, 235), 1)
                cv2.imshow("Please pick 4 point of court", show)
                key = cv2.waitKey(1)
                if key == 27:
                    cv2.destroyAllWindows()
                    break

        self.court2D = np.array(self.court2D) - np.array([[50,50]]) # unpadding
        self.court3D = np.array(self.court3D)
        self.H, status = cv2.findHomography(self.court2D, self.court3D)
