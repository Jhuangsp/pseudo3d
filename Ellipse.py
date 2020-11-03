
from ObliqueCone import ObliqueCone

class Ellipse(object):
    """docstring for Ellipse"""

    def __init__(self, lengthOfAxis, center, theta, imgSize):
        super(Ellipse, self).__init__()
        self.lengthOfAxis = lengthOfAxis
        self.center = center
        self.theta = theta
        self.imgSize = imgSize
        self.boxEllipses = {
            'axis': self.lengthOfAxis,
            'center': self.center,
            'theta': self.theta,
            'imgSize': self.imgSize
        }
        self.info = "others"
        self.cone = ObliqueCone()