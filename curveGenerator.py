import numpy as np

class curveGenerator(object):
    """docstring for curveGenerator"""
    def __init__(self):
        super(curveGenerator, self).__init__()
        self.tracks = []

    def addTrack(self, start, end, points=21):
        track = []
        unit = (end - start) / (points-1)
        x = np.linspace(-1.5, 1.5, points)
        y = -x**2 + 4.25

        for p in range(points):
            x_ = start[0] + p*unit[0]
            y_ = start[1] + p*unit[1]
            z_ = y[p]
            track.append((x_,y_,z_))
        track = np.array(track)
        self.tracks.append(track)
        return track