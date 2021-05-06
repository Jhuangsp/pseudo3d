import numpy as np
import os
import cv2
import json
from h2pose.Hfinder import Hfinder

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    ret, img = cap.read()
    if ret:
        court2D = []
        court3D = [[-3.05, 6.7], [3.05, 6.7], [3.05, -6.7], [-3.05, -6.7]]
        hf = Hfinder(img, court2D=court2D, court3D=court3D)
        Hmtx = hf.getH()
        print(Hmtx)
        with open('Hmtx_o.json', 'w') as f:
            json.dump(Hmtx.tolist(), f)
    cap.release()