import numpy as np
from math import sin, cos, pi, sqrt
import cv2

class ObliqueCone(object):
    """docstring for ObliqueCone"""

    def __init__(self):
        super(ObliqueCone, self).__init__()
        self.Q = np.zeros((3, 3))       # 3d ellipse corn equation
        self.Q_2d = np.zeros((3, 3))    # 2d ellipse equation
        self.translations = np.zeros((2, 3))  # two plausible translations solutions
        self.normals = np.zeros((2, 3))       # two plausible normal vector solutions
        self.projections = np.zeros((2, 2))   # two plausible projection circle center
        self.R = np.zeros((2, 3, 3))          # two plausible rotations solutions

        self.sol = -1  # true solution index (0 or 1), invalid (-1)

    def set(self, box):
        a, b = box['axis']
        u, v = box['center']
        angle = pi * box['theta'] / 180.0
        ca = cos(angle)
        sa = sin(angle)
        cx = box['imgSize'][0]/2
        cy = box['imgSize'][1]/2
        Re = np.array([[ca, -sa],
                       [sa,  ca]])
        ABInvTAB = np.array([[1./(a*a), 0.],
                             [0., 1./(b*b)]])
        X0 = np.array([u-cx, v-cy])
        M = Re @ ABInvTAB @ Re.T
        Mf = X0.T @ M @ X0
        A = M[0, 0]
        B = M[0, 1]
        C = M[1, 1]
        D = - A * X0[0] - B * X0[1]
        E = - B * X0[0] - C * X0[1]
        F = Mf - 1.0

        self.Q_2d = np.array([[A, B, D],
                              [B, C, E],
                              [D, E, F]])

    def pose(self, intrinsic, radius):
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        f = (fx + fy) / 2.0
        self.Q = self.Q_2d * np.array([[1,   1,   1/f],
                                       [1,   1,   1/f],
                                       [1/f, 1/f, 1/(f*f)]])

        ret, E, V = cv2.eigen(self.Q)
        V = V.T
        e1 = E[0, 0]
        e2 = E[1, 0]
        e3 = E[2, 0]
        S1 = [+1, +1, +1, +1, -1, -1, -1, -1]
        S2 = [+1, +1, -1, -1, +1, +1, -1, -1]
        S3 = [+1, -1, +1, -1, +1, -1, +1, -1]
        g = sqrt((e2-e3)/(e1-e3))
        h = sqrt((e1-e2)/(e1-e3))

        k = 0
        for i in range(8):
            z0 = S3[i] * (e2 * radius) / sqrt(-e1*e3)

            # Rotated center vector
            Tx = S2[i] * e3/e2 * h
            Ty = 0.0
            Tz = -S1[i] * e1/e2 * g

            # Rotated normal vector
            Nx = S2[i] * h
            Ny = 0.0
            Nz = -S1[i] * g

            t = z0 * V @ np.array([Tx, Ty, Tz])  # Center of circle in CCS
            n = V @ np.array([Nx, Ny, Nz])  # Normal vector unit in CCS

            # identify the two possible solutions
            if (t[2] > 0) and (n[2] < 0):  # Check facing constrain
                if k > 1:
                    continue
                self.translations[k] = t
                self.normals[k] = n

                # Projection
                Pc = intrinsic @ t
                self.projections[k, 0] = Pc[0]/Pc[2]
                self.projections[k, 1] = Pc[1]/Pc[2]
                k += 1

    def rotation2Normal(self, i):
        pass

    def normal2Rotation(self, i):
        unitZ = np.array([0, 0, 1])
        nvec = np.copy(self.normals[i])
        nvec = nvec/cv2.norm(nvec)
        c2 = nvec
        c1 = np.cross(unitZ, c2)
        c0 = np.cross(c1, c2)
        c1 = c1/cv2.norm(c1)
        c0 = c0/cv2.norm(c0)
        self.R[i] = np.array([[c0[0], c1[0], c2[0]],
                              [c0[1], c1[1], c2[1]],
                              [c0[2], c1[2], c2[2]]])
        pass
