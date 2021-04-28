import numpy as np
import os
import cv2
import argparse, json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from math import pi, tan, sin, cos
from synthetic_pseudo3D import Pseudo3d_Synth
from h2pose.H2Pose import H2Pose
from triangulation import multiCamTriang

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

def toRad(deg):
    return deg*pi/180

def dEuler(pred, gt):
    error = pred - gt
    distances = np.sqrt((error**2).sum(-1))
    return distances.sum()/distances.shape[-1]

def curve(start, end):
    points = 15
    unit = (end - start) / (points-1)
    x = np.linspace(-1.5, 1.5, points)
    y = -x**2 + 4.25

    tracks = []
    for p in range(points):
        x_ = start[0] + p*unit[0]
        y_ = start[1] + p*unit[1]
        z_ = y[p]
        tracks.append([x_, y_, z_])
    return tracks

def draw(perr, merr, name):
    # perr = np.concatenate((np.random.rand(100), np.random.rand(10)*3))
    # merr = np.random.rand(100)
    dfperr = pd.DataFrame({'Error per shot (m)':perr})
    dfmerr = pd.DataFrame({'Error per shot (m)':merr})
    dfperr = dfperr.merge(pd.DataFrame({'Algorithm':["Pseudo3D"]}), how='cross')
    dfmerr = dfmerr.merge(pd.DataFrame({'Algorithm':["MultiCam"]}), how='cross')
    df3 = pd.concat([dfperr, dfmerr])
    print(df3)
    sns.set(rc={'figure.figsize':(5.5,8)})
    ax = sns.boxplot(x=df3["Algorithm"], y=df3["Error per shot (m)"], width=0.2)
    ax.set(ylim=(-0.01, 0.2))
    # plt.show()
    plt.savefig("result_{}.png".format(name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--json", type=str, help='pose config json file')
    # parser.add_argument("--homo", type=str, help='homography matrix')
    parser.add_argument("--data", type=str, help='track data file')
    parser.add_argument("--cam1", type=str, help='camera to perform pseudo3D')
    parser.add_argument("--cam2", type=str, help='second camera for triangulation')
    parser.add_argument("--fovy", type=float, default=40, help='fovy of image')
    parser.add_argument("--height", type=int, default=1060, help='height of image')
    parser.add_argument("--width", type=int, default=1920, help='width of image')

    parser.add_argument("--task", type=str, help='task name')
    parser.add_argument("--noise", default=0, type=int, help='2D detection noise (pixel)')
    parser.add_argument("--anoise", default=0, type=int, help='Anchor noise (pixel)')
    parser.add_argument("--pnoise", default=0, type=float, help='Pose noise (degree)')
    args = parser.parse_args()

    # GT data
    f = open(args.data)
    anchors = json.load(f)
    starts = np.array(anchors["start"])
    ends = np.array(anchors["end"])
    f.close()

    # H matrix
    f = open(os.path.join(args.cam1, 'Hmtx.json'))
    Hmtx_cam1 = np.array(json.load(f))
    f.close()

    # H matrix
    f = open(os.path.join(args.cam2, 'Hmtx.json'))
    Hmtx_cam2 = np.array(json.load(f))
    f.close()


    # Prepare Intrinsic matix of video
    video_h = args.height
    video_w = args.width
    video_focal_length = video_h / (2*tan(pi*args.fovy/360))
    Kmtx = np.array(
        [[video_focal_length, 0, video_w/2],
         [0, video_focal_length, video_h/2],
         [0,                  0,         1]]
    )

    # 2d data
    track2ds_cam1 = np.load(os.path.join(args.cam1, 'data2d.npy')) # (1000, 15, 2)
    noise = np.random.randint(args.noise*2+1, size=track2ds_cam1.shape)-args.noise
    track2ds_cam1 += noise

    # 2d data
    track2ds_cam2 = np.load(os.path.join(args.cam2, 'data2d.npy')) # (1000, 15, 2)
    noise = np.random.randint(args.noise*2+1, size=track2ds_cam2.shape)-args.noise
    track2ds_cam2 += noise

    perr = []
    for i,(As, Ae, track2d) in enumerate(zip(starts, ends, track2ds_cam1)):
        if args.anoise > 0:
            As_ = np.linalg.inv(Hmtx_cam1)@np.array([[As[0]], [As[1]], [1]])
            As_ = (As_/As_[2,0])
            anoise = np.random.randint(args.anoise*2+1, size=As_.shape)-args.anoise
            As_ = As_ + anoise
            As_[2,0] = 1
            As_ = Hmtx_cam1@As_
            As_ = (As_/As_[2,0]).reshape(-1)[0:2].tolist()
            Ae_ = np.linalg.inv(Hmtx_cam1)@np.array([[Ae[0]], [Ae[1]], [1]])
            Ae_ = (Ae_/Ae_[2,0])
            anoise = np.random.randint(args.anoise*2+1, size=Ae_.shape)-args.anoise
            Ae_ = Ae_ + anoise
            Ae_[2,0] = 1
            Ae_ = Hmtx_cam1@Ae_
            Ae_ = (Ae_/Ae_[2,0]).reshape(-1)[0:2].tolist()
        else:
            As_ = As
            Ae_ = Ae

        tf = Pseudo3d_Synth(
            track2D=track2d,
            args=args, 
            H=Hmtx_cam1, 
            K=Kmtx,
            # path_j=os.path.join(args.cam1, 'camera4.json'), 
            anchors=(As_, Ae_)
        )
        # print(tf.updateF(silence=True))

        if args.pnoise > 0:
            err = dEuler(tf.updateF(silence=True, noise=args.pnoise), np.array(curve(As, Ae)))
        else:
            err = dEuler(tf.updateF(silence=True), np.array(curve(As, Ae)))
        print('{} MEAN ERROR in one shot (Pseudo3D):'.format(i), err)
        perr.append(err)
        # print(As, Ae)
        # break
    perr.sort()
    quad = len(perr)//4
    center = np.array(perr[quad:-quad])    
    print('Overall:', center.sum()/len(center))




    # get C2W
    h2p1 = H2Pose(Kmtx, Hmtx_cam1)
    h2p2 = H2Pose(Kmtx, Hmtx_cam2)
    print(h2p1.getC2W())
    print(h2p1.getCamera().T)
    print(h2p2.getC2W())
    print(h2p2.getCamera().T)

    # Start triangulation
    merr = []
    for i, (As, Ae, track2d_cam1, track2d_cam2) in enumerate(zip(starts, ends, track2ds_cam1, track2ds_cam2)):
        if args.pnoise > 0:
            c2w1 = h2p1.getC2W()
            c2w2 = h2p2.getC2W()
            d = toRad(args.pnoise)
            d = (np.random.rand((1))*2-0.5)*d
            Rx = np.array([[1,0,0],[0,cos(d),-sin(d)],[0,sin(d),cos(d)]])
            Ry = np.array([[cos(d),0,sin(d)],[0,1,0],[-sin(d),0,cos(d)]])
            Rz = np.array([[cos(d),-sin(d),0],[sin(d),cos(d),0],[0,0,1]])
            c2w1 = (Rz@Ry@Rx@c2w1.T).T
            c2w2 = (Rz@Ry@Rx@c2w2.T).T
            mct = multiCamTriang(
                track2Ds=np.array([track2d_cam1, track2d_cam2]), 
                poses=np.array([c2w1, c2w2]), 
                eye=np.array([h2p1.getCamera().T, h2p2.getCamera().T]),
                Ks=np.array([Kmtx, Kmtx]),
                silence=True
            )
        else:
            mct = multiCamTriang(
                track2Ds=np.array([track2d_cam1, track2d_cam2]), 
                poses=np.array([h2p1.getC2W(), h2p2.getC2W()]), 
                eye=np.array([h2p1.getCamera().T, h2p2.getCamera().T]),
                Ks=np.array([Kmtx, Kmtx]),
                silence=True
            )
        err = dEuler(mct.track3D, np.array(curve(As, Ae)))
        # print(mct.track3D)
        print('{} MEAN ERROR in one shot (MultiCam):'.format(i), err)
        merr.append(err)
        # break
    merr.sort()
    quad = len(merr)//4
    center = np.array(merr[quad:-quad])    
    print('Overall:', center.sum()/len(center))

    draw(np.array(perr), np.array(merr), args.task)