import numpy as np
import os
import cv2
import argparse, json, glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from math import pi, tan, sin, cos
from pseudo3D import Pseudo3d
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
    ax.set(ylim=(-0.01, 1.2))
    # plt.show()
    plt.savefig("result_{}.png".format(name))

def loadTrack(cams, path):
    track2ds = {}
    for cam in cams:
        track2ds[cam] = []
    rallys = [os.path.join(path, f) for f in os.listdir(path) if not os.path.isfile(os.path.join(path, f))]
    for rally in rallys:
        files = [os.path.join(rally, f) for f in os.listdir(rally) if os.path.isfile(os.path.join(rally, f))]
        for cam in cams:
            for f_cam in files:
                if cam in f_cam:
                    df = pd.read_csv(f_cam)
                    track2ds[cam].append(df.to_numpy())
    return track2ds

def loadHit(cams, path):
    hits = []
    rallys_hits = [os.path.join(path, f) for f in os.listdir(path) if not os.path.isfile(os.path.join(path, f))]
    for rally in rallys_hits:
        files = [os.path.join(rally, f) for f in os.listdir(rally) if os.path.isfile(os.path.join(rally, f))]
        for cam in cams:
            for f_cam in files:
                if cam in f_cam:
                    df = pd.read_csv(f_cam)
                    hit_index = np.where(df.to_numpy()[:,1]>0)
                    hits.append(hit_index[0].tolist())
    return hits

def getShot(cam, track2ds, hits):
    for r in range(len(track2ds[cam])):
        print('Rally {}'.format(r))
        for i, (s,e) in enumerate(zip(hits[r][:-1], hits[r][1:])):
            print(i)
            # print(s,e)
            yield track2ds[cam][r][s:e]

def getShotAll(cams, track2ds, hits):
    cam_iters = [getShot(cam, track2ds, hits) for cam in cams]
    count = 0
    try:
        while 1:
            payload = []
            for cam_iter in cam_iters:
                payload.append(next(cam_iter))
                # print(payload)
            assert all( map(lambda x: len(x) == len(payload[0]), payload ) ), 'the shot frames lengh is not consist between cameras'
            yield payload
    except StopIteration:
        pass
    finally:
        del cam_iters
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", type=str, default="real_track/stereo_thesis/track_gt/", help='2D detection result folder')
    parser.add_argument("--mono", type=str, default="cam00", help='pseudo3D camera')
    parser.add_argument("--stereo", type=str, nargs='+', default=["cam00", "cam01"], help='multicam cameras')
    parser.add_argument("--gt", type=str, nargs='+', default=["cam00", "cam01", "cam02", "cam03"], help='multicam cameras for ground truth')
    parser.add_argument("--hit", type=str, default="real_track/stereo_thesis/hit/", help='shot segmentation folder')
    
    parser.add_argument("--homo", type=str, default="real_track/stereo_thesis/homography/", help='homography matrix folder')
    parser.add_argument("--kmtx", type=str, default="real_track/stereo_thesis/intrinsic/", help='intrinsic matrix folder')
    
    parser.add_argument("--fovy", type=float, default=40, help='fovy of image')
    parser.add_argument("--height", type=int, default=1060, help='height of image')
    parser.add_argument("--width", type=int, default=1920, help='width of image')

    parser.add_argument("--task", type=str, help='task name')
    parser.add_argument("--noise", default=0, type=int, help='2D detection noise (pixel)')
    parser.add_argument("--anoise", default=0, type=int, help='Anchor noise (pixel)')
    parser.add_argument("--pnoise", default=0, type=float, help='Pose noise (degree)')
    args = parser.parse_args()

    # H matrix
    Hmtxs = {}
    for cam in args.gt:
        f = open(os.path.join(args.homo, cam+'.json'))
        Hmtxs[cam] = np.array(json.load(f))
        f.close()

    # K matrix
    Kmtxs = {}
    for cam in args.gt:
        f = open(os.path.join(args.kmtx, cam+'.json'))
        Kmtxs[cam] = np.array(json.load(f)["Kmtx"])
        f.close()

    # 2d data
    track2ds = loadTrack(args.gt, args.track)
    # print(track2ds)

    # hit point fpr pseudo3d
    hits = loadHit(args.gt, args.hit)
    # print(hits)

    # Form Ground Truth
    print('Forming Ground Truth')
    track3D_gt = []
    visibale_mask = []
    for idx, shot_all_cam in enumerate(getShotAll(args.gt, track2ds, hits)):
        # print('Shot:', idx)
        track2Ds = [shot_1_cam[:,2:] for shot_1_cam in shot_all_cam]
        h2ps = [H2Pose(np.array(Kmtxs[cam]), np.array(Hmtxs[cam])) for cam in args.gt]
        poses = [h2p.getC2W() for h2p in h2ps]
        eye = [h2p.getCamera().T for h2p in h2ps]
        Ks = [np.array(Kmtxs[cam]) for cam in args.gt]
        mct = multiCamTriang(
            track2Ds=np.array(track2Ds), 
            poses=np.array(poses), 
            eye=np.array(eye),
            Ks=np.array(Ks),
            silence=True
        )
        tmpmask = shot_all_cam[0][:,1]
        for mask in [shot_1_cam[:,1] for shot_1_cam in shot_all_cam]:
            tmpmask = np.logical_and(tmpmask, mask)
        visibale_mask.append(tmpmask)
        # print(visibale_mask.astype(int))
        # print(mct.track3D)
        # print(mct.track3D[visibale_mask])
        track3D_gt.append(mct.track3D)
        # if idx == 0:
        #     print(tmpmask)
        #     print(mct.track3D)
        #     break
            # exit()

    # Form Stereo Prediction
    print('Forming Stereo Prediction')
    track3D_base = []
    for idx, shot_stereo_cam in enumerate(getShotAll(args.stereo, track2ds, hits)):
        # print('Shot:', idx)
        track2Ds = np.array([shot_1_cam[:,2:] for shot_1_cam in shot_stereo_cam])
        h2ps = [H2Pose(np.array(Kmtxs[cam]), np.array(Hmtxs[cam])) for cam in args.stereo]
        poses = np.array([h2p.getC2W() for h2p in h2ps])
        # print(poses[0]); exit(0)
        eye = np.array([h2p.getCamera().T for h2p in h2ps])
        Ks = np.array([np.array(Kmtxs[cam]) for cam in args.stereo])
        track2Ds += np.random.randint(args.noise*2+1, size=track2Ds.shape)-args.noise # add 2d detection noise

        if args.pnoise > 0:
            d = toRad(args.pnoise)
            d = (np.random.rand((1))*2-0.5)*d
            Rx = np.array([[1,0,0],[0,cos(d),-sin(d)],[0,sin(d),cos(d)]])
            Ry = np.array([[cos(d),0,sin(d)],[0,1,0],[-sin(d),0,cos(d)]])
            Rz = np.array([[cos(d),-sin(d),0],[sin(d),cos(d),0],[0,0,1]])
            poses = np.array([(Rz@Ry@Rx@c2w.T).T for c2w in poses])
        mct = multiCamTriang(
            track2Ds=track2Ds, 
            poses=poses, 
            eye=eye,
            Ks=Ks,
            silence=True
        )
        track3D_base.append(mct.track3D)
        # if idx == 0:
        #     print(mct.track3D)
        #     break
            # exit()

    # Form Pseudo3D Prediction
    print('Forming Pseudo3D Prediction')
    track3D_pred = []
    for idx, shot_mono_cam in enumerate(getShot(args.mono, track2ds, hits)):
        # print('Shot:', idx)
        As = track3D_gt[idx][visibale_mask[idx]][0]
        Ae = track3D_gt[idx][visibale_mask[idx]][-1]
        if args.anoise > 0:
            As_ = np.linalg.inv(Hmtxs[args.mono])@np.array([[As[0]], [As[1]], [1]])
            As_ = (As_/As_[2,0])
            anoise = np.random.randint(args.anoise*2+1, size=As_.shape)-args.anoise
            As_ = As_ + anoise
            As_[2,0] = 1
            As_ = Hmtxs[args.mono]@As_
            As_ = (As_/As_[2,0]).reshape(-1)
            Ae_ = np.linalg.inv(Hmtxs[args.mono])@np.array([[Ae[0]], [Ae[1]], [1]])
            Ae_ = (Ae_/Ae_[2,0])
            anoise = np.random.randint(args.anoise*2+1, size=Ae_.shape)-args.anoise
            Ae_ = Ae_ + anoise
            Ae_[2,0] = 1
            Ae_ = Hmtxs[args.mono]@Ae_
            Ae_ = (Ae_/Ae_[2,0]).reshape(-1)
            As_[2] = 0
            Ae_[2] = 0
        else:
            As_ = As
            Ae_ = Ae
            As_[2] = 0
            Ae_[2] = 0
        # print(As_)
        # print(Ae_)
        track2D = shot_mono_cam[:,2:]
        track2D += np.random.randint(args.noise*2+1, size=track2D.shape)-args.noise # add 2d detection noise
        tf = Pseudo3d(start_wcs=As_, 
            end_wcs=Ae_, 
            track2D=track2D,
            args=args, 
            H=Hmtxs[args.mono], 
            K=Kmtxs[args.mono]
        )
        track3D_pred.append(tf.updateF(silence=True, noise=args.pnoise))
        # if idx == 0:
        #     print(tf.updateF(silence=True, noise=0))
        #     exit()
        #     break

    # Calculate error
    merrs = []
    perrs = []
    for i, (gt, mask, base, pred) in enumerate(zip(track3D_gt, visibale_mask, track3D_base, track3D_pred)):
        merr = dEuler(base[mask], gt[mask])
        perr = dEuler(pred[mask], gt[mask])
        print('{} MEAN ERROR in one shot (MultiCam):'.format(i), round(merr,2))
        print('{} MEAN ERROR in one shot (Pseudo3D):'.format(i), round(perr,2))
        merrs.append(merr)
        perrs.append(perr)

    merrs.sort()
    quad = len(merrs)//4
    center = np.array(merrs[quad:-quad])   
    print('Overall (Multicam):', center.sum()/len(center))

    perrs.sort()
    quad = len(perrs)//4
    center = np.array(perrs[quad:-quad])   
    print('Overall (Pseudo3D):', center.sum()/len(center))

    draw(np.array(perrs), np.array(merrs), args.task)



