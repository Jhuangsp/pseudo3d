import numpy as np
import os
import cv2
import argparse, csv, json
from pseudo3D import Pseudo3d

from math import sqrt, pi, sin, cos, tan

from Hfinder import Hfinder
from trackDetector import trackDetector
from tracker2D import tracker2D

import glob
import pprint
pp = pprint.PrettyPrinter(indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", type=str, help='output folder of TrackNetV2')
    parser.add_argument("--fovy", type=float, default=40, help='fovy of visualize window')
    parser.add_argument("--height", type=int, default=1060, help='height of visualize window')
    parser.add_argument("--width", type=int, default=1920, help='width of visualize window')
    parser.add_argument("--out", type=str, default='track_pseudo3D', help='output folder')
    args = parser.parse_args()

    if not os.path.isdir(args.out):
        os.makedirs(args.out)

    tracks = glob.glob(os.path.join(args.tracks, 'set_*.csv'))

    flaw_list = []
    with open(os.path.join(args.tracks, 'flaw_list.csv')) as fflaw:
        rows = csv.DictReader(fflaw)
        for row in rows:
            flaw_list.append(row['Flaw'])

    for track in tracks:
        rallyname = track.split(os.sep)[-1].split('.')[0]
        shot_folder = os.path.join(args.out, rallyname)
        print(rallyname)
        if not os.path.isdir(shot_folder):
            os.makedirs(shot_folder)

        if rallyname in flaw_list:
            continue

        # Prepare TrcakNetV2 result
        shots = []
        shots_start = []
        shots_end = []
        shots_frame = []
        with open(track, newline='') as ral_f:
            ral = csv.DictReader(ral_f)
            served = False
            for f in ral:
                if not served:
                    if f['Hit'] == 'True':
                        tmp = []
                        tmp_frame = []
                        tmp.append([float(f['X']), float(f['Y'])])
                        tmp_frame.append(float(f['Frame']))
                        try:
                            shots_start.append([float(f['EndX']), float(f['EndY'])])
                            shots_end.append([float(f['EndX']), float(f['EndY'])])
                        except:
                            shots_start.append([-1,-1])
                            shots_end.append([-1,-1])
                        served = True
                    else: continue
                else:
                    if f['Hit'] == 'True':
                        shots.append(tmp)
                        shots_frame.append(tmp_frame)
                        tmp = []
                        tmp_frame = []
                        tmp.append([float(f['X']), float(f['Y'])])
                        tmp_frame.append(float(f['Frame']))
                        shots_start.append([float(f['StartX']), float(f['StartY'])])
                        shots_end.append([float(f['EndX']), float(f['EndY'])])
                    else:
                        if f['Visibility'] == '1':
                            tmp.append([float(f['X']), float(f['Y'])])
                            tmp_frame.append(float(f['Frame']))
            else:
                shots.append(tmp)
                shots_frame.append(tmp_frame)

        # Prepare Homography matrix (image(pixel) -> court(meter))
        # 4 Court corners of CHEN_Long_CHOU_Tien_Chen_Denmark_Open_2019_QuarterFinal recorded in homography_matrix.csv
        # court2D = [[448, 256.6], [818.2, 256.2], [981.2, 646.4], [278.8, 649]] 
        # 4 Court corners of CHEN_Long_CHOU_Tien_Chen_World_Tour_Finals_Group_Stage
        # court2D = [[415.4,358.2], [863.4,358.2], [1018.4,672], [262.8,672]]
        # 4 Court corners of CHEN_Yufei_TAI_Tzu_Ying_Malaysia_Masters_2020_Finals
        # court2D = [[416.2,422.4], [879.6,422.2], [945.4,723.8], [225.8,718]]
        # 4 Court corners of CHEN_Yufei_TAI_Tzu_Ying_World_Tour_Finals_Finals
        # court2D = [[340.8,354.4], [823.2,353.2], [837.6,677.2], [120,675.6]]
        # 4 Court corners of CHOU_Tien_Chen_Anthony_Sinisuka_GINTING_World_Tour_Finals_Group_Stage
        court2D = [[338.4,329.4], [889.8,332.2], [952.4,705.6], [133.8,695.2]]
        court3D = [[-3.05, 6.7], [3.05, 6.7], [3.05, -6.7], [-3.05, -6.7]]
        hf = Hfinder(None, court2D=court2D, court3D=court3D)
        Hmtx = hf.getH()

        # Prepare Intrinsic matix of video
        video_h = 720
        video_w = 1280
        video_focal_length = 4000
        Kmtx = np.array(
            [[video_focal_length, 0, video_w/2],
             [0, video_focal_length, video_h/2],
             [0,                  0,         1]]
        )

        with open(os.path.join(shot_folder,'focal_length.txt'), 'w') as ffocal:
            ffocal.write(str(video_focal_length))

        # pp.pprint(shots_frame)
        # Pseudo3D trajectory transform (2D->3D)
        for shot_idx in range(1,len(shots_start)):
            tf = Pseudo3d(start_wcs=np.array(shots_start[shot_idx]), 
                end_wcs=np.array(shots_end[shot_idx]), 
                track2D=np.array(shots[shot_idx]), 
                args=args, 
                H=Hmtx, 
                K=Kmtx
            )
            pred = tf.updateF()

            with open(os.path.join(shot_folder, '{:02d}.csv'.format(shot_idx)), 'w', newline='') as csvfile:
                fieldnames = ['Frame', 'Visibility', 'X', 'Y', 'Z']

                writer = csv.writer(csvfile)
                writer.writerow(fieldnames)

                count = 0
                for Frame in range(int(shots_frame[shot_idx][0]),int(shots_frame[shot_idx][-1]+1)):
                    v = 1 if float(Frame) in shots_frame[shot_idx] else 0
                    if v == 1:
                        writer.writerow([Frame, v, pred[count,0], pred[count,1], pred[count,2]])
                        count += 1
                    else:
                        writer.writerow([Frame, v, '', '', ''])