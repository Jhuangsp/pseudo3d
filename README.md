# Pseudo3D reconstruction for badminton trajectory

## Quick Start Demo
For the `synthetic_pseudo3D.py` demo, first point out the 4 corner of the badminton court in order of `left-top`, `right-top`, `right-bottom`, `left-bottom`.
If you want to adjust your corner position, please complete the 4 corner first, then adjust the corner by clicking on the new position. The program will find the nearest corner and modify it with new position.
```
# Synthetic monocular visualization
$ python synthetic_pseudo3D.py --json synthetic_track/mono/track.json

# Real-scene monocular visualization
$ python pseudo3D.py --track real_track/CHEN_Long_CHOU_Tien_Chen_Denmark_Open_2019_QuarterFinal/set_1_00_01.csv --height 720 --width 1280
```

## How to generate synthetic data
```
# Generate 1 synthetic image with json pose config
$ mkdir mydata
$ cp synthetic_track/mono/track.json mydata/track.json
$ python generator.py --json mydata/track.json --out mydata/ --num 1
```