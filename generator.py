from OpenGL.GLU import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from math import sqrt, pi, sin, cos, tan
from PIL import Image
from PIL import ImageOps

import os
import numpy as np
import argparse
import json, yaml
import random as rd
import cv2

quadric = gluNewQuadric()

def startGL(args):
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_SINGLE)
    glutInitWindowSize(args.width, args.height)
    glutCreateWindow("Hello")
    init(args)


def init(args):
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glClearColor(0.1, 0.1, 0.1, 1.)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    # Tell the Opengl, we are going to set PROJECTION function
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()  # always call this before gluPerspective
    # set intrinsic parameters (fovy, aspect, zNear, zFar)
    gluPerspective(args.fovy, args.width/args.height, 0.1, 100000.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def setupPath(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path

def drawLogo(data):
    bitmap_tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, bitmap_tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,data.shape[1],data.shape[0],0,GL_BGRA,GL_UNSIGNED_BYTE,data)

    glEnable(GL_TEXTURE_2D)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0)
    glVertex3f(-2.4, 0.37, 0.03)
    glTexCoord2f(0, 1)
    glVertex3f(-2.4, -0.37, 0.03)
    glTexCoord2f(1, 1)
    glVertex3f(2.4, -0.37, 0.03)
    glTexCoord2f(1, 0)
    glVertex3f(2.4, 0.37, 0.03)
    glEnd()
    glDisable(GL_TEXTURE_2D)

def drawNet():
    glColor3f(0.6, 0.6, 0)
    glBegin(GL_QUADS)
    glVertex3f(-3, 0.02, 0)
    glVertex3f(-(3-0.04), 0.02, 0)
    glVertex3f(-(3-0.04), 0.02, 1.55)
    glVertex3f(-3, 0.02, 1.55)
    glEnd()
    glBegin(GL_QUADS)
    glVertex3f(-3, -0.02, 0)
    glVertex3f(-(3-0.04), -0.02, 0)
    glVertex3f(-(3-0.04), -0.02, 1.55)
    glVertex3f(-3, -0.02, 1.55)
    glEnd()
    glBegin(GL_QUADS)
    glVertex3f(-3, 0.02, 0)
    glVertex3f(-3, -0.02, 0)
    glVertex3f(-3, -0.02, 1.55)
    glVertex3f(-3, 0.02, 1.55)
    glEnd()
    glBegin(GL_QUADS)
    glVertex3f(-(3-0.04), 0.02, 0)
    glVertex3f(-(3-0.04), -0.02, 0)
    glVertex3f(-(3-0.04), -0.02, 1.55)
    glVertex3f(-(3-0.04), 0.02, 1.55)
    glEnd()

    glBegin(GL_QUADS)
    glVertex3f(3, 0.02, 0)
    glVertex3f((3-0.04), 0.02, 0)
    glVertex3f((3-0.04), 0.02, 1.55)
    glVertex3f(3, 0.02, 1.55)
    glEnd()
    glBegin(GL_QUADS)
    glVertex3f(3, -0.02, 0)
    glVertex3f((3-0.04), -0.02, 0)
    glVertex3f((3-0.04), -0.02, 1.55)
    glVertex3f(3, -0.02, 1.55)
    glEnd()
    glBegin(GL_QUADS)
    glVertex3f(3, 0.02, 0)
    glVertex3f(3, -0.02, 0)
    glVertex3f(3, -0.02, 1.55)
    glVertex3f(3, 0.02, 1.55)
    glEnd()
    glBegin(GL_QUADS)
    glVertex3f((3-0.04), 0.02, 0)
    glVertex3f((3-0.04), -0.02, 0)
    glVertex3f((3-0.04), -0.02, 1.55)
    glVertex3f((3-0.04), 0.02, 1.55)
    glEnd()

    glColor3f(1, 1, 1)
    glBegin(GL_QUADS)
    glVertex3f(-3, 0, 1.55)
    glVertex3f(3, 0, 1.55)
    glVertex3f(3, 0, 1.51)
    glVertex3f(-3, 0, 1.51)
    glEnd()

    glColor4f(0, 0, 0, 0.3);
    glBegin(GL_QUADS)
    glVertex3f(-3, 0, 1.55-0.04)
    glVertex3f(3, 0, 1.55-0.04)
    glVertex3f(3, 0, 1.55-0.76)
    glVertex3f(-3, 0, 1.55-0.76)
    glEnd()

def drawCourt():
    glColor3f(3/255, 102/255, 71/255)
    glBegin(GL_QUADS)
    glVertex3f(-5, 8.7, -0.02)
    glVertex3f(5, 8.7, -0.02)
    glVertex3f(5, -8.7, -0.02)
    glVertex3f(-5, -8.7, -0.02)
    glEnd()

    glColor3f(1, 1, 1)
    glBegin(GL_QUADS)
    glVertex3f(-3, 6.7, -0.01)
    glVertex3f(3, 6.7, -0.01)
    glVertex3f(3, 6.66, -0.01)
    glVertex3f(-3, 6.66, -0.01)
    glEnd()
    glBegin(GL_QUADS)
    glVertex3f(-3, -6.7, -0.01)
    glVertex3f(3, -6.7, -0.01)
    glVertex3f(3, -6.66, -0.01)
    glVertex3f(-3, -6.66, -0.01)
    glEnd()
    glBegin(GL_QUADS)
    glVertex3f(3, -6.7, -0.01)
    glVertex3f(3, 6.7, -0.01)
    glVertex3f(2.96, 6.7, -0.01)
    glVertex3f(2.96, -6.7, -0.01)
    glEnd()
    glBegin(GL_QUADS)
    glVertex3f(-3, -6.7, -0.01)
    glVertex3f(-3, 6.7, -0.01)
    glVertex3f(-2.96, 6.7, -0.01)
    glVertex3f(-2.96, -6.7, -0.01)
    glEnd()

    glColor3f(1, 1, 1)
    glBegin(GL_QUADS)
    glVertex3f(-3, (6.7-0.76), -0.01)
    glVertex3f(3, (6.7-0.76), -0.01)
    glVertex3f(3, (6.66-0.76), -0.01)
    glVertex3f(-3, (6.66-0.76), -0.01)
    glEnd()
    glBegin(GL_QUADS)
    glVertex3f(-3, -(6.7-0.76), -0.01)
    glVertex3f(3, -(6.7-0.76), -0.01)
    glVertex3f(3, -(6.66-0.76), -0.01)
    glVertex3f(-3, -(6.66-0.76), -0.01)
    glEnd()

    glColor3f(1, 1, 1)
    glBegin(GL_QUADS)
    glVertex3f(-3, (6.7-4.68), -0.01)
    glVertex3f(3, (6.7-4.68), -0.01)
    glVertex3f(3, (6.66-4.68), -0.01)
    glVertex3f(-3, (6.66-4.68), -0.01)
    glEnd()
    glBegin(GL_QUADS)
    glVertex3f(-3, -(6.7-4.68), -0.01)
    glVertex3f(3, -(6.7-4.68), -0.01)
    glVertex3f(3, -(6.66-4.68), -0.01)
    glVertex3f(-3, -(6.66-4.68), -0.01)
    glEnd()

    glBegin(GL_QUADS)
    glVertex3f((3-0.46), -6.7, -0.01)
    glVertex3f((3-0.46), 6.7, -0.01)
    glVertex3f((2.96-0.46), 6.7, -0.01)
    glVertex3f((2.96-0.46), -6.7, -0.01)
    glEnd()
    glBegin(GL_QUADS)
    glVertex3f(-(3-0.46), -6.7, -0.01)
    glVertex3f(-(3-0.46), 6.7, -0.01)
    glVertex3f(-(2.96-0.46), 6.7, -0.01)
    glVertex3f(-(2.96-0.46), -6.7, -0.01)
    glEnd()

    glBegin(GL_QUADS)
    glVertex3f(-0.02, 6.7, -0.01)
    glVertex3f(0.02, 6.7, -0.01)
    glVertex3f(0.02, 1.98, -0.01)
    glVertex3f(-0.02, 1.98, -0.01)
    glEnd()

    glBegin(GL_QUADS)
    glVertex3f(-0.02, -1.98, -0.01)
    glVertex3f(0.02, -1.98, -0.01)
    glVertex3f(0.02, -6.7, -0.01)
    glVertex3f(-0.02, -6.7, -0.01)
    glEnd()

    # glColor3f(1, 1, 1)
    # glBegin(GL_LINES)
    # glVertex3f(-3, 6.7, 0)
    # glVertex3f(3, 6.7, 0)
    # glEnd()
    # glBegin(GL_LINES)
    # glVertex3f(-3, -6.7, 0)
    # glVertex3f(3, -6.7, 0)
    # glEnd()
    # glBegin(GL_LINES)
    # glVertex3f(3, -6.7, 0)
    # glVertex3f(3, 6.7, 0)
    # glEnd()
    # glBegin(GL_LINES)
    # glVertex3f(-3, -6.7, 0)
    # glVertex3f(-3, 6.7, 0)
    # glEnd()


def rotateRYP(points, roll, yaw, pitch):
    R_pitch = np.array([[1,          0,           0],
                        [0, cos(pitch), -sin(pitch)],
                        [0, sin(pitch),  cos(pitch)]])
    R_yaw = np.array([[cos(yaw), -sin(yaw), 0],
                      [sin(yaw),  cos(yaw), 0],
                      [0,         0, 1]])
    R_roll = np.array([[cos(roll), 0, sin(roll)],
                       [0, 1,         0],
                       [-sin(roll), 0, cos(roll)]])

    R = R_roll @ R_yaw @ R_pitch
    if len(points.shape) == 1:  # only one point
        new_points = R @ points.reshape(3, 1)
    else:
        new_points = R @ points.T
    new_points = new_points.T
    return R, new_points


def readPose(cfg, args, num):
    lats, lons = randomRotate(cfg['rotation']['pitch'], cfg['rotation']['yaw'], num)
    t_xs, t_ys = randomTrans(cfg['shift'], num)
    for la, lo, tx, ty in zip(lats, lons, t_xs, t_ys):
        look_default = np.array([[0,0,-1]])
        up_default = np.array([[0,1,0]])
        directions = np.concatenate((look_default,up_default), axis=0)
        
        R, directions = rotateRYP(directions, 0, lo, (-la)+toRad(90)) # opengl default pitch is -90 degree
        
        look_new = directions[0]
        up_new = directions[1]

        cam_gl = - cfg['distance'] * look_new + np.array([tx,ty,0])
        obj_gl = np.array([tx,ty,0])
        up_gl = up_new
        
        # generate label data
        label = getLabel(cam_gl, obj_gl, up_gl, R)
        label['pattern'] = cfg['pattern']
        if cfg['pattern'] == "two_circle":
            label['radius'] = cfg['size']
            label['circle_shift'] = cfg['space']
        elif cfg['pattern'] == "chessboard":
            label['cube_size'] = cfg['size']
            label['chess'] = cfg['chess']
        label['intrinsic'] = {
            'focal':args.height / (2*tan(pi*args.fovy/360)),
            'h':args.height,
            'w':args.width,
            'fovy':args.fovy
        }
        label['extrinsic'] = {
            'init_distance':cfg['distance'],
            'rotation':[0.0, float(lo), float(la)],
            'translation':[float(tx), float(ty)]
        }
        yield (cam_gl, obj_gl, up_gl), label


def randomRotate(rangelat, rangelon, num):
    lats = np.random.uniform(
        sin(toRad(rangelat[0])), sin(toRad(rangelat[1])), num)
    lats = np.arcsin(lats)
    lons = np.random.uniform(
        toRad(rangelon[0]), toRad(rangelon[1]), num)
    return lats, lons


def randomTrans(range, num):
    t_xs = np.random.uniform(range[0], range[1], num)
    t_ys = np.random.uniform(range[0], range[1], num)
    return t_xs, t_ys


def toRad(deg):
    return deg*pi/180


def toDeg(rad):
    return rad*180/pi


def getLabel(cam_gl, obj_gl, up_gl, R):
    pose_label = {}
    
    cam_right_spcs = (R @ np.array([1,0,0]).reshape(3,1)).reshape(3)
    cam_down_spcs = (R @ np.array([0,-1,0]).reshape(3,1)).reshape(3)
    cam_in_spcs = (R @ np.array([0,0,-1]).reshape(3,1)).reshape(3)
    pose_label['spcs'] = {
        'cam_center':cam_gl.tolist(),
        'axis':[cam_right_spcs.tolist(), 
               cam_down_spcs.tolist(), 
               cam_in_spcs.tolist()]
    }

    T = np.array([cam_right_spcs, cam_down_spcs, cam_in_spcs])
    world_position_ccs = T @ (np.array([0,0,0])-cam_gl)
    world_RG_ccs = T.T[0]
    world_N_ccs = T.T[1]
    world_NxRG_ccs = T.T[2]
    pose_label['ccs'] = {
        'wor_center':world_position_ccs.tolist(),
        'axis':[world_RG_ccs.tolist(), 
                world_N_ccs.tolist(), 
                world_NxRG_ccs.tolist()]
    }
    return pose_label


def writeLabel(label, path):
    with open(path, 'w') as f:
        yaml.dump(label, f, default_flow_style=False, sort_keys=False)


def drawCircle(center, radius, color):
    sides = 120  # how fineness the circle is
    glColor3f(color[0]/255, color[1]/255, color[2]/255)
    glBegin(GL_POLYGON)
    for i in range(sides):
        cosine = radius * cos(i*2*pi/sides) + center[0]
        sine = radius * sin(i*2*pi/sides) + center[1]
        glVertex3f(cosine, sine, 0)
    glEnd()


def getImg(cfg, args):
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    data = glReadPixels(0, 0, args.width, args.height, GL_RGBA, GL_UNSIGNED_BYTE)
    image = Image.frombytes("RGBA", (args.width, args.height), data)
    # in my case image is flipped top-bottom for some reason
    image = ImageOps.flip(image)
    # image.save(outfolder+'/glutout_r{}_{}.png'.format(radius, 0), 'PNG')
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    return image


def getPose():
    camera_pos = np.zeros((3))
    matModelView = glGetDoublev(GL_MODELVIEW_MATRIX)
    matProjection = glGetDoublev(GL_PROJECTION_MATRIX)
    print(matModelView)
    print(matProjection)
    viewport = glGetIntegerv(GL_VIEWPORT)
    print(viewport)
    x, y, z = gluUnProject((viewport[2]-viewport[0])/2, (viewport[3]-viewport[1])/2, 0.0,
                           matModelView, matProjection, viewport)
    print(x, y, z)

def patternCircle(radius, space, chance=0.5):
    # draw center circles
    drawCircle((0, 0), radius, [255, 0, 0])

    # draw x direction circle
    drawCircle((space, 0), radius, [0, 255, 0])

    # draw rest of circle
    for c in range(-2, 3, 1):
        for r in range(-3, 4, 1):
            if c == 1 and r == 0:
                continue
            if rd.random() > chance:
                continue
            drawCircle((space*c, space*r), radius, [0, 0, 255])

def patternChess(cube_size, chess_size):
    assert chess_size[0] % 4 == 1 and chess_size[1] % 4 == 1, 'chess size must be 4n+1'
    # draw cubes of chessboard
    w, h = chess_size
    glColor3f(0, 0, 0)
    for i in range(-(w//4), (w//4)+1):
        for j in range(-(h//4), (h//4)+1):
            glBegin(GL_QUADS)
            glVertex3f((i*2+0)*cube_size, (j*2+0)*cube_size,0)
            glVertex3f((i*2+0)*cube_size, (j*2+1)*cube_size,0)
            glVertex3f((i*2+1)*cube_size, (j*2+1)*cube_size,0)
            glVertex3f((i*2+1)*cube_size, (j*2+0)*cube_size,0)
            glEnd()
            glBegin(GL_QUADS)
            glVertex3f((i*2+0)*cube_size, (j*2+0)*cube_size,0)
            glVertex3f((i*2+0)*cube_size, (j*2-1)*cube_size,0)
            glVertex3f((i*2-1)*cube_size, (j*2-1)*cube_size,0)
            glVertex3f((i*2-1)*cube_size, (j*2+0)*cube_size,0)
            glEnd()

def sphere(x,y,z,size=0.02):
    global quadric
    glColor3f(1, 0, 1)
    glTranslatef(x,y,z)
    gluSphere(quadric,size,32,32)
    glTranslatef(-x,-y,-z)

def drawTrack(start, end):
    points = 21
    unit = (end - start) / (points-1)
    x = np.linspace(-1.5, 1.5, points)
    y = -x**2 + 4.25

    for p in range(points):
        x_ = start[0] + p*unit[0]
        y_ = start[1] + p*unit[1]
        z_ = y[p]
        sphere(x_,y_,z_)
        # print(x_,y_,z_)

def draw(cfg, args, cam, obj, up):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(cam[0], cam[1], cam[2],
              obj[0], obj[1], obj[2],
              up[0], up[1], up[2])

    # draw badminton court
    drawCourt()
    drawNet()

    if cfg['pattern'] == "two_circle":
        # draw pattern for two circle method
        patternCircle(radius=cfg['size'], space=cfg['space'])
    elif cfg['pattern'] == "chessboard":
        # draw pattern for chessboard method
        patternChess(cube_size=cfg['size'], chess_size=cfg['chess'])
    image = getImg(cfg, args)

    start = np.array([-1.5, 2])
    end   = np.array([2, -5.5])
    drawCircle(start, 0.05, [0, 255, 255])
    drawCircle(end, 0.05, [255, 255, 0])
    drawTrack(start, end)
    track = getImg(cfg, args)

    glutSwapBuffers()
    return image, track


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, help='pose config json file')
    parser.add_argument("--fovy", type=float, default=40, help='fovy of image')
    parser.add_argument("--height", type=int, default=1060, help='height of image')
    parser.add_argument("--width", type=int, default=1920, help='width of image')
    parser.add_argument("--num", type=int, default=10000,
                        help='number of samples')
    parser.add_argument("--out", type=str, required=True, help='output path')
    args = parser.parse_args()

    f = open(args.json)
    cfg = json.load(f)
    f.close()

    focal = args.height / (2*tan(pi*args.fovy/360))
    print("focal", focal)
    print("Generate {} pattern".format(cfg['pattern']))

    startGL(args)
    setupPath(args.out)
    setupPath(os.path.join(args.out, 'labels'))

    for i, (pose_gl, label) in enumerate(readPose(cfg, args, args.num)):
        img, track = draw(cfg, args, pose_gl[0], pose_gl[1], pose_gl[2])
        cv2.imwrite(os.path.join(args.out,'calib_{:08d}.png'.format(i)), img)
        cv2.imwrite(os.path.join(args.out,'track_{:08d}.png'.format(i)), track)
        writeLabel(label, os.path.join(args.out, 'labels', 'label_{:08d}.yml'.format(i)))
