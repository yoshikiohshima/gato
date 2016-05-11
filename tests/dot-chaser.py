#!/usr/bin/env python

import numpy as np
import cv2
from time import clock
import sys
import video
import datetime
import sphero

s = None

lastDirs = [0]
lastPositions = [(0, 0)]


def connect():
    if not s:
        print 'sphero variable is not initialized'
        return

    print("connecting to sphero")
    try:
        s.connect()
    except:
        print("err!")
        s.close()

    print( """Bluetooth info:name: %s \nbta: %s """ %
           (s.get_bluetooth_info().name, s.get_bluetooth_info().bta))

def wrap(degrees):
    return degrees % 360

def dir(newPos, oldPos):
   (newX, newY) = newPos
   (oldX, oldY) = oldPos

   diffX = newX - oldX
   diffY = newY - oldY

   rad = np.arctan2(diffX, diffY)
   return wrap(rad * 180.0 / np.pi)

def distance(p1, p2):
   (newX, newY) = p1
   (oldX, oldY) = p2

   diffX = newX - oldX
   diffY = newY - oldY

   return np.sqrt((diffX * diffX) + (diffY * diffY))

errors = []
error = 0
integral = 0
derivative = 0

def spheroStep(s, tx, ty, lastDirs, dotx, doty, lastPositions, threshold):

    global error, errors, integral, derivative

    samples = 5

    if (abs(tx - dotx) < threshold) and (abs(ty - doty) < threshold):
        s.roll(0x0, lastDirs[-1])
        return

    if len(lastDirs) < samples:
        print 'early', lastDirs[-1]
        if s:
            s.roll(0x0F, lastDirs[-1])
        lastDirs.append(lastDirs[-1])
        lastPositions.append((dotx, doty))
        return

    prevError = error
    error = distance((tx, ty), (dotx, doty))
    derivative = error - prevError
    integral += error

    errors.append(error)
    if (len(errors) > 50):
        errors.pop(0)

    kP = 1
    kI = 0
    kD = 0

    speed = error * kP + integral * kI + derivative * kD;

    lastPositions.pop(0)
    lastDirs.pop(0)

    d = dir(lastPositions[-1], lastPositions[0])
    # the direction sphero moved in last samples steps within the camera frame

    x = y = 0
    for angle in lastDirs:
        x += np.cos(degToRad(angle))
        y += np.sin(degToRad(angle))
    sd = wrap(radToDeg(np.arctan2(y, x)))
    # the direction that sphero thinks it went in last samples frames

    cd = dir((tx, ty), (dotx, doty))
    # the direction sphero needs to go in the camera frame

    offset = d - sd
    # sphero thinks it went toward sd, but it actually went toward d
    # IOW, offset is how much off sphero's idea is.
    # e.g., if sd is zero, sphero thinks it is moving toward its north,
    # but it is actually moving toward d.

    next = wrap(cd - offset)
    # the new direction sphero should go in sphero's frame

    if wrap(next - lastDirs[-1]) < 180:
        next = wrap(lastDirs[-1] + 10)
    if wrap(next - lastDirs[-1]) >= 180:
        next = wrap(lastDirs[-1] - 10)
    
    if s:
        s.roll(0x0F, int(next))
    lastPositions.append((dotx, doty))
    lastDirs.append(int(next))
    return cd, sd, d, next, offset, speed

def degToRad(deg):
    return deg / 180.0 * np.pi

def radToDeg(rad):
    return rad * 180.0 / np.pi

def drawVec(image, dir, l, point, color):
    sx, sy = point
    rad = degToRad(dir)
    cv2.line(image, point, (int(sx + l * np.cos(rad)), int(sy + l * np.sin(rad))), color, 3)

targetX = 0
targetY = 0

smallSize = (640, 360)

def recordTarget(event, x, y, flags, param):
    global targetX, targetY
    if event == cv2.EVENT_LBUTTONDOWN:
        targetX = y
        targetY = x

def drawSparkLine(image, values, r, point, height, lineColor, dotColor):
    deltaX = 3
    x, y = point

    px, py = point
    for value in values:
        cx = px + deltaX
        cy = y + height * (1 - scale(value, r))

        cv2.line(image, (int(px), int(py)), (int(cx), int(cy)), lineColor, 1);

        px = cx
        py = cy

    cv2.circle(image, (int(px),int(py)), 3, dotColor)

def scale(value, r):
    mx, mn = r
    if (mx - mn > 0):
        return (value - mn) / (mx - mn)
    else:
        return 0

if __name__ == '__main__':

    try: fn = sys.argv[1]
    except: fn = 0

    try: noSphero = sys.argv[2]
    except: noSphero = ''

    doSphero = not noSphero

#    cam = cv2.VideoCapture(fn)
#    By reading video.py, the line above should be equivalent to the line below;  but somehow it does not create a valid camera object

    cam = video.create_capture(fn, fallback='synth:bg=../cpp/baboon.jpg:class=chess:noise=0.05')

    flag, frame = cam.read()

    small = cv2.resize(frame, smallSize, interpolation = cv2.INTER_LINEAR)

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    values = hsv[:,:,2]

    # we are still creating a new array for 'values' in the loop.

    xcord, ycord = np.indices(values.shape)
    bright = np.zeros(values.shape, values.dtype)
    prod = np.zeros(values.shape, np.int64)
    overlaid = np.zeros(small.shape, small.dtype)
    red = [0,0,255]
    cv2.imshow('hsv', values)

    cv2.namedWindow("overlay")
    cv2.setMouseCallback("overlay", recordTarget)
    cv2.imshow('overlay', overlaid)

    targetX = bright.shape[0] / 2
    targetY = bright.shape[1] / 2

    if doSphero:
        s = sphero.Sphero()
    connect()

    last = startTime = datetime.datetime.now()
    frames = 0
    fps = 0
    while True:
        flag, frame = cam.read()
        cv2.resize(frame, smallSize, small, interpolation = cv2.INTER_LINEAR)
        #small = cv2.pyrDown(frame)

        cv2.cvtColor(small, cv2.COLOR_BGR2HSV, hsv)
        values = hsv[:,:,2]

        cv2.imshow('hsv', values)

        bright.fill(0)
        mask = (values > 240)
        bright[mask] = 1
        count = np.sum(mask)
	
	for i in range(3):
	    overlaid[:,:,i] = bright * 255

        if count > 0:  # meaning that this is a valid frame
            np.multiply(xcord, bright, prod)
            cx = np.sum(prod) / count
            np.multiply(ycord, bright, prod)
            cy = np.sum(prod) / count

            val = spheroStep(s, targetX, targetY, lastDirs, cx, cy, lastPositions, smallSize[0]/15)
            if val:
                (cd, sd, d, next, offset, speed) = val
                cv2.putText(overlaid, 'dir in camera frame: ' + str(int(d)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(overlaid, 'sphero dir: ' + str(int(sd)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(overlaid, 'dir to target: ' + str(int(cd)), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(overlaid, 'next: ' + str(int(next)), (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(overlaid, 'offset: ' + str(int(offset)), (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                drawVec(overlaid, d, 20, (250, 20), (174, 139, 103))
                drawVec(overlaid, sd, 20, (250, 50), (127, 109, 201))
                drawVec(overlaid, cd, 20, (250, 80), (174, 133, 130))
                drawVec(overlaid, next, 20, (250, 110), (158, 119, 175))

                drawVec(overlaid, d, 20, (cy, cx), (174, 139, 103))
                drawVec(overlaid, sd, 20, (cy, cx), (127, 109, 201))
                drawVec(overlaid, cd, 20, (cy, cx), (174, 133, 130))
                drawVec(overlaid, next, 20, (cy, cx), (158, 119, 175))

                drawSparkLine(overlaid, errors, (300, 0),
                              (10, 170), 40, (255, 255, 255), (0, 255, 0))
                cv2.putText(overlaid, 'error: ' + str(int(error)), (100, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            l = len(lastPositions)
            sx, sy = lastPositions[0]
            for i in range(l-1):
                (dx, dy) = lastPositions[i+1]
                cv2.line(overlaid, (sy, sx), (dy, dx), (0, 0, (255*i/l)), 1)
                sx, sy = dx, dy

            overlaid[cx-1:cx+1,cy-1,:] = red
            overlaid[cx-1:cx+1,cy,:] = red
            overlaid[cx-1:cx+1,cy+1,:] = red


        frames += 1
        now = datetime.datetime.now()
        diff = now - last
        if diff.seconds >= 1:
            micro = float((diff.seconds * 1000000) + diff.microseconds) / 1000000.0
            fps = (float(frames) / micro)
            frames = 0
            last = now

        cv2.putText(overlaid, "{0:.1f}".format(fps), (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow('overlay', overlaid)

        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break

    cv2.destroyAllWindows()
