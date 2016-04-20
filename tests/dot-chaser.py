#!/usr/bin/env python

import numpy as np
import cv2
from time import clock
import sys
import video

import sphero

s = None

lastDirs = [0]
lastPositions = [[0, 0]]

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
    if degrees >= 360:
        return degrees - int(int(degrees) / 360) * 360
    if degrees < 0:
        return degrees + int(int(degrees) / 360) * -360
    return degrees

def dir(newPos, oldPos):
   newX = newPos[0]
   newY = newPos[1]
   oldX = oldPos[0]
   oldY = oldPos[1]

   diffX = newX - oldX
   diffY = newY - oldY

   rad = np.arctan2(diffX, diffY)
   return wrap(rad * 180.0 / np.pi)

def spheroStep(s, tx, ty, lastDirs, dotx, doty, lastPositions):
    if not s:
        return

    print 'step', tx, ty, lastDirs, dotx, doty, lastPositions

    if (abs(tx - dotx) < 50) and (abs(ty - doty) < 50):
        return

    if len(lastDirs) < 10:
        print 'early', lastDirs[-1]
        s.roll(0x0F, lastDirs[-1])
        lastDirs.append(lastDirs[-1])
        lastPositions.append([dotx, doty])
        return

    lastPositions.pop(0)
    lastDirs.pop(0)

    d = dir(lastPositions[-1], lastPositions[0])
    # the direction sphero moved in last 10 steps within the camera frame

    sd = wrap(sum(lastDirs) / len(lastDirs))
    # the direction that sphero thinks it went in last 10 frames

    cd = dir([dotx, doty], [tx, ty])
    # the direction sphero needs to go in the camera frame

    diff = d - cd
    if (diff > 180):
        next = wrap(lastDirs[-1] + 10)
    else:
        next = wrap(lastDirs[-1] - 10)

    print 'after', d, sd, cd, diff, next
    s.roll(0x0F, next)
    lastPositions.append([dotx, doty])
    lastDirs.append(next)

if __name__ == '__main__':

    try: fn = sys.argv[1]
    except: fn = 0

    try: noSphero = sys.argv[2]
    except: noSphero = ''

    doSphero = not noSphero

    if doSphero:
        s = sphero.Sphero()

    connect()

#    cam = cv2.VideoCapture(fn)
#    By reading video.py, the line above should be equivalent to the line below;  but somehow it does not create a valid camera object

    cam = video.create_capture(fn, fallback='synth:bg=../cpp/baboon.jpg:class=chess:noise=0.05')

    flag, frame = cam.read()

    small = cv2.pyrDown(frame)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    values = hsv[:,:,2]

    # we are still creating new arrays for 'small' and 'values' in the loop.

    xcord, ycord = np.indices(values.shape)
    xcord = np.subtract(xcord, (values.shape[0] / 2))
    ycord = np.subtract(ycord, (values.shape[1] / 2))
    bright = np.zeros(values.shape, values.dtype)
    prod = np.zeros(values.shape, np.int64)
    dot = np.zeros(values.shape, values.dtype)
    overlaid = np.zeros(small.shape, small.dtype)

    while True:
        flag, frame = cam.read()
        small = cv2.pyrDown(frame)

	cv2.imshow('rgb', small)

        cv2.cvtColor(small, cv2.COLOR_BGR2HSV, hsv)
        values = hsv[:,:,2]


        cv2.imshow('hsv', values)

        bright.fill(0)
        mask = (values > 240)
        bright[mask] = 1
        count = np.sum(mask)
	
	for i in range(3):
	    overlaid[:,:,i] = bright * 255

        cv2.imshow('bright', np.multiply(bright, 255))

        if small.shape[0] > 0:  # meaning that this is a valid frame
            np.multiply(xcord, bright, prod)
            cx = (np.sum(prod) / count) + (values.shape[0] / 2)
            np.multiply(ycord, bright, prod)
            cy = (np.sum(prod) / count) + (values.shape[1] / 2)
            
            red = [0,0,255]

            #dot.fill(0)
            overlaid[cx-1:cx+1,cy-1,:] = red
            overlaid[cx-1:cx+1,cy,:] = red
            overlaid[cx-1:cx+1,cy+1,:] = red
            
            cv2.imshow('overlay', overlaid)

        spheroStep(s, bright.shape[0] / 2, bright.shape[1] / 2, lastDirs, cx, cy, lastPositions)
            
        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break
    cv2.destroyAllWindows()
