#!/usr/bin/env python

import numpy as np
import cv2
from time import clock
import sys
import video

import sphero

s = sphero.Sphero()

def connect():
    print("connect sphero")
    try:
        s.connect()
    except:
        print("err!")
        s.close()

    print( """Bluetooth info:name: %s \nbta: %s """ %
           (s.get_bluetooth_info().name, s.get_bluetooth_info().bta))


if __name__ == '__main__':

    try: fn = sys.argv[1]
    except: fn = 0

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

    print xcord, ycord

    while True:
        flag, frame = cam.read()
        small = cv2.pyrDown(frame)
        cv2.cvtColor(small, cv2.COLOR_BGR2HSV, hsv)
        values = hsv[:,:,2]

        cv2.imshow('hsv', values)

        bright.fill(0)
        mask = (values > 240)
        bright[mask] = 1
        count = np.sum(mask)

        cv2.imshow('bright', np.multiply(bright, 255))

        if small.shape[0] > 0:  # meaning that this is a valid frame
            np.multiply(xcord, bright, prod)
            cx = (np.sum(prod) / count) + (values.shape[0] / 2)
            np.multiply(ycord, bright, prod)
            cy = (np.sum(prod) / count) + (values.shape[1] / 2)

            print 'center', cx, cy

            dot.fill(0)
            dot[cx-1, cy-1] = 255
            dot[cx, cy-1] = 255
            dot[cx+1, cy-1] = 255
            dot[cx-1, cy] = 255
            dot[cx, cy] = 255
            dot[cx+1, cy] = 255
            dot[cx-1, cy+1] = 255
            dot[cx, cy+1] = 255
            dot[cx+1, cy+1] = 255
            cv2.imshow('dot', dot)

        if (abs(cx - (bright.shape[0] / 2)) > 50):
            if (cx < (bright.shape[0] / 2)):
                s.roll(0x0F, 270)
            else:
                s.roll(0x0F, 90)
        elif (abs(cy - (bright.shape[1] / 2)) > 50):
                if (cy < (bright.shape[1] / 2)):
                    s.roll(0x0F, 0)
                else:
                    s.roll(0x0F, 180)
            
        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break
    cv2.destroyAllWindows()
