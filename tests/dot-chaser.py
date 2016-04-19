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

    cam = video.create_capture(fn, fallback='synth:bg=../cpp/baboon.jpg:class=chess:noise=0.05')

    while True:
        flag, frame = cam.read()
        cv2.imshow('camera', frame)

        small = cv2.pyrDown(frame)

        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        values = hsv[:,:,2]

        cv2.imshow('hsv', values)

        bright = np.copy(values)
        bright.fill(0)

        mask = (values > 240)
        bright[mask] = 255

        sumx = 0
        sumy = 0
        count = 0
        for (x, y) in np.ndindex(bright.shape[0], bright.shape[1]):
            if mask[x, y]:
                sumx += x
                sumy += y
                count += 1

        cx = sumx / count
        cy = sumy / count

        print 'center', cx, cy

        cv2.imshow('bright', bright)
        
        dot = np.copy(values)
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
