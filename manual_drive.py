# Author: Brian Lee <joonl4@uw.edu>

import os
import time
import cv2
import curses
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera

from pwm_controller import PWMController

  

def record(control):
    throttle = 0.0
    angle = 0.0
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32
    rawCapture = PiRGBArray(camera, size=(640, 480))
    win = curses.initscr()
    curses.noecho()
    curses.cbreak()
    win.keypad(True)
    for frame in camera.capture_continuous(rawCapture, format = "bgr", use_video_port = True):
        os.system('clear')
        print('SPEED %f' %(throttle))
        print('STEER %f' %(angle))
        txt = win.getch()
        frame = np.asarray(frame.array)
        if (txt == curses.KEY_UP):
            throttle += 0.05
        if (txt == curses.KEY_DOWN):        
            throttle -= 0.05
        if (txt == curses.KEY_LEFT):
            angle += 0.1
        if (txt == curses.KEY_RIGHT):
            angle -= 0.1
        angle = np.clip(angle, -1.0, 1.0)
        throttle = np.clip(throttle, -1.0, 0.25)
        control.steer(angle)
        control.drive(throttle)
        cv2.imshow('frame', frame)
        rawCapture.truncate(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            control.steer(0)
            control.drive(0)
            break
    cv2.destroyAllWindows()
    return None    



def main():
    throttle = 0.0
    angle = 0.0  
    control = PWMController()
    win = curses.initscr()
    curses.noecho()
    curses.cbreak()
    win.keypad(True)
    while True:
        os.system('clear')
        print('SPEED %f' %(throttle))
        print('STEER %f' %(angle))
        txt = win.getch()
        if (txt == ord('q')):
            control.steer(0)
            control.drive(0)
            break
        
        if (txt == ord('r')):
            print("Starting to record...")
            record(control); break
        
        if (txt == curses.KEY_UP):
            throttle += 0.05
        if (txt == curses.KEY_DOWN):        
            throttle -= 0.05
        if (txt == curses.KEY_LEFT):
            angle += 0.1
        if (txt == curses.KEY_RIGHT):
            angle -= 0.1
        angle = np.clip(angle, -1.0, 1.0)
        throttle = np.clip(throttle, -1.0, 0.25)
        control.steer(angle)
        control.drive(throttle)
        

if __name__ == '__main__':
    main()

