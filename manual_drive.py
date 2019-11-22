# Author: Brian Lee <joonl4@uw.edu>

import time, os
import curses
#import cv2
import numpy as np
#from picamera.array import PiRGBArray
#from picamera import PiCamera
from pwm_controller import PWMController
    

def main(stdscr):
    control = PWMController()
    throttle = 0.0
    angle = 0.0

    stdscr.clear()
    stdscr.nodelay(True)
    while True:
        txt = stdscr.getch()
        print(txt)
        if (txt == ord('q')):
            control.steer(0)
            control.drive(0)
            break
        
        if (txt == curses.KEY_UP):
            throttle += 0.1
        if (txt == curses.KEY_DOWN):        
            throttle -= 0.1
        if (txt == curses.KEY_LEFT):
            angle += 0.1
        if (txt == curses.KEY_RIGHT):
            angle -= 0.1
        angle = np.clip(angle, -1.0, 1.0)
        throttle = np.clip(throttle, -1.0, 0.25)
        control.steer(angle)
        control.drive(throttle)

if __name__ == '__main__':
    curses.wrapper(main)
