# Author: Brian Lee <joonl4@uw.edu>

import time, os
import curses
#import cv2
import numpy as np
#from picamera.array import PiRGBArray
#from picamera import PiCamera
from pwm_controller import PWMController

#def add_c(message):
    
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
    main()

