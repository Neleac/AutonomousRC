# Author: Brian Lee <joonl4@uw.edu>

import csv
import curses
import cv2
import numpy as np
#from picamera.array import PiRGBArray
#from picamera import PiCamera
from pwm_controller import PWMController
    

def main(stdscr):
    frame_count = -1
    cap = cv2.VideoCapture(1)
    csv_file = open('/home/nvidia/Desktop/rc_data/labels.csv', mode='w')
    csv_writer = csv.writer(csv_file, delimiter=',')

    control = PWMController()
    throttle = 0.0
    angle = 0.0

    stdscr.clear()
    stdscr.nodelay(True)
    while True:
        frame_count += 1
        ret, frame = cap.read()
        cv2.imwrite('/home/nvidia/Desktop/rc_data/images/' + str(frame_count) + '.jpg', frame)
        csv_writer.writerow([throttle, angle])

        txt = stdscr.getch()
        if (txt == ord('q')):
            control.steer(0)
            control.drive(0)
            break
        
        if (txt == curses.KEY_UP and throttle < 0.5):
            throttle += 0.1
        if (txt == curses.KEY_DOWN and throttle > 0):        
            throttle -= 0.1
        if (txt == curses.KEY_LEFT and angle < 1):
            angle += 0.1
        if (txt == curses.KEY_RIGHT and angle > -1):
            angle -= 0.1
        control.steer(angle)
        control.drive(throttle)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    curses.wrapper(main)
