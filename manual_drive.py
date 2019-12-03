# Authors: Brian Lee <joonl4@uw.edu>
#          Caelen Wang <wangc21@uw.edu>

import csv
import curses
import cv2
import numpy as np
import pyrealsense2 as rs
#from picamera.array import PiRGBArray
#from picamera import PiCamera
from pwm_controller import PWMController

DATA_PATH = '/home/nvidia/Desktop/rc_data/'

def main(stdscr):
    '''
    cap = cv2.VideoCapture(1)
    '''
    pipeline = rs.pipeline()
    pipeline.start()
    frame_count = -1
    csv_file = open(DATA_PATH + 'labels.csv', mode='w')
    csv_writer = csv.writer(csv_file, delimiter=',')

    control = PWMController()
    throttle = 0.0
    angle = 0.0

    stdscr.clear()
    stdscr.nodelay(True)
    while True:
        '''
        ret, frame = cap.read()
        '''
        frame = pipeline.wait_for_frames()
        img = frame.get_color_frame().as_frame().get_data()
        img = np.asanyarray(img)
        depth = frame.get_depth_frame().as_frame().get_data()
        depth = np.asanyarray(depth)
        depth = cv2.resize(depth, (640, 480), interpolation=cv2.INTER_AREA)

        frame_count += 1
        cv2.imwrite(DATA_PATH + 'images/' + str(frame_count) + '.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(DATA_PATH + 'depth/' + str(frame_count) + '.jpg', depth)
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
        if (txt == curses.KEY_RIGHT and angle < 1):
            angle += 0.1
        if (txt == curses.KEY_LEFT and angle > -1):
            angle -= 0.1
        control.steer(angle)
        control.drive(throttle)

    #cap.release()
    cv2.destroyAllWindows()
    pipeline.stop()

if __name__ == '__main__':
    curses.wrapper(main)
