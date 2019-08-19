# Author: Caelen Wang <wangc21@uw.edu>

import sys
import time
sys.path.append("..")

import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera

from haar_cascade_classifier import HaarCascadeClassifier
from pwm_controller import PWMController

def main():
    control = PWMController()
    
    face_cascade_file = 'haarcascade_frontalface_default.xml'
    face_detector = HaarCascadeClassifier( \
        face_cascade_file)
    
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32
    rawCapture = PiRGBArray(camera, size=(640, 480))
    time.sleep(0.1)
    
    #print("Press Q to quit")
    for frame in camera.capture_continuous(rawCapture, format = "bgr", use_video_port = True):
        frame = np.asarray(frame.array)
        
        faces = face_detector.classify(frame)
        max_face = (0, 0, 0, 0)
        for (x, y, w, h) in faces:
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0))
            
            # find largest face
            if w * h > max_face[2] * max_face[3]:
                max_face = (x, y, w, h)
        
        # follow largest face
        if max_face != (0, 0, 0, 0):
            # steer towards middle of face
            face_point = max_face[0] + int(max_face[2] / 2)
            angle = -float(face_point - 320) / 320
            control.steer(angle)
            
            # drive towards face
            face_size = max_face[2] * max_face[3]
            face_ratio = float(face_size) / (640 * 480)
            throttle = np.clip((-1/0.04) * face_ratio + 1, 0, 0.2)
            control.drive(throttle)
        else:
            #control.steer(0)
            control.drive(0)
            
        #cv2.imshow('frame', frame)
        rawCapture.truncate(0)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #break
    #cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
