# Author: Caelen Wang <wangc21@uw.edu>

import time

import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera

import haar_cascade_classifier

def main():
    face_cascade_file = 'haarcascade_frontalface_default.xml'
    face_detector = haar_cascade_classifier.HaarCascadeClassifier( \
        face_cascade_file)
    
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32
    rawCapture = PiRGBArray(camera, size=(640, 480))
    time.sleep(0.1)
    
    print("Press Q to quit")
    for frame in camera.capture_continuous(rawCapture, format = "bgr", use_video_port = True):
        frame = np.asarray(frame.array)
        
        faces = face_detector.classify(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0))
            
        cv2.imshow('frame', frame)
        rawCapture.truncate(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
