# Author: Caelen Wang <wangc21@uw.edu>

import cv2
import numpy as np

import haar_cascade_classifier

def main():
    face_cascade_file = 'haarcascade_frontalface_default.xml'
    face_detector = haar_cascade_classifier.HaarCascadeClassifier( \
        face_cascade_file)
    
    cap = cv2.VideoCapture(0)
    print("Press Q to quit")
    while True:
        frame = cap.read()[1]
        faces = face_detector.classify(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0))
        
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
