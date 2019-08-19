# Author: Caelen Wang <wangc21@uw.edu>

import cv2
import numpy as np

class HaarCascadeClassifier:
    # params: cascade_file - trained haar cascade classifier (.xml)
    #         scale_factor - ratio between each image scale
    #         min_neighbors - number of neighbors required to retain detection
    # effects: instantiates cascade classifier
    def __init__(self, cascade_file, scale_factor = 1.3, min_neighbors = 4):
        self.cascade_classifier = cv2.CascadeClassifier(cascade_file)
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

    # params: image - numpy array of input image
    # returns: classifier output detections, list of rectangles (x, y, w, h)
    def classify(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        output = self.cascade_classifier.detectMultiScale( \
            image, scaleFactor = self.scale_factor, minNeighbors = self.min_neighbors)
        return output
