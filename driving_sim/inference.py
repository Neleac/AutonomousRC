# Author: Caelen Wang <wangc21@uw.edu>

import sys
sys.path.append("..")

import csv
import cv2
import numpy as np
#import pyrealsense2 as rs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from model import CaeLeNet, Darknet, DarkLSTM
#from pwm_controller import PWMController

def main(model_path, video_path = None, depth_path = None):
    csv_file = open('/home/wangc21/Desktop/lstm.csv', mode='w')
    csv_writer = csv.writer(csv_file, delimiter=',')

    if video_path:
        vid_cap = cv2.VideoCapture(video_path)
        depth_cap = cv2.VideoCapture(depth_path)
    else:
        pipeline = rs.pipeline()
        pipeline.start()

    device = torch.device("cuda")
    model = DarkLSTM().to(device)
    model.load_model(model_path)

    #control = PWMController()
    throttle = 0.0
    angle = 0.0

    while True:
        if video_path:
            _, img = vid_cap.read()
            _, depth = depth_cap.read()
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        else:
            frame = pipeline.wait_for_frames()
            img = frame.get_color_frame().as_frame().get_data()
            img = np.asanyarray(img)
            depth = frame.get_depth_frame().as_frame().get_data()
            depth = np.asanyarray(depth)
            depth = cv2.resize(depth, (640, 480), interpolation=cv2.INTER_AREA)
        
        img = transforms.ToTensor()(img)
        depth = transforms.ToTensor()(depth)
        input_data = torch.cat((img, depth)).unsqueeze(0).to(device)
        output = model(input_data)
        csv_writer.writerow([output[0][0].item(), output[0][1].item()])
    
    if video_path:
        vid_cap.release()
        depth_cap.release()
    else:
        pipeline.stop()

if __name__ == '__main__':
    model_path = '/home/wangc21/datasets/ARC/right_loop/lstm/033.pt'
    video_path = '/home/wangc21/datasets/ARC/right_loop/val/video.mp4'
    depth_path = '/home/wangc21/datasets/ARC/right_loop/val/depth.mp4'
    main(model_path, video_path, depth_path)
