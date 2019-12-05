# Author: Caelen Wang <wangc21@uw.edu>

import torch
import torch.nn as nn
import torch.nn.functional as F

import pt_util

class CaeLeNet(nn.Module):
    def __init__(self):
        super(CaeLeNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv3 = nn.Conv2d(128, 256, 3, 1, 1)
        self.fc1 = nn.Linear(80*60*256, 100)
        self.fc2_1 = nn.Linear(100, 1)
        self.fc2_2 = nn.Linear(100, 1)
        self.drop = nn.Dropout(0.1)
        
        self.lowest_error = float("inf")
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 80*60*256)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        throttle = self.fc2_1(x)
        angle = self.fc2_2(x)
        return torch.cat((throttle, angle), 1)
        
    def loss(self, prediction, label, reduction='mean'):
        loss = F.mse_loss(prediction, label, reduction = reduction)
        return loss
    
    def save_model(self, file_path, num_to_keep=1):
        pt_util.save(self, file_path, num_to_keep)
        
    def save_best_model(self, error, file_path, num_to_keep=1):
        if error < self.lowest_error:
          self.lowest_error = error
          pt_util.save(self, file_path, num_to_keep)

    def load_model(self, file_path):
        pt_util.restore(self, file_path)

    def load_last_model(self, dir_path):
        return pt_util.restore_latest(self, dir_path)


class Darknet(nn.Module):
    def __init__(self):
        super(Darknet, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv6 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv7 = nn.Conv2d(512, 1024, 3, 1, 1)
        self.fc_1 = nn.Linear(5*3*1024, 1)
        self.fc_2 = nn.Linear(5*3*1024, 1)
        
        self.lowest_error = float("inf")
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv6(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv7(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5*3*1024)
        throttle = self.fc_1(x)
        angle = self.fc_2(x)
        return torch.cat((throttle, angle), 1)
        
    def loss(self, prediction, label, reduction='mean'):
        loss = F.mse_loss(prediction, label, reduction = reduction)
        return loss
    
    def save_model(self, file_path, num_to_keep=1):
        pt_util.save(self, file_path, num_to_keep)
        
    def save_best_model(self, error, file_path, num_to_keep=1):
        if error < self.lowest_error:
          self.lowest_error = error
          pt_util.save(self, file_path, num_to_keep)

    def load_model(self, file_path):
        pt_util.restore(self, file_path)

    def load_last_model(self, dir_path):
        return pt_util.restore_latest(self, dir_path)


class DarkLSTM(nn.Module):
    def __init__(self):
        super(DarkLSTM, self).__init__()
        # convolutional encoder
        self.conv1 = nn.Conv2d(4, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1)
        
        # recurrent cell
        self.lstm = nn.LSTM(
            input_size = 20 * 15 * 256,
            hidden_size = 256,
            num_layers = 3,
            batch_first = True,
            dropout = 0.1
        )
        
        # linear decoder
        self.fc_1 = nn.Linear(256, 1)
        self.fc_2 = nn.Linear(256, 1)
        
        self.lowest_error = float("inf")
    
    # x is (batch_size, seq_len, (input_size))
    def forward(self, x, hidden_state = None):
        x = x.squeeze(0)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.unsqueeze(0).reshape((BATCH_SIZE, SEQ_LEN, 20 * 15 * 256))
        
        x, hidden_state = self.lstm(x, hidden_state)
        throttle = self.fc_1(x)
        angle = self.fc_2(x)
        return torch.cat((throttle, angle), 2), hidden_state

    def loss(self, prediction, label, reduction='mean'):
        loss = F.mse_loss(prediction, label, reduction = reduction)
        return loss
    
    def save_model(self, file_path, num_to_keep=1):
        pt_util.save(self, file_path, num_to_keep)
        
    def save_best_model(self, error, file_path, num_to_keep=1):
        if error < self.lowest_error:
          self.lowest_error = error
          pt_util.save(self, file_path, num_to_keep)

    def load_model(self, file_path):
        pt_util.restore(self, file_path)

    def load_last_model(self, dir_path):
        return pt_util.restore_latest(self, dir_path)