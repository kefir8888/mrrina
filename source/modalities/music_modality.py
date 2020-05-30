from modalities.modality import  Modality

#from modalities.skeleton_modalities import Skeleton_3D_Music_to_dance
from modalities.modality import WorkWithPoints

import numpy as np

import math
import os
import sys
sys.path.append("..")
import common

import pydub
from pydub import AudioSegment
from pydub.playback import play
import scipy.fftpack
import cv2

import multiprocessing
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torchaudio
#import pandas as pd
import numpy as np

#class Motion_source:
#    def __init__ (self):
#        pass

#    def get_motion (self, time):
#        return np.zeros (18, np.float32)

#from skeleton_modalities import smth
#class Cyclic
#class Markov_chain
#class Rhytmic_sine

smol_listb = ["l_sho_roll", "l_elb_roll", "l_sho_pitch",
              "r_sho_roll", "r_elb_roll", "r_sho_pitch"]

class Archive_angles (Modality):
    def __init__ (self, angles_path_ = "", logger_ = 0):
        Modality.__init__(self, logger_)

        self.all_data = []
        self.angles_path = angles_path_
        self.dataframe_num = 0

        if (self.angles_path != ""):
            if( os.path.isfile (self.angles_path) == True ):
                print( "Angles file: ", self.angles_path)

                f = open (self.angles_path, )
                data = json.load (f)

                self.folder_path = self.angles_path [:-11]

                self.all_data = data ["angles"]

            else:
                print("\nNo angles file with name: ", self.angles_path)
                exit(0)

    def name (self):
        return "archive angles"

    def _read_data (self):
        if (self.dataframe_num >= len (self.all_data)):
            read_data = 0
            return

        self.read_data = self.all_data [self.dataframe_num]
        self.dataframe_num += 2

    def get_read_data (self):
        return self.read_data

    def _process_data (self, frame = None):
        self.processed_data = self.read_data

    def _interpret_data (self):
        self.interpreted_data = self.processed_data

    def _get_command (self):
        commands = []

        smol_dict = {}

        #for key in smol_listb:
        #    smol_dict.update ({key : self.processed_data [key]})

        for key, i in zip (smol_listb, range (len (smol_listb))):
            commands.append (("/set_joint_angle", [key, str (self.processed_data [i])]))

        return commands

    def get_command (self, skip_reading_data = False):
        if (skip_reading_data == False):
            self._read_data ()

        self._process_data   ()
        self._interpret_data ()

        return self._get_command ()

    def draw (self, canvas = np.ones ((700, 700, 3), np.uint8) * 220):
        result = canvas.copy ()

        cv2.putText (result, self.angles_path, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 50, 31), 1, cv2.LINE_AA)

        return [result]


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.hidden_size = 300
#
#         self.avgPool = nn.AvgPool1d(10)  # padding=1)
#         self.fc1 = nn.Linear(3200, self.hidden_size)
#         self.lstm = nn.LSTMCell(input_size=self.hidden_size, hidden_size=self.hidden_size)
#
#         self.hidden_state = torch.zeros(self.hidden_size)#.cuda()
#         self.cell_state = torch.zeros(self.hidden_size)#.cuda()
#
#         self.fc2 = nn.Linear(self.hidden_size, 6)
#         # ------------------------------------
#         self.lstm = torch.nn.LSTM(self.hidden_size, self.hidden_size)
#         # self.fc = torch.nn.Linear(hidden_dim,output_dim)
#         # self.bn = nn.BatchNorm1d(32)
#
#     def forward(self, x):
#         x = self.avgPool(x.view(1, 1, 32000))
#
#         x = self.fc1(x)
#         x = F.relu(x)
#         # self.hidden_state, self.cell_state = self.lstm (x.view(30, 1, -1), self.hidden_state.view(30, 1, -1))
#         # x = self.fc2 (self.hidden_state)
#
#         lstm_out, (hn, cn) = self.lstm(x.view(1, 1, self.hidden_size))
#         x = self.fc2(lstm_out[:, -1, :])
#
#         return x  # F.log_softmax(x, dim = 2)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.hidden_size = 300
        self.conv_channels = 1

        self.conv1 = nn.Conv1d(1, self.conv_channels, 10, 10)
        self.conv2 = nn.Conv1d(10, 1, 1)

        self.avgPool = nn.AvgPool1d(10)  # padding=1)

        self.fc1 = nn.Linear(6400, self.hidden_size)
        self.lstm = nn.LSTMCell(input_size=self.hidden_size, hidden_size=self.hidden_size)

        self.fc3 = nn.Linear(self.hidden_size, 6)

        self.lstm = torch.nn.LSTM(self.hidden_size, self.hidden_size)

    def forward(self, x):
        x = self.avgPool(x.view(1, 1, 64000))
        x = self.fc1(x)
        x = F.relu(x)

        lstm_out, (hn, cn) = self.lstm(x.view(1, self.conv_channels, self.hidden_size))
        x = F.relu(x)
        x = self.fc3(lstm_out[:, -1, :])

        return x

#
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv1d(1, 128, 80, 4)
#         self.bn1 = nn.BatchNorm1d(128)
#         self.pool1 = nn.MaxPool1d(4)
#         self.conv2 = nn.Conv1d(128, 128, 3)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.pool2 = nn.MaxPool1d(4)
#         self.conv3 = nn.Conv1d(128, 256, 3)
#         self.bn3 = nn.BatchNorm1d(256)
#         self.pool3 = nn.MaxPool1d(4)
#         self.conv4 = nn.Conv1d(256, 512, 3)
#         self.bn4 = nn.BatchNorm1d(512)
#         self.pool4 = nn.MaxPool1d(4)
#         self.avgPool = nn.AvgPool1d(30)  # input should be 512x30 so this outputs a 512x1
#
#         self.hidden_size = 512
#         self.out_size = 6
#         self.fc1 = nn.Linear(self.hidden_size, self.out_size)
#         self.conv5 = nn.Conv1d(2, 1, 1)
#
#         self.lstm = torch.nn.LSTM(self.hidden_size, self.hidden_size)
#
#     def forward(self, x):
#         x = self.conv1(x.view(1, 1, 64000))
#
#         #x = F.relu(self.bn1(x))
#         x = F.relu(x)
#
#         x = self.pool1(x)
#         x = self.conv2(x)
#         x = F.relu(self.bn2(x))
#         x = self.pool2(x)
#         x = self.conv3(x)
#         x = F.relu(self.bn3(x))
#         x = self.pool3(x)
#         x = self.conv4(x)
#         x = F.relu(self.bn4(x))
#         x = self.pool4(x)
#         x = self.avgPool(x)
#         x = x.permute(0, 2, 1)  # change the 512x1 to 1x512
#
#         x, (hn, cn) = self.lstm(x.view(1, 2, 512))
#
#         # print ("shape", x.shape)
#         x = self.fc1(x)
#
#         # print ("shape2", x.shape)
#         x = self.conv5(x)
#
#         # print ("shape3", x.shape)
#
#         return x[0, :]

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv1d(1, 128, 80, 4)
#         self.bn1 = nn.BatchNorm1d(128)
#         self.pool1 = nn.MaxPool1d(4)
#         self.conv2 = nn.Conv1d(128, 128, 3)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.pool2 = nn.MaxPool1d(4)
#         self.conv3 = nn.Conv1d(128, 256, 3)
#         self.bn3 = nn.BatchNorm1d(256)
#         self.pool3 = nn.MaxPool1d(4)
#         self.conv4 = nn.Conv1d(256, 512, 3)
#         self.bn4 = nn.BatchNorm1d(512)
#         self.pool4 = nn.MaxPool1d(4)
#         self.avgPool = nn.AvgPool1d(30)  # input should be 512x30 so this outputs a 512x1
#
#         self.before_lstm_sz = 512
#         self.lstm_sz = int(512 / 16)
#         self.hidden = 32
#
#         self.out_size = 6
#         self.fc1 = nn.Linear(int(self.before_lstm_sz / 16), self.lstm_sz)
#         self.fc2 = nn.Linear(self.lstm_sz, self.out_size)
#         self.conv5 = nn.Conv1d(2, 1, 1)
#
#         self.lstm = torch.nn.LSTM(self.lstm_sz, self.hidden)
#
#         self.pool5 = nn.MaxPool1d(16)
#
#     def forward(self, x):
#         x = self.conv1(x.view(1, 1, 64000))
#         # x = F.relu(self.bn1(x))
#         x = F.relu(x)
#         x = self.pool1(x)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = self.pool2(x)
#         x = self.conv3(x)
#         x = F.relu(x)
#         x = self.pool3(x)
#         x = self.conv4(x)
#         x = F.relu(x)
#         x = self.pool4(x)
#         x = self.avgPool(x)
#         x = x.permute(0, 2, 1)  # change the 512x1 to 1x512
#
#         x = self.pool5(x)
#
#         # x = self.fc1 (x)
#         # x = F.relu (x)
#
#         x, (hn, cn) = self.lstm(x.view(1, 2, self.lstm_sz))
#
#         # print ("shape", x.shape)
#         x = self.fc2(x)
#         x = F.relu(x)
#
#         # print ("shape2", x.shape)
#         x = self.conv5(x)
#
#         # print ("shape3", x.shape)
#
#         return x[0, :]

#Conditional LSTM SinGAN

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 128, 80, 4)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(128, 128, 3)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(128, 256, 3)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(256, 512, 3)
        self.bn4 = nn.BatchNorm1d(512)
        self.pool4 = nn.MaxPool1d(4)
        self.avgPool = nn.AvgPool1d(30)  # input should be 512x30 so this outputs a 512x1

        self.before_lstm_sz = 512
        self.lstm_sz = int(512 / 16)
        self.hidden = 32

        self.out_size = 6
        self.fc1 = nn.Linear(int(self.before_lstm_sz / 16), self.lstm_sz)
        self.fc2 = nn.Linear(self.lstm_sz, self.out_size)
        self.conv5 = nn.Conv1d(2, 1, 1)

        self.lstm = torch.nn.LSTM(self.lstm_sz, self.hidden)

        self.pool5 = nn.MaxPool1d(16)

        self.fc0 = nn.Linear(512, self.out_size)

    def forward(self, x):
        x = self.conv1(x.view(1, 1, 64000))
        # x = F.relu(self.bn1(x))
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        # x = F.relu(self.bn2(x))
        x = F.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        # x = F.relu(self.bn3(x))
        x = F.relu(x)
        x = self.pool3(x)
        x = self.conv4(x)
        # x = F.relu(self.bn4(x))
        x = F.relu(x)
        x = self.pool4(x)
        x = self.avgPool(x)
        x = x.permute(0, 2, 1)  # change the 512x1 to 1x512

        x = self.fc0(x)
        x = F.relu(x)
        x = self.conv5(x)

        return x[0, :]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 8, 80, 4)
        # self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(8, 8, 3)
        # self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(8, 16, 3)
        # self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(16, 32, 3)
        # self.bn4 = nn.BatchNorm1d(512)
        self.pool4 = nn.MaxPool1d(4)
        self.avgPool = nn.AvgPool1d(30)  # input should be 512x30 so this outputs a 512x1

        self.before_lstm_sz = 512
        self.lstm_sz = int(512 / 16)
        self.hidden = 32

        self.out_size = 6
        self.fc1 = nn.Linear(int(self.before_lstm_sz / 16), self.lstm_sz)
        self.fc2 = nn.Linear(self.lstm_sz, self.out_size)
        self.conv5 = nn.Conv1d(2, 1, 1)

        self.lstm = torch.nn.LSTM(self.lstm_sz, self.hidden)

        self.pool5 = nn.MaxPool1d(16)

        self.fc0 = nn.Linear(32, self.out_size)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv1(x.view(1, 1, 64000))
        # x = F.relu(self.bn1(x))
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        # x = F.relu(self.bn2(x))
        x = F.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        # x = F.relu(self.bn3(x))
        x = F.relu(x)
        x = self.pool3(x)
        x = self.conv4(x)
        # x = F.relu(self.bn4(x))
        x = F.relu(x)
        x = self.pool4(x)
        x = self.avgPool(x)
        x = x.permute(0, 2, 1)  # change the 512x1 to 1x512

        x = self.fc0(x)
        x = self.lrelu(x)
        x = self.conv5(x)

        return x[0, :]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv1d(1, 128, 80, 4)
        # #self.bn1 = nn.BatchNorm1d(128)
        # self.pool1 = nn.MaxPool1d(4)
        # self.conv2 = nn.Conv1d(128, 128, 3)
        # #self.bn2 = nn.BatchNorm1d(128)
        # self.pool2 = nn.MaxPool1d(4)
        # self.conv3 = nn.Conv1d(128, 256, 3)
        # #self.bn3 = nn.BatchNorm1d(256)
        # self.pool3 = nn.MaxPool1d(4)
        # self.conv4 = nn.Conv1d(256, 512, 3)
        # #self.bn4 = nn.BatchNorm1d(512)
        # self.pool4 = nn.MaxPool1d(4)
        # self.avgPool = nn.AvgPool1d(30) #input should be 512x30 so this outputs a 512x1

        self.conv1 = nn.Conv1d(1, 8, 80, 4)
        self.bn1 = nn.BatchNorm1d(8)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(8, 8, 3)
        self.bn2 = nn.BatchNorm1d(8)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(8, 16, 3)
        self.bn3 = nn.BatchNorm1d(16)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(16, 32, 3)
        self.bn4 = nn.BatchNorm1d(32)
        self.pool4 = nn.MaxPool1d(4)
        self.avgPool = nn.AvgPool1d(30)  # input should be 512x30 so this outputs a 512x1

        self.before_lstm_sz = 512
        self.lstm_sz = int(512 / 16)
        self.hidden = 32

        self.out_size = 6
        self.fc1 = nn.Linear(int(self.before_lstm_sz / 16), self.lstm_sz)
        self.fc2 = nn.Linear(self.lstm_sz, self.out_size)
        self.conv5 = nn.Conv1d(2, 1, 1)

        self.lstm = torch.nn.LSTM(self.lstm_sz, self.hidden)

        self.pool5 = nn.MaxPool1d(16)

        self.fc0 = nn.Linear(32, self.out_size)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv1(x.view(2, 1, 64000))
        x = F.relu(self.bn1(x))

        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))

        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))

        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))

        x = self.pool4(x)
        x = self.avgPool(x)
        x = x.permute(0, 2, 1)

        x = self.fc0(x)
        x = self.conv5(x)

        return x[:, 0, :]


class Net(nn.Module):
    def __init__(self, batch_size_):
        super(Net, self).__init__()

        self.batch_size = batch_size_

        # self.conv1 = nn.Conv1d(1, 128, 80, 4)
        # #self.bn1 = nn.BatchNorm1d(128)
        # self.pool1 = nn.MaxPool1d(4)
        # self.conv2 = nn.Conv1d(128, 128, 3)
        # #self.bn2 = nn.BatchNorm1d(128)
        # self.pool2 = nn.MaxPool1d(4)
        # self.conv3 = nn.Conv1d(128, 256, 3)
        # #self.bn3 = nn.BatchNorm1d(256)
        # self.pool3 = nn.MaxPool1d(4)
        # self.conv4 = nn.Conv1d(256, 512, 3)
        # #self.bn4 = nn.BatchNorm1d(512)
        # self.pool4 = nn.MaxPool1d(4)
        # self.avgPool = nn.AvgPool1d(30) #input should be 512x30 so this outputs a 512x1

        self.conv1 = nn.Conv1d(1, 8, 80, 4)
        self.bn1 = nn.BatchNorm1d(8)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(8, 8, 3)
        self.bn2 = nn.BatchNorm1d(8)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(8, 16, 3)
        self.bn3 = nn.BatchNorm1d(16)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(16, 32, 3)
        self.bn4 = nn.BatchNorm1d(32)
        self.pool4 = nn.MaxPool1d(4)
        self.avgPool = nn.AvgPool1d(30)  # input should be 512x30 so this outputs a 512x1

        self.before_lstm_sz = 512
        self.lstm_sz = int(512 / 16)
        self.hidden = 32

        self.out_size = 6
        self.fc1 = nn.Linear(int(self.before_lstm_sz / 16), self.lstm_sz)
        self.fc2 = nn.Linear(self.lstm_sz, self.out_size)
        self.conv5 = nn.Conv1d(2, 1, 1)

        self.lstm = torch.nn.LSTM(self.lstm_sz, self.hidden)

        self.pool5 = nn.MaxPool1d(16)

        self.fc0 = nn.Linear(32, self.out_size)
        self.lrelu = nn.LeakyReLU()

        # self.conv_channels = 10

        # self.conv1 = nn.Conv1d(1, self.conv_channels, 80, 4)
        # self.conv2 = nn.Conv1d(self.conv_channels, self.conv_channels * self.conv_channels, 4, 4)
        # #self.conv2 = nn.Conv1d(10, 1, 1)

        # #self.avgPool = nn.AvgPool1d(10)#padding=1)
        # self.pool1 = nn.MaxPool1d(4)
        # self.pool2 = nn.MaxPool1d(4)

        # self.fc1 = nn.Linear (100, self.hidden_size)

        # #self.lstm = nn.LSTMCell(input_size=self.hidden_size, hidden_size=self.hidden_size)
        # #self.lstm = nn.LSTMCell(input_size=6, hidden_size=self.hidden_size)

        # #self.hidden_state = torch.zeros(self.hidden_size).cuda()
        # #self.cell_state = torch.zeros(self.hidden_size).cuda()

        # #self.fc2 = nn.Linear (self.hidden_size, self.hidden_size)
        # self.fc3 = nn.Linear (self.hidden_size, 6)

        # ------------------------------------
        # self.fc = torch.nn.Linear(hidden_dim,output_dim)
        # self.bn = nn.BatchNorm1d(32)

    def forward(self, x):
        x = self.conv1(x.view(self.batch_size, 1, 64000))
        x = F.relu(self.bn1(x))

        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))

        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))

        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))

        x = self.pool4(x)
        x = self.avgPool(x)
        x = x.permute(0, 2, 1)

        x = self.fc0(x)
        x = self.conv5(x)

        return x[:, 0, :]

class Music_data:
    def __init__(self, music_path):
        self.music_path = music_path
        self.load_data()

    def load_data(self):
        sound = torchaudio.load(self.music_path, out=None, normalization=True)
        # load returns a tensor with the sound data and the sampling frequency (44.1kHz for UrbanSound8K)
        self.sound_data = sound[0][0, :]  # self.mixer(sound[0])

        # downsample the audio to ~8kHz
        # tempData = torch.zeros([160000, 1]) #tempData accounts for audio clips that are too short

        # if soundData.numel() < 160000:
        #    tempData[:soundData.numel()] = soundData[:]
        # else:
        #    tempData[:] = soundData[:160000]

        # soundData = tempData

    def get_sample(self, index):
        # sound_sample = torch.zeros([64000])
        #
        # lower_ind = index * 1764
        # upper_ind = index * 1764 + 320000
        #
        # sound_sample[:64000] = self.sound_data[lower_ind: upper_ind]  [::5] #take every fifth sample of soundData
        #
        # return torch.tensor(sound_sample)

        sound_sample = torch.zeros([32000])
        sound_sample_ = torch.zeros([64000])

        lower_ind = index * 1764
        upper_ind = index * 1764 + 320000

        sound_sample_[:64000] = self.sound_data[lower_ind: upper_ind][::5]

        #specgram = torchaudio.transforms.MelSpectrogram()(sound_sample_)
        #sound_sample[:32000] = specgram.reshape(-1)[:32000]

        sound_sample = torch.zeros([1, 3200])
        specgram = torchaudio.transforms.MelSpectrogram(n_mels=10)(sound_sample_)

        # print ("sound shape", sample.shape, specgram.shape)

        sound_sample[:3200] = specgram.reshape(-1)[:3200]

        return torch.tensor(sound_sample)

class Net(nn.Module):
    def __init__(self, batch_size_):
        super(Net, self).__init__()

        self.batch_size = batch_size_

        self.conv1 = nn.Conv1d(1, 8, 80, 4, padding=38)
        self.bn1 = nn.BatchNorm1d(8)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(8, 8, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(8)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(16)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1)
        self.bn4 = nn.BatchNorm1d(32)
        self.pool4 = nn.MaxPool1d(4)
        self.avgPool = nn.AvgPool1d(62)

        self.out_sz = 6

        self.lstm = torch.nn.LSTM(self.out_sz, self.out_sz)

        self.pool5 = nn.MaxPool1d(16)

        self.fc0 = nn.Linear(32, self.out_sz)
        self.lrelu = nn.LeakyReLU()

        self.lstm = nn.LSTMCell(input_size=self.out_sz, hidden_size=self.out_sz)

    def forward(self, x):
        x = self.conv1(x.view(self.batch_size, 1, 64000))
        x = F.relu(self.bn1(x))

        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))

        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))

        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))

        x = self.pool4(x)
        x = self.avgPool(x)

        x = x.permute(0, 2, 1)

        x = self.fc0(x)

        return x[:, 0, :]

class Net(nn.Module):
    def __init__(self, batch_size_):
        super(Net, self).__init__()

        self.batch_size = batch_size_

        # self.conv1 = nn.Conv1d(1, 128, 80, 4)
        # #self.bn1 = nn.BatchNorm1d(128)
        # self.pool1 = nn.MaxPool1d(4)
        # self.conv2 = nn.Conv1d(128, 128, 3)
        # #self.bn2 = nn.BatchNorm1d(128)
        # self.pool2 = nn.MaxPool1d(4)
        # self.conv3 = nn.Conv1d(128, 256, 3)
        # #self.bn3 = nn.BatchNorm1d(256)
        # self.pool3 = nn.MaxPool1d(4)
        # self.conv4 = nn.Conv1d(256, 512, 3)
        # #self.bn4 = nn.BatchNorm1d(512)
        # self.pool4 = nn.MaxPool1d(4)
        # self.avgPool = nn.AvgPool1d(30) #input should be 512x30 so this outputs a 512x1

        self.conv1 = nn.Conv1d(1, 8, 80, 4, padding=38)
        self.bn1 = nn.BatchNorm1d(8)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(8, 8, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(8)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(16)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1)
        self.bn4 = nn.BatchNorm1d(32)
        self.pool4 = nn.MaxPool1d(4)
        self.avgPool = nn.AvgPool1d(31)  # input should be 512x30 so this outputs a 512x1

        # self.before_lstm_sz = 512
        # self.lstm_sz = int (512 / 16)
        # self.hidden = 32

        self.out_sz = 6
        # self.fc1 = nn.Linear (int (self.before_lstm_sz / 16), self.lstm_sz)
        # self.fc2 = nn.Linear (self.lstm_sz, self.out_sz)
        # self.conv5 = nn.Conv1d (2, 1, 1)

        self.lstm = torch.nn.LSTM(self.out_sz, self.out_sz)

        self.pool5 = nn.MaxPool1d(16)

        self.fc0 = nn.Linear(32, self.out_sz)
        self.lrelu = nn.LeakyReLU()

        # self.conv_channels = 10

        # self.conv1 = nn.Conv1d(1, self.conv_channels, 80, 4)
        # self.conv2 = nn.Conv1d(self.conv_channels, self.conv_channels * self.conv_channels, 4, 4)
        # #self.conv2 = nn.Conv1d(10, 1, 1)

        # #self.avgPool = nn.AvgPool1d(10)#padding=1)
        # self.pool1 = nn.MaxPool1d(4)
        # self.pool2 = nn.MaxPool1d(4)

        # self.fc1 = nn.Linear (100, self.hidden_size)

        self.lstm = nn.LSTMCell(input_size=self.out_sz, hidden_size=self.out_sz)
        # #self.lstm = nn.LSTMCell(input_size=6, hidden_size=self.hidden_size)

        # #self.hidden_state = torch.zeros(self.hidden_size).cuda()
        # #self.cell_state = torch.zeros(self.hidden_size).cuda()

        # #self.fc2 = nn.Linear (self.hidden_size, self.hidden_size)
        # self.fc3 = nn.Linear (self.hidden_size, 6)

        # ------------------------------------
        # self.fc = torch.nn.Linear(hidden_dim,output_dim)
        # self.bn = nn.BatchNorm1d(32)

    def forward(self, x):
        # print ("shape0", x.shape)

        x = self.conv1(x.view(self.batch_size, 1, 32000))
        x = F.relu(self.bn1(x))
        # print ("shape1", x.shape)

        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        # print ("shape2", x.shape)

        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        # print ("shape3", x.shape)

        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        # print ("shape4", x.shape)

        x = self.pool4(x)
        # print ("shape5", x.shape)

        x = self.avgPool(x)

        # print ("shape6", x.shape)

        x = x.permute(0, 2, 1)
        # print ("shape7", x.shape)

        x = self.fc0(x)
        # print ("shape8", x.shape)

        # x = self.conv5 (x)

        # print ("shape9", x.shape)

        # x, (hn,cn) = self.lstm (x.view (self.batch_size, self.out_sz, 1))
        # x, (hn,cn) = self.lstm (x)

        # print ("shape10", x.shape)
        # print ("shape11", x [:, 0, :].shape)

        return x[:, 0, :]


class Net(nn.Module):
    def __init__(self, batch_size_):
        super(Net, self).__init__()

        self.batch_size = batch_size_

        # self.conv1 = nn.Conv1d(1, 128, 80, 4)
        # #self.bn1 = nn.BatchNorm1d(128)
        # self.pool1 = nn.MaxPool1d(4)
        # self.conv2 = nn.Conv1d(128, 128, 3)
        # #self.bn2 = nn.BatchNorm1d(128)
        # self.pool2 = nn.MaxPool1d(4)
        # self.conv3 = nn.Conv1d(128, 256, 3)
        # #self.bn3 = nn.BatchNorm1d(256)
        # self.pool3 = nn.MaxPool1d(4)
        # self.conv4 = nn.Conv1d(256, 512, 3)
        # #self.bn4 = nn.BatchNorm1d(512)
        # self.pool4 = nn.MaxPool1d(4)
        # self.avgPool = nn.AvgPool1d(30) #input should be 512x30 so this outputs a 512x1

        self.conv1 = nn.Conv1d(1, 8, 8, 4, padding=2)
        self.bn1 = nn.BatchNorm1d(8)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(8, 8, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(8)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(8, 16, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(16)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(16, 32, 3, padding=1)
        self.bn4 = nn.BatchNorm1d(32)
        self.pool4 = nn.MaxPool1d(4)
        self.avgPool = nn.AvgPool1d(3)  # input should be 512x30 so this outputs a 512x1

        # self.before_lstm_sz = 512
        # self.lstm_sz = int (512 / 16)
        # self.hidden = 32

        self.out_sz = 6
        # self.fc1 = nn.Linear (int (self.before_lstm_sz / 16), self.lstm_sz)
        # self.fc2 = nn.Linear (self.lstm_sz, self.out_sz)
        # self.conv5 = nn.Conv1d (2, 1, 1)

        self.lstm = torch.nn.LSTM(self.out_sz, self.out_sz)

        self.pool5 = nn.MaxPool1d(16)

        self.fc0 = nn.Linear(32, self.out_sz)
        self.lrelu = nn.LeakyReLU()

        # self.conv_channels = 10

        # self.conv1 = nn.Conv1d(1, self.conv_channels, 80, 4)
        # self.conv2 = nn.Conv1d(self.conv_channels, self.conv_channels * self.conv_channels, 4, 4)
        # #self.conv2 = nn.Conv1d(10, 1, 1)

        # #self.avgPool = nn.AvgPool1d(10)#padding=1)
        # self.pool1 = nn.MaxPool1d(4)
        # self.pool2 = nn.MaxPool1d(4)

        # self.fc1 = nn.Linear (100, self.hidden_size)

        self.lstm = nn.LSTMCell(input_size=self.out_sz, hidden_size=self.out_sz)
        # #self.lstm = nn.LSTMCell(input_size=6, hidden_size=self.hidden_size)

        # #self.hidden_state = torch.zeros(self.hidden_size).cuda()
        # #self.cell_state = torch.zeros(self.hidden_size).cuda()

        # #self.fc2 = nn.Linear (self.hidden_size, self.hidden_size)
        # self.fc3 = nn.Linear (self.hidden_size, 6)

        # ------------------------------------
        # self.fc = torch.nn.Linear(hidden_dim,output_dim)
        # self.bn = nn.BatchNorm1d(32)

    def forward(self, x):
        # print ("shape0", x.shape)

        x = self.conv1(x.view(self.batch_size, 1, 3200))
        x = F.relu(self.bn1(x))
        # print ("shape1", x.shape)

        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        # print ("shape2", x.shape)

        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        # print ("shape3", x.shape)

        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        # print ("shape4", x.shape)

        x = self.pool4(x)
        # print ("shape5", x.shape)

        x = self.avgPool(x)

        # print ("shape6", x.shape)

        x = x.permute(0, 2, 1)
        # print ("shape7", x.shape)

        x = self.fc0(x)
        # print ("shape8", x.shape)

        # x = self.conv5 (x)

        # print ("shape9", x.shape)

        # x, (hn,cn) = self.lstm (x.view (self.batch_size, self.out_sz, 1))
        # x, (hn,cn) = self.lstm (x)

        # print ("shape10", x.shape)
        # print ("shape11", x [:, 0, :].shape)

        return x[:, 0, :]


class Net(nn.Module):
    def __init__(self, batch_size_):
        super(Net, self).__init__()

        self.batch_size = batch_size_

        # self.conv1 = nn.Conv1d(1, 128, 80, 4)
        # #self.bn1 = nn.BatchNorm1d(128)
        # self.pool1 = nn.MaxPool1d(4)
        # self.conv2 = nn.Conv1d(128, 128, 3)
        # #self.bn2 = nn.BatchNorm1d(128)
        # self.pool2 = nn.MaxPool1d(4)
        # self.conv3 = nn.Conv1d(128, 256, 3)
        # #self.bn3 = nn.BatchNorm1d(256)
        # self.pool3 = nn.MaxPool1d(4)
        # self.conv4 = nn.Conv1d(256, 512, 3)
        # #self.bn4 = nn.BatchNorm1d(512)
        # self.pool4 = nn.MaxPool1d(4)
        # self.avgPool = nn.AvgPool1d(30) #input should be 512x30 so this outputs a 512x1

        self.conv1 = nn.Conv1d(1, 16, 80, 4, padding=38)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(16, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(16)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(16, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(32, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm1d(64)
        self.pool4 = nn.MaxPool1d(4)
        self.avgPool = nn.AvgPool1d(3)  # input should be 512x30 so this outputs a 512x1

        # self.before_lstm_sz = 512
        # self.lstm_sz = int (512 / 16)
        # self.hidden = 32

        self.out_sz = 6
        # self.fc1 = nn.Linear (int (self.before_lstm_sz / 16), self.lstm_sz)
        # self.fc2 = nn.Linear (self.lstm_sz, self.out_sz)
        # self.conv5 = nn.Conv1d (2, 1, 1)

        self.lstm = torch.nn.LSTM(self.out_sz, self.out_sz)

        self.pool5 = nn.MaxPool1d(16)

        self.fc0 = nn.Linear(64, self.out_sz)
        self.lrelu = nn.LeakyReLU()

        # self.conv_channels = 10

        # self.conv1 = nn.Conv1d(1, self.conv_channels, 80, 4)
        # self.conv2 = nn.Conv1d(self.conv_channels, self.conv_channels * self.conv_channels, 4, 4)
        # #self.conv2 = nn.Conv1d(10, 1, 1)

        # #self.avgPool = nn.AvgPool1d(10)#padding=1)
        # self.pool1 = nn.MaxPool1d(4)
        # self.pool2 = nn.MaxPool1d(4)

        # self.fc1 = nn.Linear (100, self.hidden_size)

        self.lstm = nn.LSTMCell(input_size=self.out_sz, hidden_size=self.out_sz)
        # #self.lstm = nn.LSTMCell(input_size=6, hidden_size=self.hidden_size)

        # #self.hidden_state = torch.zeros(self.hidden_size).cuda()
        # #self.cell_state = torch.zeros(self.hidden_size).cuda()

        # #self.fc2 = nn.Linear (self.hidden_size, self.hidden_size)
        # self.fc3 = nn.Linear (self.hidden_size, 6)

        # ------------------------------------
        # self.fc = torch.nn.Linear(hidden_dim,output_dim)
        # self.bn = nn.BatchNorm1d(32)

    def forward(self, x):
        # print ("shape0", x.shape)

        x = self.conv1(x.view(self.batch_size, 1, 3200))
        x = F.relu(self.bn1(x))
        # print ("shape1", x.shape)

        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        # print ("shape2", x.shape)

        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        # print ("shape3", x.shape)

        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        # print ("shape4", x.shape)

        x = self.pool4(x)
        # print ("shape5", x.shape)

        x = self.avgPool(x)

        # print ("shape6", x.shape)

        x = x.permute(0, 2, 1)
        # print ("shape7", x.shape)

        x = self.fc0(x)
        # print ("shape8", x.shape)

        # x = self.conv5 (x)

        # print ("shape9", x.shape)

        # x, (hn,cn) = self.lstm (x.view (self.batch_size, self.out_sz, 1))
        # x, (hn,cn) = self.lstm (x)

        # print ("shape10", x.shape)
        # print ("shape11", x [:, 0, :].shape)

        return x[:, 0, :]


class External_model (Modality):
    def __init__ (self, model_path_ = "", music_path_ = "", logger_ = 0):
        Modality.__init__(self, logger_)

        self.all_data = []

        self.model_path = model_path_
        self.music_path = music_path_
        self.music_data = Music_data (self.music_path)

        self.sample_num = 0

        self.processed_data_history = []

        if (self.model_path != ""):
            if( os.path.isfile (self.model_path) == True ):
                #print( "Model file: ", self.model_path)

                device = torch.device('cpu')
                self.model = Net (1)
                checkpoint = torch.load (self.model_path, map_location=device)
                #print ("fgfg", checkpoint.keys ())
                self.model.load_state_dict (checkpoint)
                #self.model.to (device)
                self.model.eval()

            else:
                print("\nNo angles file with name: ", data_path)
                exit(0)

    def name (self):
        return "archive angles"

    def _read_data (self):
        # if (self.dataframe_num >= len (self.all_data)):
        #     read_data = 0
        #     return

        #self.read_data = self.model (self.music_data.get_sample (self.sample_num))

        #sample = self.music_data.get_sample (self.sample_num)
        #feed = torch.stack ([sample, sample], dim=0)
        #output = self.model.forward (feed)

        sample = self.music_data.get_sample (self.sample_num)
        self.read_data = self.model.forward (sample)

        self.sample_num += 4

    def get_read_data (self):
        return self.read_data

    def _process_data (self, frame = None):
        self.processed_data = self.read_data.detach().cpu().numpy()
        #print ("pred", self.processed_data)

    def _interpret_data (self):
        self.processed_data_history.append (self.processed_data)

        self.interpreted_data = np.median (self.processed_data_history [-1: ], axis = 0)
        #print ("inter", self.interpreted_data)

    def _get_command (self):
        commands = []

        for key, i in zip (smol_listb, range (len (smol_listb))):
            commands.append (("/set_joint_angle", [key, str (self.interpreted_data [0, i])]))

        return commands

    def get_command (self, skip_reading_data = False):
        if (skip_reading_data == False):
            self._read_data ()

        self._process_data   ()
        self._interpret_data ()

        return self._get_command ()

    def draw (self, canvas = np.ones ((700, 700, 3), np.uint8) * 220):
        result = canvas.copy ()

        cv2.putText (result, self.model_path, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 50, 31), 1, cv2.LINE_AA)

        return [result]

class Music (Modality):
    def __init__ (self, music_path_ = "", logger_ = 0):
        self.logger = logger_
        self.music_path = music_path_

        self.tick = 0

        self.commands = {"noaction": [("noaction", [""])],
                         "0": [("/increment_joint_angle", ["l_sho_roll", "-0.11"])],
                         "1": [("/increment_joint_angle", ["l_sho_roll", "0.11"])]
                         }

        #self.rate, self.audio = self.read(music_path_)
        #self._extract_rhythm ()
        self.timeout = common.Timeout_module(1)# / self.rhythm / 8)

        #song = AudioSegment.from_mp3 (music_path_)
        #play (song)

    def play_song (self):
        pass

    def read(self, f, normalized=False):
        """MP3 to numpy array"""
        a = pydub.AudioSegment.from_mp3(f)
        y = np.array(a.get_array_of_samples())
        if a.channels == 2:
            y = y.reshape((-1, 2))
        if normalized:
            return a.frame_rate, np.float32(y) / 2 ** 15
        else:
            return a.frame_rate, y

    def write(self, f, sr, x, normalized=False):
        """numpy array to MP3"""
        channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
        if normalized:  # normalized array - each item should be a float in [-1, 1)
            y = np.int16(x * 2 ** 15)
        else:
            y = np.int16(x)
        song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
        song.export(f, format="mp3", bitrate="320k")

    def _extract_rhythm (self):
        N = 2000
        an_part = self.audio [:2000, 1]
        x = np.linspace (0, 2 * np.pi, N)

        w = scipy.fftpack.rfft (an_part)
        f = scipy.fftpack.rfftfreq (N, x[1] - x[0])
        spectrum = w**2

        cutoff_idx = spectrum > (spectrum.max () / 15)
        w2 = w.copy ()
        w2 [cutoff_idx] = 0

        print("len", w2)
        print("w", len(w2))

        self.rhythm = f [1]

    def name(self):
        return "Baseline dance generation with audio input"

    def _read_data (self):
        pass

    def _process_data(self):
        pass

    def _interpret_data(self):
        pass

    def _get_command(self):
        comm = self.commands ["noaction"]

        if (self.timeout.timeout_passed ()):
            l = len (self.commands)

            comm = self.commands[str (np.random.randint (1, l))]

            self.tick += 1

        return comm

    def get_command(self, skip_reading_data=False):
        self._read_data()
        self._process_data()
        self._interpret_data()

        return self._get_command()

    def draw (self, canvas = np.ones ((700, 700, 3), np.uint8) * 220):
        result = canvas.copy ()

        cv2.putText (result, self.music_path, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 50, 31), 1, cv2.LINE_AA)

        return [result]

class Cyclic (Music):
    def __init__ (self, music_path_ = "", logger_ = 0, dance_length_ = 50):
        Music.__init__ (self, music_path_, logger_)
        self.tick = 0
        self.dance_length = dance_length_

        #self.rate, self.audio = self.read (self.music_path)
        #self._extract_rhythm ()

        #self.timeout = Timeout_module (1 / self.rhythm / 8)
        self.timeout = common.Timeout_module (0.01)

        print ("timeout:", self.timeout)

        #song = AudioSegment.from_mp3 (music_path_)
        #play (song)

        hip_ampl  = 0.07
        head_ampl = 0.15

        head_pose_1 = [("/set_joint_angle", ["head_Pitch", str (-head_ampl)]),
                       ("/set_joint_angle", ["head_Yaw",   str (-head_ampl)])]

        head_pose_2 = [("/set_joint_angle", ["head_Pitch", str (-head_ampl)]),
                       ("/set_joint_angle", ["head_Yaw",   "0.0"])]

        head_pose_3 = [("/set_joint_angle", ["head_Pitch", str (-head_ampl)]),
                       ("/set_joint_angle", ["head_Yaw",   str (head_ampl)])]

        legs_pose_1 = [("/set_joint_angle", ["l_hip_pitch", "0.0"]),
                       ("/set_joint_angle", ["l_knee_pitch", "0.0"]),
                       ("/set_joint_angle", ["l_ank_pitch", "0.0"]),
                       ("/set_joint_angle", ["r_hip_pitch", "0.0"]),
                       ("/set_joint_angle", ["r_knee_pitch", "0.0"]),
                       ("/set_joint_angle", ["r_ank_pitch", "0.0"])]

        legs_pose_2 = [("/set_joint_angle", ["l_hip_pitch", str (-hip_ampl)]),
                       ("/set_joint_angle", ["l_knee_pitch", str (2 * hip_ampl)]),
                       ("/set_joint_angle", ["l_ank_pitch", str (-hip_ampl)]),
                       ("/set_joint_angle", ["r_hip_pitch", str (-hip_ampl)]),
                       ("/set_joint_angle", ["r_knee_pitch", str (2 * hip_ampl)]),
                       ("/set_joint_angle", ["r_ank_pitch", str (-hip_ampl)])]

        legs_pose_3 = [("/set_joint_angle", ["l_hip_pitch", str (0)]),
                       ("/set_joint_angle", ["l_knee_pitch", str (0)]),
                       ("/set_joint_angle", ["l_ank_pitch", str (0)]),
                       ("/set_joint_angle", ["r_hip_pitch", str (-hip_ampl)]),
                       ("/set_joint_angle", ["r_knee_pitch", str (2 * hip_ampl)]),
                       ("/set_joint_angle", ["r_ank_pitch", str (-hip_ampl)])]

        legs_pose_4 = [("/set_joint_angle", ["l_hip_pitch", str (-hip_ampl)]),
                       ("/set_joint_angle", ["l_knee_pitch", str (2 * hip_ampl)]),
                       ("/set_joint_angle", ["l_ank_pitch", str (-hip_ampl)]),
                       ("/set_joint_angle", ["r_hip_pitch", str (0)]),
                       ("/set_joint_angle", ["r_knee_pitch", str (0)]),
                       ("/set_joint_angle", ["r_ank_pitch", str (0)])]

        self.commands = {
                         "noaction": [("noaction", [""])],

                         "0": head_pose_1 + legs_pose_1,

                         "1": head_pose_2 + legs_pose_2,

                         "2": head_pose_3 + legs_pose_3,

                         "3": head_pose_2 + legs_pose_4
                         }

    def play_song (self):
        pass

    def name(self):
        return "Cyclic moves performing"

    def _read_data (self):
        pass

    def _process_data(self):
        pass

    def _interpret_data(self):
        pass

    def _get_command(self):
        comm = self.commands ["noaction"]

        if (self.tick >= self.dance_length):
            return comm

        if (self.timeout.timeout_passed ()):
            l = len (self.commands)

            regular_part = self.commands[str (self.tick % (l - 1))]

            # cyclic_angle_1 = math.sin (float (self.tick) / 3) / 4
            # cyclic_angle_2 = math.sin (float (self.tick + 1.5) / 3) / 4

            # unique_joints = {
            #                  "l_sho_roll" : cyclic_angle_1 + 0.4,
            #                  "l_elb_roll" : cyclic_angle_1 - 0.4,
            #                  "l_sho_pitch": cyclic_angle_1 - 0.4,
            #
            #                  "r_sho_roll" : cyclic_angle_1 - 0.4,
            #                  "r_elb_roll" : cyclic_angle_1 + 0.4,
            #                  "r_sho_pitch": cyclic_angle_2 - 0.4
            #                 }
            #
            # unique_part= []
            #
            # for unique_joint in unique_joints.keys ():
            #     unique_part += [("/set_joint_angle", [unique_joint, str (unique_joints [unique_joint])])]

            unique_part = []

            comm = regular_part + unique_part

            print (comm)

            self.tick += 1

        return comm

    def get_command(self, skip_reading_data=False):
        self._read_data()
        self._process_data()
        self._interpret_data()

        return self._get_command()

class Skeleton_3D_Music_to_dance (WorkWithPoints):
    def __init__ (self, skeleton_path_ = "", logger_ = 0):
        WorkWithPoints.__init__(self, logger_, maxlen_=20)
        self.all_data         = []
        self.dataframe_num = 0
        self.previous_knee = 0
        self.previous_hip = 0
        self.previous_ankl = 0
        self.mode = 0.0

        self.skeleton_path = skeleton_path_
        self.all_angles_data = []

        self.folder_path = self.skeleton_path[:-14]

        if (skeleton_path_ != ""):
            verbose = False
            if( os.path.isfile (self.skeleton_path) == True ):
                print( "Skeleton file: ", self.skeleton_path)

                #skeleton_data = open(skeleton_path_, 'r')
                #all_skeleton_frames = self.read_skeleton_data_from_NTU(skeleton_data, verbose )

                f = open (self.skeleton_path, )
                data = json.load (f)

                config_path = self.folder_path + "config.json"
                config = open (config_path, )
                config_data = json.load (config)

                start_frame = config_data ["start_position"]
                end_frame   = config_data ["end_position"]

                all_skeleton_frames = data["skeletons"] [start_frame : end_frame]

                self.all_data = all_skeleton_frames

            else:
                print("\nNo skeleton file with name: ", data_path)

    def __del__ (self):
        data = {}
        data ["angles"] = self.all_angles_data

        with open (self.folder_path + "angles.json", 'w') as outfile:
            json.dump (data, outfile)

    def name (self):
        return "skeleton"

    def _read_data (self):
        if (self.dataframe_num >= len (self.all_data)):
            self.end_of_data_reached = True
            read_data = 0
            return

        self.read_data = self.all_data [self.dataframe_num]
        self.dataframe_num += 1

    def create_dicts_with_coords_3D(self):
        kps = {}
        if self.read_data != []:
            for kp in self.necessary_keypoints_names:
                ind = self.kpt_names.index(kp)
                if kp == 'mid_hip':
                    if (kps["l_hip"][0] > 0 and kps["r_hip"][0] > 0):
                        kps.update ({kp : [(self.read_data[6][0] + self.read_data[12][0]) / 2, (self.read_data[6][1] +
                                            self.read_data[12][1]) / 2, (self.read_data[6][2] + self.read_data[12][2]) / 2]})
                    else:
                        kps.update ({kp : [self.read_data[0][0], self.read_data[0][1] + 200,  self.read_data[0][2]]})
                else:
                    kps.update ({kp : [self.read_data[ind][0], self.read_data[ind][1],  self.read_data[ind][2]]})

                self.kps_mean[kp]["x"].append(kps[kp][0])
                self.kps_mean[kp]["y"].append(kps[kp][1])
                self.kps_mean[kp]["z"].append(kps[kp][2])
        return kps

    def get_read_data (self):
        return self.read_data

    def _process_data (self, frame = None):
        # name = self.read_data
        # self.read_data = self.read_data

        self.interpreted_data = self.create_dicts_with_coords_3D()

        kps = self.get_mean_cords(self.kps_mean)

        # self.processed_data["mode"] = 1

        ##################################################left_full_hand##############################################################
        l_hip_neck = common.create_vec(kps["mid_hip"], kps["neck"])
        neck_l_sho = common.create_vec(kps["neck"], kps["l_sho"])
        l_elb_sho = common.create_vec(kps["l_elb"], kps["l_sho"])
        l_sho_elb = common.create_vec(kps["l_sho"], kps["l_elb"])
        l_elb_wri = common.create_vec(kps["l_elb"], kps["l_wri"])

        N_l_body_plane = np.cross(neck_l_sho, l_hip_neck)
        N_neck_l_sho_elb = np.cross(neck_l_sho, l_elb_sho)
        N_l_sho_elb_wri = np.cross(l_elb_sho, l_elb_wri)

        R_l_body_plane = np.cross(l_hip_neck, N_l_body_plane)
        R_l_arm = np.cross(l_elb_sho, N_neck_l_sho_elb)
        R_lbp_lse = np.cross(R_l_body_plane, l_sho_elb)

        mod_N_neck_l_sho_elb = common.get_mod(N_neck_l_sho_elb)
        mod_N_l_body_plane = common.get_mod(N_l_body_plane)
        mod_N_l_sho_elb_wri = common.get_mod(N_l_sho_elb_wri)
        mod_l_hip_neck = common.get_mod(l_hip_neck)
        mod_l_sho_elb = common.get_mod(l_sho_elb)
        mod_l_elb_sho = common.get_mod(l_elb_sho)
        mod_l_elb_wri = common.get_mod(l_elb_wri)
        mod_R_l_body_plane = common.get_mod(R_l_body_plane)
        mod_R_lbp_lse = common.get_mod(R_lbp_lse)
        mod_R_l_arm = common.get_mod(R_l_arm)

        l_sho_pitch_raw = math.acos(np.dot(l_hip_neck, R_lbp_lse) / (mod_l_hip_neck * mod_R_lbp_lse)) - 0.65
        l_elb_yaw_raw = math.acos(
            np.dot(N_neck_l_sho_elb, N_l_sho_elb_wri) / (mod_N_neck_l_sho_elb * mod_N_l_sho_elb_wri))

        phi_lsp = math.acos(np.dot(l_sho_elb, l_hip_neck) / (mod_l_hip_neck * mod_l_sho_elb))
        phi_ley_1 = math.acos(np.dot(l_elb_wri, N_neck_l_sho_elb) / (mod_l_elb_wri * mod_N_neck_l_sho_elb))
        phi_ley_2 = math.acos(np.dot(l_elb_wri, R_l_arm) / (mod_l_elb_wri * mod_R_l_arm))

        l_elb_yaw = 0
        if phi_ley_1 <= 1.57:
            l_elb_yaw = - l_elb_yaw_raw
        if phi_ley_1 > 1.57 and phi_ley_2 > 1.57:
            l_elb_yaw = l_elb_yaw_raw
        if phi_ley_1 > 1.57 and phi_ley_2 <= 1.57:
            l_elb_yaw = l_elb_yaw_raw - 6.28

        if phi_lsp <= 1.57:
            l_sho_pitch = -l_sho_pitch_raw
        else:
            l_sho_pitch = l_sho_pitch_raw

        l_sho_roll = 1.57 - math.acos(np.dot(l_sho_elb, R_l_body_plane) / (mod_l_sho_elb * mod_R_l_body_plane))
        l_elb_roll = -(3.14 - math.acos(np.dot(l_elb_wri, l_elb_sho) / (mod_l_elb_wri * mod_l_elb_sho)))

        #####################################################################################################################
        self.angles_mean["l_sho_pitch"].append(l_sho_pitch)
        self.angles_mean["l_sho_roll"].append(l_sho_roll)
        self.angles_mean["l_elb_yaw"].append(l_elb_yaw)
        self.angles_mean["l_elb_roll"].append(l_elb_roll)

        # self.logger.update("l shoul pitch", round(self.get_mean(self.angles_mean["l_sho_pitch"]), 2))
        # self.logger.update("l shoul roll", round(self.get_mean(self.angles_mean["l_sho_roll"]), 2))
        # self.logger.update("l elb yaw", round(self.get_mean(self.angles_mean["l_elb_yaw"]), 2))
        # self.logger.update("l elb roll", round(self.get_mean(self.angles_mean["l_elb_roll"]), 2))

        self.processed_data ["l_sho_pitch"]  = round(self.get_mean(self.angles_mean["l_sho_pitch"]), 2)
        self.processed_data ["l_sho_roll"]  = round(self.get_mean(self.angles_mean["l_sho_roll"]), 2)
        self.processed_data ["l_elb_yaw"]  = round(self.get_mean(self.angles_mean["l_elb_yaw"]), 2)
        self.processed_data ["l_elb_roll"]  = round(self.get_mean(self.angles_mean["l_elb_roll"]), 2)
        ##############################################################################################################################

        ##########################################r_full_hand###############################################################
        r_hip_neck = common.create_vec(kps["mid_hip"], kps["neck"])
        neck_r_sho = common.create_vec(kps["neck"], kps["r_sho"])
        r_elb_sho = common.create_vec(kps["r_elb"], kps["r_sho"])
        r_sho_elb = common.create_vec(kps["r_sho"], kps["r_elb"])
        r_elb_wri = common.create_vec(kps["r_elb"], kps["r_wri"])

        N_r_body_plane = -np.cross(neck_r_sho, r_hip_neck)
        N_neck_r_sho_elb = np.cross(neck_r_sho, r_elb_sho)
        N_r_sho_elb_wri = np.cross(r_elb_sho, r_elb_wri)

        R_r_body_plane = np.cross(r_hip_neck, N_r_body_plane)
        R_r_arm = np.cross(r_elb_sho, N_neck_r_sho_elb)
        R_rbp_rse = np.cross(R_r_body_plane, r_sho_elb)

        mod_N_neck_r_sho_elb = common.get_mod(N_neck_r_sho_elb)
        mod_N_r_sho_elb_wri = common.get_mod(N_r_sho_elb_wri)
        mod_r_hip_neck = common.get_mod(r_hip_neck)
        mod_r_sho_elb = common.get_mod(r_sho_elb)
        mod_r_elb_sho = common.get_mod(r_elb_sho)
        mod_r_elb_wri = common.get_mod(r_elb_wri)
        mod_R_rbp_rse = common.get_mod(R_rbp_rse)
        mod_R_r_body_plane = common.get_mod(R_r_body_plane)
        mod_R_r_arm = common.get_mod(R_r_arm)

        r_sho_pitch_raw = math.acos(np.dot(r_hip_neck, R_rbp_rse) / (mod_r_hip_neck * mod_R_rbp_rse)) - 0.65
        r_elb_yaw_raw = math.acos(
            np.dot(N_neck_r_sho_elb, N_r_sho_elb_wri) / (mod_N_neck_r_sho_elb * mod_N_r_sho_elb_wri))

        phi_rsp = math.acos(np.dot(r_sho_elb, r_hip_neck) / (mod_r_hip_neck * mod_r_sho_elb))
        phi_rey_1 = math.acos(np.dot(r_elb_wri, N_neck_r_sho_elb) / (mod_r_elb_wri * mod_N_neck_r_sho_elb))
        phi_rey_2 = math.acos(np.dot(r_elb_wri, R_r_arm) / (mod_r_elb_wri * mod_R_r_arm))

        r_elb_yaw = 0
        if phi_rey_1 <= 1.57:
            r_elb_yaw = r_elb_yaw_raw
        if phi_rey_1 > 1.57 and phi_rey_2 > 1.57:
            r_elb_yaw = -r_elb_yaw_raw
        if phi_rey_1 > 1.57 and phi_rey_2 <= 1.57:
            r_elb_yaw = r_elb_yaw_raw - 6.28

        if phi_rsp <= 1.57:
            r_sho_pitch = -r_sho_pitch_raw
        else:
            r_sho_pitch = r_sho_pitch_raw

        r_sho_roll = 1.57 - math.acos(np.dot(r_sho_elb, R_r_body_plane) / (mod_r_sho_elb * mod_R_r_body_plane))
        r_elb_roll = 3.14 - math.acos(np.dot(r_elb_wri, r_elb_sho) / (mod_r_elb_wri * mod_r_elb_sho))
        #####################################################################################################################
        self.angles_mean["r_sho_pitch"].append(r_sho_pitch)
        self.angles_mean["r_sho_roll"].append(r_sho_roll)
        self.angles_mean["r_elb_yaw"].append(r_elb_yaw)
        self.angles_mean["r_elb_roll"].append(r_elb_roll)

        # self.logger.update("r shoul pitch" , round(self.get_mean(self.angles_mean["r_sho_pitch"]), 2))
        # self.logger.update("r shoul roll", round(self.get_mean(self.angles_mean["r_sho_roll"]), 2))
        # self.logger.update("r elb yaw", round(self.get_mean(self.angles_mean["r_elb_yaw"]), 2))
        # self.logger.update("r elb roll", round(self.get_mean(self.angles_mean["r_elb_roll"]), 2))

        self.processed_data ["r_sho_pitch"] = round(self.get_mean(self.angles_mean["r_sho_pitch"]), 2)
        self.processed_data ["r_sho_roll"]  = round(self.get_mean(self.angles_mean["r_sho_roll"]), 2)
        self.processed_data ["r_elb_yaw"]   = round(self.get_mean(self.angles_mean["r_elb_yaw"]), 2)
        self.processed_data ["r_elb_roll"]  = round(self.get_mean(self.angles_mean["r_elb_roll"]), 2)
        #################################################################################################################################

    def _interpret_data (self):
        self.interpreted_data = self.processed_data

    def _get_command (self):
        commands = []

        #print ("keys:", self.processed_data.keys ())

        smol_dict = {}

        for key in smol_listb:
            smol_dict.update ({key : self.processed_data [key]})

        #for key in self.processed_data.keys ():
        for key in smol_dict.keys ():
            commands.append (("/set_joint_angle", [key, str (self.processed_data [key])]))

        self.all_angles_data.append ([self.processed_data [key] for key in smol_listb])

        return commands

    def get_command (self, skip_reading_data = False):
        if (skip_reading_data == False):
            self._read_data ()

        self._process_data   ()
        self._interpret_data ()

        return self._get_command ()

    def draw (self, canvas = np.ones ((700, 700, 3), np.uint8) * 220):
        result = canvas.copy ()

        cv2.putText (result, self.skeleton_path, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 50, 31), 1, cv2.LINE_AA)

        return [result]
