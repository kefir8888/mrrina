from modalities.modality import  Modality
from modalities.skeleton_modality import  Skeleton

import numpy as np
import common
import torch
from skel_proc import  get_skel_coords
from modules.load_state import load_state
from test.with_mobilenet import PoseEstimationWithMobileNet
from models.with_mobilenet_ import PoseEstimationWithMobileNet_
import io

#from test.demo import draww
from argparse import ArgumentParser
import json
import os
from test.input_reader import VideoReader, ImageReader
from test.draw import Plotter3d, draw_poses
from test.parse_poses import parse_poses
from test.inference_engine_pytorch import InferenceEnginePyTorch

import pydub
from pydub import AudioSegment
from pydub.playback import play
import scipy.fftpack
import cv2

class Motion_source:
    def __init__ (self):
        pass

    def get_motion (self, time):
        return np.zeros (18, np.float32)

#from skeleton_modalities import smth
#class Cyclic
#class Markov_chain
#class Rhytmic_sine

#class Archieve_data
#class Archieve_data_format1
#class Archieve_data_format2

#class External_model_loader

class Music (Modality):
    def __init__ (self, music_path_ = "", logger_ = 0):
        self.logger = logger_

        self.tick = 0

        self.commands = {"noaction": [("noaction", [""])],
                         "1": [("/stand", [""])],
                         "2": [("/left_shoulder_up", [""])],
                         "3": [("/right_shoulder_up", [""])],
                         "4": [("/head_yes", [""])],
                         "5": [("/right_hand_front", [""])],
                         "6": [("/left_hand_left", [""])],
                         "7": [("/left_hand_front", [""])],
                         "8": [("/right_hand_right", [""])],
                         "9": [("/stand", [""])],
                         "10": [("/bend_right", [""])],
                         "11": [("/bend_left", [""])],
                         "12": [("/stand", [""])],
                         "13": [("/play_airplane_1", [""])],
                         "14": [("/play_airplane_2", [""])],
                         "15": [("/play_airplane_1", [""])],
                         "16": [("/hands_sides", [""])]
                         }

        rate, audio = self.read(music_path_)
        N = 2000
        an_part = audio[:2000, 1]
        x = np.linspace(0, 2 * np.pi, N)
        #print("sh", an_part.shape)

        w = scipy.fftpack.rfft(an_part)
        f = scipy.fftpack.rfftfreq(N, x[1] - x[0])
        spectrum = w ** 2

        cutoff_idx = spectrum > (spectrum.max() / 15)
        w2 = w.copy()
        w2[cutoff_idx] = 0

        #print(f[1])

        self.timeout = common.Timeout_module(1 / f[1] / 8)

        #song = AudioSegment.from_mp3 (music_path_)
        #play (song)

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

    def name(self):
        return "Dance generation with audio input"

    def _read_data (self):
        pass

    def _process_data(self):
        pass

    def _interpret_data(self):
        pass

    def _get_command(self):
        comm = self.commands ["noaction"]

        if (self.timeout.timeout_passed ()):
            l     = len (self.commands)

            comm = self.commands[str (np.random.randint (1, l))]
            self.tick += 1

        #print ("com", comm)

        return comm

    def get_command(self, skip_reading_data=False):
        self._read_data()
        self._process_data()
        self._interpret_data()

        return self._get_command()
