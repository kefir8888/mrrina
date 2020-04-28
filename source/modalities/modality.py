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

from sympy import Plane, Point3D, Line3D
import pyrealsense2 as rs
import math

# tracker = Value_tracker ()

class Modality:
    def __init__ (self, logger = 0):
        pass

    def name (self):
        return "not specified"

    def draw (self, img):
        return [np.array ((1, 1, 1), np.uint8)]
