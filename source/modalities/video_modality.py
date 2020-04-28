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

class Video (Modality):
    def __init__ (self, video_path_ = "", model_path_ = "", mode_ = "GPU", base_height_ = 512, logger_ = 0):
        self.logger = logger_

        self.read_data        = []
        self.interpreted_data = []
        self.poses_3d         = []
        self.timeout = common.Timeout_module (0.35)
        self.base_height = base_height_

        self.dataframe_num = 0

        self.processed_data = {"righthand" : 0,
                               "rightshoulder_pitch"   : 0,
                               "rightarm"  : 0,
                               "lefthand"  : 0,
                               "leftarm"   : 0,
                               "nose_x"    : 0,
                               "nose_x"    : 0,
                               "leftshoulder_pitch"  : 0,
                               "leftarm_yaw"         : 0,
                               "rightarm_yaw"        : 0}
        # if video_path_ != '':

        #get_available_cameras()
        #self.available_cameras = get_available_cameras(upper_bound=10, lower_bound=0)
        # self.all_data = cv2.VideoCapture("/home/kompaso/DEBUG/Debug/remote control/data/video/testt.mp4")#self.available_cameras[-1])

        if (video_path_ == ""):
            # self.all_data = cv2.VideoCapture("/home/kompaso/Desktop/hand_high.mp4")
            self.all_data = cv2.VideoCapture(0)#self.available_cameras[-1]) #/home/kompaso/Desktop/hand_high.mp4

        self.skel = Skeleton(logger_ = self.logger)

        self.read = False

        # self.net = PoseEstimationWithMobileNet()
        # checkpoint = torch.load("models/checkpoint_iter_370000.pth", map_location=torch.device('cpu'))
        # load_state(self.net, checkpoint)

        if (model_path_ == ""):
            model_path_ = "/home/kompaso/NAO_PROJECT/wenhai/source/test/human-pose-estimation-3d.pth"

        self.net = InferenceEnginePyTorch (model_path_, mode_)

    def name(self):
        return "video"

    def _infer_net (self, frame):
        stride = 8

        # net = InferenceEnginePyTorch("/home/kompaso/DEBUG/Debug/remote control/source/test/human-pose-estimation-3d.pth", "GPU")
        #net = InferenceEnginePyTorch(
        #    "/Users/elijah/Dropbox/Programming/RoboCup/remote control/source/test/human-pose-estimation-3d.pth",
        #    "GPU")

        canvas_3d = np.zeros((720, 1280, 3), dtype=np.uint8)
        plotter = Plotter3d(canvas_3d.shape[:2])
        canvas_3d_window_name = 'Canvas 3D'
        #cv2.namedWindow(canvas_3d_window_name)
        #cv2.setMouseCallback(canvas_3d_window_name, Plotter3d.mouse_callback)

        file_path = None
        if file_path is None:
            file_path = os.path.join('data', 'extrinsics.json')
        with open(file_path, 'r') as f:
            extrinsics = json.load(f)
        R = np.array(extrinsics['R'], dtype=np.float32)
        t = np.array(extrinsics['t'], dtype=np.float32)

        # frame_provider = ImageReader(args.images)
        # is_video = False
        # if args.video != '':
        #     frame_provider = VideoReader(args.video)
        is_video = True
        base_height = self.base_height
        fx = -1

        delay = 1
        esc_code = 27
        p_code = 112
        space_code = 32
        mean_time = 0
        # for frame in frame_provider:
        current_time = cv2.getTickCount()

        input_scale = base_height / frame.shape[0]
        scaled_img = cv2.resize(frame, dsize=None, fx=input_scale, fy=input_scale)
        scaled_img = scaled_img[:,
                     0:scaled_img.shape[1] - (scaled_img.shape[1] % stride)]  # better to pad, but cut out for demo
        if fx < 0:  # Focal length is unknown
            fx = np.float32(0.8 * frame.shape[1])
        # print(scaled_img.shape)
        inference_result = self.net.infer(scaled_img)
        poses_3d, poses_2d = parse_poses(inference_result, input_scale, stride, fx, is_video)
        edges = []

        def rotate_poses(poses_3d, R, t):
            R_inv = np.linalg.inv(R)
            for pose_id in range(len(poses_3d)):
                pose_3d = poses_3d[pose_id].reshape((-1, 4)).transpose()
                pose_3d[0:3, :] = np.dot(R_inv, pose_3d[0:3, :] - t)
                poses_3d[pose_id] = pose_3d.transpose().reshape(-1)

            return poses_3d

        if len(poses_3d):
            poses_3d = rotate_poses(poses_3d, R, t)
            poses_3d_copy = poses_3d.copy()
            x = poses_3d_copy[:, 0::4]
            y = poses_3d_copy[:, 1::4]
            z = poses_3d_copy[:, 2::4]
            poses_3d[:, 0::4], poses_3d[:, 1::4], poses_3d[:, 2::4] = -z, x, -y

            poses_3d = poses_3d.reshape(poses_3d.shape[0], 19, -1)[:, :, 0:3]
            edges = (Plotter3d.SKELETON_EDGES + 19 * np.arange(poses_3d.shape[0]).reshape((-1, 1, 1))).reshape(
                (-1, 2))
        # print("Играем с позой 3д",poses_3d[0].astype(int))
        plotter.plot(canvas_3d, poses_3d, edges)
        #cv2.imshow(canvas_3d_window_name, canvas_3d)
        self.canvas_3d = canvas_3d


        x = draw_poses(frame, poses_2d)
        # print(x)
        current_time = (cv2.getTickCount() - current_time) / cv2.getTickFrequency()
        if mean_time == 0:
            mean_time = current_time
        else:
            mean_time = mean_time * 0.95 + current_time * 0.05
        cv2.putText(frame, 'FPS: {}'.format(int(1 / mean_time * 10) / 10),
                    (40, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
        #cv2.imshow('ICV 3D Human Pose Estimation', frame)
        self.frame = frame
        self.poses_3d = poses_3d

        return x, poses_3d

    def _read_data (self):
        # if (self.dataframe_num >= len (self.all_data)):
        #     read_data = 0
        #     return
        # self.frame_skel = run_demo(self.all_data)

        if (self.read == False):
            _, img = self.all_data.read()
            self.img = img
            #self.read = True

        # self.read_data = get_skel_coords(self.net, img, 50, True, 1, 1) #self.all_data [self.dataframe_num]
        # print(get_skel_coords(self.net, img, 50, True, 1, 1))
        self.read_data, _ = self._infer_net (self.img) #draww(self.img)

        # self.dataframe_num += 1

    def _process_data(self):
        if sum (self.read_data) != -36 and self.read_data != []:
            # print ("hehm", self.read_data)
            self.skel.read_data = self.read_data
            self.skel.poses_3d = self.poses_3d
            self.skel._process_data(self.frame)
            self.processed_data = self.skel.processed_data


            # return self.processed_data
            # print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", self.processed_data)

        # else:
        #     return

    def _interpret_data(self):
        self.interpreted_data = self.processed_data

    def _get_command(self):
        commands = []

        if (self.timeout.timeout_passed ()):
            for key in self.processed_data.keys():
                commands.append(("/set_joint_angle", [key, str(self.processed_data[key])]))

            # print ("app joints", commands)

        else:
            commands.append (("noaction", [""]))

        # print ("COMMANDS: ", commands)
        return commands

    def get_command(self, skip_reading_data=False):
        if (skip_reading_data == False):
            self._read_data()

        self._process_data()
        self._interpret_data()
        #print("МЯК", self._get_command())
        return self._get_command()

    def draw(self, img):
        return [self.canvas_3d, self.frame]
