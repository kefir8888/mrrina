from modalities.modality import  Modality
from modalities.skeleton_modality import  Skeleton
from collections import deque
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

import pyrealsense2 as rs
import math

from modalities.modality import  Modality

class RealSense (Modality):
    def __init__ (self, video_path_ = "", model_path_ = "", mode_ = "GPU", base_height_ = 512, logger_ = 0):
        self.logger = logger_
        self.read_data        = []
        self.interpreted_data = []
        self.kpt_names        = ['nose', 'neck', 'r_sho', 'r_elb', 'r_wri', 'l_sho',
                                'l_elb', 'l_wri', 'r_hip', 'r_knee', 'r_ank', 'l_hip',
                                'l_knee', 'l_ank', 'r_eye', 'l_eye', 'r_ear', 'l_ear']
        self.necessary_keypoints_names = ["l_sho", "l_elb", "l_wri", "l_hip", "r_sho", "r_elb", "r_wri", "r_hip", "neck", "nose", 'r_eye', 'l_eye', "r_ear", "l_ear"]
        self.kps_mean = {kp : {"x": deque(maxlen = 25),"y": deque(maxlen = 25),"z": deque(maxlen = 25)} for kp in self.necessary_keypoints_names}

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
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.align = rs.align(rs.stream.color)

        if (video_path_ == ""):
            self.all_data = rs.pipeline()
            self.profile = self.all_data.start(self.config)

        self.read = False

        self.net = InferenceEnginePyTorch (model_path_, mode_)


    def name(self):
        return "RealSense"
    def _infer_net (self, frame):
        stride = 8

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

        return x

    def _read_data (self):
        if (self.read == False):
            frames = self.all_data.wait_for_frames()
            aligned_frames = self.align.process(frames)
            self.img = np.asanyarray(aligned_frames.get_color_frame().get_data())
            depth_img = np.asanyarray(aligned_frames.get_depth_frame().get_data())

            xy_cords = []
            xy_cords = self._infer_net(self.img).astype(int)

            kps = {}
            if xy_cords != []:


                for kp in self.necessary_keypoints_names:
                    ind = self.kpt_names.index(kp)
                    if xy_cords[ind * 2 + 1] >= 720:
                        kps.update ({kp : [xy_cords[ind * 2], xy_cords[ind * 2 + 1],  0]})
                    if xy_cords[ind * 2] != 0 and xy_cords[ind * 2 + 1] != 0 and xy_cords[ind * 2 + 1] <= 720:
                        kps.update ({kp : [xy_cords[ind * 2], xy_cords[ind * 2 + 1],  int(depth_img[xy_cords[ind * 2 + 1], xy_cords[ind * 2]])]})
                    self.kps_mean[kp]["x"].append(kps[kp][0])
                    self.kps_mean[kp]["y"].append(kps[kp][1])
                    self.kps_mean[kp]["z"].append(kps[kp][2])

            self.read_data = kps


    def get_mean(self, dict_, ind):
        return int(np.mean(np.asarray(dict_[ind])))

    def get_mean_cords(self, kps_raw):
        kps = {}
        for kp in self.necessary_keypoints_names:
            ind = self.kpt_names.index(kp)
            kps.update ({kp : [self.get_mean(kps_raw[kp], "x"), self.get_mean(kps_raw[kp], "y"),  self.get_mean(kps_raw[kp], "z")]})
        return kps

    def _process_data(self):

        kps = self.get_mean_cords(self.kps_mean)
        hips_mid = [int((kps ["r_hip"][0] + kps ["l_hip"][0]) / 2), int((kps ["r_hip"][1] + kps ["l_hip"][1]) / 2), int((kps ["r_hip"][2] + kps ["l_hip"][2]) / 2)]
#######################################################################################################        #
        hips_mid_neck = common.create_vec(hips_mid ,kps["neck"])
        neck_l_sho = common.create_vec(kps["neck"], kps["l_sho"])
        l_sho_elb = common.create_vec(kps["l_sho"], kps["l_elb"])
        l_elb_sho = common.create_vec(kps["l_elb"], kps["l_sho"])
        l_elb_wri = common.create_vec(kps["l_sho"], kps["l_wri"])
        l_elb_wri_mod = common.get_mod(l_elb_wri)

        N_256 = np.cross(neck_l_sho, l_sho_elb)
        N_256_mod = common.get_mod(N_256)

        N_125 = np.cross(neck_l_sho, hips_mid_neck)

        R_lut = np.cross( hips_mid_neck, N_125)
        R_lut_mod = common.get_mod(R_lut)

        R_lut_56 = np.cross(R_lut, l_sho_elb)
        R_lut_56_mod = common.get_mod(R_lut_56)

        thetha_lsp = math.acos(np.dot(hips_mid_neck, R_lut_56)/(R_lut_56_mod*common.get_mod(hips_mid_neck)))
        thetha_lsr = 1.57 - math.acos(np.dot(l_sho_elb, R_lut)/common.get_mod(l_sho_elb)*R_lut_mod)

        phi_lsp = math.acos(np.dot(l_sho_elb, hips_mid_neck)/(common.get_mod(l_sho_elb)*common.get_mod(hips_mid_neck)))

        if  phi_lsp <= 1.57:
            thetha_lsp = -abs(thetha_lsp)
        else:
            thetha_lsp = abs(thetha_lsp)



        N_256 = np.cross(l_elb_sho, l_elb_wri)
        R_lua = np.cross(l_elb_sho, N_256)
        R_lua_mod = common.get_mod(R_lua)

        V_6567 = np.cross(l_elb_sho, l_elb_wri)
        V_6567_mod = common.get_mod(V_6567)

        thetha_ley = math.acos(np.dot(N_256, V_6567)/(N_256_mod*V_6567_mod))
        phi_ley_1 = math.acos(np.dot(l_elb_wri, N_256)/(l_elb_wri_mod * N_256_mod))
        phi_ley_2 = math.acos(np.dot(l_elb_wri, R_lua)/(l_elb_wri_mod * R_lua_mod))

        if phi_ley_1 <= 1.57:
            thetha_ley = -abs(thetha_ley)
        elif phi_ley_1 > 1.57 and phi_ley_2 > 1.57:
            thetha_ley = abs(thetha_ley)
        else:
            thetha_ley = abs(thetha_ley) - 3.14


        thetha_ler = 3.14 - math.acos(np.dot(l_elb_wri, l_elb_sho)/(l_elb_wri_mod*common.get_mod(l_sho_elb)))

        self.logger.update("THETA", round(thetha_lsp, 2))
        self.logger.update("THETA r", round(thetha_lsr, 2))
        self.logger.update("THETA yell wrist", round(thetha_ley, 2))
        self.logger.update("THETA roll wrist", round(thetha_ler, 2))
#########################################################################################################################3
            # kps.update ({kp : (self.read_data [ind * 2], self.read_data [ind * 2 + 1])})


            # self.processed_data ["righthand"] = roll
            # self.processed_data ["lefthand"]  = pitch
            # self.processed_data ["leftshoulder_pitch" ] = yz
            # self.processed_data ["leftarm"]   = - common.angle_2_vec (sh_l_elb, elb_l_wri)
            # self.processed_data ["rightarm"]  = common.angle_2_vec (sh_r_elb, elb_r_wri)

            # #self.processed_data ["leftleg"] = -abs(common.angle_2_vec (neck_hip, sh_r_elb))
            # self.processed_data ["rightshoulder_pitch"] = pitch



    def _interpret_data(self):
        self.interpreted_data = self.processed_data

    def _get_command(self):
        commands = []

        if (self.timeout.timeout_passed ()):
            for key in self.processed_data.keys():
                commands.append(("/set_joint_angle", [key, str(self.processed_data[key])]))

        else:
            commands.append (("noaction", [""]))

        return commands

    def get_command(self, skip_reading_data=False):
        if (skip_reading_data == False):
            self._read_data()

        self._process_data()
        self._interpret_data()
        return self._get_command()

    def draw(self, img):
        return [self.frame]
