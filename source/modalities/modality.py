import numpy as np
import common
from collections import deque
import cv2
from test.inference_engine_pytorch import InferenceEnginePyTorch
from test.parse_poses import parse_poses
from test.draw import draw_poses
from test.draw import Plotter3d
import os
import json



class Modality:
    def __init__ (self, logger_=0):
        self.timeout = common.Timeout_module (0.35)
        self.logger = logger_
        self.read_data        = []
        self.interpreted_data = []

        self.processed_data = {"r_sho_roll"  : 0,
                               "r_sho_pitch" : 0,
                               "r_elb_roll"  : 0,
                               "r_elb_yaw"   : 0,

                               "r_hip_roll"  : 0,
                               "r_hip_pitch" : 0,

                               "r_knee_pitch": 0,
                               "r_ank_pitch" : 0,
                               "r_ank_roll"  : 0,

                               "l_sho_roll" : 0,
                               "l_sho_pitch": 0,
                               "l_elb_roll" : 0,
                               "l_elb_yaw"  : 0,

                               "l_hip_roll"  : 0,
                               "l_hip_pitch" : 0,

                               "l_knee_pitch": 0,
                               "l_ank_pitch" : 0,
                               "l_ank_roll"  : 0,

                               "head_Yaw"    : 0,
                               "head_Pitch"  : 0}

    def name (self):
        return "not specified"

    def draw (self, img):
        return [np.array ((1, 1, 1), np.uint8)]

class WorkWithPoints(Modality):
    def __init__ (self, logger_=0, maxlen_ = 25):
        Modality.__init__(self, logger_)
        self.necessary_keypoints_names = ["l_sho", "l_elb", "l_wri", "l_hip", "r_sho", "r_elb", "r_wri", "r_hip","neck",'mid_hip',  "nose", 'r_eye', 'l_eye', "r_ear"]
        maxlen__ = 25
        self.kps_mean = {kp : {"x": deque(maxlen = maxlen__),"y": deque(maxlen = maxlen__),"z": deque(maxlen = maxlen__)} for kp in self.necessary_keypoints_names}
        self.angles_mean    = {"r_sho_roll"  : deque(maxlen = maxlen_),
                               "r_sho_pitch" : deque(maxlen = maxlen_),
                               "r_elb_roll"  : deque(maxlen = maxlen_),
                               "r_elb_yaw"   : deque(maxlen = maxlen_),

                               "r_hip_roll"  : deque(maxlen = maxlen_),
                               "r_hip_pitch" : deque(maxlen = maxlen_),

                               "r_knee_pitch": deque(maxlen = maxlen_),
                               "r_ank_pitch" : deque(maxlen = maxlen_),
                               "r_ank_roll"  : deque(maxlen = maxlen_),

                               "l_sho_roll" : deque(maxlen = maxlen_),
                               "l_sho_pitch": deque(maxlen = maxlen_),
                               "l_elb_roll" : deque(maxlen = maxlen_),
                               "l_elb_yaw"  : deque(maxlen = maxlen_),

                               "l_hip_roll"  : deque(maxlen = maxlen_),
                               "l_hip_pitch" : deque(maxlen = maxlen_),

                               "l_knee_pitch": deque(maxlen = maxlen_),
                               "l_ank_pitch" : deque(maxlen = maxlen_),
                               "l_ank_roll"  : deque(maxlen = maxlen_),

                               "head_Yaw"    : deque(maxlen = maxlen_),
                               "head_Pitch"  : deque(maxlen = maxlen_)}


        self.kpt_names   = ['neck', 'nose', 'mid_hip',
                 'l_sho', 'l_elb',
                 'l_wri', 'l_hip',
                 'l_knee', 'l_ank',
                 'r_sho', 'r_elb',
                 'r_wri', 'r_hip',
                 'r_knee', 'r_ank',
                 'r_eye', 'l_eye',
                 'r_ear', 'l_ear']

    def get_mean(self, dict_):
        return np.mean(np.asarray(dict_))

    def get_mean_cords(self, kps_raw):
        kps = {}
        for kp in self.necessary_keypoints_names:
            ind = self.kpt_names.index(kp)
            kps.update ({kp : [int(self.get_mean(kps_raw[kp]["x"])), int(self.get_mean(kps_raw[kp]["y"])),  int(self.get_mean(kps_raw[kp]["z"]))]})
        return kps

    # def name (self):
    #     return "not specified"

    def read_data_from_file (self, path):
        with open(path, 'r') as file:
            data = file.read()
            cleared_data = ''
            for let in data:
                if let.isdigit() or let == '-':
                    cleared_data+=let
                else:
                    cleared_data+=','

            cleared_data = cleared_data.split(',')
            data = [int(i) for i in cleared_data if i]
            data = np.asarray(data)
            data = data.reshape(-1,36)

        return data
    #
    # def draw (self, img):
    #     return [np.array ((1, 1, 1), np.uint8)]


class GetPoints(Modality):
    def __init__ (self, logger_=0, model_path_="", mode_="GPU", base_height_=512, focal_length = -1):
        Modality.__init__(self, logger_)
        self.base_height = base_height_
        self.net = InferenceEnginePyTorch (model_path_, mode_)
        self.stride = 8
        self.fx = focal_length


        file_path = os.path.join('data', 'extrinsics.json')
        with open(file_path, 'r') as f:
            extrinsics = json.load(f)
        self.R = np.array(extrinsics['R'], dtype=np.float32)
        self.t = np.array(extrinsics['t'], dtype=np.float32)


    def _infer_net (self, frame):
        current_time = cv2.getTickCount()
        mean_time = 0
        input_scale = self.base_height / frame.shape[0]
        scaled_img = cv2.resize(frame, dsize=None, fx=input_scale, fy=input_scale)
        scaled_img = scaled_img[:,0:scaled_img.shape[1] - (scaled_img.shape[1] % self.stride)]
        if self.fx < 0:
            self.fx = np.float32(0.8 * frame.shape[1])
        inference_result = self.net.infer(scaled_img)
        poses_3d, poses_2d = parse_poses(inference_result, input_scale, self.stride, self.fx, True)

        def rotate_poses(poses_3d, R, t):
            R_inv = np.linalg.inv(R)
            for pose_id in range(len(poses_3d)):
                pose_3d = poses_3d[pose_id].reshape((-1, 4)).transpose()
                pose_3d[0:3, :] = np.dot(R_inv, pose_3d[0:3, :] - t)
                poses_3d[pose_id] = pose_3d.transpose().reshape(-1)

            return poses_3d

        if len(poses_3d):
            poses_3d = rotate_poses(poses_3d, self.R, self.t)
            poses_3d_copy = poses_3d.copy()
            x = poses_3d_copy[:, 0::4]
            y = poses_3d_copy[:, 1::4]
            z = poses_3d_copy[:, 2::4]
            poses_3d[:, 0::4], poses_3d[:, 1::4], poses_3d[:, 2::4] = -z, x, -y
            poses_3d = poses_3d.reshape(poses_3d.shape[0], 19, -1)[:, :, 0:3]
            edges = (Plotter3d.SKELETON_EDGES + 19 * np.arange(1).reshape((-1, 1, 1))).reshape((-1, 2))


        x = draw_poses(frame, poses_2d)
        current_time = (cv2.getTickCount() - current_time) / cv2.getTickFrequency()
        if mean_time == 0:
            mean_time = current_time
        else:
            mean_time = mean_time * 0.95 + current_time * 0.05
        cv2.putText(frame, 'FPS: {}'.format(int(1 / mean_time * 10) / 10),
                    (40, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
        self.frame = frame
        return x, poses_3d[0], edges


    #
    # def name (self):
    #     return "not specified"
    #
    # def draw (self, img):
    #     return [np.array ((1, 1, 1), np.uint8)]
