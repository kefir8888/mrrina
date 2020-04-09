import numpy as np
import common
import torch
# from skel_proc import VideoReader, infer_fast, get_skel_coords
# from test.load_state import load_state
# from test.with_mobilenet import PoseEstimationWithMobileNet
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

from common import get_available_cameras

class Modality:
    def __init__ (self):
        pass

    def name (self):
        return "not specified"

    def draw (self, img):
        return [np.array ((1, 1, 1), np.uint8)]

class Computer_keyboard (Modality):
    def __init__ (self, phrases_path = "", key_to_command_ = {"z" : "empty"}):
        self.read_data        = 0x00
        self.processed_data   = 0x00
        self.interpreted_data = 0x00

        self.curr_mode = 0

        self.all_keys = ["w", "e", "r", "t", "y", "u", "i", "o", "p", "a", "s", "d", "f", "g", "h", "j", "k", "l", "v", "b", "m", "z", "x", "c", "n"]

        self.common_commands = {"z"        : [("/stand",    ["heh"])],
                                "c"        : [("/rest",     ["kek"])],
                                #"n"        : [("next",      [""])],
                                "x"        : [("/sit",      [""])],
                                "noaction" : [("noaction",  [""])],
                                "w"        : [("/play_mp3", ["Molodec.mp3"])],
                                "e"        : [("/play_mp3", ["Otlichnopoluchaetsja.mp3"])],
                                "r"        : [("/play_mp3", ["Zdorovo.mp3"])],
                                "t"        : [("/play_mp3", ["Zamechatelno.mp3"])],
                                "y"        : [("/play_mp3", ["Poluchilos.mp3"])],
                                "u"        : [("/play_mp3", ["Prekrasno.mp3"])],
                                "i"        : [("/play_mp3", ["Molodec.mp3"])],
                                "o"        : [("/play_mp3", ["Molodec.mp3"])],
                                "p"        : [("/play_mp3", ["Horosho.mp3"])],}

        self.repeating =   {"z"        : [("/stand",    ["heh"])],
                            "c"        : [("/rest",     ["kek"])],
                            "x"        : [("/sit",      [""])],
                            "noaction" : [("noaction",  [""])],
                            "w"        : [("/play_mp3", ["Molodec.mp3"])],
                            "e"        : [("/play_mp3", ["Povtorjajzamnoj.mp3"])],
                            "r"        : [("/play_mp3", ["Zdorovo.mp3"])],
                            "o"        : [("/bend_right", [""])],
                            "p"        : [("/bend_left", [""])],
                            "v"        : [("/hands_sides", [""])],
                            "b"        : [("/hands_front", [""])],
                            "n"        : [("/left_shoulder_up", [""])],
                            "m"        : [("/right_shoulder_up", [""])],

                            "l"        : [("/play_mp3", ["Poprobujeszeraz.mp3"])],

                            "f"        : [("/play_mp3", ["Posmotrinapravo.mp3"]),
                                          ("/play_airplane_1", [""])],
                            "g"        : [("/play_mp3", ["Posmotrinalevo.mp3"]),
                                          ("/play_airplane_2", [""])],
                            "h"        : [("/play_mp3", ["Posmotrivverh.mp3"]),
                                          ("/play_car", [""])],
                            "j"        : [("/left_hand_left", ["Horosho.mp3"])],
                            "k"        : [("/right_hand_right", ["Horosho.mp3"])],

                            "w"        : [("/walk_20", [""])],
                            "a"        : [("/rot_20", [""])],
                            "s"        : [("/rot_m20", [""])],
                            "d"        : [("/walk_m30", [""])],
                            "noaction" : [("noaction",  [""])]}

        self.repeating2 =  {"z"        : [("/stand",    ["heh"])],
                            "c"        : [("/rest",     ["kek"])],
                            "x"        : [("/sit",      [""])],
                            "noaction" : [("noaction",  [""])],
                            "w"        : [("/play_mp3", ["Molodec.mp3"])],
                            "e"        : [("/play_mp3", ["Povtorjajzamnoj.mp3"])],
                            "r"        : [("/play_mp3", ["Zdorovo.mp3"])],
                            "o"        : [("/right_hand_right", [""])],
                            "p"        : [("/right_hand_front", [""])],
                            "v"        : [("/left_hand_left", [""])],
                            "b"        : [("/left_hand_front", [""])],
                            "n"        : [("/right_hand_up", [""])],
                            "m"        : [("/left_hand_up", [""])],

                            "l"        : [("/play_mp3", ["Poprobujeszeraz.mp3"])],

                            "f"        : [("/play_mp3", ["Dajpjat.mp3"]),
                                          ("/right_hand_front", [""])],
                            "g"        : [("/play_mp3", ["Posmotrinalevo.mp3"]),
                                          ("/play_airplane_2", [""])],
                            "h"        : [("/play_mp3", ["Posmotrivverh.mp3"]),
                                          ("/play_car", [""])],
                            "j"        : [("/left_hand_left", ["Horosho.mp3"])],
                            "k"        : [("/right_hand_right", ["Horosho.mp3"])],

                            "w"        : [("/walk_20", [""])],
                            "a"        : [("/rot_20", [""])],
                            "s"        : [("/rot_m20", [""])],
                            "d"        : [("/walk_m30", [""])],
                            "noaction" : [("noaction",  [""])]}

        self.exceptional = {"z"        : [("/stand",    ["heh"])],
                                "c"        : [("/rest",     ["kek"])],
                                #"n"        : [("next",      [""])],
                                "x"        : [("/sit",      [""])],
                                "noaction" : [("noaction",  [""])],
                                "w"        : [("/play_mp3", ["Molodec.mp3"])],
                                "e"        : [("/play_mp3", ["Otlichnopoluchaetsja.mp3"])],
                                "r"        : [("/play_mp3", ["Zdorovo.mp3"])],
                                "t"        : [("/play_mp3", ["Zamechatelno.mp3"])],
                                "y"        : [("/play_mp3", ["Poluchilos.mp3"])],
                                "u"        : [("/play_mp3", ["Prekrasno.mp3"])],
                                "i"        : [("/play_mp3", ["Molodec.mp3"])],
                                "o"        : [("/play_mp3", ["Molodec.mp3"])],
                                "p"        : [("/play_mp3", ["Horosho.mp3"])],
                            "a" : [("/play_mp3", ["am_nyam.mp3"])],
                            "s" : [("/play_mp3", ["Privet.mp3"])],
                            "d" : [("/play_mp3", ["Privetik.mp3"])],
                            "f" : [("/play_mp3", ["Pokauvidimsja.mp3"])],
                            "g" : [("/play_mp3", ["Najdinakakujufigurupohozhe.mp3"])],
                            "h" : [("/stop", [""])],
                            "j" : [("/play_mp3", ["am_nyam.mp3"])],
                            "k" : [("/play_mp3", ["am_nyam.mp3"])],
                            "l" : [("/play_mp3", ["am_nyam.mp3"])],

                            "m"        : [("/play_mp3", ["Poprobujeszeraz.mp3"])],

                            "noaction" : [("noaction",  [""])]}

        self.eyes = {"z" : [("/stand",    ["heh"])],
                     "c" : [("/rest",     ["kek"])],
                     "x" : [("/sit",      [""])],
                     "w" : [("/play_mp3", ["Molodec.mp3"])],
                     "e" : [("/play_mp3", ["Otlichnopoluchaetsja.mp3"])],
                     "r" : [("/play_mp3", ["Zdorovo.mp3"])],
                     "t" : [("/play_mp3", ["Zamechatelno.mp3"])],
                     "y" : [("/play_mp3", ["Poluchilos.mp3"])],
                     "u" : [("/play_mp3", ["Prekrasno.mp3"])],
                     "i" : [("/play_mp3", ["Molodec.mp3"])],
                     "o" : [("/play_mp3", ["Molodec.mp3"])],
                     "p" : [("/play_mp3", ["Horosho.mp3"])],
                     "a" : [("/red",    [""])],
                     "s" : [("/green",  [""])],
                     "d" : [("/blue",   [""])],
                     "f" : [("/orange", [""])],
                     "g" : [("/yellow", [""])],
                     "h" : [("/white",  [""])],
                     "j" : [("/lightblue", [""])],
                     "k" : [("/violet", [""])],
                     "l" : [("/play_mp3", ["Posmotrinaglazainajdikarti.mp3"])],

                     "n"        : [("/play_mp3", ["Poprobujeszeraz.mp3"])],

                     "noaction" : [("noaction",  [""])]}

        self.direct_control =  {"z"        : [("/stand",   ["heh"])],
                                "c"        : [("/rest",    ["kek"])],
                                "w"        : [("/increment_joint_angle", ["lefthand", "0.21"])],
                                "e"        : [("/increment_joint_angle", ["lefthand", "-0.21"])],
                                "r"        : [("/increment_joint_angle", ["leftarm", "0.21"])],
                                "t"        : [("/increment_joint_angle", ["leftarm", "-0.21"])],
                                "s"        : [("/increment_joint_angle", ["righthand", "0.21"])],
                                "d"        : [("/increment_joint_angle", ["righthand", "-0.21"])],
                                "f"        : [("/increment_joint_angle", ["rightarm", "0.21"])],
                                "g"        : [("/increment_joint_angle", ["rightarm", "-0.21"])],
                                "p"        : [("/increment_joint_angle", ["nose_x", "0.1"])],
                                "l"        : [("/increment_joint_angle", ["nose_x", "-0.1"])],
                                "o"        : [("/increment_joint_angle", ["nose_y", "0.1"])],
                                "k"        : [("/increment_joint_angle", ["nose_y", "-0.1"])],
                                "n"        : [("next",     [""])],
                                "noaction" : [("noaction", [""])]}


        self.key_to_command = []

        self.key_to_command.append(self.direct_control)
        self.key_to_command.append(self.exceptional)
        self.key_to_command.append (self.repeating)
        self.key_to_command.append (self.repeating2)
        self.key_to_command.append (self.eyes)

        if (phrases_path != ""):
            f = io.open (phrases_path, "r", encoding='utf-8')
            f1 = f.readlines()

            available_keys = [x for x in self.all_keys if x not in self.common_commands.keys ()]

            phrase_name = []

            for line in f1:
                out = common.rus_line_to_eng (line)
                filename = out [:26] + '.mp3'

                phrase_name.append ((line, filename))

            #print (phrase_name)

            new_list = self.common_commands.copy ()

            while (len (phrase_name) != 0):
                for key in available_keys:
                    if (len (phrase_name) == 0):
                        break

                    el = phrase_name [0]

                    new_list.update ({key : [("/play_mp3", [el [1], el [0]])]})

                    phrase_name.pop (0)

                self.key_to_command.append (new_list)
                new_list = self.common_commands.copy ()

        else:
            self.key_to_command = [self.common_commands]

        if (key_to_command_ ["z"] != "empty"):
            self.key_to_command = key_to_command_

    def name (self):
        return "computer keyboard"

    def _read_data (self):
        self.read_data = cv2.waitKey (1)

    def get_read_data (self):
        return self.read_data

    def _process_data (self):
        self.processed_data = self.read_data

    def _interpret_data (self):
        self.interpreted_data = self.processed_data

    def _get_command (self):
        if (self.interpreted_data >= 0):
            key = str (chr (self.interpreted_data))

            if (key in self.key_to_command [self.curr_mode].keys ()):
                return self.key_to_command [self.curr_mode] [key]

            if (key in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]):
                self.curr_mode = int (key) % len (self.key_to_command)
                #print ("curr mode ", self.curr_mode)

        return self.key_to_command [self.curr_mode] ["noaction"]

    def get_command (self, skip_reading_data = False):
        if (skip_reading_data == False):
            self._read_data ()

        self._process_data   ()
        self._interpret_data ()

        return self._get_command ()

    def draw (self, canvas = np.ones ((700, 700, 3), np.uint8) * 220):
        result = canvas.copy ()

        #if (canvas is None):
        #    result = np.ones ((700, 700, 3), np.uint8) * 220

        cv2.putText (result, "curr mode: " + str (self.curr_mode), (30, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 50, 31), 1, cv2.LINE_AA)

        str_num = 0

        for key in self.key_to_command [self.curr_mode].keys ():
            text = key + str (self.key_to_command [self.curr_mode] [key])

            cv2.putText (result, text, (30, 60 + 20 * str_num),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 50, 31), 1, cv2.LINE_AA)

            str_num += 1

        return [result]

class Skeleton (Modality):
    def __init__ (self, skeleton_path_ = ""):
        self.read_data        = []
        self.interpreted_data = []
        self.all_data         = []

        self.dataframe_num = 0

        self.processed_data = {"righthand" : 0,
                               "rightarm"  : 0,
                               "lefthand"  : 0,
                               "leftarm"   : 0,
                               "nose_x"      : 0}

        if (skeleton_path_ != ""):
            self.all_data = self.read_data_from_file (skeleton_path_)

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

    def name (self):
        return "skeleton"

    def _read_data (self):
        if (self.dataframe_num >= len (self.all_data)):
            read_data = 0
            return

        self.read_data = self.all_data [self.dataframe_num]
        self.dataframe_num += 1

    def get_read_data (self):
        return self.read_data

    def _process_data (self):
        kpt_names = ['nose', 'neck', 'r_sho', 'r_elb', 'r_wri', 'l_sho',
                     'l_elb', 'l_wri', 'r_hip', 'r_knee', 'r_ank', 'l_hip',
                     'l_knee', 'l_ank', 'r_eye', 'l_eye', 'r_ear', 'l_ear']

        necessary_keypoints_names = ["l_sho", "l_elb", "l_wri", "l_hip", "r_sho", "r_elb", "r_wri", "r_hip", "neck", "nose", 'r_eye', 'l_eye', "r_ear", "l_ear"]
        kps = {}

        #print ("kps", kps)

        for kp in necessary_keypoints_names:
            ind = kpt_names.index (kp)
            kps.update ({kp : (self.read_data [ind * 2], self.read_data [ind * 2 + 1])})
        # print("Head rotation", (round((kps["l_eye"][0] - kps["nose"][0])/(kps["l_eye"][0] - kps["r_eye"][0]),3)))
        # head_pose = (round((kps["l_eye"][0] - kps["nose"][0])/(kps["l_eye"][0] - kps["r_eye"][0]),3))
        hips_mid  = ((kps ["r_hip"] [0] + kps ["l_hip"] [0]) / 2, (kps ["r_hip"] [1] + kps ["l_hip"] [1]) / 2)
        neck_hip  = (  hips_mid[0] - kps ["neck"]  [0],  hips_mid[1] - kps ["neck"]  [0])

        neck_nose = (kps ["nose"]  [0] - kps ["neck"][0], kps ["nose"]  [1] - kps ["neck"][1])
        # should_mid = (abs(kps ["r_sho"]  [0] - kps ["l_sho"][0])//2, abs(kps ["r_sho"]  [1] - kps ["l_sho"][1])//2)
        #
        # should_neck =

        sh_r_elb  = (kps ["r_elb"] [0] - kps ["r_sho"] [0], kps ["r_elb"] [1] - kps ["r_sho"] [1])
        sh_l_elb  = (kps ["l_elb"] [0] - kps ["l_sho"] [0], kps ["l_elb"] [1] - kps ["l_sho"] [1])
        elb_r_wri = (kps ["r_wri"] [0] - kps ["r_elb"] [0], kps ["r_wri"] [1] - kps ["r_elb"] [1])
        elb_l_wri = (kps ["l_wri"] [0] - kps ["l_elb"] [0], kps ["l_wri"] [1] - kps ["l_elb"] [1])

        #print("Kak tak, Rektor Kudryavsev", -common.angle_2_vec (neck_hip, sh_r_elb))#angle_2_vec_head (-1*neck_hip[0],-1*neck_hip[1], neck_nose[0],neck_nose[1]))
        self.processed_data ["nose_x"] = (kps["neck"][0] - kps["nose"][0])/40

        #print ("vectors", neck_hip, sh_r_elb)

        self.processed_data ["righthand"] = -common.angle_2_vec (neck_hip, sh_r_elb)
        self.processed_data ["lefthand"]  = common.angle_2_vec (neck_hip, sh_l_elb)
        self.processed_data ["rightarm"]  = common.angle_2_vec (sh_r_elb, elb_r_wri)
        self.processed_data ["leftarm"]   = - common.angle_2_vec (sh_l_elb, elb_l_wri)
        self.processed_data ["leftleg"] = -abs(common.angle_2_vec (neck_hip, sh_r_elb))
        # self.processed_data ["righthand"] = -(angle_2_vec (neck_hip, sh_r_elb)  + 1.57)
        # self.processed_data ["lefthand"]  = angle_2_vec (neck_hip, sh_l_elb) #+ 1.57
        #
        #
        # self.processed_data ["rightarm"]  =  angle_2_vec (sh_l_elb, elb_l_wri)
        # # self.processed_data ["rightarm"]  = -2.0
        # if - angle_2_vec (sh_r_elb, elb_r_wri)  < -1.2:
        #     self.processed_data ["leftarm"] = -1.2
        # else:
        #     self.processed_data ["leftarm"]   =  - angle_2_vec (sh_r_elb, elb_r_wri)
        #
        # # print((- angle_2_vec (sh_r_elb, elb_r_wri) ) )
        #
        # print("rightarm angle: ", self.processed_data ["righthand"])
        # print(self.processed_data ["lefthand"])
        # print(self.processed_data ["rightarm"])
        # print(self.processed_data ["leftarm"])


        # self.processed_data ["head"] = head_rotation (head_pose) #, kps["l_ear"],kps["r_ear"])

    def _interpret_data (self):
        self.interpreted_data = self.processed_data

    def _get_command (self):
        commands = []

        for key in self.processed_data.keys ():
            commands.append (("/set_joint_angle", [key, str (self.processed_data [key])]))

        return commands

    def get_command (self, skip_reading_data = False):
        if (skip_reading_data == False):
            self._read_data ()

        self._process_data   ()
        self._interpret_data ()

        return self._get_command ()

class Video (Modality):
    def __init__ (self, video_path_ = "", model_path_ = "", mode_ = "GPU"):
        self.read_data        = []
        self.interpreted_data = []
        #self.all_data        = []
        self.timeout = common.Timeout_module (3.5)

        self.dataframe_num = 0

        self.processed_data = {"righthand" : 0,
                               "lefttleg"  : 0,
                               "rightarm"  : 0,
                               "lefthand"  : 0,
                               "leftarm"   : 0,
                               "nose_x"    : 0}
        # if video_path_ != '':

        #get_available_cameras()
        #self.available_cameras = get_available_cameras(upper_bound=10, lower_bound=0)
        # self.all_data = cv2.VideoCapture("/home/kompaso/DEBUG/Debug/remote control/data/video/testt.mp4")#self.available_cameras[-1])

        if (video_path_ == ""):
            self.all_data = cv2.VideoCapture(0)#self.available_cameras[-1])

        self.skel = Skeleton()

        self.read = False

        # self.net = PoseEstimationWithMobileNet()
        # checkpoint = torch.load("models/checkpoint_iter_370000.pth", map_location=torch.device('cpu'))
        # load_state(self.net, checkpoint)

        if (model_path_ == ""):
            model_path_ = "/Users/elijah/Dropbox/Programming/RoboCup/remote control/source/test/human-pose-estimation-3d.pth"

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
        base_height = 50  # 256
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
            self.skel._process_data()
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

class Markov_chain (Modality):
    def __init__ (self, video_path_ = ""):
        self.read_data        = []
        self.interpreted_data = []

        self.timeout = common.Timeout_module (0.2)
        self.tick = 0

        self.commands = {"noaction": [("noaction", [""])],
                         "1": [("/stand", [""])],
                         "2": [("/left_shoulder_up", [""])],
                         "3": [("/right_shoulder_up", [""])],
                         "4": [("/left_shoulder_up", [""])],
                         "5": [("/stand", [""])],
                         "6": [("/left_hand_left", [""])],
                         "7": [("/stand", [""])],
                         "8": [("/right_hand_right", [""])],
                         "9": [("/stand", [""])],
                         "10": [("/bend_right", [""])],
                         "11": [("/bend_left", [""])],
                         "12": [("/stand", [""])],
                         "13": [("/play_airplane_1", [""])],
                         "14": [("/play_airplane_2", [""])],
                         "15": [("/play_airplane_1", [""])],
                         "16": [("/hands_sides", [""])],

                         }

    def name(self):
        return "Markov chain"

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

            comm = self.commands[str (self.tick % (l - 1) + 1)]
            self.tick += 1

        #print ("com", comm)

        return comm

    def get_command(self, skip_reading_data=False):
        self._read_data()
        self._process_data()
        self._interpret_data()

        return self._get_command()

    # def draw(self, img):
    #     pass


    def __init__ (self, skeleton_path_ = ""):
        self.read_data        = []
        self.interpreted_data = []

        self.timeout = common.Timeout_module (1)

        self.dataframe_num = 0

        self.commands = {"noaction": [("noaction", [""])],
                         "1": [("/stand", [""])],
                         "2": [("/hands_sides", [""])]}

        self.processed_data = {"righthand": 0,
                               "rightarm": 0,
                               "lefthand": 0,
                               "leftarm": 0}

        if (skeleton_path_ != ""):
            self.all_data = self.read_data_from_file(skeleton_path_)

    def read_data_from_file(self, path):
        with open(path, 'r') as file:
            data = file.read()
            cleared_data = ''
            for let in data:
                if let.isdigit() or let == '-':
                    cleared_data += let
                else:
                    cleared_data += ','

            cleared_data = cleared_data.split(',')
            data = [int(i) for i in cleared_data if i]
            data = np.asarray(data)
            data = data.reshape(-1, 36)

        return data

    def name(self):
        return "response to skeleton"

    def _read_data(self):
        if (self.dataframe_num >= len(self.all_data)):
            read_data = 0
            return

        self.read_data = self.all_data[self.dataframe_num]
        self.dataframe_num += 1

    def get_read_data(self):
        return self.read_data

    def _process_data (self):
        kpt_names = ['nose', 'neck', 'r_sho', 'r_elb', 'r_wri', 'l_sho',
                     'l_elb', 'l_wri', 'r_hip', 'r_knee', 'r_ank', 'l_hip',
                     'l_knee', 'l_ank', 'r_eye', 'l_eye', 'r_ear', 'l_ear']

        necessary_keypoints_names = ["l_sho", "l_elb", "l_wri", "l_hip", "r_sho", "r_elb", "r_wri", "r_hip", "neck"]
        kps = {}

        #print ("kps", kps)

        for kp in necessary_keypoints_names:
            ind = kpt_names.index (kp)
            kps.update ({kp : (self.read_data [ind * 2], self.read_data [ind * 2 + 1])})

        hips_mid  = ((kps ["r_hip"] [0] + kps ["l_hip"] [0]) / 2, (kps ["r_hip"] [1] + kps ["l_hip"] [1]) / 2)
        neck_hip  = (kps ["neck"]  [0] - hips_mid      [0], kps ["neck"]  [1] - hips_mid      [1]) #????????
        sh_r_elb  = (kps ["r_elb"] [0] - kps ["r_sho"] [0], kps ["r_elb"] [1] - kps ["r_sho"] [1])
        sh_l_elb  = (kps ["l_elb"] [0] - kps ["l_sho"] [0], kps ["l_elb"] [1] - kps ["l_sho"] [1])
        elb_r_wri = (kps ["r_wri"] [0] - kps ["r_elb"] [0], kps ["r_wri"] [1] - kps ["r_elb"] [1])
        elb_l_wri = (kps ["l_wri"] [0] - kps ["l_elb"] [0], kps ["l_wri"] [1] - kps ["l_elb"] [1])

        self.processed_data ["righthand"] = -angle_2_vec (neck_hip, sh_r_elb)
        self.processed_data ["lefthand"]  = angle_2_vec (neck_hip, sh_l_elb)
        self.processed_data ["rightarm"]  = angle_2_vec (sh_r_elb, elb_r_wri)
        self.processed_data ["leftarm"]   = angle_2_vec (sh_l_elb, elb_l_wri)

    def _interpret_data(self):
        pass

    def _get_command(self):
        comm = self.commands ["noaction"]

        if (self.timeout.timeout_passed ()):
            movement = 1

            #print ("aa", self.processed_data ["righthand"])
            if (self.processed_data ["righthand"] > -1):
                movement = 2

            comm = self.commands[str(movement)]

            #self.tick += 1

        #print ("com", comm)

        return comm

    def get_command(self, skip_reading_data=False):
        self._read_data()
        self._process_data()
        self._interpret_data()

        return self._get_command()

    # def draw(self, img):
    #     pass

class Music (Modality):
    def __init__ (self, music_path_ = ""):
        self.read_data        = []
        self.interpreted_data = []

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

    # def draw(self, img):
    #     pass

#class Voice (Modality):
#class Virtual_keyboard (Modality):
