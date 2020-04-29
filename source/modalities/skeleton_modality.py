from modalities.modality import WorkWithPoints

import numpy as np
import common

import cv2

class Skeleton_2D (WorkWithPoints):
    def __init__ (self, skeleton_path_ = "", logger_ = 0):
        WorkWithPoints.__init__(self)
        self.all_data         = []
        self.dataframe_num = 0
        self.read_data_from_file_ = False

        if (skeleton_path_ != ""):
            self.all_data = self.read_data_from_file(skeleton_path_)


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

    def hand_up_angles(self, angle, hand):
        hand_roll  = angle
        hand_pitch = 0

        if hand == "righthand" :
            k = 1
            if (angle <= -1.3*k and angle > -1.8*k):
                hand_roll = -1.3*k
                hand_pitch = (2.04 + 3.14 / 5) / 0.5 * (angle + 1.3)

            elif (angle <= -1.8*k):
                hand_roll = -1.3*k - (angle + 1.8*k)
                hand_pitch = - (2.04 + 3.14 / 5)

            return hand_roll, hand_pitch

        if hand == "lefthand" :
            k = -1
        if (angle >= 1.3 and angle < 1.8):
            hand_roll = -1.3*k
            hand_pitch = (2.04 + 3.14 / 5) / 0.5 * (angle*k + 1.3)

        elif (angle >= -1.8*k):
            hand_roll = -1.3*k - (angle + 1.8*k)
            hand_pitch = - (2.04 + 3.14 / 5)

        return hand_roll, hand_pitch

    def _process_data (self, frame = None):
        kps = {}
        for kp in self.necessary_keypoints_names:
            ind = self.kpt_names.index (kp)
            kps.update ({kp : (self.read_data [ind * 2], self.read_data [ind * 2 + 1])})
        # print("Head rotation", (round((kps["l_eye"][0] - kps["nose"][0])/(kps["l_eye"][0] - kps["r_eye"][0]),3)))
        # head_pose = (round((kps["l_eye"][0] - kps["nose"][0])/(kps["l_eye"][0] - kps["r_eye"][0]),3))
        # hips_mid  = ((kps ["r_hip"] [0] + kps ["l_hip"] [0]) / 2, (kps ["r_hip"] [1] + kps ["l_hip"] [1]) / 2)
        # neck_hip  = (  hips_mid[0] - kps ["neck"]  [0],  hips_mid[1] - kps ["neck"]  [1])
        #
        # neck_nose = (kps ["nose"]  [0] - kps ["neck"][0], kps ["nose"]  [1] - kps ["neck"][1])
        #
        # ##########################################3d block#########################################################################
        # neck_hip_xy = (self.poses_3d[0][2][0] - self.poses_3d[0][0][0], self.poses_3d[0][2][2] - self.poses_3d[0][0][2])
        # neck_hip_xz = (self.poses_3d[0][2][0] - self.poses_3d[0][0][0], self.poses_3d[0][2][1] - self.poses_3d[0][0][1])
        # neck_hip_yz = (self.poses_3d[0][2][2] - self.poses_3d[0][0][2], self.poses_3d[0][2][1] - self.poses_3d[0][0][1])
        #
        # sh_r_elb_xy = (self.poses_3d[0][10][0] - self.poses_3d[0][9][0], self.poses_3d[0][10][2] - self.poses_3d[0][9][2])
        # sh_r_elb_yz = (self.poses_3d[0][10][2] - self.poses_3d[0][9][2], self.poses_3d[0][10][1] - self.poses_3d[0][9][1])
        #
        # # elb_r_wri
        #
        # xy = common.angle_2_vec(neck_hip_xy,sh_r_elb_xy)
        # yz = -(common.angle_2_vec(neck_hip_yz,sh_r_elb_yz))
        #
        # self.logger.update ("xy", xy)
        # self.logger.update ("yz", yz)
        # ##########################################################################################################################3
        #
        # sh_r_elb  = (kps ["r_elb"] [0] - kps ["r_sho"] [0], kps ["r_elb"] [1] - kps ["r_sho"] [1])
        # sh_l_elb  = (kps ["l_elb"] [0] - kps ["l_sho"] [0], kps ["l_elb"] [1] - kps ["l_sho"] [1])
        # elb_r_wri = (kps ["r_wri"] [0] - kps ["r_elb"] [0], kps ["r_wri"] [1] - kps ["r_elb"] [1])
        # elb_l_wri = (kps ["l_wri"] [0] - kps ["l_elb"] [0], kps ["l_wri"] [1] - kps ["l_elb"] [1])
        #
        # #print("Kak tak, Rektor Kudryavsev", -common.angle_2_vec (neck_hip, sh_r_elb))#angle_2_vec_head (-1*neck_hip[0],-1*neck_hip[1], neck_nose[0],neck_nose[1]))
        # self.processed_data ["nose_x"] = (kps["neck"][0] - kps["nose"][0])/40
        #
        # #print ("vectors", neck_hip, sh_r_elb)
        #
        # if (frame is not None):
        #     cv2.arrowedLine (frame, (int (hips_mid[0]), int (hips_mid[1])),
        #         (int (kps ["neck"]  [0]), int (kps ["neck"][1])),
        #         (100, 10, 200), thickness=2)
        #
        # skel_angle = -(common.angle_2_vec (neck_hip, sh_r_elb))
        # skel_angle_ = common.angle_2_vec (neck_hip, sh_l_elb)
        #
        # roll, pitch = self.hand_up_angles (skel_angle, "righthand")
        # roll_l, pitch_l = self.hand_up_angles (skel_angle_, "lefthand")
        #
        # self.processed_data ["righthand"] = roll
        # self.processed_data ["lefthand"]  = pitch
        # self.processed_data ["leftshoulder_pitch" ] = yz
        # self.processed_data ["leftarm"]   = - common.angle_2_vec (sh_l_elb, elb_l_wri)
        # self.processed_data ["rightarm"]  = common.angle_2_vec (sh_r_elb, elb_r_wri)

        # #self.processed_data ["leftleg"] = -abs(common.angle_2_vec (neck_hip, sh_r_elb))
        # self.processed_data ["rightshoulder_pitch"] = pitch

        # self.logger.update ("rh roll", self.poses_3d[0])




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
