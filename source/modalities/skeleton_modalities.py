from modalities.modality import WorkWithPoints

import numpy as np
import common

import cv2

import math

class Skeleton_2D(WorkWithPoints):
    def __init__ (self, skeleton_path_ = "", logger_ = 0):
        WorkWithPoints.__init__(self)
        self.all_data         = []
        self.poses_3d         = []

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

class Skeleton_3D(WorkWithPoints):
    def __init__ (self, skeleton_path_ = "", logger_ = 0):
        WorkWithPoints.__init__(self, logger_)
        self.all_data         = []

        if (skeleton_path_ != ""):
            self.all_data = self.read_data_from_file(skeleton_path_)


    def name (self):
        return "skeleton"

    def _read_data (self):
        self.read_data = self.all_data

    def create_dicts_with_coords_3D(self):
        kps = {}
        if self.read_data != []:
            for kp in self.necessary_keypoints_names:
                ind = self.kpt_names.index(kp)
                if kp == 'mid_hip':
                    if (kps["l_hip"][0] > 0 and kps["r_hip"][0] > 0):
                        kps.update ({kp : [int((self.read_data[6][0] + self.read_data[12][0]) / 2), int((self.read_data[6][1] + self.read_data[12][1]) / 2), int((self.read_data[6][2] + self.read_data[12][2]) / 2)]})
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
        name = self.read_data[1]
        self.read_data = self.read_data[0]
        self.processed_data = self.create_dicts_with_coords_3D()

        kps = self.get_mean_cords(self.kps_mean)
        #################################################################################################################################

            # hips_mid = [int((kps ["r_hip"][0] + kps ["l_hip"][0]) / 2), int((kps ["r_hip"][1] + kps ["l_hip"][1]) / 2), int((kps ["r_hip"][2] + kps ["l_hip"][2]) / 2)]
        hips_mid_neck = common.create_vec(kps["mid_hip"],kps["neck"])
        neck_l_sho = common.create_vec(kps["neck"], kps["l_sho"])
        l_sho_elb = common.create_vec(kps["l_sho"], kps["l_elb"])
        l_elb_sho = common.create_vec(kps["l_elb"], kps["l_sho"])
        l_elb_wri = common.create_vec(kps["l_sho"], kps["l_wri"])
        l_elb_wri_mod = common.get_mod(l_elb_wri)
        #############################################################################################################################
        N_256 = np.cross(neck_l_sho, l_sho_elb)
        N_256_mod = common.get_mod(N_256)
        N_125 = np.cross(neck_l_sho, hips_mid_neck)
        R_lut = np.cross( hips_mid_neck, N_125)
        R_lut_mod = common.get_mod(R_lut)
        R_lut_56 = np.cross(R_lut, l_sho_elb)
        R_lut_56_mod = common.get_mod(R_lut_56)

        # if (self.img is not None):
        #     cv2.arrowedLine (self.img, (int (kps["mid_hip"][0]), int (kps["mid_hip"][1])),
        #         (int (kps ["neck"]  [0]), int (kps ["neck"][1])),
        #         (100, 10, 200), thickness=5)


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


        x = np.dot(N_256, V_6567)/(N_256_mod*V_6567_mod)

        if x <= 1:
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
            self.logger.update("yell wrist " + name, round(thetha_ley, 2))
            self.logger.update("roll wrist", round(thetha_ler, 2))

        self.logger.update("shoul pitch " + name, round(thetha_lsp, 2))
        self.logger.update("shoul roll " + name, round(thetha_lsr, 2))
######################################################################################################
        # kps = create_dicts_with_coords_3D(self, self.read_data)
        # kpt_names = ['nose', 'neck', 'r_sho', 'r_elb', 'r_wri', 'l_sho',
        #              'l_elb', 'l_wri', 'r_hip', 'r_knee', 'r_ank', 'l_hip',
        #              'l_knee', 'l_ank', 'r_eye', 'l_eye', 'r_ear', 'l_ear']
        #
        # necessary_keypoints_names = ["l_sho", "l_elb", "l_wri", "l_hip", "r_sho", "r_elb", "r_wri", "r_hip", "neck", "nose", 'r_eye', 'l_eye', "r_ear", "l_ear"]


            # [[0, 1],  # neck - nose
            #  [1, 16], [16, 18],  # nose - l_eye - l_ear
            #  [1, 15], [15, 17],  # nose - r_eye - r_ear
            #  [0, 3], [3, 4], [4, 5],     # neck - l_shoulder - l_elbow - l_wrist
            #  [0, 9], [9, 10], [10, 11],  # neck - r_shoulder - r_elbow - r_wrist
            #  [0, 6], [6, 7], [7, 8],        # neck - l_hip - l_knee - l_ankle
            #  [0, 12], [12, 13], [13, 14]])  # neck - r_hip - r_knee - r_ankle


        # for kp in self.necessary_keypoints_names:
        #     ind = self.kpt_names.index (kp)
        #     kps.update ({kp : (self.read_data [ind * 2], self.read_data [ind * 2 + 1])})
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
