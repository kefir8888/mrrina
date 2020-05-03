from modalities.modality import WorkWithPoints
import common
import os

import numpy as np
import cv2
import math

class Skeleton_2D(WorkWithPoints):
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

class Skeleton_3D(WorkWithPoints):
    def __init__ (self, skeleton_path_ = "", logger_ = 0):
        WorkWithPoints.__init__(self, logger_, maxlen_=20)
        self.all_data         = []
        self.dataframe_num = 0

        if (skeleton_path_ != ""):
            verbose = False
            if( os.path.isfile(skeleton_path_) == True ):
                print( "Skeleton file: ", skeleton_path_)
                skeleton_data = open(skeleton_path_, 'r')
                all_skeleton_frames = self.read_skeleton_data_from_NTU(skeleton_data, verbose )
                self.all_data = all_skeleton_frames
            else:
                print("\nNo skeleton file with name: ", data_path)
                exit(0)


    def name (self):
        return "skeleton"


    def _read_data (self):
        if (self.dataframe_num >= len (self.all_data)):
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
                        kps.update ({kp : [(self.read_data[6][0] + self.read_data[12][0]) / 2, (self.read_data[6][1] + self.read_data[12][1]) / 2, (self.read_data[6][2] + self.read_data[12][2]) / 2]})
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

        l_sho_pitch_raw = math.acos(np.dot(l_hip_neck, R_lbp_lse)/(mod_l_hip_neck*mod_R_lbp_lse))-0.6
        l_elb_yaw_raw = math.acos(np.dot(N_neck_l_sho_elb, N_l_sho_elb_wri)/(mod_N_neck_l_sho_elb*mod_N_l_sho_elb_wri))

        phi_lsp = math.acos(np.dot(l_sho_elb,l_hip_neck)/(mod_l_hip_neck * mod_l_sho_elb))
        phi_ley_1 = math.acos(np.dot(l_elb_wri, N_neck_l_sho_elb)/(mod_l_elb_wri * mod_N_neck_l_sho_elb))
        phi_ley_2 = math.acos(np.dot(l_elb_wri, R_l_arm)/(mod_l_elb_wri * mod_R_l_arm))

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

        l_sho_roll = 1.57 - math.acos(np.dot(l_sho_elb, R_l_body_plane)/(mod_l_sho_elb *mod_R_l_body_plane ))
        l_elb_roll = -(3.14 - math.acos(np.dot(l_elb_wri, l_elb_sho)/(mod_l_elb_wri*mod_l_elb_sho)))

#####################################################################################################################
        self.angles_mean["l_sho_pitch"].append(l_sho_pitch)
        self.angles_mean["l_sho_roll"].append(l_sho_roll)
        self.angles_mean["l_elb_yaw"].append(l_elb_yaw)
        self.angles_mean["l_elb_roll"].append(l_elb_roll)

        self.logger.update("l shoul pitch", round(self.get_mean(self.angles_mean["l_sho_pitch"]), 2))
        self.logger.update("l shoul roll", round(self.get_mean(self.angles_mean["l_sho_roll"]), 2))
        self.logger.update("l elb yaw", round(self.get_mean(self.angles_mean["l_elb_yaw"]), 2))
        self.logger.update("l elb roll", round(self.get_mean(self.angles_mean["l_elb_roll"]), 2))

        self.processed_data ["l_sho_pitch"]  = round(self.get_mean(self.angles_mean["l_sho_pitch"]), 2)
        self.processed_data ["l_sho_roll"]  = round(self.get_mean(self.angles_mean["l_sho_roll"]), 2)
        self.processed_data ["l_elb_yaw"]  = round(self.get_mean(self.angles_mean["l_elb_yaw"]), 2)
        self.processed_data ["l_elb_roll"]  = round(self.get_mean(self.angles_mean["l_elb_roll"]), 2)
###############################################################################################################################

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

        r_sho_pitch_raw = math.acos(np.dot(r_hip_neck, R_rbp_rse)/(mod_r_hip_neck*mod_R_rbp_rse))-0.6
        r_elb_yaw_raw = math.acos(np.dot(N_neck_r_sho_elb, N_r_sho_elb_wri)/(mod_N_neck_r_sho_elb*mod_N_r_sho_elb_wri))

        phi_rsp = math.acos(np.dot(r_sho_elb,r_hip_neck)/(mod_r_hip_neck * mod_r_sho_elb))
        phi_rey_1 = math.acos(np.dot(r_elb_wri, N_neck_r_sho_elb)/(mod_r_elb_wri * mod_N_neck_r_sho_elb))
        phi_rey_2 = math.acos(np.dot(r_elb_wri, R_r_arm)/(mod_r_elb_wri * mod_R_r_arm))

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

        r_sho_roll = 1.57 - math.acos(np.dot(r_sho_elb, R_r_body_plane)/(mod_r_sho_elb *mod_R_r_body_plane ))
        r_elb_roll = 3.14 - math.acos(np.dot(r_elb_wri, r_elb_sho)/(mod_r_elb_wri*mod_r_elb_sho))
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

        ################################################l_hip###########################################################################
        l_mhip_lhip = common.create_vec(kps["mid_hip"],kps["l_hip"])
        l_hip_knee = common.create_vec(kps["l_hip"],kps["l_knee"])
        N_mh_neck_lhip = np.cross(l_mhip_lhip, l_hip_neck)
        R_l_hip = np.cross(l_hip_neck, N_mh_neck_lhip)

        N_neck_l_mh_hk = np.cross(N_mh_neck_lhip, l_hip_knee)


        mod_l_mhip_lhip = common.get_mod(l_mhip_lhip)
        mod_l_hip_knee = common.get_mod(l_hip_knee)
        mod_R_l_hip = common.get_mod(R_l_hip)
        mod_N_mh_neck_lhip = common.get_mod(N_mh_neck_lhip)
        mod_N_neck_l_mh_hk = common.get_mod(N_neck_l_mh_hk)

        left_hip_roll_raw = math.acos(np.dot(R_l_hip, N_neck_l_mh_hk)/(mod_R_l_hip*mod_N_neck_l_mh_hk))

        phi_lhr = math.acos(np.dot(l_hip_knee, R_l_hip)/(mod_l_hip_knee*mod_R_l_hip))
        if phi_lhr <= 1.57:
            left_hip_roll = left_hip_roll_raw
        else:
            left_hip_roll = -left_hip_roll_raw

        left_hip_pitch = - 1.57 + math.acos(np.dot(l_hip_knee,N_mh_neck_lhip)/(mod_l_hip_knee*mod_N_mh_neck_lhip))
        ###########################################################################################################################
        self.angles_mean["l_hip_roll"].append(left_hip_roll)
        self.angles_mean["l_hip_pitch"].append(left_hip_pitch)
        # # self.angles_mean["r_elb_yaw"].append(r_elb_yaw)
        # # self.angles_mean["r_elb_roll"].append(r_elb_roll)
        #
        # self.logger.update("l hip roll" , round(self.get_mean(self.angles_mean["l_hip_roll"]), 2))
        # self.logger.update("l hip pitch", round(self.get_mean(self.angles_mean["l_hip_pitch"]), 2))
        # # self.logger.update("r elb yaw", round(self.get_mean(self.angles_mean["r_elb_yaw"]), 2))
        # # self.logger.update("r elb roll", round(self.get_mean(self.angles_mean["r_elb_roll"]), 2))
        #
        self.processed_data ["l_hip_roll"] = round(self.get_mean(self.angles_mean["l_hip_roll"]), 2)
        # self.processed_data ["l_ank_roll"] = -round(self.get_mean(self.angles_mean["l_hip_roll"]), 2)
        self.processed_data ["l_hip_pitch"]  = round(self.get_mean(self.angles_mean["l_hip_pitch"]), 2)
        # # self.processed_data ["r_elb_yaw"]   = round(self.get_mean(self.angles_mean["r_elb_yaw"]), 2)
        # # self.processed_data ["r_elb_roll"]  = round(self.get_mean(self.angles_mean["r_elb_roll"]), 2)
        #############################################################################################################################

        ###############################################l_knee#####################################################################
        l_knee_hip = common.create_vec(kps["l_knee"], kps["l_hip"])
        l_knee_ankle = common.create_vec(kps["l_knee"], kps["l_ank"])
        N_l_hip_knee_ankle = np.cross(l_knee_ankle, l_knee_hip)
        R_l_leg = np.cross(N_l_hip_knee_ankle, l_knee_hip)

        mod_l_knee_hip = common.get_mod(l_knee_hip)
        mod_l_knee_ankle = common.get_mod(l_knee_ankle)

        left_knee_pitch = 3.14 - math.acos(np.dot(l_knee_ankle, l_knee_hip)/(mod_l_knee_ankle*mod_l_knee_hip))
        ##########################################################################################################################
        self.angles_mean["l_knee_pitch"].append(left_knee_pitch)

        # self.logger.update("l knee pitch", round(self.get_mean(self.angles_mean["l_knee_pitch"]), 2))

        self.processed_data ["l_knee_pitch"]  = round(self.get_mean(self.angles_mean["l_knee_pitch"]), 2)
        ###########################################################################################################################3

        ##############################################r_knee####################################################################
        r_knee_hip = common.create_vec(kps["r_knee"], kps["r_hip"])
        r_knee_ankle = common.create_vec(kps["r_knee"], kps["r_ank"])
        N_r_hip_knee_ankle = np.cross(r_knee_ankle, r_knee_hip)
        R_r_leg = np.cross(N_r_hip_knee_ankle, r_knee_hip)

        mod_r_knee_hip = common.get_mod(r_knee_hip)
        mod_r_knee_ankle = common.get_mod(r_knee_ankle)

        right_knee_pitch = 3.14 - math.acos(np.dot(r_knee_ankle, r_knee_hip)/(mod_r_knee_ankle*mod_r_knee_hip))
        ###########################################################################################################################
        self.angles_mean["r_knee_pitch"].append(right_knee_pitch)

        # self.logger.update("r knee pitch", round(self.get_mean(self.angles_mean["r_knee_pitch"]), 2))

        self.processed_data ["r_knee_pitch"]  = round(self.get_mean(self.angles_mean["r_knee_pitch"]), 2)
        #########################################################################################################################
        ########################################test ankle#################################################################

        # x = math.acos(np.dot(N_l_body_plane, l_knee_ankle)/(mod_N_l_body_plane * mod_l_knee_ankle))
        # self.logger.update("ankle pitch", round(x, 2))
        ###############################################r_hip#########################################################################
        r_mhip_rhip = common.create_vec(kps["mid_hip"],kps["r_hip"])
        r_hip_knee = common.create_vec(kps["r_hip"],kps["r_knee"])
        N_mh_neck_rhip = np.cross(r_mhip_rhip, r_hip_neck)
        R_r_hip = np.cross(r_hip_neck, N_mh_neck_rhip)

        N_neck_r_mh_hk = np.cross(N_mh_neck_rhip, r_hip_knee)


        mod_r_mhip_rhip = common.get_mod(r_mhip_rhip)
        mod_r_hip_knee = common.get_mod(r_hip_knee)
        mod_R_r_hip = common.get_mod(R_r_hip)
        mod_N_mh_neck_rhip = common.get_mod(N_mh_neck_rhip)
        mod_N_neck_r_mh_hk = common.get_mod(N_neck_r_mh_hk)

        right_hip_roll_raw = math.acos(np.dot(R_r_hip, N_neck_r_mh_hk)/(mod_R_r_hip*mod_N_neck_r_mh_hk))

        phi_rhr = math.acos(np.dot(r_hip_knee, R_r_hip)/(mod_r_hip_knee*mod_R_r_hip))
        if phi_rhr <= 1.57:
            right_hip_roll = -right_hip_roll_raw
        else:
            right_hip_roll = right_hip_roll_raw

        right_hip_pitch = - 1.57 + math.acos(np.dot(r_hip_knee,N_mh_neck_rhip)/(mod_r_hip_knee*mod_N_mh_neck_rhip))
        ###########################################################################################################################
        self.angles_mean["r_hip_roll"].append(right_hip_roll)
        self.angles_mean["r_hip_pitch"].append(right_hip_pitch)
        self.angles_mean["r_ank_pitch"].append(right_hip_pitch)
        # # self.angles_mean["r_elb_yaw"].append(r_elb_yaw)
        # # self.angles_mean["r_elb_roll"].append(r_elb_roll)
        #
        # self.logger.update("r hip roll" , -round(self.get_mean(self.angles_mean["r_hip_roll"]), 2))
        # self.logger.update("r hip pitch", round(self.get_mean(self.angles_mean["r_hip_pitch"]), 2))
        # # self.logger.update("r elb yaw", round(self.get_mean(self.angles_mean["r_elb_yaw"]), 2))
        # # self.logger.update("r elb roll", round(self.get_mean(self.angles_mean["r_elb_roll"]), 2))
        #
        self.processed_data ["r_hip_roll"] = -round(self.get_mean(self.angles_mean["r_hip_roll"]), 2)
        # self.processed_data ["r_ank_pitch"] = round(self.get_mean(self.angles_mean["r_ank_pitch"]), 2)
        self.processed_data ["r_hip_pitch"]  = round(self.get_mean(self.angles_mean["r_hip_pitch"]), 2)

        self.processed_data ["l_hip_roll"] = -round(self.get_mean(self.angles_mean["r_hip_roll"]), 2)
        # self.processed_data ["l_ank_pitch"] = round(self.get_mean(self.angles_mean["r_ank_pitch"]), 2)
        self.processed_data ["l_hip_pitch"]  = round(self.get_mean(self.angles_mean["r_hip_pitch"]), 2)
        # # self.processed_data ["r_elb_yaw"]   = round(self.get_mean(self.angles_mean["r_elb_yaw"]), 2)
        # # self.processed_data ["r_elb_roll"]  = round(self.get_mean(self.angles_mean["r_elb_roll"]), 2)
        #############################################################################################################################

        ###################################################head#######################################################################
        neck_l_sho = common.create_vec(kps["neck"], kps["l_sho"])
        neck_nose = common.create_vec(kps["neck"], kps["nose"])
        l_ear_eye = common.create_vec(kps["l_ear"], kps["l_eye"])
        mid_hip_neck = common.create_vec(kps["mid_hip"], kps["neck"])


        mod_neck_l_sho = common.get_mod(neck_l_sho)
        mod_neck_nose = common.get_mod(neck_nose)
        mod_l_ear_eye = common.get_mod(l_ear_eye)
        mod_mid_hip_neck = common.get_mod(mid_hip_neck)

        nose_neck_sho = np.dot(neck_l_sho, neck_nose)
        # norm_l_ear_eye = np.dot(, l_ear_eye)
        mid_hip_neck_nose = np.dot(mid_hip_neck, neck_nose)

        head_Yaw = -(1.57 - math.acos(nose_neck_sho/(mod_neck_l_sho*mod_neck_nose)))
####################численное решение и это очень плохо#########################
        head_pitch = -(math.acos(mid_hip_neck_nose/(mod_neck_nose * mod_mid_hip_neck))-0.6)


        ############################################################################################################################3
        self.angles_mean["head_Yaw"].append(head_Yaw)
        self.angles_mean["head_Pitch"].append(head_pitch)

        # self.logger.update("Head Yaw", round(self.get_mean(self.angles_mean["head_Yaw"]), 2))
        # self.logger.update("Head Pitch", round(self.get_mean(self.angles_mean["head_Pitch"]), 2))

        self.processed_data ["head_Yaw"]  = round(self.get_mean(self.angles_mean["head_Yaw"]), 2)
        self.processed_data ["head_Pitch"]  = round(self.get_mean(self.angles_mean["head_Yaw"]), 2)
        #############################################################################################################################

        #############################################################################################################################
        #
        # l_elb_sho = common.create_vec(kps["l_elb"], kps["l_sho"])
        # l_elb_wri = common.create_vec(kps["l_sho"], kps["l_wri"])
        # l_elb_wri_mod = common.get_mod(l_elb_wri)
        # ###########################################################################################################################
        # neck_r_sho = common.create_vec(kps["neck"], kps["r_sho"])
        # r_sho_elb = common.create_vec(kps["r_sho"], kps["r_elb"])
        # r_elb_sho = common.create_vec(kps["r_elb"], kps["r_sho"])
        # r_elb_wri = common.create_vec(kps["r_sho"], kps["r_wri"])
        # r_elb_wri_mod = common.get_mod(r_elb_wri)
        # #############################################################################################################################
        #
        #
        #
        # R_lut_mod = common.get_mod(R_lut)
        # R_lut_56 = np.cross(R_lut, l_sho_elb)
        # R_lut_56_mod = common.get_mod(R_lut_56)
        # ########################################################################################################################
        # # if (self.img is not None):
        # #     cv2.arrowedLine (self.img, (int (kps["mid_hip"][0]), int (kps["mid_hip"][1])),
        # #         (int (kps ["neck"]  [0]), int (kps ["neck"][1])),
        # #         (100, 10, 200), thickness=5)
        #
        #
        # thetha_lsp = math.acos(np.dot(hips_mid_neck, R_lut_56)/(R_lut_56_mod*common.get_mod(hips_mid_neck)))
        # thetha_lsr = 1.57 - math.acos(np.dot(l_sho_elb, R_lut)/common.get_mod(l_sho_elb)*R_lut_mod)
        #
        # self.angles_mean["l_sho_roll"].append(thetha_lsr)
        # self.angles_mean["l_sho_pitch"].append(thetha_lsp)
        #
        # phi_lsp = math.acos(np.dot(l_sho_elb, hips_mid_neck)/(common.get_mod(l_sho_elb)*common.get_mod(hips_mid_neck)))
        #
        # if  phi_lsp <= 1.57:
        #     thetha_lsp = -abs(thetha_lsp)
        # else:
        #     thetha_lsp = abs(thetha_lsp)
        #
        # thetha_ler = 0.0
        # thetha_ley = 0.0
        #
        # R_lua = np.cross(l_elb_sho, N_256)
        # R_lua_mod = common.get_mod(R_lua)
        #
        # V_6567 = np.cross(l_elb_sho, l_elb_wri)
        # V_6567_mod = common.get_mod(V_6567)
        # x = np.dot(N_256, V_6567)/(N_256_mod*V_6567_mod)
        #
        # if x <= 1:
        #     thetha_ley = math.acos(np.dot(N_256, V_6567)/(N_256_mod*V_6567_mod))
        #     phi_ley_1 = math.acos(np.dot(l_elb_wri, N_256)/(l_elb_wri_mod * N_256_mod))
        #     phi_ley_2 = math.acos(np.dot(l_elb_wri, R_lua)/(l_elb_wri_mod * R_lua_mod))
        #
        #     if phi_ley_1 <= 1.57:
        #         thetha_ley = -thetha_ley
        #     elif phi_ley_1 > 1.57 and phi_ley_2 > 1.57:
        #         thetha_ley = thetha_ley
        #     else:
        #         thetha_ley = thetha_ley - 3.14
        #
        #     thetha_ler = 3.14 - math.acos(np.dot(l_elb_wri, l_elb_sho)/(l_elb_wri_mod*common.get_mod(l_elb_sho)))
        #     self.angles_mean["l_elb_roll"].append(thetha_ler)
        #     self.angles_mean["l_elb_yaw"].append(thetha_ley)
        #
        #
        # self.logger.update("roll wrist " + name, -round(self.get_mean(self.angles_mean["l_elb_roll"]), 2))
        #     # self.logger.update("roll wrist", round(thetha_ler, 2))
        # #
        # self.logger.update("shoul pitch " + name, round(self.get_mean(self.angles_mean["l_sho_roll"]), 2))
        # self.logger.update("shoul roll " + name, round(self.get_mean(self.angles_mean["l_sho_pitch"]), 2))

        # self.processed_data ["l_sho_roll"]  = round(self.get_mean(self.angles_mean["l_sho_roll"]), 2)
        # self.processed_data ["l_sho_pitch"] = round(self.get_mean(self.angles_mean["l_sho_pitch"]), 2)
        # self.processed_data ["l_elb_roll" ] = -round(self.get_mean(self.angles_mean["l_elb_roll"]), 2)
        # self.processed_data ["l_elb_yaw"]   = round(self.get_mean(self.angles_mean["l_elb_yaw"]), 2)
        #
        # self.processed_data ["r_sho_roll"]  = round(self.get_mean(self.angles_mean["r_sho_roll"]), 2)
        # self.processed_data ["r_sho_pitch"] = round(self.get_mean(self.angles_mean["r_sho_pitch"]), 2)
        # self.processed_data ["r_elb_roll" ] = -round(self.get_mean(self.angles_mean["r_elb_roll"]), 2)
        # self.processed_data ["r_elb_yaw"]   = round(self.get_mean(self.angles_mean["r_elb_yaw"]), 2)

        # self.processed_data ["rightarm"]  = common.angle_2_vec (sh_r_elb, elb_r_wri)
        #
        # #self.processed_data ["leftleg"] = -abs(common.angle_2_vec (neck_hip, sh_r_elb))
        # self.processed_data ["rightshoulder_pitch"] = pitch
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
