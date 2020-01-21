from common import *
import torch
from demo import VideoReader, infer_fast, run_demo
from modules.load_state import load_state
from models.with_mobilenet import PoseEstimationWithMobileNet


class Modality:
    def __init__ (self):
        pass

    def name (self):
        return "not specified"

class Computer_keyboard (Modality):
    def __init__ (self, key_to_command_ = {"z" : "empty"}):
        self.read_data        = 0x00
        self.processed_data   = 0x00
        self.interpreted_data = 0x00

        self.key_to_command = {"z"        : ("/stand",   ["heh"]),
                               "c"        : ("/rest",    ["kek"]),
                               "w"        : ("/increment_joint_angle", ["lefthand", "0.21"]),
                               "e"        : ("/increment_joint_angle", ["lefthand", "-0.21"]),
                               "r"        : ("/increment_joint_angle", ["leftarm", "0.21"]),
                               "t"        : ("/increment_joint_angle", ["leftarm", "-0.21"]),
                               "s"        : ("/increment_joint_angle", ["righthand", "0.21"]),
                               "d"        : ("/increment_joint_angle", ["righthand", "-0.21"]),
                               "f"        : ("/increment_joint_angle", ["rightarm", "0.21"]),
                               "g"        : ("/increment_joint_angle", ["rightarm", "-0.21"]),
                               "n"        : ("next",     [""]),
                               "noaction" : ("noaction", [""])}

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

            if (key in self.key_to_command.keys ()):
                return self.key_to_command [key]

        return [self.key_to_command ["noaction"]]

    def get_command (self, skip_reading_data = False):
        if (skip_reading_data == False):
            self._read_data ()

        self._process_data   ()
        self._interpret_data ()

        return self._get_command ()

    def draw (self, img):
        pass

class Skeleton (Modality):
    def __init__ (self, skeleton_path_ = ""):
        self.read_data        = []
        self.interpreted_data = []
        self.all_data         = []

        self.dataframe_num = 0

        self.processed_data = {"righthand" : 0,
                               "rightarm"  : 0,
                               "lefthand"  : 0,
                               "leftarm"   : 0}

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

    def angle_2_vec_ (self, x1, y1, x2, y2):
        dot = x1*x2 + y1*y2
        det = x1*y2 - y1*x2
        angle = math.atan2(det, dot)

        return angle

    def angle_2_vec (self, vec1, vec2):
        return self.angle_2_vec_ (vec1 [0], vec1 [1], vec2 [0], vec2 [1])

    def _process_data (self):
        kpt_names = ['nose', 'neck', 'r_sho', 'r_elb', 'r_wri', 'l_sho',
                     'l_elb', 'l_wri', 'r_hip', 'r_knee', 'r_ank', 'l_hip',
                     'l_knee', 'l_ank', 'r_eye', 'l_eye', 'r_ear', 'l_ear']

        necessary_keypoints_names = ["l_sho", "l_elb", "l_wri", "l_hip", "r_sho", "r_elb", "r_wri", "r_hip", "neck"]
        kps = {}

        print ("kps", kps)

        for kp in necessary_keypoints_names:
            ind = kpt_names.index (kp)
            kps.update ({kp : (self.read_data [ind * 2], self.read_data [ind * 2 + 1])})

        hips_mid  = ((kps ["r_hip"] [0] + kps ["l_hip"] [0]) / 2, (kps ["r_hip"] [1] + kps ["l_hip"] [1]) / 2)
        neck_hip  = (kps ["neck"]  [0] - hips_mid      [0], kps ["neck"]  [1] - hips_mid      [1])
        sh_r_elb  = (kps ["r_elb"] [0] - kps ["r_sho"] [0], kps ["r_elb"] [1] - kps ["r_sho"] [1])
        sh_l_elb  = (kps ["l_elb"] [0] - kps ["l_sho"] [0], kps ["l_elb"] [1] - kps ["l_sho"] [1])
        elb_r_wri = (kps ["r_wri"] [0] - kps ["r_elb"] [0], kps ["r_wri"] [1] - kps ["r_elb"] [1])
        elb_l_wri = (kps ["l_wri"] [0] - kps ["l_elb"] [0], kps ["l_wri"] [1] - kps ["l_elb"] [1])

        self.processed_data ["righthand"] = -self.angle_2_vec (neck_hip, sh_r_elb)
        self.processed_data ["lefthand"]  = -self.angle_2_vec (neck_hip, sh_l_elb)
        self.processed_data ["rightarm"]  = -self.angle_2_vec (sh_r_elb, elb_r_wri)
        self.processed_data ["leftarm"]   = -self.angle_2_vec (sh_l_elb, elb_l_wri)

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

    def draw (self, img):
        pass

class Video (Modality):
    def __init__ (self, video_path_ = ""):
        self.read_data        = []
        self.interpreted_data = []
        #self.all_data         = []

        self.dataframe_num = 0

        self.processed_data = {"righthand" : 0,
                               "rightarm"  : 0,
                               "lefthand"  : 0,
                               "leftarm"   : 0}
        # if video_path_ != '':
        self.all_data = cv2.VideoCapture(0)

        self.skel = Skeleton()
        self.net = PoseEstimationWithMobileNet()
        checkpoint = torch.load("models/checkpoint_iter_370000.pth", map_location='cpu')
        load_state(self.net, checkpoint)

    def name(self):
        return "video"

    def _read_data (self):
        # if (self.dataframe_num >= len (self.all_data)):
        #     read_data = 0
        #     return
        # self.frame_skel = run_demo(self.all_data)


        _, img = self.all_data.read()


        self.read_data = run_demo(self.net, img, 256, False, 1, 1) #self.all_data [self.dataframe_num]

        # self.dataframe_num += 1

    def _process_data(self):
        if sum (self.read_data) != -36 and self.read_data != []:
            # print ("hehm", self.read_data)
            self.skel.read_data = self.read_data
            self.skel._process_data()
            self.processed_data = self.skel.processed_data
            print(self.processed_data)

        else:
            return

    def _interpret_data(self):
        self.interpreted_data = self.processed_data

    def _get_command(self):
        commands = []

        for key in self.processed_data.keys():
            commands.append(("/set_joint_angle", [key, str(self.processed_data[key])]))

        print ("com", commands)

        return commands

    def get_command(self, skip_reading_data=False):
        if (skip_reading_data == False):
            self._read_data()

        self._process_data()
        self._interpret_data()

        return self._get_command()

    def draw(self, img):
        pass






#class Voice (Modality):
#class Virtual_keyboard (Modality):
