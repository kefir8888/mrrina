from common import *
import torch
from skel_proc import VideoReader, infer_fast, get_skel_coords
from modules.load_state import load_state
from models.with_mobilenet import PoseEstimationWithMobileNet
import io

class Modality:
    def __init__ (self):
        pass

    def name (self):
        return "not specified"

    def draw (self, img):
        return np.array ((1, 1, 1), np.uint8)

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

#        self.key_to_command = {"z"        : ("/stand",   ["heh"]),
#                               "c"        : ("/rest",    ["kek"]),
#                               "w"        : ("/increment_joint_angle", ["lefthand", "0.21"]),
#                               "e"        : ("/increment_joint_angle", ["lefthand", "-0.21"]),
#                               "r"        : ("/increment_joint_angle", ["leftarm", "0.21"]),
#                               "t"        : ("/increment_joint_angle", ["leftarm", "-0.21"]),
#                               "s"        : ("/increment_joint_angle", ["righthand", "0.21"]),
#                               "d"        : ("/increment_joint_angle", ["righthand", "-0.21"]),
#                               "f"        : ("/increment_joint_angle", ["rightarm", "0.21"]),
#                               "g"        : ("/increment_joint_angle", ["rightarm", "-0.21"]),
#                               "n"        : ("next",     [""]),
#                               "noaction" : ("noaction", [""])}
#

        self.key_to_command = []
        self.key_to_command.append (self.exceptional)
        self.key_to_command.append (self.repeating)
        self.key_to_command.append (self.repeating2)
        self.key_to_command.append (self.eyes)

        if (phrases_path != ""):
            f = io.open (phrases_path, "r", encoding='utf-8')
            f1 = f.readlines()
    
            available_keys = [x for x in self.all_keys if x not in self.common_commands.keys ()]

            phrase_name = []

            for line in f1:
                out = rus_line_to_eng (line)
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

        return result

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
        angle = math.atan2(det,dot)
        # print(angle)

        return angle

    def angle_2_vec (self, vec1, vec2):
        return self.angle_2_vec_ (vec1 [0], vec1 [1], vec2 [0], vec2 [1])

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


        self.processed_data ["righthand"] = self.angle_2_vec (neck_hip, sh_r_elb)
        self.processed_data ["lefthand"]  = self.angle_2_vec (neck_hip, sh_l_elb)
        self.processed_data ["rightarm"]  = self.angle_2_vec (sh_r_elb, elb_r_wri)
        self.processed_data ["leftarm"]   = self.angle_2_vec (sh_l_elb, elb_l_wri)

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
    #
    # def draw (self, img):
    # return


def get_available_cameras(upper_bound=10, lower_bound=0):
    available = []

    for i in range(lower_bound, upper_bound):
        cap = cv2.VideoCapture(i)

        if (cap.isOpened()):
            available.append(i)

        cap.release()

    return available

class Video (Modality):
    def __init__ (self, video_path_ = ""):
        self.read_data        = []
        self.interpreted_data = []
        #self.all_data        = []

        self.dataframe_num = 0

        self.processed_data = {"righthand" : 0,
                               "rightarm"  : 0,
                               "lefthand"  : 0,
                               "leftarm"   : 0}
        # if video_path_ != '':

        get_available_cameras()
        self.available_cameras = get_available_cameras(upper_bound=10, lower_bound=0)
        self.all_data = cv2.VideoCapture(self.available_cameras[-1])

        self.skel = Skeleton()
        self.net = PoseEstimationWithMobileNet()
        checkpoint = torch.load("models/checkpoint_iter_370000.pth", map_location='cuda')
        load_state(self.net, checkpoint)

    def name(self):
        return "video"

    def _read_data (self):
        # if (self.dataframe_num >= len (self.all_data)):
        #     read_data = 0
        #     return
        # self.frame_skel = run_demo(self.all_data)


        _, img = self.all_data.read()


        self.read_data = get_skel_coords(self.net, img, 256, False, 1, 1) #self.all_data [self.dataframe_num]

        # self.dataframe_num += 1

    def _process_data(self):
        if sum (self.read_data) != -36 and self.read_data != []:
            # print ("hehm", self.read_data)
            self.skel.read_data = self.read_data
            self.skel._process_data()
            self.processed_data = self.skel.processed_data
            # print(self.processed_data)

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

class Markov_chain (Modality):
    def __init__ (self, video_path_ = ""):
        self.read_data        = []
        self.interpreted_data = []

        self.timeout = Timeout_module (0.7)
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

        print ("com", comm)

        return comm

    def get_command(self, skip_reading_data=False):
        self._read_data()
        self._process_data()
        self._interpret_data()

        return self._get_command()

    # def draw(self, img):
    #     pass

class Response_to_skeleton (Modality):
    def __init__ (self, video_path_ = ""):
        self.read_data        = []
        self.interpreted_data = []

        self.timeout = Timeout_module (0.7)

        self.dataframe_num = 0

        self.commands = {"noaction": [("noaction", [""])],
                         "1": [("/stand", [""])],
                         "2": [("/hands_sides", [""])]}

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

    def _process_data(self):
        pass

    def _interpret_data(self):
        pass

    def _get_command(self):
        comm = self.commands ["noaction"]

        if (self.timeout.timeout_passed ()):
            movement = classifier.classify (self.read_data)

            comm = self.commands[str(1)]

            self.tick += 1

        print ("com", comm)

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
