from modalities.keyboard_modality import Computer_keyboard
from modalities.video_modality import Video
from modalities.skeleton_modalities import Skeleton_3D
from modalities.realsense_modality import RealSense
import fsm
import robots
from common import *
from time import time, sleep
import sys
from value_tracker import Value_tracker

paths = {"kompaso" : {"model_path"   : "/home/kompaso/NAO_PROJECT/wenhai/source/test/human-pose-estimation-3d.pth",
                      "phrases_path" : "/home/kompaso/NAO_PROJECT/wenhai/data/sounds/phrases.txt",
                      "vision_path"  : "/home/kompaso/NAO_PROJECT/wenhai/robotics_course/modules/"},

         "elijah"  : {"model_path"   : "/Users/elijah/Dropbox/Programming/RoboCup/remote control/source/test/human-pose-estimation-3d.pth",
                      "phrases_path" : "/Users/elijah/Dropbox/Programming/RoboCup/remote control/data/sounds/phrases.txt",
                      "vision_path"  : "/Users/elijah/Dropbox/Programming/robotics_course/modules/"}}
user = "elijah"
#user = "kompaso"

sys.path.append (paths [user] ["vision_path"])
import input_output

class Manager:
    def __init__ (self, config_ = "", silent_mode_ = True, time_to_not_silent_ =  0, color_ = 230):
        self.inputs = {}
        self.robots_list = {}
        self.silent_mode = silent_mode_
        self.time_to_not_silent = time_to_not_silent_
        self.color = color_
        self.quit = False

    def __del__ (self):
        self.logfile.close ()
        cv2.destroyAllWindows()

    def create_window (self, WIND_X, WIND_Y):
        self.WIND_X = WIND_X
        self.WIND_Y = WIND_Y

        cv2.namedWindow("remote_controller", cv2.WINDOW_AUTOSIZE)
        cv2.resizeWindow("remote_controller", (WIND_Y, WIND_X))
        self.canvas = np.ones ((WIND_Y, WIND_X, 3), np.uint8) * self.color

    def init (self):
        self.curr_time = time()
        self.logfile = open("log/" + str(self.curr_time) + ".txt", "w+")
        self.tracker = Value_tracker()
        self.fsm_processor = fsm.FSM_processor ()
        self.start_time = self.curr_time

    def add_inputs (self, inputs):
        self.inputs.update (inputs)

    def add_robots (self, robots):
        self.robots_list.update (robots)

    def form_output_image (self, window_x_sz = -1, one_img_x_sz = -1):
        result = input_output.form_grid (self.output_images, window_x_sz, one_img_x_sz)

        return result

    def handle_keyboard (self):
        self.inputs["computer keyboard"][0]._read_data()

        keyboard_data = self.inputs ["computer keyboard"] [0].get_read_data ()

        if (keyboard_data == ord("q")):
            self.quit = True

        if (keyboard_data == ord("-")):
            self.silent_mode = not self.silent_mode

        if (self.curr_time - self.start_time >= self.time_to_not_silent):
            self.silent_mode = False
            self.time_to_not_silent = 1000000000

    def handle_modalities (self):
        self.output_images = []
        self.output_names  = []

        for modality in self.inputs.keys ():
            skip_reading_data = False

            if (modality == "computer keyboard"):
                skip_reading_data = True

            command = self.inputs [modality] [0].get_command (skip_reading_data)

            self.logfile.write (str (self.curr_time) + str (command))

            action = self.fsm_processor.handle_command (command)

            if (self.silent_mode == False):
                for key in self.inputs [modality] [1]:
                    if (key in self.robots_list.keys ()):
                        self.robots_list [key].add_action (action)

            modality_frames = self.inputs [modality] [0].draw (self.canvas)

            if (modality_frames [0].shape [0] > 1):
                self.output_images += modality_frames
                self.output_names.append (modality)

    def handle_robots (self):
        self.canvas = np.ones ((self.WIND_Y, self.WIND_X, 3), np.uint8) * self.color
        canvas_ = self.canvas.copy ()
        output_images += tracker.draw (canvas)

        if (self.silent_mode == False):
            for key in self.robots_list.keys ():
                self.robots_list [key].on_idle ()

        list (robots_list.items ()) [0] [1].plot_state (canvas_, 150, 40, 2.5)

        self.output_images.append (canvas_)
        self.output_names.append  ("remote controller")

    def on_idle (self):
        self.curr_time = time ()
        # tracker.update("time", curr_time)

        self.handle_keyboard   ()
        self.handle_modalities ()
        self.handle_robots     ()

        if (silent_mode == True):
            self.canvas = cv2.putText (self.canvas, "silent mode", (30, 100), cv2.FONT_HERSHEY_SIMPLEX,
                   1, (0, 255, 0), 2, cv2.LINE_AA)

        sleep  (0.02)

        return {"quit" : self.quit}

def main():
    AUTONOMOUS = True #without physical robot

    manager = Manager ()
    manager.create_window (800, 700)
    manager.init ()

    inputs = {"computer keyboard" : (Computer_keyboard (paths [user] ["phrases_path"],
                                    logger_ = manager.tracker), ["physical", "simulated2"]) ,

              "video input": (Video(video_path_ = "", model_path_ = paths [user] ["model_path"],
              base_height_ = 30, logger_ = manager.tracker), ["physical", "simulated2"]) }
              # "Realsense input": (RealSense(video_path_ = "", model_path_ = paths [user] ["model_path"],
              # base_height_ = 300, logger_ = tracker), ["physical", "simulated2"])}

              # "archive skeleton"  : (Skeleton_3D (skeleton_path_ = "/home/kompaso/diplom_modules/S001C001P001R001A010.skeleton", logger_ = tracker),
              #                       ["simulated2"])}

    manager.add_inputs (inputs)
    manager.add_robots ({"simulated2" : robots.Simulated_robot (logger_ = manager.tracker)})

    if (AUTONOMOUS == False):
        ip = "192.168.1.66"
        manager.add_robots ({"physical" : robots.Real_robot (ip, "9569", logger_ = manager.tracker)})

    while (True):
        manager.handle_keyboard ()

        if (manager.on_idle () ["quit"] == True):
            break

        cv2.imshow ("remote_controller", manager.form_output_image (1400))

if __name__ == '__main__':
    main()
