from modalities.keyboard_modality import Computer_keyboard
from modalities.video_modality import Video
from modalities.realsense_modality import RealSense
import fsm
import robots
from common import *
from time import time, sleep
import sys

#class Manager:
#    def __init__ (self, config_ = ""):

class Value_tracker:
    def __init__ (self):
        self.tracked = {}

    def name (self):
        return "value_tracker"

    def update (self, value_name, value):
        self.tracked.update ({value_name : value})

    def draw (self, img):
        result = np.array (img)

        i = 0
        for k, v in self.tracked.items():
            result = cv2.putText (result, k + ": " + str (v) [:5], (30, 60 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX,
                                 2, (100, 25, 130), 2, cv2.LINE_AA)
            i += 1

        return [result]

paths = {"kompaso" : {"model_path"   : "/home/kompaso/NAO_PROJECT/wenhai/source/test/human-pose-estimation-3d.pth",
                      "phrases_path" : "/home/kompaso/NAO_PROJECT/wenhai/data/sounds/phrases.txt",
                      "vision_path"  : "/home/kompaso/NAO_PROJECT/wenhai/robotics_course/modules/"},

         "elijah"  : {"model_path"   : "/Users/elijah/Dropbox/Programming/RoboCup/remote control/source/test/human-pose-estimation-3d.pth",
                      "phrases_path" : "/Users/elijah/Dropbox/Programming/RoboCup/remote control/data/sounds/phrases.txt",
                      "vision_path"  : "/Users/elijah/Dropbox/Programming/robotics_course/modules/"}}

# user = "elijah"
user = "kompaso"

sys.path.append (paths [user] ["vision_path"])
import input_output

def main():
    AUTONOMOUS = True #without physical robot

    WIND_X = 800
    WIND_Y = 500
    cv2.namedWindow  ("remote_controller", cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow ("remote_controller", (WIND_Y, WIND_X))
    canvas = np.ones ((WIND_Y, WIND_X, 3), np.uint8) * 230

    curr_time = time ()
    logfile = open ("log/" + str (curr_time) + ".txt", "w+")

    tracker = Value_tracker ()

    inputs = {"computer keyboard" : (Computer_keyboard (paths [user] ["phrases_path"],
                                    logger_ = tracker), ["physical", "simulated2"]), #}

              #"response" : (modalities.Response_to_skeleton ("/Users/elijah/Dropbox/Programming/RoboCup/remote control/data/skeletons/skel_up_ponomareva.txt"),
              #              ["simulated1", "physical"]),
              #"music": (modalities.Music(
              #    "/Users/elijah/Dropbox/Programming/RoboCup/remote control/data/music/gorillaz_collar_part.mp3"),
              #             ["simulated1", "physical"])}

              #"video input" : (modalities.Video(), ["physical", "simulated2"])}

              #../data/videos/female_dancer_studio.mp4
              "video input": (Video(video_path_ = "", model_path_ = paths [user] ["model_path"],
              base_height_ = 100, logger_ = tracker), ["physical", "simulated2"]) ,
              "Realsense input": (RealSense(video_path_ = "", model_path_ = paths [user] ["model_path"],
              base_height_ = 300, logger_ = tracker), ["physical", "simulated2"])}

    #"archive skeleton"  : modalities.Skeleton ("/home/kompaso/Desktop/ISP/lightweight-human-pose-estimation_2/skel/skel_robot_ponomareva.txt")}
              #"archive skeleton"  : (modalities.Skeleton ("/Users/elijah/Dropbox/Programming/RoboCup/remote control/data/skeletons/skel_up_ponomareva.txt"),
              #                       ["simulated2"])}

    robots_list = {}

    #if (AUTONOMOUS == True):

    robots_list.update ({"simulated2" : robots.Simulated_robot (logger_ = tracker)})

    if (AUTONOMOUS == False):
        ip = "192.168.43.152"
        #ip = "10.6.255.230"
        # ip = "10.0.0.101"

        robots_list.update ({"physical" : robots.Real_robot (ip, "9569", logger_ = tracker)})
        #robots_list.update({"simulated1": robots.Simulated_robot()})

    fsm_processor = fsm.FSM_processor ()

    silent_mode = True
    time_to_not_silent = 5
    start_time = curr_time

    ###пример трекера
    #tracker.update ("asda", 5)

    while (True):
        curr_time = time ()
        #tracker.update("time", curr_time)

        inputs ["computer keyboard"] [0]._read_data ()

        keyboard_data = inputs ["computer keyboard"] [0].get_read_data ()
        if (keyboard_data == ord ("q")):
            break

        if (keyboard_data == ord ("-")):
            silent_mode = not silent_mode

        if (curr_time - start_time >= time_to_not_silent):
            silent_mode = False
            time_to_not_silent = 1000000000

        output_images = []
        output_names  = []

        # print ("keys", inputs.keys ())

        for modality in inputs.keys ():
            skip_reading_data = False

            if (modality == "computer keyboard"):
                skip_reading_data = True

            command = inputs [modality] [0].get_command (skip_reading_data)

            # print ("modality: ", modality)
            # print (command)
            logfile.write (str (curr_time) + str (command))

            action = fsm_processor.handle_command (command)
            #print ("ACTION)0))))0)))=========|-)", action)

            if (silent_mode == False):
                for key in inputs [modality] [1]:
                    if (key in robots_list.keys ()):
                        robots_list [key].add_action (action)

            canvas = np.ones((WIND_Y, WIND_X, 3), np.uint8) * 200
            output_images += tracker.draw (canvas)

            modality_frames = inputs [modality] [0].draw (canvas)
            #print (modality, "mod")
            if (modality_frames [0].shape [0] > 1):
                # cv2.imshow (modality, modality_frame)
                output_images += modality_frames
                output_names.append  (modality)

        if (silent_mode == False):
            for key in robots_list.keys ():
                robots_list [key].on_idle ()

        list (robots_list.items ()) [0] [1].plot_state (canvas, 150, 40, 2.5)

        if (silent_mode == True):
            canvas = cv2.putText (canvas, "silent mode", (30, 100), cv2.FONT_HERSHEY_SIMPLEX,
                   1, (0, 255, 0), 2, cv2.LINE_AA)

        # cv2.imshow ("remote_controller", canvas)
        output_images.append (canvas)

        output_names.append ("remote controller")

        cv2.imshow ("remote_controller", input_output.form_grid (output_images, 1400, -1))

        sleep  (0.02)

    #logfile.close ()
    cv2.destroyAllWindows ()

if __name__ == '__main__':
    main()
