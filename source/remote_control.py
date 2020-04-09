import modalities
import fsm
import robots
from common import *
from time import time, sleep
import sys

#sys.path.append("/home/kompaso/DEBUG/Debug/remote control/robotics_course/modules/")
sys.path.append("/Users/elijah/Dropbox/Programming/robotics_course/modules/")

import input_output

#class Manager:
#    def __init__ (self, config_ = ""):

#names

def main():
    AUTONOMOUS = True #without physical robot

    WIND_X = 800
    WIND_Y = 500
    cv2.namedWindow  ("remote_controller", cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow ("remote_controller", (WIND_Y, WIND_X))
    canvas = np.ones ((WIND_Y, WIND_X, 3), np.uint8) * 230

    curr_time = time ()
    logfile = open ("log/" + str (curr_time) + ".txt", "w+")

    inputs = {"computer keyboard" : (modalities.Computer_keyboard ("/Users/elijah/Dropbox/Programming/RoboCup/remote control/data/sounds/phrases.txt"),
                                     ["physical", "simulated2"]), #}

              #"response" : (modalities.Response_to_skeleton ("/Users/elijah/Dropbox/Programming/RoboCup/remote control/data/skeletons/skel_up_ponomareva.txt"),
              #              ["simulated1", "physical"]),
              #"music": (modalities.Music(
              #    "/Users/elijah/Dropbox/Programming/RoboCup/remote control/data/music/gorillaz_collar_part.mp3"),
              #             ["simulated1", "physical"])}

              "video input" : (modalities.Video(), ["physical", "simulated2"])}
              #"archive skeleton"  : modalities.Skeleton ("/home/kompaso/Desktop/ISP/lightweight-human-pose-estimation_2/skel/skel_robot_ponomareva.txt")}
              #"archive skeleton"  : (modalities.Skeleton ("/Users/elijah/Dropbox/Programming/RoboCup/remote control/data/skeletons/skel_up_ponomareva.txt"),
              #                       ["simulated2"])}

    robots_list = {}

    #if (AUTONOMOUS == True):
    robots_list.update ({"simulated2" : robots.Simulated_robot ()})

    if (AUTONOMOUS == False):
        # ip = "192.168.1.70"
        #ip = "10.6.255.230"
        ip = "10.0.0.105"

        robots_list.update ({"physical" : robots.Real_robot (ip, "9569")})
        #robots_list.update({"simulated1": robots.Simulated_robot()})

    fsm_processor = fsm.FSM_processor ()

    while (True):
        curr_time = time ()

        inputs ["computer keyboard"] [0]._read_data ()

        keyboard_data = inputs ["computer keyboard"] [0].get_read_data ()
        if (keyboard_data == ord ("q")):
            break

        output_images = []
        output_names  = []

        # print ("keys", inputs.keys ())

        for modality in inputs.keys ():
            skip_reading_data = False

            if (modality == "computer keyboard"):
                skip_reading_data = True

            command = inputs [modality] [0].get_command (skip_reading_data)

            #print ("modality: ", modality)
            #print(command)
            logfile.write (str (curr_time) + str (command))

            action = fsm_processor.handle_command (command)
            #print ("ACTION)0))))0)))=========|-)", action)

            for key in inputs [modality] [1]:
                if (key in robots_list.keys ()):
                    robots_list [key].add_action (action)

            canvas = np.ones((WIND_Y, WIND_X, 3), np.uint8) * 200

            modality_frames = inputs [modality] [0].draw (canvas)
            #print (modality, "mod")
            if (modality_frames [0].shape [0] > 1):
                # cv2.imshow (modality, modality_frame)
                output_images += modality_frames
                output_names.append  (modality)

        for key in robots_list.keys ():
            robots_list [key].on_idle ()

        list (robots_list.items ()) [0] [1].plot_state (canvas, 150, 40, 2.5)

        # cv2.imshow ("remote_controller", canvas)
        output_images.append (canvas)

        output_names.append ("remote controller")

        cv2.imshow ("remote_controller", input_output.form_grid (output_images, 1200, -1))

        sleep  (0.02)

    #logfile.close ()
    cv2.destroyAllWindows ()

if __name__ == '__main__':
    main()
