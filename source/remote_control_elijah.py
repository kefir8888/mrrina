from modalities.keyboard_modality   import Computer_keyboard
from modalities.video_modality      import Video
from modalities.skeleton_modalities import Skeleton_3D
#from modalities.realsense_modality  import RealSense
from modalities.music_modality import Cyclic, Skeleton_3D_Music_to_dance, Archive_angles, External_model
from modalities.skeleton_modalities import Skeleton_3D

import robots
import service.input_output as input_output

from service.manager import Manager
from common import *

user = "elijah"

WIND_X, WIND_Y = 800, 700

def main():
    AUTONOMOUS = False
    AUTONOMOUS = True

    KB_ONLY = False
    #KB_ONLY = True

    manager = Manager (draw_tracker_ = True)
    manager.create_window (WIND_X, WIND_Y)
    manager.init ()

    manager.add_inputs ({"computer keyboard" : (Computer_keyboard (paths [user] ["phrases_path"],
                         logger_ = manager.tracker), ["physical", "simulated2", "simulated3d"])})

    if (KB_ONLY == False):
        manager.add_inputs ({"music": (Cyclic ("/Users/elijah/Dropbox/Programming/RoboCup/remote control/data/music/gorillaz_collar_part.mp3",
            logger_ = manager.tracker, dance_length_ = 15000), ["physical", "simulated3d"])})

        #manager.add_inputs ({"skeleton": (Skeleton_3D_Music_to_dance ( "/Users/elijah/Downloads/dataset/DANCE_R_10/skeletons.json",
        #                     logger_ = manager.tracker), ["simulated3d"])})

        #manager.add_inputs ({"angles": (Archive_angles ( "/Users/elijah/Downloads/dataset/DANCE_R_6/angles.json",
        #                     logger_ = manager.tracker), ["simulated3d"])})

        #manager.add_inputs({"model": (External_model("/Users/elijah/Dropbox/Programming/RoboCup/remote control/source/generation/trained39-2.pth",
        manager.add_inputs({"model": (External_model("/Users/elijah/Dropbox/Programming/RoboCup/remote control/source/generation/trained59.pth",
                                                    "/Users/elijah/Downloads/dataset/DANCE_R_4/audio.mp3",
                             logger_ = manager.tracker), ["physical", "simulated2", "simulated3d2"])})

    manager.add_robots ({"simulated2" : robots.Simulated_robot (logger_ = manager.tracker, omit_warnings_ = True)})

    manager.add_robots ({"simulated3d" : robots.Simulated_robot_3D (WIND_X_ = WIND_X, WIND_Y_ = WIND_Y,
                                            logger_ = manager.tracker, omit_warnings_ = True)})
    manager.add_robots ({"simulated3d2" : robots.Simulated_robot_3D (WIND_X_ = WIND_X, WIND_Y_ = WIND_Y,
                                            logger_ = manager.tracker, omit_warnings_ = True)})

    if (AUTONOMOUS == False):
        ip = paths [user] ["robot_ip"]
        manager.add_robots ({"physical" : robots.Real_robot_qi (ip, "9569", logger_ = manager.tracker,
                            action_time_ = 0.11, omit_warnings_ = True)})

    # common_prefix = "/Users/elijah/Downloads/dataset/DANCE_"
    # common_infix = "_"
    # common_postfix = "/skeletons.json"
    #
    # unique = {"C" : [str (i) for i in range (6, 10)] + ["1"],
    #           "T" : [str (i) for i in range (1, 10)],
    #           "W" : [str (i) for i in range (1, 35)],}
    #
    # data_paths = []
    # file_num = 0
    #
    # for letter in unique.keys ():
    #     for num in unique [letter]:
    #         path = common_prefix + letter + common_infix + num + common_postfix
    #         data_paths.append (path)

    while (True):
        # if ("skeleton" not in manager.inputs.keys ()):
        #     if (file_num >= len (data_paths)):
        #         print ("total uptime: ", manager.tracker.get_value ("uptime"))
        #         return
        #
        #     print ("reading ", data_paths [file_num])
        #     manager.add_inputs ({"skeleton": (Skeleton_3D_Music_to_dance (data_paths [file_num],
        #                         logger_ = manager.tracker), ["simulated3d"])})
        #
        #     file_num += 1
        #
        # if (manager.inputs ["skeleton"] [0].end_of_data () == True or
        #     manager.inputs["skeleton"][0].data_loaded () == False):
        #     del manager.inputs ["skeleton"]

        if (manager.on_idle () ["quit"] == True):
            break

        cv2.imshow ("remote_controller", manager.form_output_image (2700))

if __name__ == '__main__':
    main()

    # manager = Manager (draw_tracker_ = True)
    # manager.create_window (WIND_X, WIND_Y)
    # manager.init ()
    #
    # manager.add_inputs ({"computer keyboard" : (Computer_keyboard (paths [user] ["phrases_path"],
    #                      logger_ = manager.tracker), ["physical", "simulated2", "simulated3d"])})
    #
    # manager.add_inputs ({"music": (Cyclic ("/data/music/gorillaz_collar_part.mp3",
    #                      logger_ = manager.tracker, dance_length_ = 150), ["physical", "simulated2"])})
    #
    # manager.add_inputs ({"angles": (External_model ( "/generation/trained.pth",
    #                      "/Users/elijah/Downloads/dataset/DANCE_W_3/audio.mp3",
    #                      logger_ = manager.tracker), ["physical", "simulated2", "simulated3d"])})
    #
    # manager.add_robots ({"simulated3d" : robots.Simulated_robot_3D (WIND_X_ = WIND_X, WIND_Y_ = WIND_Y,
    #                                         logger_ = manager.tracker, omit_warnings_ = True)})
    #
    # manager.add_robots ({"physical" : robots.Real_robot_qi (paths [user] ["robot_ip"], "9569",
    #                      logger_ = manager.tracker, action_time_ = 0.21, omit_warnings_ = True)})
    #
    # while (True):
    #     if (manager.on_idle () ["quit"] == True):
    #         break
    #
    #     cv2.imshow ("remote_controller", manager.form_output_image (2700))