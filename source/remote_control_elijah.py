from modalities.keyboard_modality   import Computer_keyboard
from modalities.video_modality      import Video
from modalities.skeleton_modalities import Skeleton_3D
#from modalities.realsense_modality  import RealSense
from modalities.music_modality import Cyclic
from modalities.skeleton_modalities import Skeleton_3D, Skeleton_3D_Music_to_dance

import robots
import service.input_output

from service.manager import Manager
from common import *

user = "elijah"

def main():
    AUTONOMOUS = False
    AUTONOMOUS = True

    KB_ONLY = False
    #KB_ONLY = True

    manager = Manager (draw_tracker_ = False)
    manager.create_window (800, 700)
    manager.init ()

    manager.add_inputs ({"computer keyboard" : (Computer_keyboard (paths [user] ["phrases_path"],
                         logger_ = manager.tracker), ["physical", "simulated2"])})

    if (KB_ONLY == False):
        manager.add_inputs ({"music": (Cyclic ("/Users/elijah/Dropbox/Programming/RoboCup/remote control/data/music/gorillaz_collar_part.mp3",
            logger_ = manager.tracker, dance_length_ = 150), ["physical", "simulated2"])})

        #manager.add_inputs ({"skeleton": (Skeleton_3D_Music_to_dance ( "/Users/elijah/Downloads/dataset/DANCE_C_9/skeletons.json",
        #                     logger_ = manager.tracker), ["physical", "simulated2"])})

    manager.add_robots ({"simulated2" : robots.Simulated_robot (logger_ = manager.tracker)})

    if (AUTONOMOUS == False):
        ip = paths [user] ["robot_ip"]
        manager.add_robots ({"physical" : robots.Real_robot_qi (ip, "9569", logger_ = manager.tracker, action_time_ = 0.21)})

    while (True):
        if (manager.on_idle () ["quit"] == True):
            break

        cv2.imshow ("remote_controller", manager.form_output_image (1400))

if __name__ == '__main__':
    main()
