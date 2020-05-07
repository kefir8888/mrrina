from modalities.keyboard_modality   import Computer_keyboard
from modalities.video_modality      import Video
from modalities.skeleton_modalities import Skeleton_3D
from modalities.realsense_modality  import RealSense

import robots
from manager import Manager

from common import *

user = "kompaso"

sys.path.append (paths [user] ["vision_path"])
import input_output

def main():
    AUTONOMOUS = True #without physical robot

    manager = Manager ()
    manager.create_window (800, 700)
    manager.init ()

    inputs = {"computer keyboard" : (Computer_keyboard (paths [user] ["phrases_path"],
                                    logger_ = manager.tracker), ["physical", "simulated2"])}#},

              #"Realsense input": (RealSense(video_path_ = "", model_path_ = paths [user] ["model_path"],
              #base_height_ = 300, logger_ = manager.tracker), ["physical", "simulated2"])}

              # "archive skeleton"  : (Skeleton_3D (skeleton_path_ = "/home/kompaso/diplom_modules/S001C001P001R001A010.skeleton", logger_ = tracker),
              #                       ["simulated2"])}

    manager.add_inputs (inputs)
    manager.add_robots ({"simulated2" : robots.Simulated_robot (logger_ = manager.tracker)})

    if (AUTONOMOUS == False):
        ip = "192.168.1.66"
        manager.add_robots ({"physical" : robots.Real_robot (ip, "9569", logger_ = manager.tracker)})

    while (True):
        if (manager.on_idle () ["quit"] == True):
            break

        cv2.imshow ("remote_controller", manager.form_output_image (900))

if __name__ == '__main__':
    main()
