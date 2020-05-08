from modalities.keyboard_modality   import Computer_keyboard
from modalities.video_modality      import Video
from modalities.skeleton_modalities import Skeleton_3D
#from modalities.realsense_modality  import RealSense

import robots
import input_output

from manager import Manager
from common import *

user = "elijah"

def main():
    AUTONOMOUS = False

    manager = Manager ()

    manager.create_window (800, 700)
    manager.init ()

    manager.add_inputs ({"computer keyboard" : (Computer_keyboard (paths [user] ["phrases_path"],
                         logger_ = manager.tracker), ["physical", "simulated2"])})

    manager.add_robots ({"simulated2" : robots.Simulated_robot (logger_ = manager.tracker)})

    if (AUTONOMOUS == False):
        ip = paths [user] ["robot_ip"]
        manager.add_robots ({"physical" : robots.Real_robot_qi (ip, "9569", logger_ = manager.tracker)})

    while (True):
        if (manager.on_idle () ["quit"] == True):
            break

        cv2.imshow ("remote_controller", manager.form_output_image (1400))

if __name__ == '__main__':
    main()
