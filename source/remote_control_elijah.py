# coding=utf-8

from modalities.keyboard_modality   import Computer_keyboard
from modalities.video_modality      import Video
from modalities.skeleton_modalities import Skeleton_3D
#from modalities.realsense_modality  import RealSense
from modalities.music_modality import Cyclic, Skeleton_3D_Music_to_dance, Archive_angles, External_model
from modalities.skeleton_modalities import Skeleton_3D
from modalities.voice_recognition import Voice_recognition

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

    manager.add_input ({"computer keyboard" : (Computer_keyboard (paths [user] ["phrases_path"],
                         logger_ = manager.tracker), ["physical1", "simulated2", "simulated3d"])})

    manager.add_input ({"voice recognition" : (Voice_recognition (), ['fake'])})

    if (KB_ONLY == False):
        manager.add_input ({"music": (Cyclic ("/Users/elijah/Dropbox/Programming/RoboCup/remote control/data/music/gorillaz_collar_part.mp3",
            logger_ = manager.tracker, dance_length_ = 940), ["physical1", "physical2", "simulated3d"])})

        #manager.add_input ({"skeleton": (Skeleton_3D_Music_to_dance ( "/Users/elijah/Downloads/dataset/DANCE_R_10/skeletons.json",
        #                      logger_ = manager.tracker), ["simulated3d", "physical1"])})

        manager.add_input ({"angles": (Archive_angles ( "/Users/elijah/Downloads/dataset/lezginka/angles_generated.json",
                             logger_ = manager.tracker), ["simulated3d", "physical1", "physical2"])})

        #manager.add_input ({"angles": (Archive_angles ( "/Users/elijah/Downloads/dataset/DANCE_W_2/angles.json",
        #                     logger_ = manager.tracker), ["simulated3d", "physical1", "physical2"])})

        #manager.add_input({"model": (External_model("/Users/elijah/Dropbox/Programming/RoboCup/remote control/source/generation/trained39-2.pth",
        #manager.add_input({"model": (External_model("/Users/elijah/Dropbox/Programming/RoboCup/remote control/source/generation/trained59.pth",
                                                    #"/Users/elijah/Downloads/dataset/DANCE_T_4/audio.mp3",
        #                                            "/Users/elijah/Downloads/dataset/lezginka/lezgi.mp3",
        #                                            logger_ = manager.tracker, length_ = 2370000), ["physical1", "simulated2", "simulated3d"])})

    #manager.add_data_stream (["angles"], ["physical"], process_all_data = True)

    manager.add_robot ({"fake" : robots.Fake ()})

    manager.add_robot ({"simulated2" : robots.Simulated_robot (logger_ = manager.tracker, omit_warnings_ = True)})

    manager.add_robot ({"simulated3d" : robots.Simulated_robot_3D (WIND_X_ = WIND_X, WIND_Y_ = WIND_Y,
                                            logger_ = manager.tracker, omit_warnings_ = True)})
    manager.add_robot ({"simulated3d2" : robots.Simulated_robot_3D (WIND_X_ = WIND_X, WIND_Y_ = WIND_Y,
                                            logger_ = manager.tracker, omit_warnings_ = True)})

    if (AUTONOMOUS == False):
        ip = paths [user] ["robot_ip"]

        #manager.add_robot({"physical1": robots.Real_robot_qi("10.0.0.102", logger_=manager.tracker,
        #                                                    action_time_=0.05, omit_warnings_=True,
        #                                                     additional_delay_ = 0.07)})

        #manager.add_robot({"physical1": robots.Real_robot_semiautonomous("10.0.0.102", logger_=manager.tracker,
        #        action_time_=0.05, omit_warnings_=True, additional_delay_ = 0.07)})

        #manager.add_input ({"robot_view" : manager.Video_source (manager.robots ["physical1"].get_exchange_folder (), )})

        #manager.add_robot({"physical2": robots.Real_robot_qi("10.0.0.104", logger_=manager.tracker,
        #                                                    action_time_=0.05, omit_warnings_=True)})

        #manager.add_robot({"physical": robots.Real_robot_timeline(ip, "9569", logger_=manager.tracker,
        #                    action_time_=0.11, omit_warnings_=True)})

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

"""#сделать закрытие queue
#свое окно (если это все-таки возможно) или передача картинки в родительский процесс
#переделать менеджер основного проекта
#
#еще можно по фану добавить генерацию питоновского скрипта, который выполняет пришедшую на вход последовательность действий
#локальная обработка данных, связь с интернетом, связь с железом, многопоточность
#
#Кажется, наступил момент, когда нужно делать интерфейс для изменения потоков данных, которые идут на роботов,
#и не только этих потоков, а вообще всех на уровне модальностей."""

    # manager.add_input ({"computer keyboard" : (Computer_keyboard (), ["real1", "real2", "sim3d"])})
    # manager.add_input ({"safe movements" : (Cyclic ("song1.mp3"),  ["real1", "real2", "simd"])})
    # manager.add_input ({"generated" : (External_model ("model.pth", "song1.mp3"), ["sim3d", "real2"])})
    #
    # manager.add_robot ({"sim3d" : robots.Simulated_robot_3D ()})
    # manager.add_robot ({"real1" : robots.Real_robot_qi ("10.0.0.102")})
    # manager.add_robot ({"real2" : robots.Real_robot_qi ("10.0.0.104")})