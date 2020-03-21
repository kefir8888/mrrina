import cv2
import numpy as np
import time
import math
from pathlib import Path

#from IPython.display import clear_output

#URL-requests to the robot
import requests

#speech generation
import os.path
import cyrtranslit
from gtts import gTTS

#.mp3 files playing
#from pygame import mixer

class Timeout_module:
    def __init__ (self, timeout_):
        self.curr_time = 0
        self.last_action_time = 0
        self.timeout = timeout_

        self._update ()

    def _update (self):
        self.curr_time = time.time ()

    def _update_last_action_time (self, new_last_action_time = -1):
        if (new_last_action_time == -1):
            self.last_action_time = self.curr_time

        else:
            self.last_action_time = new_last_action_time

    def timeout_passed (self, additional_condition = True, print_time = False):
        self._update ()

        time_from_last_action = self.curr_time - self.last_action_time

        if (print_time == True):
            print ("time from last command: ", time_from_last_action)

        if (time_from_last_action > self.timeout and additional_condition == True):
            self._update_last_action_time ()
            return True

        else:
            return False

def rus_line_to_eng (line):
    out = cyrtranslit.to_latin (line, 'ru')
    out = "".join (c for c in out if c not in ['!', '.', '#', ':', "'", '?', ' ', '-', '\'', ',', '\n'])
    return out

def angle_2_vec_ (x1, y1, x2, y2):
    dot = x1*x2 + y1*y2
    det = x1*y2 - y1*x2
    angle = math.atan2(det,dot)
    # print(angle)

    return angle

def angle_2_vec (vec1, vec2):
    return angle_2_vec_ (vec1 [0], vec1 [1], vec2 [0], vec2 [1])
