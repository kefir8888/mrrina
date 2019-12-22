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

class Robot:
    def __init__(self, timeout_ = 0):
        self.queue         = []
        self.commands_sent = 0

        self.available_commands = {"/rest"  : ("/action=/rest&text=", "a"),
                                   "/stand" : ("/action=/stand&text=", "a"),
                                   "/free"  : ("/action=/free&text=", "a")}

        self.timeout_module = Timeout_module (timeout_)

    def _send_command (self, command, args):
        pass

    def on_idle (self):
        if (self.timeout_module.timeout_passed (len (self.queue) > self.commands_sent)):
            next_command = self.queue [self.commands_sent]
            self.commands_sent += 1

            #if (next_command [0] != "noaction"):
            self._send_command (next_command)

    def add_action (self, action):
        #print ("appending ", action)
        for act in action:
            if (act [0] != "noaction"):
                self.queue = self.queue + action

class Fake_robot(Robot):
    def __init__(self, timeout_ = 0.5):
        Robot.__init__ (self, timeout_)

    def _send_command (self, action):
        if (action [0] in self.available_commands.keys ()):
            print ("sending command [fake]: ", action)

        else:
            print ("action :", action, " is not implemented")

class Real_robot(Robot):
    def __init__(self, ip_num, port_ = 9559, timeout_ = 4.5):
        Robot.__init__ (self, timeout_)

        self.ip_prefix = "http://"
        self.ip_postfix = ":"

        self.ip   = self.ip_prefix + ip_num + self.ip_postfix
        self.port = port_

        self.free = False
        self.free_timeout_module = Timeout_module (0.4)

    def _send_command (self, action):
        r = -1

        #if (action [0] == "noaction"):
        #    pass

        if (action in self.available_commands.keys ()):
            r = requests.get (self.ip + self.port + "/?" + "action="
                + action [0] + "&" + "text=" + str(action [1]))

        else:
            print ("action :", action, " is not implemented")

        return r

    def on_idle (self):
        if (self.free_timeout_module.timeout_passed ()):
            r = self._send_command ("/free")
            #print ("resp", r)

            free = int (str (r) [13:14]) #6 free, 7 not free; don't ask, don't tell

            if (free == 6):
                self.free = True

            else:
                self.free = False

        print ("queue", self.queue [self.commands_sent:])

        if (self.timeout_module.timeout_passed (len (self.queue) > self.commands_sent) and
            self.free == True):
            command = self.queue [self.commands_sent]

            print ("azaz", command)

            self._send_command (command)
            self.commands_sent += 1

class Modality:
    def __init__ (self):
        pass

    def name (self):
        return "not specified"

class Computer_keyboard (Modality):
    def __init__ (self, key_to_command_ = {"z" : "empty"}):
        self.read_data        = 0x00
        self.processed_data   = 0x00
        self.interpreted_data = 0x00

        self.key_to_command = {"z"        : ("/stand",   [""]),
                               "c"        : ("/rest",    [""]),
                               "n"        : ("next",     [""]),
                               "noaction" : ("noaction", [""])}

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

            if (key in self.key_to_command.keys ()):
                return self.key_to_command [key]

        return self.key_to_command ["noaction"]

    def get_command (self, skip_reading_data = False):
        if (skip_reading_data == False):
            self._read_data ()

        self._process_data   ()
        self._interpret_data ()

        return self._get_command ()

    def draw (self, img):
        pass

#class Voice (Modality):
#class Virtual_keyboard (Modality):

class FSM_processor:
    def __init__ (self, config_file = ""):
        self.fsms = []
        self.active_fsm = -1
        self.current_state = 0

        if (config_file == ""):
            files = sorted (Path ("/Users/elijah/Dropbox/Programming/RoboCup/remote control/tasks/").glob('*.txt'))
            print (files)

            fsm_num = 0

            for filename in files:
                print (filename)
                file = open (filename)
                ln = file.readline ()

                self.fsms.append ([])

                while (ln != ""):
                    self.fsms [fsm_num].append (ln [:-1])
                    ln = file.readline ()

                print ("fsmmmm", self.fsms [fsm_num])

                fsm_num += 1

            if (len (self.fsms) > 0):
                self.active_fsm = 0

        else:
            print ("to be implemented")

    def handle_command (self, command):
        #print ("command to fsm proc: ", command)

        if (command [0] [0] == "/"):
            return [command]

        elif (command [0] == "noaction"):
            return [command]

        else:
            if (command [0] == "next"):
                print ("command next")
                curr_st = self.fsms [self.active_fsm] [self.current_state]

                self.current_state += 1

                if (self.current_state >= len (self.fsms [self.active_fsm])):
                    self.current_state = 0

                return [curr_st [0], curr_st [1]]

def main():
    AUTONOMOUS = True #without robot

    WIND_X = 500
    WIND_Y = 500
    cv2.namedWindow  ("remote_controller", cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow ("remote_controller", (WIND_Y, WIND_X))
    canvas = np.ones ((WIND_Y, WIND_X, 3), np.uint8) * 230
        
    #logfile = open ("log/" + str (curr_time) + ".txt", "w+")
    
    modalities = {"computer keyboard" : Computer_keyboard ()}

    robot = Fake_robot ()

    if (AUTONOMOUS == False):
        robot = Real_robot ("10.0.0.105", "9569")

    fsm_processor = FSM_processor ()

    while (True):
        modalities ["computer keyboard"]._read_data ()

        keyboard_data = modalities ["computer keyboard"].get_read_data ()
        if (keyboard_data == ord ("q")):
            break

        #change task, go to next question

        for modality in modalities.keys ():
            skip_reading_data = False

            if (modality == "computer keyboard"):
                skip_reading_data = True

            command = modalities [modality].get_command (skip_reading_data)

            #print ("command: ", command)

            action = fsm_processor.handle_command (command)
            #print ("action", action)

            robot.add_action (action)

        robot.on_idle ()

        canvas = np.ones ((WIND_Y, WIND_X, 3), np.uint8) * 200
        
        cv2.imshow ("remote_controller", canvas)
        
        time.sleep  (0.9)        
            
    #logfile.close ()
    cv2.destroyAllWindows ()

if __name__ == '__main__':
    main()
