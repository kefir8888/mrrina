# coding=utf-8

from tkinter import Tk
import os
import requests

from interface import Main_window
from speech_processing import Words_processor, Dialogue_system#, Speech_parser

from multiprocessing import Process, freeze_support, SimpleQueue
#from multiprocessing.queues import 
import speech_recognition as sr

import time

import cv2
import numpy as np

from pathlib import Path

connection = SimpleQueue ()

class Robot:
    def __init__ (self, ip_num_, port_, autonomous_ = False):
        self.ip_prefix = "http://"
        self.ip_postfix = ":"

        #self.ip_num = ip_num_
        #self.ip     = self.ip_prefix + self.ip_num + self.ip_postfix
        self.change_ip (ip_num_)
        
        self.port   = port_

        self.autonomous = autonomous_

    def change_ip (self, new_ip):
        self.ip_num = new_ip
        self.ip     = self.ip_prefix + self.ip_num + self.ip_postfix

    def _send_command (self, command):
        if (self.autonomous == True):
            print ("Robot", self.ip, "in autonomous mode, skipping command", command)
            return

        try:
            r = requests.get (command)

        except:
            print ("cannot send command", command, "to robot", self.ip, self.port)

    def send_command (self, action, text):
        command = self.ip + str (self.port) + "/?action=" + action + "&text=" + text
        self._send_command (command)

    def copy_file_to_robot (self, filename_local, path_remote):
        #copy_str = "pscp -pw nao 1.mp3 nao@192.168.43.169:/home/nao/remote_control/sounds"
        copy_str = "pscp -pw nao " + filename_local + " nao@" + self.ip_num +\
            self.ip_postfix + path_remote

        time.sleep (0.1)

        print ("copy str", copy_str)
        #os.system ("")
        os.system (copy_str)

    def copy_file_from_robot (self, filename_remote, path_local):
        copy_str = "pscp -pw nao " + "nao@" + self.ip_num + self.ip_postfix +\
            filename_remote + " " + path_local

        os.system (copy_str + "> nothing.txt")

class Robot_state:
    def __init__ (self, speech_parser_, robot_, words_processor_, connection_, dialogue_system_):
        self.robot_state = "waiting"
        self.states_list = ["waiting", "playing_football", "wiki_search"]
        self.search_for_red_card = False

        self.speech_parser = speech_parser_
        
        self.robot = robot_
        self.words_processor = words_processor_
        self.connection = connection_
        self.dialogue_system  = dialogue_system_

        self.queue = []
        self.to_search_in_wiki = []
        
        self.lth = [100, 100, 100]
        self.hth = [200, 200, 200]
        
        cv2.namedWindow ("trackbars")
        cv2.createTrackbar ("rl", "trackbars", 100, 255, lambda n : self.change_lth (0, n))
        cv2.createTrackbar ("rh", "trackbars", 200, 255, lambda n : self.change_hth (0, n))
        cv2.createTrackbar ("gl", "trackbars", 100, 255, lambda n : self.change_lth (1, n))
        cv2.createTrackbar ("gh", "trackbars", 200, 255, lambda n : self.change_hth (1, n))
        cv2.createTrackbar ("bl", "trackbars", 100, 255, lambda n : self.change_lth (2, n))
        cv2.createTrackbar ("bh", "trackbars", 200, 255, lambda n : self.change_hth (2, n))

    def change_lth (self, ind, new_val):
        self.lth [ind] = new_val

    def change_hth (self, ind, new_val):
        self.hth [ind] = new_val

    def change_ip (self, new_ip):
        self.robot.change_ip (new_ip)

    def join (self):
        self.speech_parser.join ()
        self.connection.close ()

    def change_state (self, new_state):
        if (new_state in self.states_list):
            self.robot_state = new_state
            print ("setting state", self.robot_state)

        else:
            print ("cannot set state ", new_state)
            return

    def do_action (self, command):
        print ("do_action")
        
        if (self.robot_state == "wiki_search"):
            if (len (self.to_search_in_wiki) > 0):
                filename = "1.mp3"
                
                message = self.to_search_in_wiki [0]
                
                succ       = message ["success"]
                recognized = message ["recognized"]
                
                if (succ == False):
                    return
                
                print ("recognized")
                
                succ, content = self.words_processor.get_wiki_content (recognized)
        
                if (succ == True):
                    print ("content extracted")
                    succ, name = self.words_processor.generate_mp3 (content, filename)
        
                else:
                    print ("aborting")
                    return "nothing.nothing", succ
                
                self.robot.copy_file_to_robot (filename, "/home/nao/remote_control/sounds")
                
                command = {"type"           : "action",
                           "execution_time" : 0,
                           "contents"       : "/play_mp3",
                           "parameter"      : "1.mp3"}
                
                self.queue.append (command)
        
                self.change_state ("waiting")
                
                self.to_search_in_wiki = []

        elif (self.robot_state == "play_football"):
            print ("play_football state is a placeholder for now")

        elif (self.robot_state == "waiting"):
            self.robot.send_command (command ["contents"], command ["parameter"])

        else:
            print ("state", self.robot_state, "is supported, but not implemented")

    def send_ths (self):
        #send_command
        command = "/setths"
        text = str (self.lth [0])
        
        for i, value in enumerate ([self.lth [1], self.lth [2], self.hth [0], self.hth [1], self.hth [2]]):
            text += "&t" + str (i + 1) + "="
            text += str (value)
        
        self.robot.send_command (command, text)
        
        #refresh local image
        
        self.robot.send_command ("/makesavephoto", "a")
        
        time.sleep (0.6)
        
        local_path = "img.jpg"
        remote_path = "/home/nao/img.jpg"
        
        self.robot.copy_file_from_robot (remote_path, local_path)

    def execute_command (self, command):
        if (command ["type"] == "state_change"):
            self.change_state (command ["contents"])
        
        elif (command ["type"] == "action"):
            self.do_action (command)

    def on_idle (self):
        #print ("idle called")
        
        """
        while (self.connection.empty () == False):
            message = self.connection.get ()
            print ("command in idle")

            if (message ["success"] == False):
                continue
            
            elif (self.robot_state == "waiting"):
                commands = self.dialogue_system.response_in_dialogue (message ["recognized"])
            
                for command in commands:
                    self.queue.append (command)
"""

        if (self.robot_state == "wiki_search"):
            audio = get_audio ()
                
            succ, recognized = recognize (audio)
            
            print ("data for wiki search", recognized)
            
            succ, result = self.words_processor.get_wiki_content (recognized)
            to_gen = result [:200]
            
            print ("wiki content for mp3 generation", to_gen)
            succ, filename = self.words_processor.generate_mp3 (to_gen)
            
            self.robot.copy_file_to_robot ('C:/Users/Admin/Desktop/' + filename,
                                           "/home/nao/remote_control/sounds/")
            
            curr = time.time ()
        
            play_command = {"type"           : "action",
                            "execution_time" : curr,
                            "contents"       : "/play_mp3",
                            "parameter"      : filename}
            
            self.queue.append (play_command)
            
            self.change_state ("waiting")

        curr_time = time.time ()

        to_del = []

        for i in range (len (self.queue)):
            command = self.queue [i]
            
            if (len (command.keys ()) == 0):
                continue
            
            print ("command in idle: ", command)
            
            if (command ["execution_time"] <= curr_time):
                self.execute_command (command)
                to_del.append (command)
    
        for c in to_del:
            self.queue.remove (c)
        
        #self.robot.send_command ("/makesavephoto", "a")
        
        #time.sleep (0.6)
        
        local_path = "img.jpg"
        #remote_path = "/home/nao/img.jpg"
        
        #self.robot.copy_file_from_robot (remote_path, local_path)
        
        img_path = Path (local_path)
        
        if (img_path.is_file ()):
            image = cv2.imread (local_path)
            
            #mask = cv2.inRange (image, tuple (self.lth), tuple (self.hth))
            mask_ = cv2.inRange (image, tuple (self.lth), tuple (self.hth))

            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.erode (mask_, kernel, iterations = 1)
        
            mask_3ch = np.zeros_like (image)
            mask_3ch [:, :, 0] = mask [:, :]
            mask_3ch [:, :, 1] = mask [:, :]
            mask_3ch [:, :, 2] = mask [:, :]
            
            cv2.imshow ("mask", np.concatenate ((image, mask_3ch), axis=1))
        
        if (self.robot_state == "playing_football"):
            self.robot.send_command ("/play_football", "a")
    
    def add_commands_to_queue (self, commands):
        for command in commands:
            self.queue.append (command)
    
    def stop (self):
        self.robot.send_command ("/stop", "a")

recognizer = sr.Recognizer()
microphone = sr.Microphone (device_index=0)

def get_audio ():#microphone, recognizer):
    with microphone as source:
        print ("listening...")
        audio = recognizer.listen (source)

        return audio

def recognize (audio):
    success = True

    try:
        recognized = recognizer.recognize_google (audio, language = 'ru-RU')
        print (u"recognized %s" % recognized)

    except:
        print ("cannot recognize")
        success = False
        recognized = 0

    return success, recognized

#recognizer = None
#microphone = None
#connection = None
    
def run_audio_recognition (connection):
    #global recognizer
    #global microphone
    #global connection
    
    print ("started listening loop")
        
    while (True):
        time.sleep (1)
        print ("tick")
        
        audio = get_audio ()#smicrophone, recognizer)
            
        succ, recognized = recognize (audio)
            
        message = {"success"    : succ,
                   "recognized" : recognized}
            
        connection.put (message)

def main ():
    #global recognizer
    #global microphone
    #global connection
    
    #recognizer = sr.Recognizer()
    #microphone = sr.Microphone (device_index=0)
    connection = SimpleQueue ()
    
    #speech_parser   = Speech_parser ()
    #speech_parser.set_connection (connection)
    
    #speech_parser   = Process (target=run_audio_recognition, args=(connection,))
    speech_parser = 0
    #freeze_support ()
    #speech_parser.start ()
    
    robot           = Robot ("192.168.137.150", 9555)
    #robot           = Robot ("10.0.0.102", 9555)
    words_processor = Words_processor ()
    dialogue_system = Dialogue_system  ()
    
    robot_state = Robot_state (speech_parser, robot, words_processor, connection, dialogue_system)
    
    root = Tk ()
    root.geometry ("730x580")
    
    GUI  = Main_window (root, robot_state)
    
    root.mainloop ()
    
    #speech_parser.join ()

if __name__ == '__main__':
    main ()
