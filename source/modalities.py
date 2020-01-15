from common import *

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

        self.key_to_command = {"z"        : ("/stand",   ["heh"]),
                               "c"        : ("/rest",    ["kek"]),
                               "w"        : ("/increment_joint_angle", ["lefthand", "0.4"]),
                               "e"        : ("/increment_joint_angle", ["lefthand", "-0.4"]),
                               "r"        : ("/increment_joint_angle", ["leftarm", "0.4"]),
                               "t"        : ("/increment_joint_angle", ["leftarm", "-0.4"]),
                               "s"        : ("/increment_joint_angle", ["righthand", "0.4"]),
                               "d"        : ("/increment_joint_angle", ["righthand", "-0.4"]),
                               "f"        : ("/increment_joint_angle", ["rightarm", "0.4"]),
                               "g"        : ("/increment_joint_angle", ["rightarm", "-0.4"]),
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

#class Video (Modality):
#class Voice (Modality):
#class Virtual_keyboard (Modality):
