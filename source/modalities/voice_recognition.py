import numpy as np
import common

from modalities.modality import Modality
from multiprocessing import Process
import speech_recognition as sr

class Voice_recognition (Modality, Process):
    def __init__(self, language_ = 'ru-RU'):
        Modality.__init__ (self)
        Process.__init__ (self)

        self.connection_set = False
        self.connection = 0

        self.language = language_

        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone(device_index=0)

    def set_connection (self, connection):
        self.connection = connection

    def get_audio():
        with microphone as source:
            print("listening...")
            audio = recognizer.listen(source)

            return audio

    def recognize(audio):
        success = True

        try:
            recognized = recognizer.recognize_google (audio, language = self.language)
            print(u"recognized %s" % recognized)

        except:
            print("cannot recognize")
            success = False
            recognized = 0

        return success, recognized

    def run (self):
        print("started listening loop")

        while (True):
            time.sleep(1)
            print("tick")

            audio = get_audio()  # smicrophone, recognizer)

            succ, recognized = recognize(audio)

            message = {"success": succ,
                       "recognized": recognized}

            connection.put(message)
