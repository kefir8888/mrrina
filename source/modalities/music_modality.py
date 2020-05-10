from modalities.modality import  Modality

import numpy as np
import common

import pydub
from pydub import AudioSegment
from pydub.playback import play
import scipy.fftpack
import cv2

import multiprocessing

#class Motion_source:
#    def __init__ (self):
#        pass

#    def get_motion (self, time):
#        return np.zeros (18, np.float32)

#from skeleton_modalities import smth
#class Cyclic
#class Markov_chain
#class Rhytmic_sine

#class Archieve_data
#class Archieve_data_format1
#class Archieve_data_format2

#class External_model_loader

class Music (Modality):
    def __init__ (self, music_path_ = "", logger_ = 0):
        self.logger = logger_
        self.music_path = music_path_

        self.tick = 0

        self.commands = {"noaction": [("noaction", [""])],
                         "0": [("/increment_joint_angle", ["l_sho_roll", "-0.11"])],
                         "1": [("/increment_joint_angle", ["l_sho_roll", "0.11"])]
                         }

        self.rate, self.audio = self.read(music_path_)
        self._extract_rhythm ()
        self.timeout = common.Timeout_module(1 / self.rhythm / 8)

        #song = AudioSegment.from_mp3 (music_path_)
        #play (song)

    def play_song (self):
        pass

    def read(self, f, normalized=False):
        """MP3 to numpy array"""
        a = pydub.AudioSegment.from_mp3(f)
        y = np.array(a.get_array_of_samples())
        if a.channels == 2:
            y = y.reshape((-1, 2))
        if normalized:
            return a.frame_rate, np.float32(y) / 2 ** 15
        else:
            return a.frame_rate, y

    def write(self, f, sr, x, normalized=False):
        """numpy array to MP3"""
        channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
        if normalized:  # normalized array - each item should be a float in [-1, 1)
            y = np.int16(x * 2 ** 15)
        else:
            y = np.int16(x)
        song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
        song.export(f, format="mp3", bitrate="320k")

    def _extract_rhythm (self):
        N = 2000
        an_part = self.audio [:2000, 1]
        x = np.linspace (0, 2 * np.pi, N)

        w = scipy.fftpack.rfft (an_part)
        f = scipy.fftpack.rfftfreq (N, x[1] - x[0])
        spectrum = w**2

        cutoff_idx = spectrum > (spectrum.max () / 15)
        w2 = w.copy ()
        w2 [cutoff_idx] = 0

        self.rhythm = f [1]

    def name(self):
        return "Baseline dance generation with audio input"

    def _read_data (self):
        pass

    def _process_data(self):
        pass

    def _interpret_data(self):
        pass

    def _get_command(self):
        comm = self.commands ["noaction"]

        if (self.timeout.timeout_passed ()):
            l = len (self.commands)

            comm = self.commands[str (np.random.randint (1, l))]

            self.tick += 1

        return comm

    def get_command(self, skip_reading_data=False):
        self._read_data()
        self._process_data()
        self._interpret_data()

        return self._get_command()

    def draw (self, canvas = np.ones ((700, 700, 3), np.uint8) * 220):
        result = canvas.copy ()

        cv2.putText (result, self.music_path, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 50, 31), 1, cv2.LINE_AA)

        return [result]

class Cyclic (Music):
    def __init__ (self, music_path_ = "", logger_ = 0):
        Music.__init__ (self, music_path_, logger_)

        self.tick = 0

        self.rate, self.audio = self.read (self.music_path)
        self._extract_rhythm ()
        self.timeout = common.Timeout_module(1 / self.rhythm / 8)

        print ("timeout:", self.timeout)

        #song = AudioSegment.from_mp3 (music_path_)
        #play (song)

    def play_song (self):
        pass


    def name(self):
        return "Cyclic moves performing"

    def _read_data (self):
        pass

    def _process_data(self):
        pass

    def _interpret_data(self):
        pass

    def _get_command(self):
        comm = self.commands ["noaction"]

        if (self.timeout.timeout_passed ()):
            l = len (self.commands)

            comm = self.commands[str (self.tick % (l - 1))]
            #[("/increment_joint_angle", ["l_sho_roll", "-0.11"])]

            self.tick += 1

        return comm

    def get_command(self, skip_reading_data=False):
        self._read_data()
        self._process_data()
        self._interpret_data()

        return self._get_command()
