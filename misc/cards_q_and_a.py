import cv2
import time
import os
import math
import sys

from IPython.display import clear_output
import os

import requests

#speech generation
import os.path
import cyrtranslit
from gtts import gTTS

sys.path.append("/Users/elijah/Dropbox/Programming/detectors/modules/")

from input_output import Source
import detectors
import tracker

X_WIND = 640
Y_WIND = 480

def to_eng (line):
    out = cyrtranslit.to_latin(line, 'ru')
    out = "".join(c for c in out if c not in ['!', '.', ':', "'", '?', ' ', '-', '\'', ',', '\n'])
    
    return out

def get_filename (text):
    eng = to_eng (text)
    filename = "sounds/" + eng [:26] + ".mp3"
    
    return filename

#def make_command_printable (command):
#    if ("/say_local_ru" not in command):
#        return command
    
#    else:
#        text, filename = get_text_and_filename (command)
#        text_start = command.find ("text")
        
#        result = command [:text_start] + filename [:-4]
        
#        return result

#def get_score ():


ip_prefix  = "http://"
ip_num = "10.0.0.105"
ip_postfix = ":"
ip = ip_prefix + ip_num + ip_postfix
port = "9569"

def main ():
    source = Source ("1")

    config = open ("cards.txt", "r")

    cards = []

    curr_card = 0

    while (True):
        file = config.readline () [:-1]
        text = config.readline () [:-1]
        greet = config.readline () [:-1]
        nope = config.readline () [:-1]

        if (file == None or text == None or len (file) < 3):
            break

        #print ("a", file, "b")

        filename = get_filename (text)

        templ  = Source (str (file))
        template = templ.get_frame ()

        print (text)

        cards.append ([template, text, [], 0.0, greet, nope])

        for sen in [text, greet, nope]:
            filename = get_filename (sen)
            
            if (os.path.exists (filename) and os.path.isfile (filename)):
                print ("already exists: ", filename)
                continue
            
            else:
                print ("generating: ", filename)
                tts = gTTS (text, lang='ru')
                tts.save (filename)

    print (cards)

    cv2.namedWindow ("ff", cv2.WINDOW_NORMAL)
    cv2.resizeWindow ("ff", (X_WIND, Y_WIND))

    starting_frames = 15
    frame_num = 0

    for i in range (starting_frames):
        frame = source.get_frame ()

        for card in cards:
            meth = 'cv2.TM_CCOEFF'
            method = eval (meth)

            res = cv2.matchTemplate (frame, card [0], method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc (res)

            top_left = max_loc
            bottom_right = (top_left[0] + 300, top_left[1] + 430)

            print (card [1], max_val, min_val)

            if (frame_num < starting_frames):
                card [2].append (max_val)

            elif (frame_num == starting_frames):
                card [3] = float (sum (card [2])) / len (card [2])

            frame_num += 1

    to_next = True
    q_given = False
    finished = False

    while (True):
        frame = source.get_frame ()

        print ("damn", finished, to_next)

        if (to_next == True):
            to_next = False
            finished = False
            text = to_eng (cards [curr_card] [1]) + ".mp3"

            print (text)
            r = requests.get (ip + port + "/" + "?" + "action=/" + "play_mp3" + "&" + "text=" + text)
            print ("ask")

        elif (finished == False):
            for card in cards:
                meth = 'cv2.TM_CCOEFF'
                method = eval (meth)

                res = cv2.matchTemplate (frame, card [0], method)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc (res)

                top_left = max_loc
                bottom_right = (top_left[0] + 300, top_left[1] + 430)

                print (card [1], max_val, min_val)
            
                if (max_val > card [3] * 2):
                    cv2.rectangle (frame, top_left, bottom_right, (0, 255, 0), 2)

                    text = ""

                    if (card == cards [curr_card]):
                        text = to_eng (card [4]) + ".mp3"
                        finished = True
                        curr_card += 1

                    else:
                        text = to_eng (card [5]) + ".mp3"

                    print (text)
                    r = requests.get (ip + port + "/" + "?" + "action=/" + "play_mp3" + "&" + "text=" + text)
                    time.sleep (3)
            
        cv2.imshow ("ff", frame)

        cv2.waitKey (1)
        
        if (curr_card >= len (cards)):
            break

        time.sleep (0.02)

        #clear_output (wait=True)
        #os.system ('clear')

        keyb = cv2.waitKey (1) & 0xFF
        
        if (keyb == ord('a')):
            to_next = True

        if (keyb == ord('q')):
            break

        frame_num += 1

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main ()