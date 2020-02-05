from gtts import gTTS
import cyrtranslit
import io
import os
from common import rus_line_to_eng

def main():
    f = io.open ("/Users/elijah/Dropbox/Programming/RoboCup/remote control/data/sounds/phrases.txt", "r", encoding='utf-8')
    f1 = f.readlines()
    
    for line in f1:
        out = rus_line_to_eng (line)

        filename = "/Users/elijah/Dropbox/Programming/RoboCup/remote control/data/sounds/" + out [:26] + '.mp3'

        #tts = gTTS (line, lang='ru')
        #tts.save (filename)

        if (os.path.exists (filename) and os.path.isfile (filename)):
            print ("already exists: ", filename)
            continue
            
        else:
            print ("generating: ", filename)
            tts = gTTS (line, lang='ru')
            tts.save (filename)

if __name__ == '__main__':
    main()