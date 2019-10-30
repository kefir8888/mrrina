import cv2
import numpy as np
import time
import math

from IPython.display import clear_output

#URL-requests to the robot
import requests

#speech generation
import os.path
import cyrtranslit
from gtts import gTTS

#.mp3 files playing
from pygame import mixer

#TODO: отрефакторить этот кусок говна
#TODO: выводить назначение кнопок

#TODO: кнопки на движения рук, последовательности упражнений
#TODO: кнопка на поощрение
#TODO: скопировать фразы на робота

#управление:
#    - поощрение dfgh
#    - ходьба ijkm
#    - встать/сесть zxc
#    - упражнения op
#    - вкл/выкл голосовой контроль vb
#    - 

dict_of_commands = {'a': '/?action=/stand&text=qwer', # встать
                   'd': '/?action=/stand&text=qwer', # встать
                   'n': '/?action=/stand&text=qwer', # встать
                   'q': '/?action=/stand&text=qwer', # встать
                   's': '/?action=/stand&text=qwer', # встать
                   'z': '/?action=/stand&text=qwer', # встать
                   'z': '/?action=/stand&text=qwer', # встать
                   'z': '/?action=/stand&text=qwer', # встать
                   'z': '/?action=/stand&text=qwer', # встать
                   'w': '/?action=/dance&text=qwer', # танцевать
                   'e': '/?action=/hands_front&text=qwer', # руки вперед
                   'r': '/?action=/hands_sides&text=open_right', # руки вбок
                   't': '/?action=/hands&text=close_right', # закрыть правую руку
                   'x': '/?action=/sit&text=qwer', # сесть
                   'c': '/?action=/rest&text=qwer', # корточки
                    
                    'd' : "/?action=/say_local_ru&text=Молодец",
                    'f' : "/?action=/say_local_ru&text=Отлично",
                    'g' : "/?action=/say_local_ru&text=Здорово",
                    'h' : "/?action=/say_local_ru&text=Хорошо получается",
                    
                   'm': '/?action=/walk_m30&text=qwer', # назад 30
                   'j': '/?action=/rot_20&text=qwer', # повернуться на 20 влево
                   'u': '/?action=/walk_50&text=qwer', # вперед 50
                   'i': '/?action=/walk_20&text=qwer', # вперед 20
                   'k': '/?action=/rot_m20&text=qwer', # 
                   
                   'l': '/?action=/say_local_ru&text=Попробуй еще раз', # 
                   #'o': '/?action=/say_local_ru&text=Маркус, подожди, сейчас очередь Бориса', # открыть правую руку
                   #'p': '/?action=/say_local_ru&text=Борис, подожди, сейчас очередь Маркуса', # открыть правую руку
                   '1': '/?action=/M1',
                   '2': '/?action=/M2',
                   '3': '/?action=/M3',
                   '4': '/?action=/M4',
                   '5': '/?action=/M5',
                   '6': '/?action=/M6',
                   }

response_word = {666 : ".",
                 667 : "yes",
                 668 : "no",
                 669 : "привет",
                 670 : "подними",
                 671 : "правую",
                 672 : "левую",
                 673 : "руку",
                 674 : "вперед",
                 675 : "вправо",
                 676 : "влево",
                 677 : "встань",
                 678 : "сядь",
                 679 : "красный",
                 680 : "зеленый",
                 681 : "синий",
                 682 : "разожми",
                 689 : "сожми",
                 684 : "пальцы",
                 685 : "иди",
                 686 : "назад",
                 687 : "поверни",
                 688 : "направо",
                 683 : "прямо"}

st = "/?action=/stop&text=qwer"

phrases_actions = {#"правуюрукувправо" : ["/?action=/right_hand_right&text=qwer"],
                   #"левуюрукувлево" : ["/?action=/left_hand_left&text=qwer"],
                   #"правуюрукувперед" : ["/?action=/right_hand_front&text=qwer"],
                   #"левуюрукувперед" : ["/?action=/left_hand_front&text=qwer"],
                   #"разожмипальцы" : ["/?action=/hands&text=open_right"],
                   #"сожмипальцы" : ["/?action=/hands&text=close_right"],
                   #"прямо" : ["/?action=/walk_20&text=qwer", st],
                   #"идиназад" : ["/?action=/walk_m30&text=qwer", st],
                   #"повернивправо" : ["/?action=/rot_m20&text=qwer", st],
                   #"повернивлево" : ["/?action=/rot_20&text=qwer", st],
                   
                   "вперед" : ["/?action=/walk_20&text=qwer"],
                   "назад" : ["/?action=/walk_m30&text=qwer"],
                   "вправо" : ["/?action=/rot_m20&text=qwer"],
                   "влево" : ["/?action=/rot_20&text=qwer"],

                   "подними" : ["/?action=/hands_sides&text=qwer"],

                   "встань" : ["/?action=/stand&text=qwer"],
                   "сядь" : ["/?action=/rest&text=qwer"],
                   "красный" : ["/?action=/red&text=qwer"],
                   "зеленый" : ["/?action=/green&text=qwer"],
                   "синий" : ["/?action=/blue&text=qwer"]}

activities = {}

activities.update ({"greeting" : ["/?action=/stand&text=qwer",
                                  "/?action=/say_local_ru&text=Привет!"]})

ex_list_1 = ["/?action=/right_shoulder_up&text=qwer",
            "/?action=/say_local_ru&text=Раз",
            
            "/?action=/stand&text=qwer",
            "/?action=/say_local_ru&text=Два",
            
            "/?action=/left_shoulder_up&text=qwer",
            "/?action=/say_local_ru&text=Три",
            
            "/?action=/stand&text=qwer",
            "/?action=/say_local_ru&text=Четыре",
            
            "/?action=/right_hand_front&text=qwer",
            "/?action=/say_local_ru&text=Пять",
            
            "/?action=/stand&text=qwer",
            "/?action=/say_local_ru&text=Шесть",
            
            "/?action=/left_hand_front&text=qwer",
            "/?action=/say_local_ru&text=Семь",
            
            "/?action=/stand&text=qwer",
            "/?action=/say_local_ru&text=Восемь"]

ex_list_2 = ["/?action=/right_hand_right&text=qwer",
            "/?action=/say_local_ru&text=Раз",
            "/?action=/stand&text=qwer",
            "/?action=/say_local_ru&text=Два",
            "/?action=/left_hand_left&text=qwer",
            "/?action=/say_local_ru&text=Три",
            "/?action=/stand&text=qwer",
            "/?action=/say_local_ru&text=Четыре",
            "/?action=/rest&text=qwer",
            "/?action=/say_local_ru&text=Пять",
            "/?action=/stand&text=qwer",
            "/?action=/say_local_ru&text=Шесть",
            "/?action=/rest&text=qwer",
            "/?action=/say_local_ru&text=Семь",
            "/?action=/stand&text=qwer",
            "/?action=/say_local_ru&text=Восемь"]

activities.update ({"exercises" : ["/?action=/say_local_ru&text=Повторяй за мной"] +
                                   ex_list_1 +
                                   ["/?action=/say_local_ru&text=Молодец"] +
                                   ex_list_2# +
                                   #["/?action=/say_local_ru&text=Теперь только Маркус"] +
                                   #ex_list_1 +
                                   #["/?action=/say_local_ru&text=Я "] +
                                   #ex_list_2
                                   })

activities.update ({"complex_exercises" : ["/?action=/stand&text=qwer",
                                           "/?action=/say_local_ru&text=Повторяйте за мной!",
                                           "/?action=/M1",
                                           "/?action=/say_local_ru&text=Отлично получается",
                                           "/?action=/M2"#,
                                           #"/?action=/say_local_ru&text=Боря, теперь только ты",
                                           #"/?action=/M1",
                                           #"/?action=/say_local_ru&text=Маркус, теперь только ты",
                                           #"/?action=/M2"
                                           ]})

#activities.update ({"greeting" : ["/?action=/stand&text=qwer",
#                                  "/?action=/say_local_ru&text=Привет!"]})

def to_eng (line):
    out = cyrtranslit.to_latin(line, 'ru')
    out = "".join(c for c in out if c not in ['!', '.', ':', "'", '?', ' ', '-', '\'', ',', '\n'])
    
    return out

def get_text_and_filename (command):
    text_start = command.find ("text")
    text = command [text_start + 5:]    
    eng = to_eng (text)
    filename = "sounds/" + eng [:26] + ".mp3"
    
    return text, filename

def make_command_printable (command):
    if ("/say_local_ru" not in command):
        return command
    
    else:
        text, filename = get_text_and_filename (command)
        text_start = command.find ("text")
        
        result = command [:text_start] + filename [:-4]
        
        return result

for key in activities.keys ():
    activity = activities [key]
    
    for command in activity:
        if ("/say_local_ru" in command):
            text, filename = get_text_and_filename (command)
            
            if (os.path.exists (filename) and os.path.isfile (filename)):
                print ("already exists: ", filename)
                continue
            
            else:
                print ("generating: ", filename)
                tts = gTTS (text, lang='ru')
                tts.save (filename)

# for a in range (int ('a'), int ('z')):
#     activity = dict_of_commands [str (a)]
    
#     for command in activity:
#         if ("/say_local_ru" in command):
#             text, filename = get_text_and_filename (command)
            
#             if (os.path.exists (filename) and os.path.isfile (filename)):
#                 print ("already exists: ", filename)
#                 continue
            
#             else:
#                 print ("generating: ", filename)
#                 tts = gTTS (text, lang='ru')
#                 tts.save (filename)

def list_and_dict_with_ord (dict_of_commands):
    list_of_keys = []
    dict_with_ord = {}
    for i in list(dict_of_commands.keys()):
        list_of_keys.append(ord(i))
        dict_with_ord.update({ord(i): dict_of_commands[i]})
    return list_of_keys, dict_with_ord

#free = 6

def main():
    queue  = []
    queue_ = []
    
    WIND_X = 500
    WIND_Y = 500
    cv2.namedWindow  ("remote_controller", cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow ("remote_controller", (WIND_Y, WIND_X))
    canvas = np.ones ((WIND_Y, WIND_X, 3), np.uint8) * 200
    
    ip_prefix  = "http://"
    ip_num = "192.168.1.29"
    #ip_num = "10.0.0.102"
    ip_postfix = ":"
    ip = ip_prefix + ip_num + ip_postfix
    port = "9568"
    
    words_queue = []
    
    AUTONOMOUS         = True #without robot
    to_next_operation  = True
    mode_without_queue = False
    
    curr_time          = time.time ()
    time_of_prev_press = 0.0
    
    voice_recognition = False
    last_speech_request_time = 0
    
    logfile = open ("log/" + str (curr_time) + ".txt", "w+")
    
    r = "66666666666666666666666"
    
    list_of_keys, dict_with_ord = list_and_dict_with_ord(dict_of_commands)
    
    mixer.init ()
    
    sounds_queue = []
    
    last_sound  = []
    last_action = []

    while (True):    
        cv2.waitKey (1)
        
        curr_time = time.time ()
        
        canvas = np.ones ((WIND_Y, WIND_X, 3), np.uint8) * 200
        
        for i in range (len (queue)):
            command = make_command_printable (queue [i])
            cv2.putText (canvas, command, (10, 30 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX,
                         0.4, (200,0,0), 1, cv2.LINE_AA)
        
        cv2.putText (canvas, "to next operation:    " + str (to_next_operation),
                     (10, 400), cv2.FONT_HERSHEY_SIMPLEX,
                      0.8, (20, 20, 250), 1, cv2.LINE_AA)
    
        cv2.putText (canvas, "mode without queue: " + str (mode_without_queue),
                     (10, 430), cv2.FONT_HERSHEY_SIMPLEX,
                      0.8, (20, 20, 250), 1, cv2.LINE_AA)
        
        cv2.putText (canvas, "voice recognition:   " + str (voice_recognition),
                     (10, 370), cv2.FONT_HERSHEY_SIMPLEX,
                      0.8, (20, 20, 250), 1, cv2.LINE_AA)
        
        cv2.imshow ("remote_controller", canvas)
        
        time.sleep  (0.03)
        
        #handle keyboard events
        keyb = cv2.waitKey (1)
        
        if (len (sounds_queue) != 0 and
            mixer.music.get_busy () == False):# and
            #to_next_operation == True):
            mixer.music.load (sounds_queue [0])
            mixer.music.play ()
            
            print (sounds_queue [0])
            
            tex, _ = get_text_and_filename (sounds_queue [0])
            
            r = requests.get (ip + port + "/" + "?" + "action=/" + "play_mp3" + "&" + "text=" + tex)
            
            last_sound = [sounds_queue [0]]
            
            sounds_queue.remove (sounds_queue [0])
        
        if (len (queue) != 0 and (to_next_operation == True or "stop" in queue [0])):
            free = 6
            
            if (AUTONOMOUS == False):
                action = 'free'
                text = 'qwer'
                r = requests.get (ip + port + "/" + "?" + "action=/" + action + "&" + "text=" + text)
                free = int (str (r) [13:14]) #6 free, 7 not free; don't ask, don't tell
                print ("fuck", free)
            
            if ("/say_local_ru" in queue [0]):
                _, filename = get_text_and_filename (queue [0])
                
                sounds_queue.append (filename)
                queue.remove (queue [0])
                
                if not mode_without_queue:
                    to_next_operation = False
                
                continue
    
            if ("/play" in queue [0]):
                text_start = command.find ("text")
                text = command [text_start + 5:]    
                
                print (text)
                
                sounds_queue.append (text)
                queue.remove (queue [0])
                
                if not mode_without_queue:
                    to_next_operation = False
                
                continue
            
            if free == 6 or "stop" in queue [0]:
                time.sleep(0.1)
                
                logfile.write (str (curr_time) + queue [0] + "\n")
                
                if (AUTONOMOUS == False):
                    requests.get (ip + port + queue [0])
                    
                else:
                    print (ip + port + queue [0])
                
                if not mode_without_queue:
                    to_next_operation = False
                
                last_action = [queue [0]]
                
                queue.remove (queue [0])
        
        if (keyb != -1):
            upd = True
        
        if (keyb & 0xFF == ord ('q')):
            break
        
        elif (keyb & 0xFF == ord(' ')):
            mode_without_queue = not mode_without_queue
            
            #if (mode_without_queue == True):
            #    to_next_operation = True
        
        elif (keyb & 0xFF == ord ('s')):
            queue [:] = []
            
            stop_request = ip + port + "/?action=/stop&text=qwer"
            
            logfile.write (str (curr_time) + stop_request + "\n")
            
            if (AUTONOMOUS == False):
                r = requests.get (stop_request) # остановить действие
            
            else:
                print (stop_request)
    
        elif (keyb & 0xFF == ord ('a')):
            queue  [:] = []
            queue_ [:] = []
    
        elif (keyb & 0xFF == ord ('v')):
            r = requests.get (ip + port + "/?" + "action=/" + "start_voice_recognition" + "&" + "text=" + "m")
            #queue += activities ["exercises"]
            voice_recognition = True
        
        elif (keyb & 0xFF == ord ('n') or
              (voice_recognition == True and
               curr_time - last_speech_request_time >= 0.5)):# and
             #free == 6):
            print (curr_time - last_speech_request_time)
            last_speech_request_time = curr_time
            
            r = requests.get (ip + port + "/?" + "action=/" + "word_if_any" + "&" + "text=" + "m")
            print (r)
            word = response_word [int (str (r) [11:14])]
            print (word)
            
            if (word != "."):
                words_queue.append (word)
    
        elif (keyb & 0xFF == ord ('b')):
            r = requests.get (ip + port + "/?" + "action=/" + "stop_voice_recognition" + "&" + "text=" + "m")
            voice_recognition = False
            
            #queue += activities ["complex_exercises"]
        
        #elif (keyb & 0xFF == ord ('h')):
        #    queue = last_action + queue
        #    to_next_operation = True
    
        #elif (keyb & 0xFF == ord ('j')):
        #    sounds_queue = last_sound + sounds_queue
            #to_next_operation = True
    
        elif (keyb & 0xFF == ord ('o')):
            queue += activities ["complex_exercises"]
    
        elif (keyb & 0xFF == ord ('p')):
            queue += activities ["exercises"]
            
        elif (keyb in list_of_keys):
            #queue.append(dict_with_ord[keyb])
            queue = [dict_with_ord[keyb]] + queue
        
        elif (keyb & 0xFF == ord ('0')):
            to_next_operation = True
            
            cv2.waitKey (100)
        
        if (voice_recognition == True and len (words_queue) != 0):
            print (words_queue, "keke")
            
            if (len (words_queue) >= 1):
                if (words_queue [-1] in phrases_actions.keys ()):
                    phrase = words_queue [-1]
                    
                    queue += phrases_actions [phrase]
                    words_queue = []
            
            if (len (words_queue) >= 2):
                phrase = words_queue [-2] + words_queue [-1]
                
                print (phrase)
                
                if (phrase in phrases_actions.keys ()):
                    queue += phrases_actions [phrase]
                    words_queue = []
            
            if (len (words_queue) >= 3):
                phrase = words_queue [-3] + words_queue [-2] + words_queue [-1]
                
                print (phrase)
                
                if (phrase in phrases_actions.keys ()):
                    queue += phrases_actions [phrase]
                    words_queue = []
    
    logfile.close ()
    #cv2.waitKey           (1)
    cv2.destroyAllWindows ()

if __name__ == '__main__':
    main()
