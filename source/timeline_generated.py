from multiprocessing import Process, Pipe
from multiprocessing.queues import Queue
import cv2
import time
import numpy as np

class Window (Process):
    def __init__ (self, name_, connection_, letter_ = "a"):
        Process.__init__ (self)
        self.name  = name_
        self.connection = connection_
        self.letter = letter_
        self.quit_required = False

        self.lifetime = 5

    def run (self):
        print ("work started", self.letter)
        #cv2.namedWindow ("a", cv2.WINDOW_AUTOSIZE)
        #cv2.resizeWindow ("a", (300, 300))

        while (True):
            print("idle", self.letter)
            canvas = np.ones ((300, 300, 3), np.uint8) * 55

            cv2.putText (canvas, self.name + self.letter, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 25), 1, cv2.LINE_AA)

            #cv2.imshow ("a", canvas)
            #cv2.waitKey (10)
            #print ("img dsplayed)0")


            message = self.connection.get ()

            print (self.name + "received" + message ["text"])
            time.sleep(1)

            self.lifetime -= 1

            if (self.lifetime <= 0):
                break

class Manager:
    def __init__ (self, description_):
        self.description = description_

        self.turn_num = 0

        # self.load_configuration ()

    def __del__ (self):
        for p in self.processes:
            p.join ()

    def quit (self):
        return self.quit_required

    #def load_configuration (self):


    def work (self):
        cv2.namedWindow ("manager", cv2.WINDOW_AUTOSIZE)
        cv2.resizeWindow ("manager", (400, 400))

        self.connections = []
        self.processes = []
        self.messages = []

        for i in range (len (self.description)):
            instance_description = self.description [i]
            print ("description:", instance_description)

            queue = Queue ()
            self.connections.append (queue)
            self.messages.append (instance_description["message"])

            name = instance_description ["name"]
            letter = instance_description ["letter"]

            p = Window (name, queue, letter)
            p.start ()

            self.processes.append (p)

        while (True):
            time.sleep (1)
            canvas = np.ones ((400, 400, 3), np.uint8) * 55

            cv2.imshow ("manager", canvas)
            cv2.waitKey (10)
            print ("manager")

            for i in range (len (self.connections)):
                connection = self.connections [i]
                message    = self.messages    [i]

                connection.put ({"text" : message})
                print ("manager sent" + message + "to" + str (i) + "process")

if __name__ == '__main__':
    description = [{"name" : '1st',
                    "letter" : "a",
                    "message" : "A"}]#,
                    #{"name": '2nd',
                    # "letter": "b"}]

    manager = Manager (description)
    manager.work ()

#сделать закрытие queue
#свое окно (если это все-таки возможно) или передача картинки в родительский процесс
#переделать менеджер основного проекта
#
#еще можно по фану добавить генерацию питоновского скрипта, который выполняет пришедшую на вход последовательность действий
#локальная обработка данных, связь с интернетом, связь с железом, многопоточность
#
#Кажется, наступил момент, когда нужно делать интерфейс для изменения потоков данных, которые идут на роботов,
#и не только этих потоков, а вообще всех на уровне модальностей.
#
#Пока не совсем ясно, как именно нужно реализовать собирание поведения в кучку (последовательные команды от разных
#модальностей). Это должна быть какая-то штука между модальностями и роботами.
#
#
#
#
#
#
#
#
#
