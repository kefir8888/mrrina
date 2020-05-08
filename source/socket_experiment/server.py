import zmq
import random
import sys
import time

from adam_sender import *

# context = zmq.Context()
# socket = context.socket(zmq.PAIR)
# socket.bind("tcp://*:%s" % port)

img = cv2.resize (cv2.imread ("test_img.png"), (300, 200))

sender = zmqConnect (port1)
receiver = zmqImageShowServer (port2)

# while True:
    # socket.send_string("Server message to client3")
    # msg = socket.recv()
    # print (msg)
# time.sleep(1)

print ("sending image")
sender.send_image ("img", img)

num_list = receiver.receive_image ()
print ("received list: ", num_list)

key = cv2.waitKey(1) & 0xFF



socket.close ()
cv2.destroyAllWindows ()
