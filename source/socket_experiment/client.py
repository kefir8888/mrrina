import zmq
import random
import sys
import time

from common import *

# context = zmq.Context()
# socket = context.socket(zmq.PAIR)
# socket.connect()

receiver = zmqImageShowServer (port1)
sender = zmqConnect (port2)

while True:
    # msg = socket.recv()
    # print (msg)
    # socket.send_string("client message to server1")

    image = receiver.receive_image ()
    cv2.imshow ("received image", image)

    list_to_send = [3, 4, 5]
    sender.send_image ("list", list_to_send)

    time.sleep(1)