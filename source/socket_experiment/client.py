import zmq
import random
import sys
import time

from common import *

# context = zmq.Context()
# socket = context.socket(zmq.PAIR)
# socket.connect()

receiver = zmqImageShowServer (port)

while True:
    # msg = socket.recv()
    # print (msg)
    # socket.send_string("client message to server1")

    receiver.imshow ()

    time.sleep(1)