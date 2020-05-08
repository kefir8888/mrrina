from modalities.modality import GetPoints
import zmq
import random
import sys
import time

from adam_sender import *

# context = zmq.Context()
# socket = context.socket(zmq.PAIR)
# socket.connect()

receiver = zmqImageShowServer (port1)
sender = zmqConnect (port2)
net = GetPoints(model_path_ = "/home/kompaso/wenhai/wenhai/source/test/human-pose-estimation-3d.pth")

while True:
    # msg = socket.recv()
    # print (msg)
    # socket.send_string("client message to server1")

    image = receiver.receive_image ()
    x, y, z = net._infer_net(image)
    # cv2.imshow ("received image", image)

    # list_to_send = [3, 4, 5]
    sender.send_image ("list", x)
