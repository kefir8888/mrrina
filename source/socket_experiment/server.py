import zmq
import random
import sys
import time

from common import *

# context = zmq.Context()
# socket = context.socket(zmq.PAIR)
# socket.bind("tcp://*:%s" % port)

img = cv2.imread ("test_img.png")

sender = zmqConnect (port)

while True:
    # socket.send_string("Server message to client3")
    # msg = socket.recv()
    # print (msg)
    time.sleep(1)

    sender.imshow ("img", img)

    key = cv2.waitKey(1) & 0xFF

    if (key == ord ('q')):
        break

socket.close ()
cv2.destroyAllWindows ()