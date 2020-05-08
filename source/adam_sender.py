import zmq
import numpy as np
import cv2

port_num1 = "5558"
port1 = "tcp://127.0.0.1:%s" % port_num1

port_num2 = "5557"
port2 = "tcp://127.0.0.1:%s" % port_num2

class SerializingSocket(zmq.Socket):
    """A class with some extra serialization methods

    send_array sends numpy arrays with metadata necessary
    for reconstructing the array on the other side (dtype,shape).
    Also sends array name for display with cv2.show(image).
    recv_array receives dict(arrayname,dtype,shape) and an array
    and reconstructs the array with the correct shape and array name.
    """

    def send_array(self, A, arrayname="NoName", flags=0, copy=True, track=False):
        """send a numpy array with metadata and array name"""
        md = dict(
            arrayname=arrayname,
            dtype=str(A.dtype),
            shape=A.shape,
        )
        self.send_json(md, flags | zmq.SNDMORE)
        return self.send(A, flags, copy=copy, track=track)

    def recv_array(self, flags=0, copy=True, track=False):
        """recv a numpy array, including arrayname, dtype and shape"""
        md = self.recv_json(flags=flags)
        msg = self.recv(flags=flags, copy=copy, track=track)
        A = np.frombuffer(msg, dtype=md['dtype'])
        return (md['arrayname'], A.reshape(md['shape']))


class SerializingContext(zmq.Context):
    _socket_class = SerializingSocket


class zmqConnect():
    '''A class that opens a zmq REQ socket on the headless computer
    '''

    def __init__(self, connect_to="tcp://jeff-mac:5555"):
        '''initialize zmq socket for sending images to display on remote computer'''
        '''connect_to is the tcp address:port of the display computer'''
        self.zmq_context = SerializingContext()
        self.zmq_socket = self.zmq_context.socket(zmq.REQ)
        self.zmq_socket.connect(connect_to)

    def send_image(self, arrayname, array):
        '''send image to display on remote server'''

        # if array.flags['C_CONTIGUOUS']:
        #     # if array is already contiguous in memory just send it
        #     self.zmq_socket.send_array(array, arrayname, copy=False)
        # else:
        #     # else make it contiguous before sending
        #     array = np.ascontiguousarray(array)
        #     self.zmq_socket.send_array(array, arrayname, copy=False)

        array = np.ascontiguousarray(array)
        self.zmq_socket.send_array(array, arrayname, copy=False)

        message = self.zmq_socket.recv()

    def receive_list(self, copy=False):
        '''receive and show image on viewing computer display'''
        arrayname, list = self.zmq_socket.recv_array(copy=False)
        # print "Received Array Named: ", arrayname
        # print "Array size: ", image.shape
        #cv2.imshow(arrayname, image)

        #print("img displayed")

        cv2.waitKey(1)
        self.zmq_socket.send(b"OK")

        return list

class zmqImageShowServer():
    '''A class that opens a zmq REP socket on the display computer to receive images
    '''

    def __init__(self, open_port="tcp://*:5555"):
        '''initialize zmq socket on viewing computer that will display images'''
        self.zmq_context = SerializingContext()
        self.zmq_socket = self.zmq_context.socket(zmq.REP)
        self.zmq_socket.bind(open_port)

    def receive_image(self, copy=False):
        '''receive and show image on viewing computer display'''
        arrayname, image = self.zmq_socket.recv_array(copy=False)
        # print "Received Array Named: ", arrayname
        # print "Array size: ", image.shape
        #cv2.imshow(arrayname, image)

        #print ("img displayed")

        cv2.waitKey(1)
        self.zmq_socket.send(b"OK")

        return image

    def send_list(self, arrayname, array):
        '''send image to display on remote server'''
        #if array.flags['C_CONTIGUOUS']:
        #    # if array is already contiguous in memory just send it
        #    self.zmq_socket.send_array(array, arrayname, copy=False)
        #else:
            # else make it contiguous before sending
        array = np.ascontiguousarray(array)
        self.zmq_socket.send_array(array, arrayname, copy=False)

        message = self.zmq_socket.recv()