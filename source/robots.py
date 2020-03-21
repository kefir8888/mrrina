from common import *

class Robot:
    def __init__(self, timeout_ = 0):
        self.queue         = []
        self.commands_sent = 0
        self.name = "base"

        self.available_commands = {"/rest"  : ("/action=/rest&text=", "a"),
                                   "/stand" : ("/action=/stand&text=", "a"),
                                   "/sit" : ("/action=/stand&text=", "a"),
                                   "/free"  : ("/action=/free&text=", "a"),
                                   "/increment_joint_angle" : (),
                                   "/set_joint_angle" : (),
                                   "/walk_20" : (),
                                   "/walk_m30" : (),
                                   "/rot_20" : (),
                                   "/rot_m20" : (),
                                   "/right_hand_right" : (),
                                   "/left_hand_left" : (),
                                   "/right_hand_up" : (),
                                   "/right_shoulder_up" : (),
                                   "/left_shoulder_up" : (),
                                   "/play_mp3" : (),
                                   "/play_airplane_1" : (),
                                   "/play_airplane_2" : (),
                                   "/play_car" : (),
                                   "/red" : (),
                                   "/green" : (),
                                   "/blue" : (),
                                   "/orange" : (),
                                   "/yellow" : (),
                                   "/lightblue" : (),
                                   "/violet" : (),
                                   "/pink" : (),
                                   "/brown" : (),
                                   "/white" : (),
                                   "/wipe_forehead" : (),
                                   "/right_hand_front" : (),
                                   "/right_hand_right" : (),
                                   "/left_hand_left" : (),
                                   "/left_hand_front" : (),
                                   "/right_hand_up" : (),
                                   "/left_hand_up" : (),
                                   "/hands_sides" : (),
                                   "/hands_front" : (),
                                   "/bend_right" : (),
                                   "/bend_left" : ()}

        self.timeout_module = Timeout_module (timeout_)

    def _send_command (self, command, args):
        pass

    def plot_state (self, img):
        pass

    def on_idle (self):
        if (self.timeout_module.timeout_passed (len (self.queue) > self.commands_sent)):
            while (len (self.queue) > self.commands_sent):
                next_command = self.queue [self.commands_sent]
                self.commands_sent += 1

                if (next_command [0] != "noaction"):
                    #print ("sending ", next_command)
                    #print (self.queue)
                    #print ("")
                    self._send_command (next_command)

                if (self.name == "real"):
                    break

    def add_action (self, action):
        #print ("appending ", action)

        if (action is None):
            return

        for act in action [0]:
            if (act [0] != "noaction"):
                self.queue.append ([act])

class Fake_robot(Robot):
    def __init__(self, timeout_ = 0.5):
        Robot.__init__ (self, timeout_)
        self.name = "fake"

    def _send_command (self, action):
        if (action [0] in self.available_commands.keys ()):
            print ("sending command [fake]: ", action)

        else:
            print ("action :", action, " is not implemented")

class Joint:
    def __init__(self, length_, angle_, angle_multiplier_, col1_, col2_, name_, min_angle_, max_angle_):
        self.length     = length_
        self.init_angle = angle_
        self.angle      = 0
        self.angle_multiplier = angle_multiplier_
        self.min_angle = min_angle_
        self.max_angle = max_angle_
        self.col1       = col1_
        self.col2       = col2_
        self.joint_name = name_

        self.children = []

    def name (self):
        return self.joint_name

    def draw (self, img, x, y, parent_angle, scale = 1):
        #print ("joint: ", self.length, self.angle)
        angle = self.init_angle + self.angle + parent_angle

        x1 = x + self.length * math.cos (angle)
        y1 = y + self.length * math.sin (angle)

        x_  = int (x * scale)
        y_  = int (y * scale)
        x1_ = int (x1 * scale)
        y1_ = int (y1 * scale)

        cv2.line (img, (int (x_), int (y_)), (int (x1_), int (y1_)), self.col1, 3)
        cv2.circle (img, (int (x1_), int (y1_)), 5, self.col2, -1)
        cv2.putText (img, self.joint_name, (int (x1_) + 0, int (y1_) + 0),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 250, 231), 1, cv2.LINE_AA)

        for child in self.children:
            child.draw (img, x1, y1, angle, scale)

    def set_angle (self, new_angle):
        if (new_angle >= self.min_angle and new_angle <= self.max_angle):
            self.angle = new_angle

        else:
            print ("warning: cannot set joint " + self.joint_name + " to " + str (new_angle) +\
                ": the constraints are [" + str (self.min_angle) + ", " + str (self.max_angle) + "]")

        return self.angle

    def add_child (self, name, length, angle, angle_multiplier, min_angle, max_angle):
        self.children.append (Joint (length, angle, angle_multiplier, self.col1, self.col2, name, min_angle, max_angle))

class Simulated_robot(Robot):
    def __init__(self, timeout_ = 0.01, path_ = ""):
        Robot.__init__ (self, timeout_)

        self.config_path = path_

        self.base_point = Joint (0, 0, 1, (10, 100, 200), (230, 121, 2), "base", -10, 10)
        self.load_configuration (self.config_path)

        self.joints_to_track = ["righthand", "lefthand", "leftarm", "rightarm"]

        self.updated = False
        self.name = "simulated"

    def load_configuration (self, path = ""):
        if (path == ""):
            path = "/Users/elijah/Dropbox/Programming/RoboCup/remote control/source/robot_configuration.txt"

        config = open (path, "r")

        string = config.readline ()

        while (string != ""):
            data = string [:-1].split (" ")

            #print ("dat", data)

            parent = str   (data [0])
            name   = str   (data [1])
            length = float (data [2])
            angle  = float (data [3])
            angle_multiplier = float (data [4])
            min_angle = float (data [5])
            max_angle = float (data [6])

            self.add_joint (parent, name, length, angle, angle_multiplier, min_angle, max_angle)

            string = config.readline ()

    def find_joint (self, joint_name = ""):
        stack = [self.base_point]
        all_joints = []

        target = stack [0]
        found  = False

        if (joint_name == ""):
            found = True

        while (len (stack) != 0):
            if (len (stack) >= 1000):
                print ("Stack size has reached 1000. Probably smth went wrong, \
for instance the robot model is recursive. Aborting operation.")
                break

            curr = stack [0]
            
            if (joint_name != ""):
                if (curr.name () == joint_name):
                    target = curr
                    found = True

                    break
            else:
                all_joints.append (curr)

            for child in curr.children:
                stack.append (child)

            stack.remove (curr)

        if (found == False):
            print ("Warning: requested joint ", joint_name, " not found")

        if (joint_name != ""):
            return target, found

        return all_joints

    def set_joint_angle (self, joint_name, new_angle, increment = False):
        target_joint, succ = self.find_joint (joint_name)

        if (succ == False):
            print ("Unable to set ", joint_name, " to ", new_angle, ": no such joint")

        if (increment == True):
            new_angle += target_joint.angle

        set_angle = target_joint.set_angle (new_angle)

        if (set_angle == new_angle):
            self.updated = True

        return set_angle

    def add_joint (self, parent_name, new_joint_name, length, angle, angle_multiplier, min_angle, max_angle):
        target_joint, succ = self.find_joint (parent_name)

        if (succ == False):
            print ("Unable to add child ", new_joint_name, " to ", parent_name, ": no such joint")

        target_joint.add_child (new_joint_name, length, angle, angle_multiplier, min_angle, max_angle)

    def _send_command (self, actions):
        for action in actions:
            #print ("action [0]: ", action [0])
            if (action [0] in self.available_commands.keys ()):
                self.updated = True
                #print ("sending command [simulated]: ", action)
            
                if (action [0] == "/increment_joint_angle"):
                    self.set_joint_angle (action [1] [0], float (action [1] [1]), increment = True)

                if (action [0] == "/set_joint_angle"):
                    self.set_joint_angle (action [1] [0], float (action [1] [1]))

                elif (action [0] == "/stand"):
                    #self.set_joint_angle ("righthand", -0.2)
                    self.base_point.children = []
                    self.load_configuration (self.config_path)
                    self.set_joint_angle ("base", 0)

                elif (action [0] == "/rest"):
                    self.set_joint_angle ("base", 5)

                elif (action [0] == "/hands_sides"):
                    self.set_joint_angle ("righthand", 1)

            else:
                print ("action :", action, " is not supported")

    def plot_state (self, img, x, y, scale = 1):
        line_num = 0

        for joint in self.find_joint (""):
            if (joint.name () not in self.joints_to_track):
                continue

            text = joint.name () + ":  [" + "{0:.2f}".format(joint.min_angle) \
                                 + " / " + "{0:.2f}".format(joint.angle) \
                                 + " /  " + "{0:.2f}".format(joint.max_angle) + "]"

            cv2.putText (img, text, (30, 30 * (1 + line_num)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 50, 231), 1, cv2.LINE_AA)

            line_num += 1

        self.base_point.draw (img, x, y, 0, scale)

class Real_robot(Robot):
    def __init__(self, ip_num, port_ = 9559, timeout_ = 0.3):
        Robot.__init__ (self, timeout_)

        self.ip_prefix = "http://"
        self.ip_postfix = ":"

        self.ip   = self.ip_prefix + ip_num + self.ip_postfix
        self.port = port_

        self.free = False
        self.free_timeout_module = Timeout_module (0.3)

        self.simulated = Simulated_robot ()

        self.synchronized_joints = {"righthand" : "r_shoulderroll",
                                    "rightarm"  : "r_elbowroll",
                                    "lefthand"  : "l_shoulderroll",
                                    "leftarm"   : "l_elbowroll",
                                    
                                    "leftleg"   : "r_shoulderpitch",
                                    "rightleg"  : "r_elbowyaw",
                                    "leftfoot"  : "l_shoulderpitch",
                                    "rightfoot" : "l_elbowyaw"}

        self.init_positions = {"r_shoulderpitch" : 1.57,
                               "r_shoulderroll"  : 0,
                               "r_elbowroll"     : 0,
                               "r_elbowyaw"      : 0,
                               "l_shoulderpitch" : 1.57,
                               "l_shoulderroll"  : 3.14,
                               "l_elbowroll"     : 0,
                               "l_elbowyaw"      : 0}

        self.name = "real"

    def _send_command (self, action):
        r = -1

        #if (action [0] == "noaction"):
        #    pass

        self.simulated._send_command (action)

        #print ("lalala")
        #print (action [0] [0])
        #print (self.available_commands.keys())

        if (action [0] [0] == "/increment_joint_angle" or
            action [0] [0] == "/set_joint_angle"):
            action_str = "/raise_hands"
            text_str   = ""

            for key in self.synchronized_joints.keys ():
                joint, _ = self.simulated.find_joint (key)
                robot_joint = self.synchronized_joints [key]
                init_angle = self.init_positions [robot_joint]

                if (key == "righthand"):
                    print ("hand", joint.angle, joint.angle_multiplier, init_angle)
                    text_str += "&" + robot_joint + "=" + str(abs(joint.angle) * joint.angle_multiplier + init_angle)

                else:
                    text_str += "&" + robot_joint + "=" + str(abs(joint.angle) * joint.angle_multiplier + init_angle)

        elif (action [0] [0] in self.available_commands.keys ()):
            action_str = action [0] [0]
            text_str   = str (action [0] [1] [0])
            #print ("text_str", text_str, "act", action [0])

        else:
            print ("action :", action, " is not implemented")
            return -1

        if (self.simulated.updated == True or action [0] == "/free"):
            if (action [0] [0] != "/free"):
                print ("sending command [physical]: ", action)

            r = requests.get (self.ip + self.port + "/?" + "action="
                + action_str + "&" + "text=" + text_str)
            self.simulated.updated = False

        return r

    def on_idle (self):
        if (self.free_timeout_module.timeout_passed ()):
            #r = self._send_command ([["/free", "a"]])
            #print ("resp", r)

            free = 6#int (str (r) [13:14]) #6 free, 7 not free; don't ask, don't tell

            if (free == 6):
                self.free = True

            else:
                self.free = True

        #print ("queue", self.queue [self.commands_sent:])
        #print (len (self.queue), self.commands_sent, self.free)

        print ("len and sent", len (self.queue), self.commands_sent)

        if (self.timeout_module.timeout_passed (len (self.queue) > self.commands_sent) and
            self.free == True):
            command = self.queue [self.commands_sent]

            #while (command == "/noaction" and len (self.queue) > self.commands_sent):
            #    self.commands_sent += 1
            #    command = self.queue[self.commands_sent]

            print ("command", command)

            self._send_command (command)
            self.commands_sent += 1

            #self.commands_sent = len (self.queue)

    def plot_state (self, img, x, y, scale = 1):
        self.simulated.plot_state (img, x, y, scale)
