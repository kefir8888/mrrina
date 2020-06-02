# -*- coding: utf-8 -*-

from naoqi import ALProxy
from common import *
import copy

markup_color = (250, 250, 250)
axes = {"x" : 0, "y" : 1, "z" : 2}

from modalities.music_modality import Archive_angles

class Robot:
    def __init__(self, timeout_ = 0.001, logger_ = 0):
        self.queue         = []
        self.commands_sent = 0
        self.name = "base"

        self.logger = logger_

        self.available_commands = {"/Rest"  : ("/action=/rest&text=", "a"),
                                   "/Stand" : ("/action=/stand&text=", "a"),
                                   "/Sit" : ("/action=/stand&text=", "a"),
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

    def _send_command (self, command):
        pass

    def plot_state (self, img):
        pass

    def on_idle (self):
        #If the robot is simulated, it is supposed to perform all the available
        #actions from the queue. If the robot is real, it (to this date)
        #performs only one action

        #print ("on_idle, ", len (self.queue), self.commands_sent)

        if (self.timeout_module.timeout_passed (len (self.queue) > self.commands_sent)):
            #print ("TIMEOUT_PASSED")
            while (len (self.queue) > self.commands_sent):
                # print ("len >")
                next_command = self.queue [self.commands_sent]
                self.commands_sent += 1

                if (next_command [0] != "noaction"):
                    # print ("sending ", next_command)
                    #print (self.queue)
                    #print ("")
                    self._send_command (next_command)

                if (self.name == "real"):
                    break

    def add_action (self, action):
        #print ("appending ", action)

        if (action is None):
            return

        act_list = []

        for act in action [0]:
            if (act [0] != "noaction"):
                #print ("appending action", act)
                act_list.append (act)

        if (len (act_list) > 0):
            self.queue.append (act_list)

class Fake_robot(Robot):
    def __init__(self, timeout_ = 0.5):
        Robot.__init__ (self, timeout_)
        self.name = "fake"

    def _send_command (self, action):
        action=action
        # if (action [0] in self.available_commands.keys ()):
        #     # print ("sending command [fake]: ", action)
        #
        # else:
        #     print ("action :", action, " is not implemented")

class Joint:
    def __init__(self, length_, angle_, angle_multiplier_, col1_, col2_, name_, min_angle_, max_angle_, angle_shift_):
        self.length     = length_
        self.init_angle = angle_
        self.angle      = 0
        self.angle_multiplier = angle_multiplier_
        self.min_angle = min_angle_
        self.max_angle = max_angle_
        self.col1       = col1_
        self.col2       = col2_
        self.joint_name = name_
        self.angle_shift = angle_shift_

        self.children = []

    def name (self):
        return self.joint_name

    def draw (self, img, x, y, parent_angle, scale = 1):
        # print("DRAW", y)
        if y != None and x is not None: #тут была проверка на флоат, но сейчас координаты уже в инт

            # print ("joint: ", self.length, self.angle)

            angle = self.init_angle - self.angle + parent_angle

            #print("HENLO", x, self.init_angle, self.angle, parent_angle, self.name())
            x1 = x + self.length * math.cos (angle)
            y1 = y + self.length * math.sin (angle)

            x_  = int (x * scale)
            y_  = int (y * scale)


            x1_ = int (float (x1 * scale))
            y1_ = int (float (y1 * scale))

            cv2.line (img, (int (x_), int (y_)), (int (x1_), int (y1_)), self.col1, 3)
            cv2.circle (img, (int (x1_), int (y1_)), 5, self.col2, -1)
            cv2.putText (img, self.joint_name, (int (x1_) + 0, int (y1_) + 0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 250, 231), 1, cv2.LINE_AA)

            for child in self.children:
                #print("Child", angle)
                child.draw (img, x1, y1, angle, scale)

    def set_angle (self, new_angle, omit_warnings = False):
        if (new_angle < self.min_angle):
            self.angle = self.min_angle

            if (omit_warnings == False):
                print ("warning: cannot set joint " + self.joint_name + " to " + str (new_angle) +\
                         ": the constraints are [" + str (self.min_angle) + ", " + str (self.max_angle) + "]")

        elif (new_angle > self.max_angle):
            self.angle=self.max_angle

            if (omit_warnings == False):
                print ("warning: cannot set joint " + self.joint_name + " to " + str (new_angle) +\
                         ": the constraints are [" + str (self.min_angle) + ", " + str (self.max_angle) + "]")
        else:
            self.angle = new_angle
        # if (new_angle >= self.min_angle and new_angle <= self.max_angle):
        #     self.angle = new_angle
        #
        # else:
        #     print ("warning: cannot set joint " + self.joint_name + " to " + str (new_angle) +\
        #         ": the constraints are [" + str (self.min_angle) + ", " + str (self.max_angle) + "]")

        return self.angle

    def add_child (self, name, length, angle, angle_multiplier, min_angle, max_angle, angle_shift):
        self.children.append (Joint (length, angle, angle_multiplier, self.col1, self.col2, name, min_angle, max_angle, angle_shift))

class Simulated_robot(Robot):
    def __init__(self, timeout_ = 0.0, path_ = "", logger_ = 0, omit_warnings_ = False):
        Robot.__init__ (self, timeout_)

        self.config_path = path_
        self.logger = logger_
        self.omit_warnings = omit_warnings_

        self.base_point = Joint (0, 0, 1, (10, 100, 200), (230, 121, 2), "base", -10, 10, 69*420)
        #(self, length_, angle_, angle_multiplier_, col1_, col2_, name_, min_angle_, max_angle_)
        self.load_configuration (self.config_path)

        self.joints_to_track = ["r_sho_roll", "l_sho_roll", "r_sho_pitch", "l_sho_pitch", "r_elb_roll", "l_elb_roll", "r_elb_yaw", "l_elb_yaw"]

        self.updated = False
        self.name = "simulated"

    def load_configuration (self, path = ""):
        if (path == ""):
            path = "robot_configuration.txt"

        config = open (path, "r")

        string = config.readline ()

        while (string != ""):
            data = string [:-1].split (" ")

            # print ("dat", data)

            parent = str   (data [0])
            name   = str   (data [1])
            length = float (data [2])
            angle  = float (data [3])
            # print("HElllLO", angle)
            angle_multiplier = float (data [4])
            min_angle = float (data [5])
            max_angle = float (data [6])
            shift_angle = float (data [7])

            self.add_joint (parent, name, length, angle, angle_multiplier, min_angle, max_angle, shift_angle)

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
            # print(new_angle)

        set_angle = target_joint.set_angle (new_angle, self.omit_warnings)

        # print ("joint: ", joint_name, set_angle)

        if (set_angle == new_angle):
            self.updated = True

        return set_angle

    def add_joint (self, parent_name, new_joint_name, length, angle, angle_multiplier, min_angle, max_angle, angle_shift):
        target_joint, succ = self.find_joint (parent_name)

        if (succ == False):
            print ("Unable to add child ", new_joint_name, " to ", parent_name, ": no such joint")

        target_joint.add_child (new_joint_name, length, angle, angle_multiplier, min_angle, max_angle, angle_shift)

    def _send_command (self, actions):
        for action in actions:
            #print ("Sim action [0]: ", action [0])
            if (action [0] in self.available_commands.keys ()):
                self.updated = True
                # print ("sending command [simulated]: ", action)

                if (action [0] == "/increment_joint_angle"):
                    self.set_joint_angle (action [1] [0], float (action [1] [1]), increment = True)
                    #print((action [1] [0], float (action [1] [1])))

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
        #print("HENLO")
        self.base_point.draw(img, x, y, 0, scale)

        return img

class Vector:
    def __init__(self, coords_=[]):
        self.coords = coords_

    def get_coords(self):
        return self.coords

    def dotproduct(self, v):
        return sum((a * b) for a, b in zip(self.coords, v.coords))

    def length(self):
        return math.sqrt(self.dotproduct(self))

    def subtr(self, v):
        return Vector([a - b for a, b in zip(self.coords, v.coords)])

    def add(self, v):
        return Vector([a + b for a, b in zip(self.coords, v.coords)])

    def mul(self, coeff):
        return Vector([a * coeff for a in self.coords])

    def cos(self, v):
        return self.dotproduct(v) / (self.length() * v.length() + 0.0001)

    def change_coord(self, coord, val=0, increment=False, invert=False):
        if (invert == True):
            self.coords[axes[coord]] *= -1

        else:
            if (increment == True):
                self.coords[axes[coord]] += val

            else:
                self.coords[axes[coord]] = val

    def rotate_2d(self, axis1, axis2, angle):
        orig1 = self.coords[axes[axis1]]
        orig2 = self.coords[axes[axis2]]

        rot1 = orig1 * math.cos(angle) + orig2 * math.sin(angle)
        rot2 = - orig1 * math.sin(angle) + orig2 * math.cos(angle)

        self.coords[axes[axis1]] = rot1
        self.coords[axes[axis2]] = rot2

    def copy(self):
        return Vector(copy.deepcopy(self.coords))

    def scale(self, coeff):
        for i in range(len(self.coords)):
            self.coords[i] *= coeff

class Canvas:
    def __init__(self, xsz_, ysz_, zsz_, centerx_, centery_, WIND_X_, WIND_Y_):
        self.xsz = xsz_
        self.ysz = ysz_
        self.zsz = zsz_
        self.centerx = centerx_
        self.centery = centery_
        self.WIND_X = WIND_X_
        self.WIND_Y = WIND_Y_

        self.canvas = np.ones ((self.WIND_Y, self.WIND_X, 3), np.uint8) * 55
        self.canvas_ = np.ones ((self.WIND_Y, self.WIND_X, 3), np.uint8) * 55

        self.tick = 0

    def get_canvas(self):
        return self.canvas

    def refresh(self):
        self.canvas = np.ones ((self.WIND_Y, self.WIND_X, 3), np.uint8) * 55

        # if (self.tick > 0):
        #     #root.draw(canvas, int(440), 700, 2, tick)
        #
        #     shadow = self.canvas.astype("float") * 0.99
        #     new_canvas = self.canvas_.copy()
        #
        #     shift = 13
        #     new_canvas[: -7, shift:, :] = shadow[7:, : -shift, :]
        #
        #     self.canvas = new_canvas
        #
        # cv2.imwrite ("/Users/elijah/Dropbox/Programming/RoboCup/remote control/skel_imgs/img" + str (self.tick) + ".jpg", self.canvas)

        self.tick += 1

        #self.draw_space_box()

    def _transform_point(self, p):
        coords = p.get_coords()

        x = int((coords[0] / (coords[2] + 0) + self.centerx) * self.WIND_X / self.xsz)
        y = int((coords[1] / (coords[2] + 0) + self.centery) * self.WIND_Y / self.ysz)

        return x, y

    def draw_3d_line(self, p1, p2, color = markup_color, thickness=1):
        if (thickness == - 1):
            return

        x1, y1 = self._transform_point(p1)
        x2, y2 = self._transform_point(p2)

        cv2.line(self.canvas, (x1, y1), (x2, y2), color, thickness)

    def draw_3d_triangle(self, p1, p2, p3, color):
        x1, y1 = self._transform_point(p1)
        x2, y2 = self._transform_point(p2)
        x3, y3 = self._transform_point(p3)

        contour = np.array([(x1, y1), (x2, y2), (x3, y3)])

        cv2.drawContours(self.canvas, [contour], 0, color, -1)

    def draw_3d_circle(self, p, r, color):
        x, y = self._transform_point(p)

        cv2.circle(self.canvas, (x, y), int(r / p.get_coords()[2]), color, 1)

    def draw_space_box(self):
        lucc = Vector([- self.centerx, - self.centery, 1])  # left-upper-close corner
        ludc = Vector([- self.centerx, - self.centery, self.zsz])
        ldcc = Vector([- self.centerx, self.ysz - self.centery, 1])
        lddc = Vector([- self.centerx, self.ysz - self.centery, self.zsz])

        rucc = Vector([self.xsz - self.centerx, - self.centery, 1])
        rudc = Vector([self.xsz - self.centerx, - self.centery, self.zsz])
        rdcc = Vector([self.xsz - self.centerx, self.ysz - self.centery, 1])
        rddc = Vector([self.xsz - self.centerx, self.ysz - self.centery, self.zsz])

        self.draw_3d_line(lucc, ludc, markup_color)
        self.draw_3d_line(ldcc, lddc, markup_color)
        self.draw_3d_line(rucc, rudc, markup_color)
        self.draw_3d_line(rdcc, rddc, markup_color)

        self.draw_3d_line(ludc, rudc, markup_color)
        self.draw_3d_line(rudc, rddc, markup_color)
        self.draw_3d_line(rddc, lddc, markup_color)
        self.draw_3d_line(lddc, ludc, markup_color)

    def put_text (self, text, x, y, color=(100, 250, 130)):
        cv2.putText(self.canvas, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)

    def put_text_3d (self, text, p, color=(100, 250, 130)):
        x, y = self._transform_point(p)

        self.put_text (text, x, y)

# class Limb_3D:
#     def __init__ (self, name_, limb_, coords_, init_angle_, axis_ = "x"):
#         self.name   = name_
#         self.limb   = limb_
#         self.coords = coords_
#         self.angle  = init_angle_
#         self.axis   = axis_
#
#         #self.children = []
#
#     def draw (self, canvas):
#         canvas.draw_3d_line (self.coords, self.coords.add (self.limb), thickness = 3)

class Limb_3D:
    def __init__(self, name_, limb_, axis_, col1_, col2_, angle_multiplier_):
        self.init_limb = limb_
        self.limb = copy.deepcopy (self.init_limb)
        self.angle_multiplier = angle_multiplier_

        self.axis  = axis_
        self.angle = 0

        self.col1       = col1_
        self.col2       = col2_
        self.joint_name = name_

        self.children = []

    def name (self):
        return self.joint_name

    #def translate (self, vec):
    #    self.limb = self.limb.add (vec)

    def rot_inds (self):
        ind1, ind2 = [ax for ax in axes if ax not in [self.axis]][:]

        return ind1, ind2

    def construct_configuration (self):
        configuration = []

        ind1, ind2 = self.rot_inds ()

        for child in self.children:
            configuration += copy.deepcopy (child.construct_configuration ())

        for el in configuration:
            el [1].rotate_2d (ind1, ind2, self.angle)
            el[1] = el[1].add (self.limb)

            el [2].rotate_2d (ind1, ind2, self.angle)
            el [2] = el [2].add (self.limb)

        #print(self.name (), self.limb.get_coords())

        configuration.append ([self.joint_name, Vector ([0, 0, 0]), self.limb])

        return configuration

    def set_angle (self, new_angle):
        self.angle = new_angle * self.angle_multiplier

        ind1, ind2 = self.rot_inds ()

        self.limb = self.init_limb.copy ()
        self.limb.rotate_2d (ind1, ind2, self.angle)

        #print ("axis", self.axis, self.angle, self.limb.get_coords ())

        return self.angle

    def add_child (self, name, limb, axis, angle_multiplier):
        self.children.append (Limb_3D (name, limb, axis, self.col1, self.col2, angle_multiplier))

class Simulated_robot_3D (Robot):
    def __init__(self, timeout_ = 0.01, path_ = "", logger_ = 0, WIND_X_ = 800, WIND_Y_ = 700, omit_warnings_ = False):
        Robot.__init__ (self, timeout_)

        self.config_path = path_
        self.logger = logger_

        self.canvas = Canvas (1.6, 1.4, 2.5, 0.8, 0.7, WIND_X_, WIND_Y_)

        self.col1 = (10, 100, 200)
        self.col2 = (230, 121, 2)

        self.base_point = Limb_3D ("base", Vector ([-0.25, -0.4, 1.0]), "z", self.col1, self.col2, 1)
        self.load_configuration (self.config_path)

        self.joints_to_track = ["r_sho_roll", "l_sho_roll", "r_sho_pitch", "l_sho_pitch", "r_elb_roll", "l_elb_roll", "r_elb_yaw", "l_elb_yaw"]
        #self.joints_to_track = ["l_sho_roll"]

        self.updated = False
        self.name = "simulated3d"

        self.simulated = Simulated_robot (logger_=self.logger, omit_warnings_ = omit_warnings_)

    def load_configuration (self, path = ""):
        if (path == ""):
            path = "robot_configuration_3d.txt"

        config = open (path, "r")

        string = config.readline ()

        while (string != ""):
            data = string [:-1].split (" ")

            parent = str (data [0])
            name   = str (data [1])

            x = float (data [2])
            y = float (data [3])
            z = float (data [4])
            limb = Vector ([x, y, z])

            axis = str (data [5])
            angle_multiplier = float (data [6])

            #print (name, x, y, z)

            self.add_limb (parent, name, limb, axis, self.col1, self.col2, angle_multiplier)

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

    def add_limb (self, parent_name, new_limb_name, limb, axis, col1, col2, angle_multiplier):
        target_joint, succ = self.find_joint (parent_name)

        if (succ == False):
            print ("Unable to add child ", new_limb_name, " to ", parent_name, ": no such joint")

        target_joint.add_child (new_limb_name, limb, axis, angle_multiplier)

    def _send_command (self):
        action = self.queue[self.commands_sent]

        while (True):
            action_ = self.queue[self.commands_sent]
            self.commands_sent += 1

            self.simulated._send_command(action_)

            if (not ((action[0][0] == "/increment_joint_angle" or
                      action[0][0] == "/set_joint_angle") and
                     action[0][0] == action_[0][0] and
                     len(self.queue) > self.commands_sent)):
                break

        if (action[0][0] == "/increment_joint_angle" or
                action[0][0] == "/set_joint_angle"):
            action_str = "/raise_hands"
            text_str = ""

            #for key in self.synchronized_joints.keys():
            for j in self.joints_to_track:
                joint, _ = self.simulated.find_joint (j)
                #robot_joint = self.synchronized_joints [key]
                #init_angle = self.init_positions [robot_joint]

                #if (joint.angle is None):
                #    joint.angle = 0

                #angle = joint.angle * joint.angle_multiplier + init_angle
                #names.append(robot_joint)
                #angles.append([angle])

            #self.motionProxy.angleInterpolation(names, angles, timeList, True)
            #if (action[0] == "/increment_joint_angle"):
                #print ("ang", joint.angle)
                self.set_joint_angle (j, joint.angle)

        # elif (action[0][0] in self.available_commands.keys()):
        #     action_str = action[0][0]
        #
        #     if action_str[1:] == "Rest":
        #         self.motionProxy.rest()
        #
        #     elif action_str[1:] == "Stand":
        #         self.postureProxy.goToPosture(action_str[1:], 2)
        #
        #     text_str = str(action[0][1][0])
        #
        # else:
        #     print("action :", action, " is not implemented")
        #     return -1
        #
        # if (self.simulated.updated == True or action[0] == "/free"):
        #     request_str = self.ip_num + "/?" + "action=" \
        #                   + action_str + "&" + "text=" + text_str
        #
        #     self.simulated.updated = False
        #
        # for action in actions:
        #     if (action [0] in self.available_commands.keys ()):
        #         self.updated = True
        #         # print ("sending command [simulated]: ", action)
        #
        #         if (action [0] == "/increment_joint_angle"):
        #             self.set_joint_angle (action [1] [0], float (action [1] [1]), increment = True)
        #             #print((action [1] [0], float (action [1] [1])))
        #
        #         if (action [0] == "/set_joint_angle"):
        #             self.set_joint_angle (action [1] [0], float (action [1] [1]))
        #
        #         elif (action [0] == "/stand"):
        #             #self.set_joint_angle ("righthand", -0.2)
        #             self.base_point.children = []
        #             self.load_configuration (self.config_path)
        #             self.set_joint_angle ("base", 0)
        #
        #         elif (action [0] == "/rest"):
        #             self.set_joint_angle ("base", 5)
        #
        #         elif (action [0] == "/hands_sides"):
        #             self.set_joint_angle ("righthand", 1)
        #
        #     else:
        #         print ("action :", action, " is not supported")

    def plot_state (self, img, x, y, scale = 1):
        # line_num = 0
        #
        # for joint in self.find_joint (""):
        #     if (joint.name () not in self.joints_to_track):
        #         continue
        #
        #     text = joint.name () + ":  [" + "{0:.2f}".format(joint.min_angle) \
        #                          + " / " + "{0:.2f}".format(joint.angle) \
        #                          + " /  " + "{0:.2f}".format(joint.max_angle) + "]"
        #
        #     cv2.putText (img, text, (30, 30 * (1 + line_num)),
        #         cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 50, 231), 1, cv2.LINE_AA)
        #
        #     line_num += 1
        #print("HENLO")

        self.canvas.refresh ()
        skeleton = self.base_point.construct_configuration ()

        #print ("aa")

        for element in skeleton:
            thickness = 3

            if (element [0] == "base"):
                thickness = -1

            #print (element [0], element [1].get_coords ())

            self.canvas.draw_3d_line (element [1], element [2], self.col1, thickness)
            self.canvas.draw_3d_circle (element [2], 9, self.col2)
            #self.canvas.put_text_3d (element [0] [:7], element [2])

        return self.canvas.get_canvas ()

    def on_idle (self):
        if (len (self.queue) > self.commands_sent):
            self._send_command ()#command)

            self.commands_sent = len (self.queue)

class Real_robot(Robot):
    def __init__(self, ip_num, port_ = 9559, timeout_ = 0.04, logger_ = 0):
        Robot.__init__ (self, timeout_)
        self.logger = logger_

        self.ip_prefix = "http://"
        self.ip_postfix = ":"

        self.ip   = self.ip_prefix + ip_num + self.ip_postfix
        self.port = port_

        self.free = False
        self.free_timeout_module = Timeout_module (0.3)

        self.simulated = Simulated_robot (logger_ = self.logger)

        self.synchronized_joints = {"head_Yaw"    : "head_Yaw",
                                    "head_Pitch"  : "head_Pitch",

                                    "l_sho_roll"  : "l_shoulderroll",
                                    "l_sho_pitch" : "l_shoulderpitch",
                                    "l_elb_roll"  : "l_elbowroll",
                                    "l_elb_yaw"   : "l_elbowyaw" ,

                                    "l_hip_roll"  : "l_hiproll",
                                    "l_hip_pitch" : "l_hippitch",

                                    "l_knee_pitch": "l_kneepitch",
                                    "l_ank_pitch" : "l_ankpitch",
                                    "l_ank_roll"  : "l_ankroll",

                                    "r_sho_roll"  : "r_shoulderroll",
                                    "r_sho_pitch" : "r_shoulderpitch",
                                    "r_elb_roll"  : "r_elbowroll",
                                    "r_elb_yaw"   : "r_elbowyaw",

                                    "r_hip_roll"  : "r_hiproll",
                                    "r_hip_pitch" : "r_hippitch",

                                    "r_knee_pitch": "r_kneepitch",
                                    "r_ank_pitch" : "r_ankpitch",
                                    "r_ank_roll"  : "r_ankroll"
                                    }

        self.init_positions = {"r_shoulderpitch" : 0,
                               "r_shoulderroll"  : 0,
                               "r_elbowroll"     : 0,
                               "r_elbowyaw"      : 0,
                               "r_hiproll"       : 0,
                               "r_hippitch"      : 0,
                               "r_kneepitch"     : 0,
                               "r_ankpitch"      : 0,
                               "r_ankroll"       : 0,

                               "l_shoulderpitch" : 0,
                               "l_shoulderroll"  : 0,
                               "l_elbowroll"     : 0,
                               "l_elbowyaw"      : 0,
                               "l_hiproll"       : 0,
                               "l_hippitch"      : 0,
                               "l_kneepitch"     : 0,
                               "l_ankpitch"      : 0,
                               "l_ankroll"       : 0,

                               "head_Yaw"        : 0,
                               "head_Pitch"      : -0.3}

        self.name = "real"

    def _send_command (self):#, action):
        r = -1

        #if (action [0] == "noaction"):
        #    pass

        # print ("command to simulated: ", action)
        action = self.queue [self.commands_sent]
        #self.commands_sent += 1
        action_ = action

        # print ("queue", self.queue [self.commands_sent :])

        while (True):
            action_ = self.queue [self.commands_sent]
            self.commands_sent += 1

            self.simulated._send_command (action_)
            print ("action: ", action_)

            if (not ((action [0] [0] == "/increment_joint_angle" or
                 action [0] [0] == "/set_joint_angle") and
                 action [0] [0] == action_ [0] [0] and
                 len (self.queue) > self.commands_sent)):
                break

        #print ("lalala")
        #print (action [0] [0])
        #print (self.available_commands.keys())

        if (action [0] [0] == "/increment_joint_angle" or
            action [0] [0] == "/set_joint_angle"):
            action_str = "/raise_hands"
            text_str   = ""

            for key in self.synchronized_joints.keys ():
                # print ("кий", key)
                joint, _ = self.simulated.find_joint (key)
                robot_joint = self.synchronized_joints [key]
                init_angle = self.init_positions [robot_joint]
                angle_shift = joint.angle_shift
                min_angle = joint.min_angle
                max_angle = joint.max_angle

                # print("PIRKOVA ZA CHTO: ", robot_joint)

                if (joint.angle is None):
                    joint.angle = 0

                angle = joint.angle * joint.angle_multiplier + init_angle
                #if key == "righthand":
                #    print("SHANKOV ZA CHTO: ", joint.angle, angle)
                # if (angle < min_angle):
                #     angle = min_angle
                #
                # if (angle > max_angle):
                #     angle = max_angle
                # print("Send_commsnd posle:", angle)

                # if (key == "righthand"):
                    # print ("hand", joint.angle, joint.angle_multiplier, init_angle)
                    # ang_ = joint.angle * (joint.angle_multiplier) - 1.57
                    # ang = ang_

                    # ang = -0.6 + ang_ / 3
                    # print ("ANG", ang)

                # text_str += "&" + robot_joint + "=" + str(angle)

                #else:
                text_str += "&" + robot_joint + "=" + str(angle)
                #print(text_str)

        elif (action [0] [0] in self.available_commands.keys ()):
            action_str = action [0] [0]
            text_str   = str (action [0] [1] [0])
            #print ("text_str", text_str, "act", action [0])

        else:
            print ("action :", action, " is not implemented")
            return -1

        if (self.simulated.updated == True or action [0] == "/free"):
            #if (action [0] [0] != "/free"):
            #    print ("sending command [physical]: ", action)

            request_str = self.ip + self.port + "/?" + "action="\
                + action_str + "&" + "text=" + text_str

            # print ("Final", request_str)

            r = 5

            try:
                r = requests.get (request_str)

            except:
                print ("Request", request_str, " failed, skipping. Check if the robot's behaviour is alive")

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

        # print ("len and sent", len (self.queue), self.commands_sent)

        if (self.timeout_module.timeout_passed (len (self.queue) > self.commands_sent) and
            self.free == True):
            #command = self.queue [self.commands_sent]

            #print ("command", command)

            self._send_command ()#command)
            #self.commands_sent += 1

            #self.commands_sent = len (self.queue)

    def plot_state (self, img, x, y, scale = 1):
        self.simulated.plot_state (img, x, y, scale)

class Real_robot_qi(Robot):
    def __init__(self, ip_num_, timeout_ = 0.04, logger_ = 0, action_time_ = 0.8, omit_warnings_ = False):
        Robot.__init__ (self, timeout_)
        self.logger = logger_
        self.first_frame = True

        self.action_time = action_time_

        print("Hello")

        self.ip_num = ip_num_
        self.motionProxy = ALProxy("ALMotion", self.ip_num, 9559)

        self.postureProxy = ALProxy("ALRobotPosture", self.ip_num, 9559)

        self.motionProxy.wbEnable(False)
        self.postureProxy.goToPosture("Stand", 1.5)
        #self.motionProxy.wbEnable(True)
        #self.motionProxy.setBreathEnabled('Body', False)

        #self.motionProxy.wbFootState("Fixed", "Legs")
        #self.motionProxy.wbEnableBalanceConstraint(True, "Legs")

        pNames = "Body"
        pStiffnessLists = 0.6
        pTimeLists = 1.0
        self.motionProxy.stiffnessInterpolation (pNames, pStiffnessLists, pTimeLists)

        self.simulated = Simulated_robot (logger_=self.logger, omit_warnings_ = omit_warnings_)

        ########################################################################################################################################33
        self.synchronized_joints = {"head_Yaw"    : "HeadYaw",
                                    "head_Pitch"  : "HeadPitch",

                                    "l_sho_roll"  : "LShoulderRoll",
                                    "l_sho_pitch" : "LShoulderPitch",
                                    "l_elb_roll"  : "LElbowRoll",
                                    "l_elb_yaw"   : "LElbowYaw" ,

                                    "l_hip_roll"  : "LHipRoll",
                                    "l_hip_pitch" : "LHipPitch",

                                    "l_knee_pitch": "LKneePitch",
                                    "l_ank_pitch" : "LAnklePitch",
                                    "l_ank_roll"  : "LAnkleRoll",

                                    "r_sho_roll"  : "RShoulderRoll",
                                    "r_sho_pitch" : "RShoulderPitch",
                                    "r_elb_roll"  : "RElbowRoll",
                                    "r_elb_yaw"   : "RElbowYaw",

                                    "r_hip_roll"  : "RHipRoll",
                                    "r_hip_pitch" : "RHipPitch",

                                    "r_knee_pitch": "RKneePitch",
                                    "r_ank_pitch" : "RAnklePitch",
                                    "r_ank_roll"  : "RAnkleRoll"
                                    }

        self.init_positions = {"RShoulderPitch" : 1.1,
                               "RShoulderRoll"  : 0,
                               "RElbowRoll"     : 0,
                               "RElbowYaw"      : 0,
                               "RHipRoll"       : 0,
                               "RHipPitch"      : 0,
                               "RKneePitch"     : 0,
                               "RAnklePitch"    : 0,
                               "RAnkleRoll"     : 0,

                               "LShoulderPitch" : 1.1,
                               "LShoulderRoll"  : 0,
                               "LElbowRoll"     : 0,
                               "LElbowYaw"      : 0,
                               "LHipRoll"       : 0,
                               "LHipPitch"      : 0,
                               "LKneePitch"     : 0,
                               "LAnklePitch"    : 0,
                               "LAnkleRoll"     : 0,

                               "HeadYaw"        : 0,
                               "HeadPitch"      : -0.3}

        self.name = "real_qi"

    def __del__ (self):
        self.motionProxy.wbEnable(False)

    def _send_command (self):#, action):
        action = self.queue [self.commands_sent]

        names = []

        for key in self.synchronized_joints.keys():
            #print ("key1", key)
            robot_joint = self.synchronized_joints[key]
            names.append(robot_joint)

        angles = {name : [] for name in names}
        times  = {name : [] for name in names}

        first_turn = True

        t = 0

        print ("first action", action)

        while (True):
            if (len (self.queue) <= self.commands_sent):
                break

            #print ("t", t)

            action_ = self.queue [self.commands_sent]
            self.commands_sent += 1

            #print ("self.queue", self.queue)

            self.simulated._send_command (action_)

            if (not ((action [0] [0] == "/increment_joint_angle" or
                 action [0] [0] == "/set_joint_angle") and
                 action [0] [0] == action_ [0] [0])):
                if (action[0][0] in self.available_commands.keys()):
                    action_str = action[0][0]

                    if action_str[1:] == "Rest":
                        self.motionProxy.rest()

                    elif action_str[1:] == "Stand":
                        self.postureProxy.goToPosture(action_str[1:], 2)

                    else:
                        print("action :", action, " is not implemented")
                        return -1

                break

            curr_step_t = 0
            if (self.first_frame == False or first_turn == False):
                #timeList = [self.action_time] * 20
                curr_step_t = self.action_time

            else:
                #timeList = [1.0] * 20
                curr_step_t = 1.0
                self.first_frame = False

            t += curr_step_t

            if (action [0] [0] == "/increment_joint_angle" or
                action [0] [0] == "/set_joint_angle"):
                #print ("sync")
                for key in self.synchronized_joints.keys ():
                    #print("key2", key)

                    joint, _ = self.simulated.find_joint (key)
                    robot_joint = self.synchronized_joints [key]
                    init_angle = self.init_positions [robot_joint]

                    if (joint.angle is None):
                        joint.angle = 0

                    angle = joint.angle * joint.angle_multiplier + init_angle
                    angles [robot_joint].append (angle)
                    times  [robot_joint].append (t)

            first_turn = False

        angles_ = [angles [key] for key in names]
        times_  = [times  [key] for key in names]

        #print ("times", times_, angles_)

        #curr_angles = self.motionProxy.getAngles ("Body", True)
        #print ("curr angles", curr_angles)
        #print ("angles [0]", angles_ [0])

        if (len (angles_ [0]) > 0):
            print ("sending angleinterpolation")
            self.motionProxy.angleInterpolation (names, angles_, times_, True)
            print ("angleinterpolation finished")

            #text_str   = str (action [0] [1] [0])

        # if (self.simulated.updated == True or action [0] == "/free"):
        #     request_str = self.ip_num  + "/?" + "action="\
        #         + action_str + "&" + "text=" + text_str
        #
        #     self.simulated.updated = False

    def on_idle (self):
        if (len (self.queue) > self.commands_sent):
            self._send_command ()#command)

            self.commands_sent = len (self.queue)

    def plot_state (self, img, x, y, scale = 1):
        return self.simulated.plot_state (img, x, y, scale)

# class Real_robot_timeline(Robot):
#     def __init__(self, ip_num_, timeout_ = 0.04, logger_ = 0, action_time_ = 0.8, omit_warnings_ = False):
#         Robot.__init__ (self, timeout_)
#         self.logger = logger_
#         self.action_time = action_time_
#
#         #-----------------------
#         self.ip_num = ip_num_
#         self.motionProxy = ALProxy("ALMotion", self.ip_num, 9559)
#         self.postureProxy = ALProxy("ALRobotPosture", self.ip_num, 9559)
#
#         self.motionProxy.wbEnable(False)
#         self.postureProxy.goToPosture("Stand", 1.5)
#         self.motionProxy.wbEnable(True)
#         self.motionProxy.setBreathEnabled('Body', False)
#
#         #self.motionProxy.wbFootState("Fixed", "Legs")
#         #self.motionProxy.wbEnableBalanceConstraint(True, "Legs")
#
#         #pNames = "Body"
#         #pStiffnessLists = 0.6
#         #pTimeLists = 1.0
#         #self.motionProxy.stiffnessInterpolation (pNames, pStiffnessLists, pTimeLists)
#         #------------------------
#
#         self.simulated = Simulated_robot (logger_=self.logger, omit_warnings_ = omit_warnings_)
#
#         self.smol_listb = ["l_sho_roll", "l_elb_roll", "l_sho_pitch",
#                            "r_sho_roll", "r_elb_roll", "r_sho_pitch"]
#
#         self.name = "real_timeline"
#
#         self.command_sent = False
#
#     def __del__ (self):
#         #self.motionProxy.wbEnable(False)
#         pass
#
#     def _send_command (self):
#         names = list ()
#         times = list ()
#         mot_keys  = list ()
#
#         robot_names = ["LShoulderRoll", "LElbowRoll", "LShoulderPitch"]
#                        #"RShoulderRoll", "RElbowRoll", "RShoulderPitch"]
#
#         for i, robot_angle_name in enumerate (robot_names):
#             l = 0
#
#             angles = Archive_angles("/Users/elijah/Downloads/dataset/DANCE_R_6/angles.json")
#
#             names.append (robot_angle_name)
#
#             k = []
#             t = []
#
#             tim_step = 0.05
#             tim = tim_step
#
#             while (angles.end_of_data () == False):
#                 l += 1
#                 if (l > 151):
#                     break
#
#                 command = angles.get_command ()
#                 #print ("command", command)
#                 self.simulated.add_action ([command])
#                 self.simulated.on_idle ()
#
#                 tim += tim_step
#                 t.append (tim)
#
#                 joint_key = self.smol_listb [i]
#
#                 joint, _ = self.simulated.find_joint (joint_key)
#
#                 #robot_joint = self.synchronized_joints[key]
#                 #init_angle = self.init_positions[robot_joint]
#
#                 #if (joint.angle is None):
#                 #    joint.angle = 0
#
#                 #angle = joint.angle * joint.angle_multiplier + init_angle
#                 #names.append(robot_joint)
#                 #angles.append([angle])
#
#                 joint_angle = joint.angle
#                 #joint_angle = math.sin (float (l) / 10)
#
#                 if (robot_angle_name == "RShoulderPitch" or robot_angle_name == "LShoulderPitch"):
#                     joint_angle +=  1.1
#
#                 #print("s", joint_angle)
#
#                 k.append (joint_angle)
#
#             #print ("SaS")
#
#             mot_keys.append  (k)
#             times.append (t)
#
#         # names = ["LShoulderRoll"]
#         # times = [[0.92, 2, 3.4, 4.48, 5.8, 6.32, 6.84]]
#         # keys = [[0.139697, 0.15506, 0.139697, 0.15506, 0.139697, 0.00609397, 0.116542]]
#
#         print ("a", mot_keys, times, names)
#
#         try:
#             print("starting interpolation")
#             #motion = ALProxy("ALMotion", "192.168.1.8", 9559)
#             #motion.angleInterpolation (names, mot_keys, times, True)
#             self.motionProxy.angleInterpolation(names, mot_keys, times, True)
#             print ("interpolation over")
#
#         except BaseException, err:
#             print ("mlem")
#
#     def on_idle (self):
#         if (self.command_sent == False):
#             self._send_command ()
#             self.command_sent = True
#
#     def plot_state (self, img, x, y, scale = 1):
#         return img
