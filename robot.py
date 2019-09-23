import json
import math
import time
import argparse
import numpy as np
import pandas as pd
# from naoqi import ALProxy
import requests
import pyautogui

# Global variables
listAngles = []
shoulderLeft = []
elbowLeft = []
wristLeft = []
shoulderRight = []
elbowRight = []
wristRight = []
t = 0
RobotIP = '192.168.1.29'
# RobotIP = "172.20.10.14"
PORT = '9562'

cursor = [

    (1712, 517),  # 1. start recording
    (1863, 517),  # 2. end


]


def click(x, y):
    # win32api.SetCursorPos((x,y))
    # time.sleep(.01)
    # win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
    # time.sleep(.01)
    # win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)
    pyautogui.click(x, y)

def transform_raw(file_path):
    ###how joint posotions are saved in file
    axes = ['x', 'y', 'z']
    columns = ['hip_center_' + i for i in axes] + \
              ['spine_' + i for i in axes] + \
              ['shoulder_center_' + i for i in axes] + \
              ['head_' + i for i in axes] + \
              ['left_shoulder_' + i for i in axes] + \
              ['left_elbow_' + i for i in axes] + \
              ['left_wrist_' + i for i in axes] + \
              ['left_hand_' + i for i in axes] + \
              ['right_shoulder' + i for i in axes] + \
              ['right_elbow_' + i for i in axes] + \
              ['right_wrist_' + i for i in axes] + \
              ['right_hand_' + i for i in axes] + \
              ['left_hip_' + i for i in axes] + \
              ['left_knee_' + i for i in axes] + \
              ['left_ankle_' + i for i in axes] + \
              ['left_foot_' + i for i in axes] + \
              ['right_hip_' + i for i in axes] + \
              ['right_knee_' + i for i in axes] + \
              ['right_ankle_' + i for i in axes] + \
              ['right_foot_' + i for i in axes]

    ####how we need them
    new_columns = ['head_' + i for i in axes] + \
                  ['shoulder_center_' + i for i in axes] + \
                  ['spine_' + i for i in axes] + \
                  ['hip_center_' + i for i in axes] + \
                  ['left_shoulder_' + i for i in axes] + \
                  ['left_elbow_' + i for i in axes] + \
                  ['left_wrist_' + i for i in axes] + \
                  ['left_hand_' + i for i in axes] + \
                  ['right_shoulder' + i for i in axes] + \
                  ['right_elbow_' + i for i in axes] + \
                  ['right_wrist_' + i for i in axes] + \
                  ['right_hand_' + i for i in axes] + \
                  ['left_hip_' + i for i in axes] + \
                  ['left_knee_' + i for i in axes] + \
                  ['left_ankle_' + i for i in axes] + \
                  ['left_foot_' + i for i in axes] + \
                  ['right_hip_' + i for i in axes] + \
                  ['right_knee_' + i for i in axes] + \
                  ['right_ankle_' + i for i in axes] + \
                  ['right_foot_' + i for i in axes]

    ####read from binnary file
    fid = np.fromfile(file_path, dtype=np.float32)
    jointNumber = 20
    tracks = 6

    ####reshape to present in "table" view
    Skeleton = np.reshape(fid, (-1, tracks, jointNumber, 4))
    #### we are working only with one user (client), here we are looking for him (kinect could write it in any cell)
    for i in range(len(Skeleton[0])):
        for j in range(tracks):
            if Skeleton[0][i][j][0] != 0:
                client = i
                break

    new_cord = []

    for i in range(len(Skeleton)):
        for j in range(jointNumber):
            new_cord.append(Skeleton[i][client][j][0])
            new_cord.append(Skeleton[i][client][j][1])
            new_cord.append(Skeleton[i][client][j][2])

    data = np.reshape(np.asarray(new_cord), (-1, 60))

    result = pd.DataFrame.from_records(data, columns=columns)
    #     print(result[:2])
    result = result[new_columns]
    #     print(result[:2])
    #     result.insert(0, "action", new_col)
    # print(result[-1:].values.tolist())
    print(result.shape)
    return result[-1:]

def transform_raw_every_5_motion(file_path):
    ###how joint posotions are saved in file
    axes = ['x', 'y', 'z']
    columns = ['hip_center_' + i for i in axes] + \
              ['spine_' + i for i in axes] + \
              ['shoulder_center_' + i for i in axes] + \
              ['head_' + i for i in axes] + \
              ['left_shoulder_' + i for i in axes] + \
              ['left_elbow_' + i for i in axes] + \
              ['left_wrist_' + i for i in axes] + \
              ['left_hand_' + i for i in axes] + \
              ['right_shoulder' + i for i in axes] + \
              ['right_elbow_' + i for i in axes] + \
              ['right_wrist_' + i for i in axes] + \
              ['right_hand_' + i for i in axes] + \
              ['left_hip_' + i for i in axes] + \
              ['left_knee_' + i for i in axes] + \
              ['left_ankle_' + i for i in axes] + \
              ['left_foot_' + i for i in axes] + \
              ['right_hip_' + i for i in axes] + \
              ['right_knee_' + i for i in axes] + \
              ['right_ankle_' + i for i in axes] + \
              ['right_foot_' + i for i in axes]

    ####how we need them
    new_columns = ['head_' + i for i in axes] + \
                  ['shoulder_center_' + i for i in axes] + \
                  ['spine_' + i for i in axes] + \
                  ['hip_center_' + i for i in axes] + \
                  ['left_shoulder_' + i for i in axes] + \
                  ['left_elbow_' + i for i in axes] + \
                  ['left_wrist_' + i for i in axes] + \
                  ['left_hand_' + i for i in axes] + \
                  ['right_shoulder' + i for i in axes] + \
                  ['right_elbow_' + i for i in axes] + \
                  ['right_wrist_' + i for i in axes] + \
                  ['right_hand_' + i for i in axes] + \
                  ['left_hip_' + i for i in axes] + \
                  ['left_knee_' + i for i in axes] + \
                  ['left_ankle_' + i for i in axes] + \
                  ['left_foot_' + i for i in axes] + \
                  ['right_hip_' + i for i in axes] + \
                  ['right_knee_' + i for i in axes] + \
                  ['right_ankle_' + i for i in axes] + \
                  ['right_foot_' + i for i in axes]

    ####read from binnary file
    fid = np.fromfile(file_path, dtype=np.float32)
    jointNumber = 20
    tracks = 6

    ####reshape to present in "table" view
    Skeleton = np.reshape(fid, (-1, tracks, jointNumber, 4))
    #### we are working only with one user (client), here we are looking for him (kinect could write it in any cell)
    for i in range(len(Skeleton[0])):
        for j in range(tracks):
            if Skeleton[0][i][j][0] != 0:
                client = i
                break

    new_cord = []

    for i in range(len(Skeleton)):
        for j in range(jointNumber):
            new_cord.append(Skeleton[i][client][j][0])
            new_cord.append(Skeleton[i][client][j][1])
            new_cord.append(Skeleton[i][client][j][2])

    data = np.reshape(np.asarray(new_cord), (-1, 60))

    result = pd.DataFrame.from_records(data, columns=columns)
    result = result[new_columns]
    return result[::30]


def angleRShoulderPitch(x2, y2, z2, x1, y1, z1): #calulates the Shoulderpitch value for the Right shoulder by using geometry
    if(y2<y1):
        angle = math.atan(abs(y2 - y1) / abs(z2 - z1)) 
        angle = math.degrees(angle)
        angle = -(angle)
        if(angle<-118):
            angle = -117
        return angle
    else:
        angle = math.atan((z2-z1)/(y2-y1))
        angle = math.degrees(angle)
        angle = 90-angle
        return angle

def angleRShoulderRoll(x2, y2, z2, x1, y1, z1): #calulates the ShoulderRoll value for the Right shoulder by using geometry
    if(z2<z1):
        test = z2
        anderetest = z1
        z2=anderetest
        z1=test
    if (z2 - z1 < 0.1):
        z2 = 1.0
        z1 = 0.8
    angle = math.atan((x2 - x1) / (z2 - z1))
    angle = math.degrees(angle)
    return angle

def angleLShoulderPitch(x2, y2, z2, x1, y1, z1): #calulates the Shoulderpitch value for the Left shoulder by using geometry
    if (y2 < y1):
        angle = math.atan(abs(y2 - y1) / abs(z2 - z1))
        angle = math.degrees(angle)
        angle = -(angle)
        if (angle < -118):
            angle = -117
        return angle
    else:
        angle = math.atan((z2 - z1) / (y2 - y1))
        angle = math.degrees(angle)
        angle = 90 - angle
        return angle

def angleLShouderRoll(x2, y2, z2, x1, y1, z1): #calulates the ShoulderRoll value for the Left shoulder by using geometry
    if (z2 < z1):
        test = z2
        anderetest = z1
        z2 = anderetest
        z1 = test
    if(z2-z1< 0.1):
        z2=1.0
        z1=0.8
    angle = math.atan((x2-x1)/(z2-z1))
    angle = math.degrees(angle)
    return angle

def angleRElbowYaw(x2, y2, z2, x1, y1, z1,shoulderpitch): #calulates the ElbowYaw value for the Right elbow by using geometry
    if(abs(y2-y1)<0.2 and abs(z2-z1) < 0.2 and (x1<x2) ):
        return 0
    elif(abs(x2-x1)<0.1 and abs(z2-z1)<0.1 and (y1>y2)):
        return 90
    elif(abs(x2-x1)<0.1 and abs(z2-z1)<0.1 and (shoulderpitch > 50)):
        return 90
    elif(abs(y2-y1)<0.1 and abs(z2-z1)<0.1 and (shoulderpitch < 50)):
        return 0
    elif(abs(x2-x1)<0.1 and abs(y2-y1)<0.1 and (shoulderpitch > 50)):
        return 90
    else:
        angle = math.atan((z2 - z1) / (y2 - y1))
        angle = math.degrees(angle)
        angle = - angle + (shoulderpitch)
        angle = - angle
        return angle


def angleRElbowRoll(x3, y3, z3, x2, y2, z2, x1, y1, z1): #calulates the ElbowRoll value for the Right elbow by using geometry
    a1=(x3-x2)**2+(y3-y2)**2 + (z3-z2)**2 
    lineA= a1 ** 0.5                        # calculates length of line between 2 3D coordinates
    b1=(x2-x1)**2+(y2-y1)**2 + (z2-z1)**2
    lineB= b1 ** 0.5                        # calculates length of line between 2 3D coordinates
    c1=(x1-x3)**2+(y1-y3)**2 + (z1-z3)**2
    lineC= c1 ** 0.5                        # calculates length of line between 2 3D coordinates

    cosB = (pow(lineA, 2) + pow(lineB,2) - pow(lineC,2))/(2*lineA*lineB)
    acosB = math.acos(cosB)
    angle = math.degrees(acosB)
    angle = 180 - angle
    return angle


def angleLElbowYaw(x2, y2, z2, x1, y1, z1, shoulderpitch): #calulates the ElbowYaw value for the Left elbow by using geometry
    if(abs(y2-y1)<0.2 and abs(z2-z1) < 0.2 and (x1>x2) ):
        return 0
    elif(abs(x2-x1)<0.1 and abs(z2-z1)<0.1 and (y1>y2)):
        return -90
    elif(abs(x2-x1)<0.1 and abs(z2-z1)<0.1 and (shoulderpitch > 50)):
        return -90
    elif(abs(y2-y1)<0.1 and abs(z2-z1)<0.1 and (shoulderpitch > 50)):
        return 0
    elif(abs(x2-x1)<0.1 and abs(y2-y1)<0.1 and (shoulderpitch > 50)):
        return -90
    else:
        angle = math.atan((z2 - z1) / (y2 - y1))
        angle = math.degrees(angle)
        angle = - angle + (shoulderpitch)
        angle = - angle
        return angle

def angleLElbowRoll(x3, y3, z3, x2, y2, z2, x1, y1, z1): #calulates the ElbowRoll value for the Left elbow by using geometry

    a1=(x3-x2)**2+(y3-y2)**2 + (z3-z2)**2
    lineA= a1 ** 0.5                        # calculates length of line between 2 3D coordinates
    b1=(x2-x1)**2+(y2-y1)**2 + (z2-z1)**2
    lineB= b1 ** 0.5                        # calculates length of line between 2 3D coordinates
    c1=(x1-x3)**2+(y1-y3)**2 + (z1-z3)**2
    lineC= c1 ** 0.5                        # calculates length of line between 2 3D coordinates

    cosB = (pow(lineA, 2) + pow(lineB,2) - pow(lineC,2))/(2*lineA*lineB)
    acosB = math.acos(cosB)
    angle = math.degrees(acosB)
    angle = -180+ angle
    return angle

def on_message(RobotIP, RobotPort): # Checks the mqtt message it receives and processes the json
    path = 'C:/KinectData/test5/Skel/Joint_Position.binary'
    result_all = transform_raw_every_5_motion(file_path=path)
    for i, result_of_row in result_all.iterrows():
        time.sleep(1)
        result = []
        for column in result_all.columns:
            result.append(result_of_row[column])
        shoulderLeft = [result[12], result[13], result[14]]
        elbowLeft = [result[15], result[16], result[17]]
        wristLeft = [result[18], result[19], result[20]]
        shoulderRight = [result[24], result[25], result[26]]
        elbowRight = [result[27], result[28], result[29]]
        wristRight = [result[30], result[31], result[32]]
        listAngles.append(
            angleRShoulderPitch(shoulderRight[0], shoulderRight[1], shoulderRight[2], elbowRight[0], elbowRight[1],
                                elbowRight[2]))
        listAngles.append(
            angleRShoulderRoll(shoulderRight[0], shoulderRight[1], shoulderRight[2], elbowRight[0], elbowRight[1],
                            elbowRight[2]))
        listAngles.append(
            angleRElbowRoll(shoulderRight[0], shoulderRight[1], shoulderRight[2], elbowRight[0], elbowRight[1],
                            elbowRight[2], wristRight[0], wristRight[1], wristRight[2]))
        listAngles.append(
            angleRElbowYaw(elbowRight[0], elbowRight[1], elbowRight[2], wristRight[0], wristRight[1],
                        wristRight[2], angleRShoulderPitch(shoulderRight[0], shoulderRight[1], shoulderRight[2], elbowRight[0], elbowRight[1],
                                elbowRight[2])))
        listAngles.append(
            angleLShoulderPitch(shoulderLeft[0], shoulderLeft[1], shoulderLeft[2], elbowLeft[0], elbowLeft[1],
                                elbowLeft[2]))
        listAngles.append(
            angleLShouderRoll(shoulderLeft[0], shoulderLeft[1], shoulderLeft[2], elbowLeft[0], elbowLeft[1],
                            elbowLeft[2]))
        listAngles.append(
            angleLElbowRoll(shoulderLeft[0], shoulderLeft[1], shoulderLeft[2], elbowLeft[0], elbowLeft[1],
                            elbowLeft[2], wristLeft[0], wristLeft[1], wristLeft[2]))
        listAngles.append(
            angleLElbowYaw(elbowLeft[0], elbowLeft[1], elbowLeft[2], wristLeft[0], wristLeft[1],
                        wristLeft[2], angleLShoulderPitch(shoulderLeft[0], shoulderLeft[1], shoulderLeft[2], elbowLeft[0], elbowLeft[1],
                                elbowLeft[2])))

        angleLists = [(listAngles[len(listAngles) - 8]), # all the coordinates are saved in one big list
                    (listAngles[len(listAngles) - 7]),  # and in a specific order (see list of joints)
                    (listAngles[len(listAngles) - 6]),  # this gets them out of that list and sent to the right joint
                    (listAngles[len(listAngles) - 5]),
                    (listAngles[len(listAngles) - 4]),
                    (listAngles[len(listAngles) - 3]),
                    (listAngles[len(listAngles) - 2]),
                    (listAngles[len(listAngles) - 1])]
        angleLists_rounded = [round (elem, 2) for elem in angleLists]
        url_parameteres = ''

        i = 0
        for r_l in ['r_', 'l_']:
            for part in ['shoulderpitch', 'shoulderroll', 'elbowroll', 'elbowyaw']:
                    url_parameteres = url_parameteres + '&' + r_l + part + '=' + str(angleLists_rounded[i])
                    i += 1

        # names = ["RShoulderPitch", "RShoulderRoll", "RElbowRoll", "RElbowYaw",  "LShoulderPitch", "LShoulderRoll", "LElbowRoll", "LElbowYaw"]

        sendrobot(url_parameteres, RobotIP, RobotPort) # takes userinput
        print(url_parameteres)

def sendrobot(angles, robot_ip, port):
    action = "/?action=/raise_hands" + angles
    r = requests.get('http://' + robot_ip + ':' + port + action)
    # print(action)

def main(robotIp, port):
    # while True:
        # click(cursor[0][0], cursor[0][1])
        # time.sleep(1)
        # click(cursor[1][0], cursor[1][1])
        # time.sleep(1)
        on_message(robotIp, port)


if __name__ == '__main__':
    main(RobotIP, '9562')

















    #
    #
    # def sendrobot(anglelist, robotIP="172.30.248.120", PORT=9559):
    #     try:
    #         try:
    #             motionProxy = ALProxy("ALMotion", robotIP, PORT)  # creates proxy to call specific functions
    #         except Exception, e:
    #             print "Could not create proxy to AlMotion"
    #             print "Error was: ", e
    #         try:
    #             postureProxy = ALProxy("ALRobotPosture", robotIP, PORT)  # creates proxy to call specific functions
    #         except Exception, e:
    #             print "Could not create proxy to ALRobotPosture"
    #             print "Error was: ", e
    #
    #         global t  # uses global variable t
    #
    #         if (t == 0):  # if it is the first time the robot is called upon
    #             motionProxy.setStiffnesses("Body", 0.0)  # unstiffens the joints
    #             postureProxy.goToPosture("StandInit", 0.5)  # gets the robot into his initial standing position
    #
    #         names = ["RShoulderPitch", "RShoulderRoll", "RElbowRoll", "RElbowYaw", "LShoulderPitch", "LShoulderRoll",
    #                  "LElbowRoll", "LElbowYaw"]
    #         # list of joints that will get changed
    #
    #         angleLists = [[math.radians(anglelist[len(anglelist) - 8])],
    #                       # all the coordinates are saved in one big list
    #                       [math.radians(anglelist[len(anglelist) - 7])],  # and in a specific order (see list of joints)
    #                       [math.radians(anglelist[len(anglelist) - 6])],
    #                       # this gets them out of that list and sent to the right joint
    #                       [math.radians(anglelist[len(anglelist) - 5])],
    #                       [math.radians(anglelist[len(anglelist) - 4])],
    #                       [math.radians(anglelist[len(anglelist) - 3])],
    #                       [math.radians(anglelist[len(anglelist) - 2])],
    #                       [math.radians(anglelist[len(anglelist) - 1])]]
    #         timeLists = [[0.4], [0.4], [0.4], [0.4], [0.4], [0.4], [0.4], [
    #             0.4]]  # sets the time the robot has to get to the joint location (when you give more than one coordinate for a joint, you have to give more than one timestamp for that same joint!)
    #         isAbsolute = True  # kindoff is deprecated, but makes the joint positions absolute and not relative
    #         motionProxy.angleInterpolation(names, angleLists, timeLists,
    #                                        isAbsolute)  # the function talks with the robot
    #         t += 1  # global t gets added by 1 so the joints dont get unstiffened again and the robot does not get put in its initial position
    #     except Exception:  # checks for any and all errors
    #         pass  # ignores every single one of them, except keyboardInterupt and SystemExit
    #     except (KeyboardInterrupt, SystemExit):  # when the program gets terminated
    #         postureProxy.goToPosture("StandInit", 0.5)  # set the robot in its initial position
    #         motionProxy.setStiffnesses("Body", 1.0)  # stiffen the joints
    #         raise  # actually quit
