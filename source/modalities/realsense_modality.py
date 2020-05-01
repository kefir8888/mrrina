from modalities.modality import GetPoints
from modalities.skeleton_modalities import  Skeleton_3D
from collections import deque
import numpy as np
import common
import torch
from test.draw import Plotter3d, draw_poses
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import io

import json
import os

import cv2

import pyrealsense2 as rs
import math

from modalities.modality import  Modality

class RealSense (GetPoints):
    def __init__ (self, video_path_ = "", model_path_ = "", mode_ = "GPU", base_height_ = 512, logger_ = 0, focal_length = 1.93):
        GetPoints.__init__(self, logger_, model_path_, mode_, base_height_, focal_length)
        self.skel_3d = Skeleton_3D(logger_ = self.logger)
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.align = rs.align(rs.stream.color)
        self.canvas_3d = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.plotter = Plotter3d(self.canvas_3d.shape[:2], origin=(0.5, 0.5), scale=0.1)

        if (video_path_ == ""):
            self.all_data = rs.pipeline()
            self.profile = self.all_data.start(self.config)

        self.read = False

    def name(self):
        return "RealSense"

    def _read_data (self):
        if (self.read == False):
            xy_cords = []
            frames = self.all_data.wait_for_frames()
            aligned_frames = self.align.process(frames)
            self.img = np.asanyarray(aligned_frames.get_color_frame().get_data())
            depth_img = np.asanyarray(aligned_frames.get_depth_frame().get_data())

            xy_cords, _, edges = self._infer_net(self.img)
            xy_cords_ = xy_cords.reshape(-1,2)

            coords_3D = np.zeros((xy_cords_.shape[0], 3))
            coords_3D[:,:2] = np.asarray(xy_cords_)
            coords_3D = coords_3D.astype(int)
            for i in range(xy_cords_.shape[0]):
                if coords_3D[i,1] >= 720:
                    coords_3D[i,2] = int(depth_img[719, coords_3D[i,0]])
                elif coords_3D[i,0] < 0:
                    coords_3D[i,2] = 0
                else:
                    coords_3D[i,2] = int(depth_img[coords_3D[i,1], coords_3D[i,0]])

            # print("RS", coords_3D)
            # self.plotter.plot(self.canvas_3d, coords_3D, edges)
            # cv2.imshow("kek", self.canvas_3d)

            self.read_data = coords_3D


    def _process_data(self):
        self.skel_3d.read_data = (self.read_data, "rs")
        self.skel_3d._process_data()
        self.processed_data = self.skel_3d.processed_data
        # print("Realsense", self.processed_data)

        # self.processed_data = self.skel_3d.processed_data


    def _interpret_data(self):
        self.interpreted_data = self.processed_data

    def _get_command(self):
        commands = []

        if (self.timeout.timeout_passed ()):
            for key in self.processed_data.keys():
                commands.append(("/set_joint_angle", [key, str(self.processed_data[key])]))
            print("COMM", commands)

        else:
            commands.append (("noaction", [""]))

        return commands

    def get_command(self, skip_reading_data=False):
        if (skip_reading_data == False):
            self._read_data()

        self._process_data()
        self._interpret_data()
        return self._get_command()

    def draw(self, img):
        return [self.canvas_3d, self.frame]
