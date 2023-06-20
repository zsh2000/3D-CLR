# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from turtle import forward
from typing import Any, List, Optional
from abc import ABC
import os
import json
import argparse
import logging
import pickle
from collections import defaultdict

import numpy as np
import copy
import cv2

from habitat.utils.geometry_utils import quaternion_to_list

import quaternion
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from data_utils.tools import Application

def robot2world(position, u, v , heading):
    x0, y0, z0 = position
    x1 = x0 + v * np.cos(heading + np.pi/2)
    z1 = -(-z0 + v * np.sin(heading + np.pi/2))
    x2 = x1 + u * np.cos(heading + np.pi/2 - np.pi/2)
    z2 = -(-z1 + u * np.sin(heading + np.pi/2 - np.pi/2))
    return [x2, y0, z2]


def transformation_quatrtnion2heading(rotation:quaternion):
    quat = quaternion_to_list(rotation)
    q = R.from_quat(quat)
    heading = q.as_rotvec()[1]
    return heading


def main(room_name):
    datadir_root_name = "/gpfs/u/home/LMCG/LMCGnngn/scratch-shared/masked/"
    data_dir = os.path.join(datadir_root_name, room_name)
    api = Application((512, 512), 90, 1, 0.005, 600, 1.5, 1, 2)
    from tqdm import tqdm
    for file in tqdm(os.listdir(data_dir)):
        if not "depth" in file: continue
        pose_file = json.load(open(os.path.join(data_dir, file.replace("_depth.npy", ".json"))))
  
        house_name = room_name.split('_')[0]
        ky = room_name.split('_')[1]

        bbox_dir = "room_bboxes_with_walls_revised_axis"
        bbox = json.load(open(os.path.join(bbox_dir, house_name + ".json")))[ky]

        
        min_x = bbox[0][0]
        min_y = bbox[0][1]
        min_z = bbox[0][2]
        max_x = bbox[1][0]
        max_y = bbox[1][1]
        max_z = bbox[1][2]

        rotation_0 = pose_file["rotation"][0]
        rotation_1 = pose_file["rotation"][1]
        rotation_2 = pose_file["rotation"][2]
        rotation_3 = pose_file["rotation"][3]
        position = pose_file["translation"]

        heading = transformation_quatrtnion2heading(np.quaternion(rotation_0, rotation_1, rotation_2, rotation_3))
        if heading > np.pi*2:
            heading -= np.pi*2
        elif heading < 0:
            heading += np.pi*2

        depth_map = np.load(os.path.join(data_dir, file))
        point_clouds_2current = api.transformation_camera2robotcamera(np.expand_dims(depth_map/10., axis=2))
        color_map = cv2.imread(os.path.join(data_dir, file.replace("_depth.npy", ".png")))
        color_map_out = copy.deepcopy(color_map)
        for w in range(point_clouds_2current.shape[0]):
            for h in range(point_clouds_2current.shape[1]):
                pc2r = [point_clouds_2current[w,h,j] for j in range(point_clouds_2current.shape[-1])]

                pc2w = robot2world(position, pc2r[0]*10, pc2r[1]*10, heading)
                pc2w[1] = pc2r[2]*10 + pc2w[1]

                if not ((min_x-0 < pc2w[0] < max_x+0) and (min_y-0 < pc2w[1] < max_y+0) and (min_z-0 < pc2w[2] < max_z+0)):
                    color_map_out[w, h] = np.array([0, 0, 0])

        cv2.imwrite("./masked/" + room_name + "/%s"%(file.replace("_depth.npy", ".png")), color_map_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Scene")
    parser.add_argument('--room_name', default="00234-nACV8wLu1u5_10", type=str)

    args = parser.parse_args()

    main(args.room_name)
