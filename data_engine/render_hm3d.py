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

import habitat_sim
from habitat.core.registry import registry
from habitat.core.simulator import AgentState, Simulator
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat_sim.utils.common import quat_to_angle_axis, quat_to_coeffs, quat_from_angle_axis, quat_from_coeffs
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal, ShortestPathPoint


import quaternion
import matplotlib.pyplot as plt

def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]

    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]

    agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def main(dataset):
    import argparse
    parser = argparse.ArgumentParser(description="Scene")
    parser.add_argument('-rt', '--root', default='val')
    parser.add_argument('-id', '--scene_id', default='00800TEEsavR23oF')
    parser.add_argument('--job', default=0, type=int)

    args = parser.parse_args()

    simulator = None
    
    bbox_dir = "./room_bboxes_revised_axis"
    scene_lis = os.listdir(bbox_dir)
    for scene in scene_lis:
        scene = scene.replace(".json", "")
        scene_mesh_dir = os.path.join('./{}'.format(args.root), scene, scene[6:]+'.basis'+'.glb')
        navmesh_file = os.path.join('./{}'.format(args.root), scene, scene[6:]+'.basis'+'.navmesh')
        bbox_file = os.path.join('./', 'room_bboxes_revised_axis', scene+".json")
        if not os.path.exists(navmesh_file):
            continue
        bboxes = json.load(open(bbox_file))

        for ky in bboxes.keys():
            ky2 = ky
            ky = ky.strip()
            box = bboxes[ky2]

            def dis(p1, p2):
                import math
                nor2 = (p1[0]-p2[0])*(p1[0]-p2[0])
                nor2 += (p1[1]-p2[1])*(p1[1]-p2[1])
                nor2 += (p1[2]-p2[2])*(p1[2]-p2[2])
                return math.sqrt(nor2)

            def inside_p(pt, leng):
                if pt[0] < box[0][0] or pt[0] > box[1][0]: return False
                if pt[1] < box[0][1] or pt[1] > box[1][1]: return False
                if pt[2] < box[0][2] or pt[2] > box[1][2]: return False
                if box[1][0] - pt[0] < leng or pt[0] - box[0][0] < leng: return False
                if box[1][2] - pt[2] < leng or pt[2] - box[0][2] < leng: return False
                return True

            sim_settings = {
                "scene": scene_mesh_dir,
                "default_agent": 0,
                "sensor_height": 1.5,
                "width": 512,
                "height": 512,
            }
            cfg = make_simple_cfg(sim_settings)
            simulator = habitat_sim.Simulator(cfg)

            pathfinder = simulator.pathfinder
            pathfinder.seed(13)
            pathfinder.load_nav_mesh(navmesh_file)

            agent = simulator.initialize_agent(sim_settings["default_agent"])
            agent_state = habitat_sim.AgentState()

            follower = habitat_sim.GreedyGeodesicFollower(pathfinder, agent, 
                forward_key="move_forward",
                left_key="turn_left",
                right_key="turn_right"
            )
            
            pts = []; las_pt = None; trys = 0
            while len(pts)<500:
                trys += 1
                if trys > 10000: break
                my_pt = pathfinder.get_random_navigable_point()
                if len(my_pt)==3 and inside_p(my_pt, 0.0):
                    if las_pt is None or dis(my_pt, las_pt) >= 0.5: 
                        pts.append(my_pt)
                        las_pt = my_pt
            if len(pts) == 0: continue
            agent_distance = 0
            num_acts = 0
            
            try:
                os.mkdir("./data/{}".format(scene+"_" + ky))
            except:
                pass

            for i, pt in enumerate(pts):
                if i == 0: continue
                agent_position = pts[i-1]
                angle = np.random.uniform(0, 360)
                updown_angle = 0

                agent_rotation = quat_to_coeffs(quat_from_angle_axis(np.deg2rad(angle), np.array([0, 1, 0]))*quat_from_angle_axis(np.deg2rad(updown_angle), np.array([1, 0, 0]))).tolist()

                agent_state.position = agent_position
                agent_state.rotation = agent_rotation

                path = habitat_sim.ShortestPath()
                path.requested_start = pts[i-1]
                path.requested_end = pt
                if not pathfinder.find_path(path):
                    continue
                agent.set_state(agent_state)
                
                for rr in range(12):
                    pos = agent_state.position
                    angle = rr * 30
                    updown_angle = 0
                    rot = quat_to_coeffs(quat_from_angle_axis(np.deg2rad(angle), np.array([0, 1, 0]))*quat_from_angle_axis(np.deg2rad(updown_angle), np.array([1, 0, 0]))).tolist()
                    
                    agent_state.position = pos
                    agent_state.rotation = rot
                    agent.set_state(agent_state)

                    obs = simulator.get_sensor_observations()

                    sensor = agent.get_state().sensor_states['depth_sensor']
                    quaternion_0 = sensor.rotation
                    translation_0 = sensor.position
                    mat = np.eye(4)
                    import quaternion
                    mat[:3, :3] = quaternion.as_rotation_matrix(quaternion_0)
                    mat[:3,3] = translation_0
                    cam_jsn = json.dumps({"pose": mat.tolist(), "rotation": quaternion.as_float_array(quaternion_0).tolist(), "translation": translation_0.tolist()})
                    plt.imsave('./data/{}/'.format(scene+"_"+ky)+'/{}.png'.format(num_acts), obs['color_sensor'])
                    np.save('./data/{}/'.format(scene+"_"+ky)+'/{}_depth.npy'.format(num_acts), obs['depth_sensor'])
                    with open('./data/{}/'.format(scene+"_"+ky)+'/{}.json'.format(num_acts), 'w') as fo:
                        fo.write(cam_jsn)
                    num_acts += 1
                    if num_acts >= 1000: break

                try:
                    action_list = follower.find_path(pt)
                except habitat_sim.errors.GreedyFollowerError:
                    action_list = [None]

                las_pos = pts[i-1]

                for act in action_list:
                    if act is None: break
                    
                    obs = simulator.step(act)
                    agent_distance += np.linalg.norm(las_pos - agent.state.position)
                    las_pos = agent.state.position
                    
                    cur_depth = obs["depth_sensor"]
                    cur_depth = cur_depth[cur_depth>0.0000001]
                    
                    try:
                        cur_min = cur_depth.min()
                    except:
                        continue
                    cur_max = cur_depth.max()
                    
                    if cur_min < 0.2:
                        continue
                    if np.mean(cur_depth)<1.0:
                        continue
                    if cur_max<1.5:
                        continue
                    if cur_max>4.0:
                        continue
                    
                    sensor = agent.get_state().sensor_states['depth_sensor']
                    quaternion_0 = sensor.rotation
                    translation_0 = sensor.position
                    mat = np.eye(4)
                    import quaternion
                    mat[:3, :3] = quaternion.as_rotation_matrix(quaternion_0)
                    mat[:3,3] = translation_0
                    cam_jsn = json.dumps({"pose": mat.tolist(), "rotation": quaternion.as_float_array(quaternion_0).tolist(), "translation": translation_0.tolist()})
                    
                    plt.imsave('data/{}/'.format(scene+"_"+ky)+'/{}.png'.format(num_acts), obs['color_sensor'])
                    np.save('./data/{}/'.format(scene+"_"+ky)+'/{}_depth.npy'.format(num_acts), obs['depth_sensor'])
                    with open('data/{}/'.format(scene+"_"+ky)+'/{}.json'.format(num_acts), 'w') as fo:
                        fo.write(cam_jsn)
                    
                    num_acts += 1
                    if num_acts >= 1000: break
                if num_acts >= 1000: break


if __name__ == '__main__':
    main('hm3d')
