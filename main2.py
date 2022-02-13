#!/usr/bin/env python3
"""
Shows how to toss a capsule to a container.
"""
import random

import gym.spaces
from gym.spaces import Box, Discrete
from mujoco_py import load_model_from_path, MjSim, MjViewer
import os
from itertools import count
import random

model = load_model_from_path('selfblnc2.xml')
sim = MjSim(model)

viewer = MjViewer(sim)

sim_state = sim.get_state()

motor_x = 0
motor_y = 0

sim.set_state(sim_state)
for i in count():
    # sim.data.ctrl[:] = [motor_x, motor_y]

    # motor_x += random.randint(-1, 1) if motor_x <= 90 else -1
    # motor_y += random.randint(-1, 1) if motor_y <= 90 else -1

    # sim.data.qpos[0] = 0
    # random.gauss(0, 45)
    #
    # sim.data.qpos[:2] = random_sample
    # sim.data.qpos[2] = 0
    # sim.data.qpos[3] = 0
    #
    # if i % 10 == 0:
    #     random_sample = hand_movement.sample()
    # print(motor_x, motor_y)

    sim.step()
    viewer.render()
