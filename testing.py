import json
import time
from itertools import count

import numpy as np
import serial
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

from self_balancer_env import SelfBalancerEnv

env = SelfBalancerEnv()

actor = keras.models.load_model('saved_models/actor-continuous-2022-03-07 18:44:11.742029')

num_actions = env.action_space.shape[0]
tq = tqdm(count())
observation = env.reset()

render = False

plc = serial.Serial(port='/dev/cu.usbmodem1101', baudrate=115200, timeout=.1)


def read_noise_from_serial():
    noise = plc.readline()
    try:
        if len(noise) > 0:
            noise = np.array(json.loads(noise.decode('utf-8')))
            noise = np.clip(noise, -90, 90) * np.pi / 180
            return noise
    except Exception as e:
        print(e)

    return 0


def send_target_rotation_to_serial(target_rotation):
    target_angle = target_rotation.flatten().copy()
    target_angle *= 180 / np.pi
    target_angle = target_angle.astype(int).astype(str)
    writable = ",".join(target_angle) + '\n'
    plc.write(writable.encode('utf-8'))


for i in tq:
    s = time.time()

    if render:
        env.render()

    # read human noise from the serial port
    noise = read_noise_from_serial()
    # noise = env.noise()
    # noise = 0

    env.set_noise(noise)

    logits = actor(observation.reshape(1, -1))

    mu, sigma = logits[0, :num_actions], logits[0, num_actions:]

    action = tf.random.normal((1, num_actions), mu, sigma)

    _action = action[0:, ]

    target_rotation = np.where((_action >= -np.pi / 2) & (_action <= np.pi / 2), _action, 0) - noise

    observation, reward, done, info = env.step(target_rotation)

    send_target_rotation_to_serial(target_rotation)

    tq.set_description(f'Reward:{reward}')
