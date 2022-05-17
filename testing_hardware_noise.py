import json
import time
from itertools import count

import numpy as np
import serial
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

from self_balancer_env import SelfBalancerEnv


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


def send_action_rotation_to_serial(action):
    action = action.flatten().copy()
    action *= 180 / np.pi
    action = action.astype(int).astype(str)
    writable = ",".join(action) + '\n'
    plc.write(writable.encode('utf-8'))


def sample_action(observation, action_dimension):
    logits = actor(observation)

    mu, sigma = logits[0, :action_dimension], logits[0, action_dimension:]

    action = tf.random.normal((1, action_dimension), mu, sigma)

    return logits, action


def augmented_action(action, noise):
    return action - noise


env = SelfBalancerEnv()
actor = keras.models.load_model('saved_models/actor-continuous-2022-03-12 17:55:41.714827')

plc = serial.Serial(port='/dev/cu.usbmodem1101', baudrate=115200, timeout=.1)

action_dimension = env.action_space.shape[0]
tq = tqdm(count())
observation = env.reset()

render = False

for i in tq:
    s = time.time()

    if render:
        env.render()

    # read human noise from the serial port
    noise = read_noise_from_serial()

    env.set_noise(noise)

    observation = observation.reshape(1, -1)

    logits, action = sample_action(observation, action_dimension)

    aug_action = augmented_action(action, noise)

    observation, reward, done, info = env.step(tf.reshape(aug_action, -1))

    send_action_rotation_to_serial(aug_action.numpy())

    tq.set_description(f'Reward:{reward}')
