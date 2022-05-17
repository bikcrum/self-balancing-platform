from tqdm import tqdm
from itertools import count
import tensorflow as tf
from tensorflow import keras
import numpy as np
from self_balancer_env import SelfBalancerEnv

env = SelfBalancerEnv()

actor = keras.models.load_model('saved_models/actor')

observation = env.reset()

tq = tqdm(count())
for i in tq:
    env.render()

    # random action
    # observation, reward, done, info = env.step(env.action_space.sample())

    # ppo policy
    observation = observation.reshape(1, -1)
    logits = actor(observation)
    action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)

    act = action[0].numpy()
    x = act // 3
    y = act % 3
    x = (x - 1) * np.pi / 180.0
    y = (y - 1) * np.pi / 180.0

    observation, reward, done, info = env.step(np.array([x, y]))
