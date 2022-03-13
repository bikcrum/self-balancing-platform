import time
from itertools import count

import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

from self_balancer_env_v2 import SelfBalancerEnv_v2

env = SelfBalancerEnv_v2()
actor = keras.models.load_model('saved_models/actor-continuous-2022-03-13 01:11:16.241599')

action_dimension = env.action_space.shape[0]
tq = tqdm(count())
observation = env.reset()


def sample_action(observation, action_dimension):
    logits = actor(observation)
    mu, sigma = logits[0, :action_dimension], logits[0, action_dimension:]
    action = tf.random.normal((1, action_dimension), mu, sigma)
    return logits, action


render = True
for i in tq:
    s = time.time()

    if render:
        env.render()

    observation = observation.reshape(1, -1)

    logits, action = sample_action(observation, action_dimension)

    observation, reward, done, info = env.step(tf.reshape(action, -1))

    tq.set_description(f'Reward:{reward}')
