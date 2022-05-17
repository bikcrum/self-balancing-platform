from tqdm import tqdm
from itertools import count
import tensorflow as tf
from tensorflow import keras
import numpy as np
from self_balancer_env import SelfBalancerEnv
import time
import threading

env = SelfBalancerEnv()

interpreter = tf.lite.Interpreter(model_path="actor.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

observation = env.reset()
num_actions = 2
tq = tqdm(count())

for i in tq:
    s = time.time()

    env.render()

    input_data = np.array(observation.reshape(1, -1), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    logits = interpreter.get_tensor(output_details[0]['index'])

    mu, sigma = logits[0, :num_actions], logits[0, num_actions:]

    action = tf.random.normal((1, num_actions), mu, sigma)

    _action = action[0:, ]
    target_rotation = np.where((_action >= -np.pi / 2) & (_action <= np.pi / 2), _action, 0)

    observation, reward, done, info = env.step(target_rotation)

    tq.set_description(f'Reward:{reward}:Action:{_action}')
