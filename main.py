from itertools import count

from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np


class SelfBalancerEnv:
    def __init__(self):
        model = load_model_from_path('selfblnc.xml')

        self.sim = MjSim(model)
        self.viewer = MjViewer(self.sim)

        sim_state = self.sim.get_state()
        self.sim.set_state(sim_state)

    def step(self, action):
        self.sim.data.ctrl[:] = action
        self.sim.data.qpos[:] = [0, 0]
        self.sim.step()

        # observation, reward, done, info
        return self._get_observation(), self._get_reward(), False, {}

    def _get_observation(self):
        # pos_x, pos_y, vel_x, vel_y
        ob = self.sim.get_state()
        ob = np.concatenate((ob.qpos, ob.qvel))
        return ob

    # how far from upright position
    def _get_reward(self):
        return 1 - sum(np.abs(self._get_observation()[:1]))

    def render(self):
        self.viewer.render()


env = SelfBalancerEnv()

motor_x = 0
motor_y = 0
for i in count():
    env.render()

    observation, reward, done, info = env.step([motor_x, motor_y])

    # PROBLEM
    # A simple PID controller didn't work due to inertial force in mujoco
    # BUT hardware has enough damping/stopper in servo motor that let state to hold at one position
    #
    # SOLUTION
    # 1. Model a servo motor in mujoco OR
    # 2. Change servo motor to freely rotate and use torque instead of angle inputs

    if observation[0] < 0:
        motor_x += 1
    else:
        motor_x += -1

    if observation[1] < 0:
        motor_y += 1
    else:
        motor_y += -1

    print(reward)
