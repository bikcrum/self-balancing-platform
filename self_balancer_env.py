from gym import spaces
from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np


class MultiDiscrete(spaces.MultiDiscrete):
    def sample(self):
        return (super().sample() - 1) * np.pi / 180.0


class SelfBalancerEnv:
    def __init__(self):
        model = load_model_from_path('selfblnc.xml')

        self.sim = MjSim(model)
        self.viewer = MjViewer(self.sim)

        sim_state = self.sim.get_state()
        self.sim.set_state(sim_state)

        self.observation_space = spaces.Box(low=-np.pi / 2.0, high=np.pi / 2.0, shape=(2,))

        self.action_space = MultiDiscrete((3, 3))

    def step(self, action):
        # since gear=10 in mujoco, we mupliply radian angle by 10
        # move motor from current position
        self.sim.data.ctrl[:] += action * 10

        # self.sim.data.qpos[:] = [0, 0]

        self.sim.step()

        reward = self._get_reward()
        # observation, reward, done, info
        return self._get_observation(), reward, reward < 0, {}

    def _get_observation(self):
        # pos_x, pos_y, vel_x, vel_y
        ob = self.sim.get_state()
        # ob = np.concatenate((ob.qpos, ob.qvel))
        # only pos for now
        return ob.qpos

    # how far from upright position
    def _get_reward(self):
        return 1 - sum(np.abs(self._get_observation()))

    def render(self):
        self.viewer.render()

    def reset(self):
        self.sim.reset()
        self.sim.data.qpos[:] = [0, 0]
        self.sim.data.qvel[:] = [0, 0]
        return self._get_observation()
