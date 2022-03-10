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

        self.action_space = spaces.Box(low=-np.pi / 2.0, high=np.pi / 2.0, shape=(2,))

    def step(self, action):
        # since gear=10 in mujoco, we mupliply radian angle by 10
        # move motor from current position

        self.sim.data.ctrl[2:] = action * 20

        self.sim.step()

        ob = self._get_observation()
        reward = self._get_reward()
        # observation, reward, done, info

        # todo use velocity (ob[4:]) to improve wobbliness
        return ob[:2], reward, reward < -0.5, {}

    def _get_observation(self):
        # pos_x, pos_y, vel_x, vel_y
        state = self.sim.get_state()
        # state = np.concatenate((state.qpos, state.qvel))
        # only pos for now

        # compute position of upper platform given hand movement and motor position
        return np.concatenate((state.qpos[:2] + state.qpos[2:],
                               state.qvel[:2] + state.qvel[2:]))

    # how far from upright position
    def _get_reward(self):
        # return 1 - sum(np.abs(self._get_observation()))
        obs = self._get_observation()[:2]
        # delta = obs[:2] + obs[2:]
        return 1 - max(np.abs(obs))

    @staticmethod
    def _random_angle_delta():
        return np.random.randint(-1, 2) * np.pi / 180.0

    def render(self):
        self.viewer.render()

    def noise(self):
        # random hand controller with gear=50
        x, y = self._random_angle_delta(), self._random_angle_delta()
        self.sim.data.ctrl[:2] += np.array([x, y]) * 50
        self.sim.step()

        return self.sim.data.ctrl[:2] / 50.0

    def set_noise(self, noise):
        # random hand controller with gear=50
        self.sim.data.ctrl[:2] = np.array(noise) * 50
        self.sim.step()

    def reset(self, start_pos=None):
        self.sim.reset()
        self.sim.data.qpos[:] = [0, 0, 0, 0] if start_pos is None else start_pos
        self.sim.data.qvel[:] = [0, 0, 0, 0]

        return self._get_observation()[:2]
