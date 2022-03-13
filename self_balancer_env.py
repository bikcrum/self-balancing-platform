from gym import spaces
from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np


class SelfBalancerEnv:
    def __init__(self):
        # load model from xml file
        model = load_model_from_path('sbp_model.xml')

        # load simulator that compiles dynamics of model
        self.sim = MjSim(model)

        # load viewer for visualizing model dynamics in screen
        self.viewer = MjViewer(self.sim)

        sim_state = self.sim.get_state()
        self.sim.set_state(sim_state)

        # The topmost platform is observed as polar coordinates using two angles ranging between -90 to 90
        self.observation_space = spaces.Box(low=-np.pi / 2.0, high=np.pi / 2.0, shape=(2,))

        # The action space is target rotation of motors
        self.action_space = spaces.Box(low=-np.pi / 2.0, high=np.pi / 2.0, shape=(2,))

        # gear ratio means ratio of number of rotations of a driver gear to the number of rotations of a driven gear
        self.servo_gear_ratio = 20
        self.hand_gear_ratio = 50

    def step(self, action):
        # clip action between -90 and 90 degrees
        action = np.clip(action, -np.pi / 2, np.pi / 2)

        # set action as target motor rotation (first two actions are for motor and last two action are for hand)
        self.sim.data.ctrl[2:] = action * self.servo_gear_ratio
        # execute the action
        self.sim.step()

        # get new observation
        observation = self._get_observation()

        # get reward obtained from new observation by execution of an action
        reward = self._get_reward(observation)

        # when reward is below certain threshold we mark it as done
        done = reward < -0.5
        return observation, reward, done, {}

    def _get_observation(self):
        state = self.sim.get_state()

        # The net position of topmost platform is calculated using motor position and hand offset, so we sum

        net_position = state.qpos[:2] + state.qpos[2:]
        # net_velocity = state.qvel[:2] + state.qvel[2:]

        # Since we deal with only position so we only return them
        return net_position

    # reward calculates how far from balanced position
    def _get_reward(self, observation=None):
        return 1 - max(np.abs(self._get_observation())) if observation is None else 1 - max(np.abs(observation))

    # obtain random scalar angle between -90 and 90 (used for noise)
    @staticmethod
    def _random_angle_delta():
        return np.random.randint(-1, 2) * np.pi / 180.0

    # render current dynamics
    def render(self):
        self.viewer.render()

    # randomly generate hand position by moving in random direction, 1 degree at a time
    def noise(self):
        # set action as noise (first two actions are for motor and last two action are for hand)
        x, y = self._random_angle_delta(), self._random_angle_delta()
        self.sim.data.ctrl[:2] += np.array([x, y]) * self.hand_gear_ratio
        self.sim.step()

        return self.sim.data.ctrl[:2] / self.hand_gear_ratio

    # set noise from outside
    def set_noise(self, noise):
        self.sim.data.ctrl[:2] = np.array(noise) * self.hand_gear_ratio
        self.sim.step()

    # reset sets the environment to start_pos which by default is the perfect balanced position without noise
    def reset(self, start_pos=None):
        self.sim.reset()
        self.sim.data.qpos[:] = [0, 0, 0, 0] if start_pos is None else start_pos
        self.sim.data.qvel[:] = [0, 0, 0, 0]

        return self._get_observation()
