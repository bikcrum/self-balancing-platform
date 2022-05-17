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
