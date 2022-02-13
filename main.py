import gym
# env = gym.make('CartPole-v1')
from self_balancer import SelfBalancer
from itertools import count

env = SelfBalancer()
env.reset()
for i in count():
    env.render()
    observation, reward, done, info = env.step([1 if i % 50 < 25 else -1, 0])
    # observation, reward, done, info = env.step(env.action_space.sample())
    action_space = env.action_space
    print(reward)
env.close()
