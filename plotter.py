import json

import numpy as np
from matplotlib import pyplot as plt

data = json.load(open('saved_reports/hand-balance-continuous-rewards-2022-03-14 18:14:20.048706.txt', 'r'))
data1 = json.load(open('saved_reports/hand-balance-continuous-rewards-2022-03-14 18:19:12.551098.txt', 'r'))
data = np.array(data)
data1 = np.array(data1)
plt.plot(data.mean(axis=1))
plt.plot(data1.mean(axis=1))
plt.title(
    f'Training episode vs rewards (convergence)\n{data1.shape[1]} steps/episode, Max reward=1/step')
plt.xlabel('Episode')
plt.ylabel('Episode rewards')
plt.legend(['Policy LR=0.0003, Value LR=0.001', 'Policy LR=0.0003, Value LR=0.01'])
plt.grid()
plt.show()

data = json.load(open('saved_reports/hand-balance-continuous-rewards-2022-03-14 14:37:01.655896.txt', 'r'))
data1 = json.load(open('saved_reports/hand-balance-continuous-rewards-2022-03-14 16:06:08.373991.txt', 'r'))
data = np.array(data)
data1 = np.array(data1)
plt.plot(data.mean(axis=1))
plt.plot(data1.mean(axis=1))
plt.title(
    f'Training episode vs rewards (convergence)\n{data1.shape[1]} steps/episode, Max reward=1/step')
plt.xlabel('Episode')
plt.ylabel('Episode rewards')
plt.legend(['Policy LR=0.03, Value LR=0.001', 'Policy LR=0.003, Value LR=0.01'])
plt.grid()
plt.show()

data = json.load(open('saved_reports/hand-balance-continuous-rewards-2022-03-12 17:55:41.714827.txt', 'r'))
data = np.array(data)
plt.plot(data.mean(axis=1))
plt.title(
    f'Training episode vs rewards (convergence)\n{data1.shape[1]} steps/episode, Max reward=1/step')
plt.xlabel('Episode')
plt.ylabel('Episode rewards')
plt.legend(['Policy LR=0.0003, Value LR=0.001'])
plt.grid()
plt.show()
