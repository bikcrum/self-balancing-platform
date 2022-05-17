import json

import numpy as np
from matplotlib import pyplot as plt

data = json.load(open('saved_reports/hand-balance-continuous-rewards-2022-03-12 17:55:41.714827.txt', 'r'))
data = np.array(data)
plt.plot(data.mean(axis=1))
plt.title(f'Training episode vs rewards\n{data.shape[1]} steps/episode, Max reward=1/step')
plt.xlabel('Episode')
plt.ylabel('Episode rewards')
plt.grid()
plt.show()
