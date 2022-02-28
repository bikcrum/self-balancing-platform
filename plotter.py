from matplotlib import pyplot as plt
import json
import numpy as np

data = json.load(open('reports/flat-balance-rewards.txt', 'r'))
data = np.array(data)
plt.plot(data.sum(axis=1))
plt.title('Training episode vs rewards\n4000 steps/episode, Max reward=1/step')
plt.xlabel('Episode')
plt.ylabel('Episode rewards')
plt.legend(['Reward convergence'])
plt.grid()
plt.show()
