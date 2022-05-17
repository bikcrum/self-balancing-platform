from matplotlib import pyplot as plt
import json
import numpy as np
import seaborn as sns

data = json.load(open('reports/hand-balance-continuous-rewards-1.txt', 'r'))
data = np.array(data)
data2 = json.load(open('reports/hand-balance-continuous-rewards-2.txt', 'r'))
data2 = np.array(data2)
data3 = json.load(open('reports/hand-balance-continuous-rewards-3.txt', 'r'))
data3 = np.array(data3)
plt.plot(data3.mean(axis=1))
plt.title('Training episode vs rewards\n4000 steps/episode, Max reward=1/step')
plt.xlabel('Episode')
plt.ylabel('Episode rewards')
plt.legend(['min','mean'])
plt.grid()
plt.show()