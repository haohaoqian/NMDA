import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir(os.path.split(os.path.realpath(__file__))[0])
labels = np.load('./data/labels.npy')
times = np.load('./data/times.npy')

times=np.mod(times,3600*24)

times_spike = times[labels == 0]/3600
times_HFO = times[labels == 1]/3600

plt.figure()
plt.hist(times_spike, bins=24*60)
plt.xlim(0, 24)
plt.xticks(range(25))
plt.xlabel('time')
plt.ylabel('Spike count')
plt.show()

plt.figure()
plt.hist(times_HFO, bins=24*60)
plt.xlim(0, 24)
plt.xticks(range(25))
plt.xlabel('time')
plt.ylabel('HFO count')
plt.show()