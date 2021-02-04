import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

os.chdir(os.path.split(os.path.realpath(__file__))[0])
data_3d = np.load('./data/data_3d.npy')
data_label = np.load('./data/labels.npy')

fig = plt.figure()
ax = Axes3D(fig)
for i in range(10000):
    co='red' if data_label[i] else 'blue'
    ax.scatter(data_3d[i, 0], data_3d[i, 1], data_3d[i, 2], marker='.', color=co)
    co='red' if data_label[-i] else 'blue'
    ax.scatter(data_3d[-i, 0], data_3d[-i, 1], data_3d[-i, 2], marker='.', color=co)

plt.show()