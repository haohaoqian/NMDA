import numpy as np
import os
from scipy import signal
from sklearn.cluster import KMeans

os.chdir(os.path.split(os.path.realpath(__file__))[0])

data1 = np.load('./data/S1/merge.npz', allow_pickle=True)
sigs1 = data1['merge_sig_dict']
del data1

data2 = np.load('./data/S2/merge.npz', allow_pickle=True)
sigs2 = data2['merge_sig_dict']
del data2

sigs = np.concatenate([sigs1, sigs2], axis=0)
del sigs1
del sigs2

data_resampled = signal.resample(sigs, 100, axis=1)
print(data_resampled.shape)

kmeans_model = KMeans(n_clusters=2, random_state=0, max_iter=10000).fit(data_resampled)
np.save('./data/labels.npy', kmeans_model.labels_)