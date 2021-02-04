import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import time

os.chdir(os.path.split(os.path.realpath(__file__))[0])

start=time.time()
data_processed = np.load('./data/data_max.npy')
print('Raw data shape:', data_processed.shape)
data_io = time.time()

kmeans_model = KMeans(n_clusters=2, random_state=0, max_iter=10000).fit(data_processed)
cluster = time.time()

print('Score:', metrics.calinski_harabasz_score(data_processed, kmeans_model.labels_))
scoring = time.time()

print('data_io time:', data_io - start)
print('cluster time:', cluster - data_io)
print('scoring time:', scoring - cluster)
np.save('./data/labels.npy', kmeans_model.labels_)