import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import sklearn.metrics as metrics
import time

os.chdir(os.path.split(os.path.realpath(__file__))[0])

start=time.time()
data_processed = np.load('./data/data_max.npy')
print('Raw data shape:', data_processed.shape)
data_io=time.time()

data_compressed = PCA(n_components=30).fit_transform(data_processed)
print('Compressed data shape:',data_compressed.shape)
compress = time.time()

kmeans_model = KMeans(n_clusters=2, random_state=0, max_iter=10000).fit(data_compressed)
cluster = time.time()

print('Score:', metrics.calinski_harabasz_score(data_processed, kmeans_model.labels_))
scoring = time.time()

print('data_io time:', data_io - start)
print('compress time:', compress - data_io)
print('cluster time:', cluster - compress)
print('scoring time:', scoring - cluster)

np.save('./data/labels.npy',kmeans_model.labels_)