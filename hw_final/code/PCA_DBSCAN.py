import numpy as np
import os
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import time

os.chdir(os.path.split(os.path.realpath(__file__))[0])

data_processed = np.load('./data/data_max.npy')
print('Raw data shape:', data_processed.shape)

start=time.time()
data_compressed = PCA(n_components=20).fit_transform(data_processed[:20000])
print('Compressed data shape:',data_compressed.shape)

compress=time.time()
dbscan = DBSCAN().fit(data_compressed)

cluster = time.time()
print('cluster time:', cluster - compress)
print('compress time:', compress - start)

np.save('./data/labels.npy',dbscan.labels_)