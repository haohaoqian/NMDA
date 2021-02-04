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

print('Complete loading data')

data_max=list()
data_mean=list()
for i in range(sigs.shape[0]):
    test_segment = sigs[i]
    b,a=signal.butter(10,80,btype='highpass',fs=1000)
    test_segment = signal.filtfilt(b, a, test_segment)

    f, _, test_spec = signal.spectrogram(test_segment, 1000, nperseg=30, noverlap=20, nfft=128, mode='magnitude')
    test_spec_s = test_spec[(f > 0) & (f < 500)]
    test_spec_s = test_spec_s / np.max(test_spec_s)
    data_max.append(np.max(test_spec_s, axis=-1))
    data_mean.append(np.mean(test_spec_s, axis=-1))
    print(i)

data_max = np.array(data_max)
data_mean = np.array(data_mean)
del sigs
print(data_max.shape)
print(data_mean.shape)
np.save('./data/data_max.npy', data_max)
np.save('./data/data_mean.npy',data_mean)