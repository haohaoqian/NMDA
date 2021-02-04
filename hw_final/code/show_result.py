import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import signal

os.chdir(os.path.split(os.path.realpath(__file__))[0])
data = np.load('./data/S1/merge.npz', allow_pickle=True)
sigs = data['merge_sig_dict']
del data

labels = np.load('./data/labels.npy')
class_dict=['Spike','HFO']

plt.figure()
for i in range(15):
    test_segment = sigs[i]
    b,a=signal.butter(10,80,btype='highpass',fs=1000)
    test_segment = signal.filtfilt(b, a, test_segment)

    f, t, test_spec = signal.spectrogram(test_segment, 1000, nperseg=30, noverlap=20, nfft=128, mode='magnitude')
    f_s=f[(f>0)&(f<500)]
    test_spec_s = test_spec[(f > 0) & (f < 500)]
    test_spec_s = test_spec_s/np.max(test_spec_s)
    plt.subplot(3, 5, i+1)
    plt.pcolormesh(t, f_s, test_spec_s, cmap='jet', shading='auto')
    plt.title(class_dict[labels[i]])
    plt.colorbar()
plt.show()