import numpy as np
import os
import matplotlib.pyplot as plt
import ep_utils as utils
import scipy.signal as signal

os.chdir(os.path.split(os.path.realpath(__file__))[0])
sigs_dict, times_dict, fs, startT = utils.load_npz_data('./data/S1/FA134AX4_cutSigs.npz')
signals_per2h_list=[]
for k,v in sigs_dict.items():
    signals_per2h_list.append(v)
signals_per2h=np.concatenate(signals_per2h_list,axis=0)
print('segments in 2h:', signals_per2h.shape)


plt.figure()
for i in range(10):
    test_segment = signals_per2h[i]
    b,a=signal.butter(10,80,btype='highpass',fs=1000)
    test_segment = signal.filtfilt(b, a, test_segment)

    f, t, test_spec = signal.spectrogram(test_segment, 1000, nperseg=30, noverlap=20, nfft=128, mode='magnitude')
    f_s=f[(f>0)&(f<500)]
    test_spec_s = test_spec[(f > 0) & (f < 500)]
    test_spec_s = test_spec_s/np.max(test_spec_s)
    plt.subplot(2, 5, i+1)
    plt.pcolormesh(t, f_s, test_spec_s, cmap='jet', shading='auto')
    plt.xlabel('time/s')
    plt.ylabel('frequency/Hz')
    plt.colorbar()
plt.show()