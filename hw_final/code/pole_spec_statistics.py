import numpy as np
import os
import ep_utils as utils
from scipy import signal
import matplotlib.pyplot as plt

os.chdir(os.path.split(os.path.realpath(__file__))[0])
path = './data/S2'
labels = np.load('./data/labels.npy')

spec = dict()
spec_count = dict()

HFO_spec = dict()
HFO_count = dict()

spike_spec = dict()
spike_count = dict()

pointer = 1330070

for f in os.listdir(path):
    if f!='recording_start_times.txt':
        sigs_dict, times_dict, fs, startT = utils.load_npz_data(os.path.join(path, f))
        for k,v in sigs_dict.items():
            if k in spec:
                for i in range(v.shape[0]):
                    test_segment = v[i]
                    b,a=signal.butter(10,80,btype='highpass',fs=1000)
                    test_segment = signal.filtfilt(b, a, test_segment)
                    f, t, test_spec = signal.spectrogram(test_segment, 1000, nperseg=30, noverlap=20, nfft=128, mode='magnitude')
                    test_spec_s = test_spec[(f > 0) & (f < 500)]
                    f_s = f[(f > 0) & (f < 500)]
                    spec[k] += test_spec_s
                    spec_count[k] += 1

                    if labels[pointer] == 1:
                        HFO_spec[k] += test_spec_s
                        HFO_count[k] += 1
                    else:
                        spike_spec[k] += test_spec_s
                        spike_count[k] += 1
                    pointer += 1

            else:
                for i in range(v.shape[0]):
                    test_segment = v[i]
                    b,a=signal.butter(10,80,btype='highpass',fs=1000)
                    test_segment = signal.filtfilt(b, a, test_segment)
                    f, t, test_spec = signal.spectrogram(test_segment, 1000, nperseg=30, noverlap=20, nfft=128, mode='magnitude')
                    test_spec_s = test_spec[(f > 0) & (f < 500)]
                    f_s = f[(f > 0) & (f < 500)]
                    spec[k] = test_spec_s
                    spec_count[k] = 1

                    if labels[pointer] == 1:
                        HFO_spec[k] = test_spec_s
                        HFO_count[k] = 1
                        spike_count[k] = 0
                        spike_spec[k] = np.zeros(test_spec_s.shape)
                    else:
                        spike_spec[k] = test_spec_s
                        spike_count[k] = 1
                        HFO_count[k] = 0
                        HFO_spec[k] = np.zeros(test_spec_s.shape)

                    pointer += 1

for key in spec.keys():
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.pcolormesh(t, f_s, spec[key] / spec_count[key], cmap='jet', shading='auto')
    plt.title('Average spectrum-pole ' + key)
    plt.xlabel('time/s')
    plt.ylabel('frequency/Hz')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.pcolormesh(t, f_s, spike_spec[key] / spike_count[key], cmap='jet', shading='auto')
    plt.title('Average spike spectrum-pole ' + key)
    plt.xlabel('time/s')
    plt.ylabel('frequency/Hz')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.pcolormesh(t, f_s, HFO_spec[key] / HFO_count[key], cmap='jet', shading='auto')
    plt.title('Average HFO spectrum-pole ' + key)
    plt.xlabel('time/s')
    plt.ylabel('frequency/Hz')
    plt.colorbar()

    plt.show()