import numpy as np
import os
import matplotlib.pyplot as plt
import ep_utils as utils
import scipy.signal as signal

os.chdir(os.path.split(os.path.realpath(__file__))[0])
plt.figure()

sigs_dict, times_dict, fs, startT = utils.load_npz_data('./data/S1/FA134AX4_cutSigs.npz')
signals_per2h_list=[]
for k,v in sigs_dict.items():
    signals_per2h_list.append(v)
signals_per2h=np.concatenate(signals_per2h_list,axis=0)
test_segment = signals_per2h[0]

plt.subplot(2, 4, 1)
plt.plot(np.arange(len(test_segment)) / fs, test_segment)
plt.xlabel('time/s')
plt.ylabel('amplitude')
plt.title('S1:Raw waveform')

f, t, test_spec = signal.spectrogram(test_segment, 1000, nperseg=30, noverlap=20, nfft=128, mode='magnitude')
f_s=f[(f>0)&(f<300)]
test_spec_s = test_spec[(f > 0) & (f < 300)]
plt.subplot(2, 4, 2)
plt.pcolormesh(t, f_s, test_spec_s, cmap='jet', shading='auto')
plt.xlabel('time/s')
plt.ylabel('frequency/Hz')
plt.title('S1:Raw spectrum')
plt.colorbar()

b,a=signal.butter(10,80,btype='highpass',fs=1000)
test_segment = signal.filtfilt(b, a, test_segment)

plt.subplot(2, 4, 3)
plt.plot(np.arange(len(test_segment)) / fs, test_segment)
plt.xlabel('time/s')
plt.ylabel('amplitude')
plt.title('S1:Bandpass waveform')

f, t, test_spec = signal.spectrogram(test_segment, 1000, nperseg=30, noverlap=20, nfft=128, mode='magnitude')
f_s=f[(f>0)&(f<300)]
test_spec_s = test_spec[(f > 0) & (f < 300)]
plt.subplot(2, 4, 4)
plt.pcolormesh(t, f_s, test_spec_s, cmap='jet', shading='auto')
plt.xlabel('time/s')
plt.ylabel('frequency/Hz')
plt.title('S1:Bandpass spectrum')
plt.colorbar()



sigs_dict, times_dict, fs, startT = utils.load_npz_data('./data/S2/FA1349ZH_cutSigs.npz')
signals_per2h_list=[]
for k,v in sigs_dict.items():
    signals_per2h_list.append(v)
signals_per2h=np.concatenate(signals_per2h_list,axis=0)
test_segment = signals_per2h[0]

plt.subplot(2, 4, 5)
plt.plot(np.arange(len(test_segment)) / fs, test_segment)
plt.xlabel('time/s')
plt.ylabel('amplitude')
plt.title('S2:Raw waveform')

f, t, test_spec = signal.spectrogram(test_segment, 1000, nperseg=30, noverlap=20, nfft=128, mode='magnitude')
f_s=f[(f>0)&(f<300)]
test_spec_s = test_spec[(f > 0) & (f < 300)]
plt.subplot(2, 4, 6)
plt.pcolormesh(t, f_s, test_spec_s, cmap='jet', shading='auto')
plt.xlabel('time/s')
plt.ylabel('frequency/Hz')
plt.title('S2:Raw spectrum')
plt.colorbar()

test_segment = signal.filtfilt(b, a, test_segment)

plt.subplot(2, 4, 7)
plt.plot(np.arange(len(test_segment)) / fs, test_segment)
plt.xlabel('time/s')
plt.ylabel('amplitude')
plt.title('S2:Bandpass waveform')

f, t, test_spec = signal.spectrogram(test_segment, 1000, nperseg=30, noverlap=20, nfft=128, mode='magnitude')
f_s=f[(f>0)&(f<300)]
test_spec_s = test_spec[(f > 0) & (f < 300)]
plt.subplot(2, 4, 8)
plt.pcolormesh(t, f_s, test_spec_s, cmap='jet', shading='auto')
plt.xlabel('time/s')
plt.ylabel('frequency/Hz')
plt.title('S2:Bandpass waveform')
plt.colorbar()

plt.show()