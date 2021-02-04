import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.signal as signal

os.chdir(os.path.split(os.path.realpath(__file__))[0])
data = np.load('./data/S1/merge.npz', allow_pickle=True)
sigs = data['merge_sig_dict']
del data
fs=1000

test_segment = sigs[0]

plt.subplot(4, 4, 1)
plt.plot(np.arange(len(test_segment)) / fs, test_segment)
plt.xlabel('time/s')
plt.ylabel('amplitude')
plt.title('Spike from S1')

f, t, test_spec = signal.spectrogram(test_segment, 1000, nperseg=30, noverlap=20, nfft=128, mode='magnitude')
f_s=f[(f>0)&(f<300)]
test_spec_s = test_spec[(f > 0) & (f < 300)]
test_spec_s = test_spec_s / np.max(test_spec_s)
plt.subplot(4, 4, 5)
plt.pcolormesh(t, f_s, test_spec_s, cmap='jet', shading='auto')
plt.xlabel('time/s')
plt.ylabel('frequency/Hz')
plt.colorbar()

b,a=signal.butter(10,80,btype='highpass',fs=1000)
test_segment = signal.filtfilt(b, a, test_segment)

plt.subplot(4, 4, 9)
plt.plot(np.arange(len(test_segment)) / fs, test_segment)
plt.xlabel('time/s')
plt.ylabel('amplitude')

f, t, test_spec = signal.spectrogram(test_segment, 1000, nperseg=30, noverlap=20, nfft=128, mode='magnitude')
f_s=f[(f>0)&(f<300)]
test_spec_s = test_spec[(f > 0) & (f < 300)]
test_spec_s = test_spec_s / np.max(test_spec_s)
plt.subplot(4, 4, 13)
plt.pcolormesh(t, f_s, test_spec_s, cmap='jet', shading='auto')
plt.xlabel('time/s')
plt.ylabel('frequency/Hz')
plt.colorbar()



test_segment = sigs[5]

plt.subplot(4, 4, 3)
plt.plot(np.arange(len(test_segment)) / fs, test_segment)
plt.xlabel('time/s')
plt.ylabel('amplitude')
plt.title('HFO from S1')

f, t, test_spec = signal.spectrogram(test_segment, 1000, nperseg=30, noverlap=20, nfft=128, mode='magnitude')
f_s=f[(f>0)&(f<300)]
test_spec_s = test_spec[(f > 0) & (f < 300)]
test_spec_s = test_spec_s / np.max(test_spec_s)
plt.subplot(4, 4, 7)
plt.pcolormesh(t, f_s, test_spec_s, cmap='jet', shading='auto')
plt.xlabel('time/s')
plt.ylabel('frequency/Hz')
plt.colorbar()

b,a=signal.butter(10,80,btype='highpass',fs=1000)
test_segment = signal.filtfilt(b, a, test_segment)

plt.subplot(4, 4, 11)
plt.plot(np.arange(len(test_segment)) / fs, test_segment)
plt.xlabel('time/s')
plt.ylabel('amplitude')

f, t, test_spec = signal.spectrogram(test_segment, 1000, nperseg=30, noverlap=20, nfft=128, mode='magnitude')
f_s=f[(f>0)&(f<300)]
test_spec_s = test_spec[(f > 0) & (f < 300)]
test_spec_s = test_spec_s / np.max(test_spec_s)
plt.subplot(4, 4, 15)
plt.pcolormesh(t, f_s, test_spec_s, cmap='jet', shading='auto')
plt.xlabel('time/s')
plt.ylabel('frequency/Hz')
plt.colorbar()



data = np.load('./data/S2/merge.npz', allow_pickle=True)
sigs = data['merge_sig_dict']
del data
fs=1000

test_segment = sigs[-5]

plt.subplot(4, 4, 2)
plt.plot(np.arange(len(test_segment)) / fs, test_segment)
plt.xlabel('time/s')
plt.ylabel('amplitude')
plt.title('Spike from S2')

f, t, test_spec = signal.spectrogram(test_segment, 1000, nperseg=30, noverlap=20, nfft=128, mode='magnitude')
f_s=f[(f>0)&(f<300)]
test_spec_s = test_spec[(f > 0) & (f < 300)]
test_spec_s = test_spec_s / np.max(test_spec_s)
plt.subplot(4, 4, 6)
plt.pcolormesh(t, f_s, test_spec_s, cmap='jet', shading='auto')
plt.xlabel('time/s')
plt.ylabel('frequency/Hz')
plt.colorbar()

b,a=signal.butter(10,80,btype='highpass',fs=1000)
test_segment = signal.filtfilt(b, a, test_segment)

plt.subplot(4, 4, 10)
plt.plot(np.arange(len(test_segment)) / fs, test_segment)
plt.xlabel('time/s')
plt.ylabel('amplitude')

f, t, test_spec = signal.spectrogram(test_segment, 1000, nperseg=30, noverlap=20, nfft=128, mode='magnitude')
f_s=f[(f>0)&(f<300)]
test_spec_s = test_spec[(f > 0) & (f < 300)]
test_spec_s = test_spec_s / np.max(test_spec_s)
plt.subplot(4, 4, 14)
plt.pcolormesh(t, f_s, test_spec_s, cmap='jet', shading='auto')
plt.xlabel('time/s')
plt.ylabel('frequency/Hz')
plt.colorbar()



test_segment = sigs[-11]

plt.subplot(4, 4, 4)
plt.plot(np.arange(len(test_segment)) / fs, test_segment)
plt.xlabel('time/s')
plt.ylabel('amplitude')
plt.title('HFO from S2')

f, t, test_spec = signal.spectrogram(test_segment, 1000, nperseg=30, noverlap=20, nfft=128, mode='magnitude')
f_s=f[(f>0)&(f<300)]
test_spec_s = test_spec[(f > 0) & (f < 300)]
test_spec_s = test_spec_s / np.max(test_spec_s)
plt.subplot(4, 4, 8)
plt.pcolormesh(t, f_s, test_spec_s, cmap='jet', shading='auto')
plt.xlabel('time/s')
plt.ylabel('frequency/Hz')
plt.colorbar()

b,a=signal.butter(10,80,btype='highpass',fs=1000)
test_segment = signal.filtfilt(b, a, test_segment)

plt.subplot(4, 4, 12)
plt.plot(np.arange(len(test_segment)) / fs, test_segment)
plt.xlabel('time/s')
plt.ylabel('amplitude')

f, t, test_spec = signal.spectrogram(test_segment, 1000, nperseg=30, noverlap=20, nfft=128, mode='magnitude')
f_s=f[(f>0)&(f<300)]
test_spec_s = test_spec[(f > 0) & (f < 300)]
test_spec_s = test_spec_s / np.max(test_spec_s)
plt.subplot(4, 4, 16)
plt.pcolormesh(t, f_s, test_spec_s, cmap='jet', shading='auto')
plt.xlabel('time/s')
plt.ylabel('frequency/Hz')
plt.colorbar()
plt.show()