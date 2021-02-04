import numpy as np
from scipy import signal
from scipy import ndimage
import os
import time


def load_npz_data(file_path):
    ## load npz results
    #file_path: absolute path to npz file

    dets_data=np.load(file_path,allow_pickle=True)
    sigs_dict = dets_data['merge_sig_dict'][()]
    times_dict = dets_data['merge_time_dict'][()]
    startT = int(dets_data['start_time'][()])
    fs=dets_data['fs'][()]

    startT=time.localtime(startT)
    print('hour: %d'%startT.tm_hour)
    return sigs_dict,times_dict,fs,startT

def band_filt(data,fs,freqs):
    ## band filter
    #data: eeg data, nChannels X timePoints
    #fs: sample frequency
    #freqs: list, e.g. [80,250]

    nyq=fs/2
    b,a=signal.butter(5,[freqs[0]/nyq,freqs[1]/nyq],btype='bandpass')
    return signal.filtfilt(b,a,data,axis=-1)


def return_spectrogram(data,fs,tWin,norm=False):
    ## compute spectrogram(stft) for data
    #data: one channel signal
    #fs: sample frequency
    #tWin: windown time length for computing fft in spectrogram. default overlap ration is set as 0.8, nfft is set equal to nperseg
    #norm: frequency normalization, default False

    f,t,spex=signal.spectrogram(data,fs,nperseg=int(tWin*fs),noverlap=int(0.8*tWin*fs),nfft=int(tWin*fs),mode='magnitude')
    if norm:
        spex = (spex - np.mean(spex, axis=1, keepdims=True)) / np.std(spex, axis=1, keepdims=True)
    else:
        pass
    spex=ndimage.gaussian_filter(spex,sigma=1.5)

    return t,f,spex


