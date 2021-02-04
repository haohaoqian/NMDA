import numpy as np
import os
import ep_utils as utils

os.chdir(os.path.split(os.path.realpath(__file__))[0])
path = './data/S2'
data_merge = list()

for f in os.listdir(path):
    if f!='recording_start_times.txt':
        sigs_dict, times_dict, fs, startT = utils.load_npz_data(os.path.join(path, f))
        for k,v in sigs_dict.items():
            data_merge.append(v)

data_merge = np.concatenate(data_merge, axis=0)

print(data_merge.shape)

np.savez(os.path.join(path,'merge.npz'),merge_sig_dict=data_merge)