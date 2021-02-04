import numpy as np
import os
import ep_utils as utils

os.chdir(os.path.split(os.path.realpath(__file__))[0])
time_merge = list()

path = './data/S1'
start_time_list1=np.array([7,9,11,13,15,17,19,21,23,1,3,5])*3600+245
count=0
for f in os.listdir(path):
    if f!='recording_start_times.txt':
        sigs_dict, times_dict, fs, startT = utils.load_npz_data(os.path.join(path, f))
        for k, v in times_dict.items():
            time_merge.append(start_time_list1[count] + v[:,0])
        count += 1

path = './data/S2'
start_time_list2=np.array([4,6,8,10,12,14,16,18,20,22,0,2])*3600+2428
count=0
for f in os.listdir(path):
    if f!='recording_start_times.txt':
        sigs_dict, times_dict, fs, startT = utils.load_npz_data(os.path.join(path, f))
        for k,v in times_dict.items():
            time_merge.append(start_time_list2[count] + v[:,0])
        count += 1

time_merge = np.concatenate(time_merge, axis=0)
print(time_merge.shape)

np.save('./data/times.npy', time_merge)