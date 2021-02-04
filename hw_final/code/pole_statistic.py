import numpy as np
import matplotlib.pyplot as plt
import ep_utils as utils
import os

os.chdir(os.path.split(os.path.realpath(__file__))[0])
labels = np.load('./data/labels.npy')

path = './data/S2'
spike_count = dict()
HFO_count = dict()

pointer= 1330070
for f in os.listdir(path):
    if f!='recording_start_times.txt' and f!='merge.npz':
        sigs_dict, times_dict, fs, startT = utils.load_npz_data(os.path.join(path, f))
        for k, v in sigs_dict.items():
            length = v.shape[0]
            temp_label = labels[pointer:pointer + length]
            if k in spike_count:
                spike_count[k] += (length - np.sum(temp_label))
                HFO_count[k] += (np.sum(temp_label))
            else:
                spike_count[k] =(length - np.sum(temp_label))
                HFO_count[k] = (np.sum(temp_label))
            pointer += length

print(pointer)

names = list(spike_count.keys())
spike_values = list(spike_count.values())
HFO_values = list(HFO_count.values())

plt.figure()
plt.bar(range(len(names)), spike_values, label='Spike',)
plt.bar(range(len(names)), HFO_values, bottom=spike_values, label='HFO')
plt.xlabel('Pole')
plt.xticks(range(len(names)),names,rotation=45,fontsize=8)
plt.ylabel('Count')
plt.legend()
plt.show()