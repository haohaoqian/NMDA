import numpy as np
import os
import matplotlib.pyplot as plt

data = np.zeros((63, 24*60))
spike = np.zeros((63, 24*60))
HFO = np.zeros((63, 24*60))
count = np.zeros(24*60)
count_spike = np.zeros(24*60)
count_HFO = np.zeros(24*60)

os.chdir(os.path.split(os.path.realpath(__file__))[0])
specs = np.load('./data/data_max.npy')
times = np.load('./data/times.npy')
labels = np.load('./data/labels.npy')
times = np.mod(times, 3600 * 24)

for i in range(specs.shape[0]):
    interval = int(np.floor(times[i] / 60))
    count[interval] += 1
    data[:, interval] += specs[i]
    if labels[i] == 0:
        count_spike[interval] += 1
        spike[:,interval] += specs[i]
    else:
        count_HFO[interval] += 1
        HFO[:,interval] += specs[i]
    print(i)

data = data / count
data = data / np.max(data)

HFO = HFO / count_HFO
HFO = HFO / np.max(HFO)

spike = spike / count_spike
spike = spike / np.max(spike)

f=[7.8125,15.625,23.4375,31.25,39.0625,46.875,54.6875,62.5,70.3125,78.125,85.9375,93.75,101.5625,109.375,117.1875,125.,132.8125,140.625,148.4375,156.25,164.0625,171.875,179.6875,187.5,195.3125,203.125,210.9375,218.75,226.5625,234.375,242.1875,250.,257.8125,265.625,273.4375,281.25,289.0625,296.875,304.6875,312.5,320.3125,328.125,335.9375,343.75,351.5625,359.375,367.1875,375.,382.8125,390.625,398.4375,406.25,414.0625,421.875,429.6875,437.5,445.3125,453.125,460.9375,468.75,476.5625,484.375,492.1875]

plt.plot()
plt.pcolormesh(np.array(range(24*60))/60, f, data, cmap='jet', shading='auto')
plt.xlabel('Time')
plt.ylabel('Frequency/Hz')
plt.xticks(range(25))
plt.colorbar()
plt.show()

plt.plot()
plt.pcolormesh(np.array(range(24*60))/60, f, spike, cmap='jet', shading='auto')
plt.xlabel('Time')
plt.ylabel('Frequency/Hz')
plt.xticks(range(25))
plt.colorbar()
plt.show()

plt.plot()
plt.pcolormesh(np.array(range(24*60))/60, f, HFO, cmap='jet', shading='auto')
plt.xlabel('Time')
plt.ylabel('Frequency/Hz')
plt.xticks(range(25))
plt.colorbar()
plt.show()