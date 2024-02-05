import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import find_peaks

path_to_file = "./data/DC/data.npz"
data = np.load(path_to_file)
Vg1, Vg2, curr = data['x'], data['y'], data['z']
X,Y=np.meshgrid(Vg1,Vg2)

window_length = 12
poly = 2
dist = None


def mean_std(time_series, cut_off_val):
    """
    This function allows us to play around with the height threshold and prominence of the find peaks function
    :param time_series:
    :param cut_off_val:
    :return:
    """
    cut_off = int(np.round(len(time_series)) * cut_off_val)
    mean_current = np.mean(time_series[:cut_off])
    std = np.std(time_series[:cut_off])
    return mean_current, std

def find_initial_signals(curr, Vg1, Vg2, cutoff):
    peak_1_2_list = []
    for i in range(len(curr)): #use range(len(curr)) to iterate through every Vg1 and Vg2
        signal_i = curr[i]
        smoothed_signal = savgol_filter(signal_i, window_length, poly, mode="nearest")
        mean_current, std = mean_std(smoothed_signal, cutoff)
        peaks, _ = find_peaks(smoothed_signal, height=mean_current, prominence=std, distance=dist)
        peak_1 = peaks[0]

        #peak_2 = peaks[1]

        peak_1_2_list.append(peak_1)
        #peak_1_2_list.append(peak_2)
    return np.array(peak_1_2_list)


test = find_initial_signals(curr, Vg1, Vg2, 1)
print(test)

plt.plot(test, 'o')
plt.title(f'First signal points using Savitkzy-Golay Filter, window length = {window_length}, polynomial order = {poly}')
#plt.savefig('./Figures/1_lines.png')
plt.show()