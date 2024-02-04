import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import SimpleExpSmoothing
from scipy.signal import find_peaks

# DC measurements
path_to_file = "./data/DC/data.npz"
data = np.load(path_to_file)
Vg1, Vg2, curr = data['x'], data['y'], data['z']
X,Y=np.meshgrid(Vg1,Vg2)
empty_mesh = np.zeros((len(X), len(Y)))
alpha = 0.25
prom = 0.00002
dist = None

#define functions
def exp_smoothing(time_series, alpha):
    smoother = SimpleExpSmoothing(time_series)
    smoothed_time_series = smoother.fit(smoothing_level=alpha, optimized=False).fittedvalues
    return smoothed_time_series

def mean_std(time_series, cut_off_val):
    cut_off = int(np.round(len(time_series)) * cut_off_val)
    mean_current = np.mean(time_series[:cut_off])
    std = np.std(time_series[:cut_off])
    return mean_current, std

def find_initial_signals(curr, Vg1, Vg2, alpha, prom, dist):
    peak_1_2_list= []
    for i in range(len(curr)): #use range(len(curr)) to iterate through every Vg1 and Vg2
        current_time_series = curr[i]
        Vg2_val = Vg2[i]
        #smooth signal
        smoothed_signal = exp_smoothing(current_time_series, alpha)
        mean_current, std = mean_std(smoothed_signal, 0.9)
        #find the peaks
        peaks, _ = find_peaks(smoothed_signal, height= mean_current, prominence=std, distance=dist)
        peak_1 = peaks[0]
        #peak_2 = peaks[1]

        peak_1_2_list.append(peak_1)
        #peak_1_2_list.append(peak_2)

    return np.array(peak_1_2_list)

#testing out the function



test = find_initial_signals(curr, Vg1, Vg2, alpha, prom, dist)
print(test)

plt.plot(test, 'o')
plt.title(f'First signal points, $\\alpha$ = {alpha}, prominence = {prom}')
#plt.savefig('./Figures/1_lines.png')
plt.show()