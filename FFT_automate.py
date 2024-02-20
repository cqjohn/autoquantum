import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy. fftpack import fft, ifft

#data and parameters
path_to_file = "./data/DC/data.npz"
data = np.load(path_to_file)
Vg1, Vg2, curr = data['x'], data['y'], data['z']
X,Y=np.meshgrid(Vg1,Vg2)


value = 0
value2 = 1
Vg2_value = Vg2[value]
signal = curr[value]
signal2 = curr[value2]
n = len(Vg1)

guess_a =  1e-5
guess_b = 1e-5
guess_c = 1e-5
p0 = [guess_a, guess_b, guess_c]

window_length = 20
poly = 2
threshold = 0.2e-8

def exp_fit(x, a, b, c):
    return (a*np.exp(b * x)) + c

def sig_sub_exp(signal, Vg1, p0):
    params = curve_fit(exp_fit, Vg1, signal, p0)

    signal_fit = exp_fit(Vg1, *params[0])

    sub_signal = signal - signal_fit

    return sub_signal


def sig_sub_FFT_noise(sub_signal, threshold, n):
    """
    Removes noise based on FFT
    :param sub_signal: signal with the exponential fit subtracted from it
    :param threshold: threshold for the FFT
    :param n:
    :return: signal with high freqency noise subtracted from it
    """
    freq_domain = fft(sub_signal)
    PSD = freq_domain * np.conj(freq_domain) / n
    index = PSD < threshold
    clean = PSD * index
    freq_domain = index * freq_domain
    inverse = ifft(freq_domain)

    sig_sub_noise = sub_signal - inverse

    return sig_sub_noise

def sig_sub_exp_FFT(signal, Vg1, p0, threshold, n):
    signal_sub_exp = sig_sub_exp(signal, Vg1, p0)
    signal_sub_FFT = sig_sub_FFT_noise(signal_sub_exp, threshold, n)

    return signal_sub_FFT


def mean_std(time_series, cut_off_val = 1):
    """
    This function allows us to play around with the height threshold and prominence of the find peaks function
    :param time_series:
    :param cut_off_val:
    :return: mean current, standard deviation
    """
    cut_off = int(np.round(len(time_series)) * cut_off_val)
    mean_current = np.mean(time_series[:cut_off])
    std = np.std(time_series[:cut_off])
    return mean_current, std


def smooth_peaks(sig_sub_noise, window_length = 20, poly=2):
    smooth_signal = savgol_filter(sig_sub_noise, window_length, poly, mode="nearest")

    h, prom = mean_std(smooth_signal, 1)  # prominence
    dist = None
    peaks, _ = find_peaks(smooth_signal, height=h, prominence=prom, distance=dist)
    #peaks returns the indices of the peaks in the original signal

    return smooth_signal, peaks


def find_plot_signals_CSD(current, Vg1, Vg2, X, Y, p0, threshold, n):

    #make empty array

    arr = np.zeros((len(X), len(Y)))

    for i in range(len(current)):
        signal_i = current[i]
        signal_i_filt = sig_sub_exp_FFT(signal_i, Vg2, p0, threshold, n)

        #find peaks
        smooth_signal_i, peaks_i =  smooth_peaks(signal_i_filt, window_length, poly)

        #make binary meshgrid
        for peak in peaks_i:
            arr[i][peak] = 1

    contourplot = plt.contour(Vg1, Vg2, arr, levels=20, cmap=plt.cm.Greys, origin='lower')
    plt.xlabel('Vg1')
    plt.ylabel('Vg2')
    #cbar = plt.colorbar(contourplot)
    plt.title(f'Charge Stability Diagram, FFT_Threshold = {threshold} ')
    plt.show()

    return arr


def Lorentzian_fit(x, amplitude, centre, width):
    return (amplitude*width**2 / ((x - centre)**2 + width**2))


def find_final_peak_Lorentz(signal, Vg2, p0, threshold, n):
    """
    This function finds the final peak in the signal and fits a Lorentzian to it, returning the fitted amp, centre and width
    :param signal: raw, noisy, unprocessed signal
    :param Vg2:
    :param p0: initial conditions for the exponential fit
    :param threshold: threshold for the FFT
    :param n: parameter for the FFT
    :return: optimised amplitude, centre, and width of the Lorentzian fitted to the final peak
    """

    processed_sig = sig_sub_exp_FFT(signal, Vg2, p0, threshold, n)

    smooth_sig, peaks = smooth_peaks(processed_sig)


    last_peak = peaks[-1]  #position of the centre of the last peak
    last_peak_val = Vg1[last_peak]  #Vg1 value of peak, use this as an estimate of the centre

    #Lorentzian Fit
    #--------Parameters------------
    amp = 1 #got this number by looking at a peak- how can i automate this?
    centre = last_peak_val
    width = 1e-5
    p01 = [amp, centre, width]
    #-----------------------------
    popt, pcov = curve_fit(Lorentzian_fit, Vg1, smooth_sig, p01)
    smooth_signal_fit = Lorentzian_fit(Vg1, *popt)

    errs = np.sqrt(np.diag(pcov))
    amp_fit = popt[0]
    centre_fit = popt[1]
    width_fit = popt[2]


    return amp_fit, centre_fit, width_fit


"""
plt.plot(Vg1[last_peak], smooth_sig[last_peak], 'o', label='Peaks', markersize=8, color='red')
plt.plot(Vg1, smooth_sig)
plt.plot(Vg1, smooth_signal_fit)
plt.xlabel(f'$Vg_1$')
plt.ylabel('Current (A)')
plt.title(f'Lorentzian fit to final peak for $Vg_2 = {Vg2_value}$')
plt.show()
"""

def likliness_two_peaks(signal1, signal2): #these signals are raw signals
    #find final peaks and parameters of the Lorentzian fits for both signals
    """
    This function finds the final peaks of two signals and fits a Lorentzian to this peak.
    It then checks if these 'final' peaks are the same by comparing their optimised fit parameters
    :param signal1: Raw, noisy, unprocessed signal 1
    :param signal2: Raw, noisy, unprocessed signal 2
    :return: verdict: Boolean, True if the final peaks detected in both signals are the same and False other wise
                        centre1: centre of final peak of signal 1
                        centre2: centre of final peak of signal 2

    """
    amp1, centre1, width1 = find_final_peak_Lorentz(signal1, Vg1, p0, threshold, n)
    amp2, centre2, width2 = find_final_peak_Lorentz(signal2, Vg1, p0, threshold, n)

    #verifying the difference

    #these are the threshold differences that we allow- these were found by looking at the difference between the two final peaks of the 1st and 2nd signal
    #can play around with these values
    amp_thres = 2e-5 * 4
    centre_thres = 0.02 * 4
    width_thres = 1e-3 * 4

    amp_diff = abs(amp2 - amp1)
    centre_diff = abs(centre2 - centre1)
    width_diff = abs(width2 - width1)

    verdict = True

    if amp_diff > amp_thres:
        verdict = False

    elif centre_diff > centre_thres:
        verdict = False

    elif width_diff > width_thres:
        verdict = False

    return verdict, centre1, centre2


def dist_final_peaks(signal1, signal2):
    """
    This function calculates the difference between two final peaks that have been Lorentzian fitted
    :param signal1:
    :param signal2:
    :return: distance
    """

    verdict, centre1, centre2 = likliness_two_peaks(signal1, signal2)

    final_peak_dist = abs(centre2 - centre1)

    return final_peak_dist


def calc_mean_shift(current, Vg1, Vg2):

    """
    This function calculates the mean difference between consecutive final peaks of the first 40 signals
    40 because after this the final peaks are associated with a different transition line- how can this be automated?
    :param current:
    :param Vg1:
    :param Vg2:
    :return:
    """
    #initialise array of the distance so that we can calculate the mean later
    final_peak_dist_list = []

    for i in range(40):
        signal_i = current[i]
        signal_next = current[i + 1] #next signal

        #find the difference between the final peaks of each signal
        #is the verdict true?

        verdict, centre1, centre2 = likliness_two_peaks(signal_i, signal_next)

        if verdict == True: #if the final peaks are the same calculate the difference between the peaks

            final_peak_dist = abs(centre2 - centre1)
            final_peak_dist_list.append(final_peak_dist)

    #find the mean of the array
    final_peak_dist_array = np.array(final_peak_dist_list)

    mean_peak_shift = np.mean(final_peak_dist_array)

    return mean_peak_shift

print(calc_mean_shift(curr, Vg1, Vg2))



def verify_each_peak(current, Vg1, Vg2, X, Y, p0, threshold, n):
    """
    This function goes through each signal and for each peak in the signal
    #checks if there is another signal within range of it in the signal before. Is so then it is not noise and is plotted.
    :param current:
    :param Vg1:
    :param Vg2:
    :param X:
    :param Y:
    :param p0:
    :param threshold:
    :param n:
    :return: Binary mesh grid of the signals
    """

    #make empty array- we can produce a binary meshgrid
    bin_mesh = np.zeros((len(X), len(Y)))

    mean_peak_shift = calc_mean_shift(current, Vg1, Vg2)
    
    for i in range(len(current)-1): #700
        signal_i = current[i]
        signal_next = current[i + 1]
        
        #find each peak in signali
        signal_i_filt = sig_sub_exp_FFT(signal_i, Vg1, p0, threshold, n) #process signal 1
        smooth_signal_i, peaks_i = smooth_peaks(signal_i_filt, window_length, poly) #smooth signal and find peak positions
        peaks_i_value = Vg1[peaks_i] #get peak Vg1 values
        
        #find each peak in the next signal
        signal_next_filt = sig_sub_exp_FFT(signal_next, Vg1, p0, threshold, n) #process signal 2 (the next signal)
        smooth_signal_next, peaks_next = smooth_peaks(signal_next_filt, window_length, poly)
        peaks_next_value = Vg1[peaks_next] #value of the peaks in the next signal

        #print(peaks_next_value)

        for j in range(len(peaks_next_value)):
            proposed_peak = peaks_next_value[j]
            #print(proposed_peak)

            #check if there is a 'peak' in the signal before within the mean peak shift


            result = np.any(abs(proposed_peak - peaks_i_value) <= mean_peak_shift) #removed the absolute
            #print(result)

            if result: #if there does exist a peak within range, find its position in the peaks_next_value

                peak_index = peaks_next[j]

                bin_mesh[i+1][peak_index] = 1
                print(bin_mesh)

    return bin_mesh
        

#test

binary_mesh = verify_each_peak(curr, Vg1, Vg2, X, Y, p0, threshold, n)
contourplot = plt.contour(Vg1, Vg2, binary_mesh, levels=20, cmap=plt.cm.Greys, origin='lower')
plt.xlabel('Vg1')
plt.ylabel('Vg2')
# cbar = plt.colorbar(contourplot)
plt.title(f'Charge Stability Diagram, FFT_Threshold = {threshold} ')
plt.show()



"""
#signal_mesh = find_plot_signals_CSD(curr, Vg1, Vg2, X, Y, p0, threshold, n)
proc_sig = sig_sub_exp_FFT(signal, Vg2, p0, threshold, n)
#find_final_peak_Lorentz(proc_sig)
res1 = find_final_peak_Lorentz(signal, Vg2, p0, threshold, n)
res2 = find_final_peak_Lorentz(signal2, Vg2, p0, threshold, n)
print(res1)
print(res2)
"""