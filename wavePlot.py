# import modules

import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fft
from scipy.signal import fftconvolve
from scipy.signal import firwin

def fig_time(signal, fs=48000, title="wave form", xaxis="time", label="signal", legend=False, color=None, ls=None):

    if signal.ndim != 1:
        error = "dim of signal must be 1."
        return print(error)

    if xaxis == "tap":
        plt.plot(signal, label=label, color=color, ls=ls)
        plt.xlabel("tap")
    elif xaxis == "time":
        time = np.linspace(0, signal.shape[0]/fs, signal.shape[0])
        plt.plot(time, signal, label=label, color=color, ls=ls)
        plt.xlabel("time [s]")
    else:
        error = "xaxis must be \"tap\" or \"time\""
        print (error)
        return

    plt.title(title)
    if legend:
        legend()


def fig_freqz(signal, fs=48000, title="Frequency Characteristic", label="signal", legend=False, color=None, ls=None, normalize_f=None, p_pref=2e-5):

    if signal.ndim != 1:
        error = "dim of signal must be 1."
        print(error)
        return

    signalF = fft.fft(signal)
    N = signalF.shape[0]
    f = fft.fftfreq(signalF.shape[0], d=1/fs)
    if normalize_f == None:
        norm_value = p_pref
    else:
        normalize_tap = int(normalize_f * (N//2) / (fs//2))
        norm_value = np.abs(signalF[normalize_tap])
    plt.plot(f[:N//2], 20*np.log10(np.abs(signalF[:N//2])/norm_value), label=label, color=color, ls=ls)
    plt.xscale('log')
    plt.xlim(20, fs//2)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Level [dB]")
    plt.title(title)
    if legend:
        plt.legend()


def fig_all(signal, fs=48000, num=1, time_title="Signal wave", time_xaxis="time", freqz_title="Frequency Responce", suptitle="Signal", label="signal"):
    plt.figure(num, figsize=[14,5])
    plt.subplot(121)
    fig_time(signal, fs, title=time_title, xaxis=time_xaxis, label=label)
    plt.subplot(122)
    fig_freqz(signal, fs, title=freqz_title, label=label)
    plt.suptitle(suptitle)

def fig_octbandfreq(signal, fs=48000, octband='1/3', filter_taps=2048, p_pref=2e-5):
    if signal.ndim != 1:
        error = "dim of signal must be 1."
        print(error)
        return
    center_freqs = np.array([20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000])
    if octband == '1/3':
        upper_freqs = center_freqs * np.power(2, 1/6)
        bottom_freqs = center_freqs / np.power(2, 1/6)
    elif octband == '1':
        center_freqs = center_freqs[2::2]
        upper_freqs = center_freqs * np.power(2, 1/2)
        bottom_freqs = center_freqs / np.power(2, 1/2)
    else:
        print("enter correct octband '1' or '1/3'")
        return

    band_power = np.zeros(center_freqs.size)
    for i in range(center_freqs.size):
        tmp_bandfilter = firwin(numtaps=filter_taps, cutoff=[bottom_freqs[i], upper_freqs[i]], pass_zero=False, fs=fs)
        tmp_bandsignal = fftconvolve(signal, tmp_bandfilter)
        band_power[i] = 20*np.log10(np.mean(np.abs(tmp_bandsignal)) / p_pref)

    plt.plot(center_freqs, band_power, '-o')
    plt.title("band freq characteristic")
    plt.xlabel("center freq [Hz]")
    plt.ylabel("power [dB]")
    plt.xscale('log')
