# import modules

import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fft

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


def fig_freqz(signal, fs=48000, title="Frequency Characteristic", label="signal", legend=False, color=None, ls=None, normalize_f=None):

    if signal.ndim != 1:
        error = "dim of signal must be 1."
        print(error)
        return

    signalF = fft.fft(signal)
    N = signalF.shape[0]
    f = fft.fftfreq(signalF.shape[0], d=1/fs)
    if normalize_f == None:
        norm_value = np.max(np.abs(signalF[:N//2]))
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
    plt.ylim(-70, 0)
    plt.suptitle(suptitle)
