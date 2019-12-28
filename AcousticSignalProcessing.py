"""
Functions of audio signal processing
"""
import numpy as np
import scipy.fftpack as fft
import matplotlib.pyplot as plt
import wave
from scipy import signal
import matplotlib.animation as animation
from scipy.signal import fftconvolve
import time
from tqdm import tqdm
import sys
import random
from struct import unpack


"""
List of Functions
readWave ... Read .wav file
writeWave ... write .wav file
fig_freqz ... plot a frequency charasteristic of a signal
fig_time ... convert responces of tsp signal to IR
fig_all ... execute fig_time and fig_freqz
tsp2ir ... convert responces of tsp signal to IR
mkwhite ... This program makes white noise.
mkpink ... This program makes pink noise.
convInvFast ... convolve malti channel signal with inverse filter
"""

"""
Convolution Calculation Speed

fs = 48000[Hz]
-------------------------------
time of the signal is 1 [sec]
scipy.signal.fftconvolve: 0.005183 [sec]
numpy.convolve: 0.223275 [sec]
asp.overlap_save: 0.013217 [sec]
-------------------------------
time of the signal is 10 [sec]
scipy.signal.fftconvolve: 0.066644 [sec]
numpy.convolve: 0.744487 [sec]
asp.overlap_save: 0.179715 [sec]
-------------------------------
time of the signal is 30 [sec]
scipy.signal.fftconvolve: 0.180910 [sec]
numpy.convolve: 2.281399 [sec]
asp.overlap_save: 0.571477 [sec]
-------------------------------
time of the signal is 60 [sec]
scipy.signal.fftconvolve: 0.357637 [sec]
numpy.convolve: 4.500604 [sec]
asp.overlap_save: 1.116902 [sec]
-------------------------------
"""

def readWave(filename):
    """
    Read wave file.
    Args: filename string: path to .wav file
    Return: Numpy Array(float64) [channels, length]
    """

    wr = wave.open(filename, 'r')
    params = wr.getparams()
    nchannels = params[0]   # Number of Cannels
    sampwidth = params[1]   # Quantization Bit Number
    rate = params[2]        # Sampling Frequency
    nframes =  params[3]    # Length of the Signal
    frames = wr.readframes(nframes) # Data of the Signal
    wr.close()

    # by Bit Number
    if sampwidth == 1:
        data = np.frombuffer(frames, dtype=np.uint8)
        data = (data - 128) / 128
    elif sampwidth == 2:
        data = np.frombuffer(frames, dtype=np.int16) / 32768
    elif sampwidth == 3:
        a8 = np.frombuffer(frames, dtype=np.uint8)
        tmp = np.zeros((nframes * nchannels, 4), dtype=np.uint8)
        tmp[:, 1:] = a8.reshape(-1, 3)
        data = tmp.view(np.int32)[:, 0] / 2147483648
    elif sampwidth == 4:
        data = np.frombuffer(frames, dtype=np.int32) / 2147483648

    data = np.reshape(data, (-1, nchannels)).T
    if nchannels==1:
        data = np.reshape(data, (nframes))
    return data


def writeWave(file_name, data, params=(1, 3, 48000)):
    """
    Write .wav File

    Args:
    -------------------
    filename: fullpass to write the .wav at (string)
    data: data to convert to .wav (numpy array (float64))
    params: (number of channels, amp width, framerate)
    """
    if data.ndim == 1:
        nchannels = 1
        data = np.reshape(data, [data.shape[0], 1])
    else:
        nchannels = data.shape[0]

    data = data.T
    audio = wave.Wave_write(file_name) # filenameというファイルに書き出し
    # パラメータ設定
    audio.setnchannels(params[0])
    audio.setsampwidth(params[1])
    audio.setframerate(params[2])

    data = (data*(2**(8*params[1]-1)-1)).reshape(data.size, 1)
    if params[1] == 1:
        data = data + 128
        frames = data.astype(np.uint8).tostring()
    elif params[1] == 2:
        frames = data.astype(np.int16).tostring()
    elif params[1] == 3:
        a32 = np.asarray(data, dtype = np.int32)
        a8 = ( a32.reshape(a32.shape + (1,)) >> np.array([0, 8, 16]) ) & 255
        frames = a8.astype(np.uint8).tostring()
    elif params[1] == 4:
        frames = data.astype(np.int32).tostring()

    audio.writeframes(frames) # 出力データ設定
    audio.close() # ファイルを閉じる
    return

def mkTottedashi(signal, ir, savePath):
    """
    モノラル音源に24chIRを畳み込んで24chの配列にする関数
    正規化せずに .npyファイルで保存するので注意。
    Params
    ------------
    signalPath: モノラル音源 .wav
    irPath: 24ch IR .npy ([24, ??])
    savePath: 完成した音源を保存するパス
    """


    if ir.shape[0] != 24:
        print("the Number of IR channels is %d"%(ir.shape[0]))
        print("Recheck the Number of Channels")
        return

    lenSignal = signal.shape[0]
    lenIr = ir.shape[1]

    audio = np.zeros([24, lenSignal+lenIr-1])
    start = time.time()
    for i in range(24):
        audio[i] = fftconvolve(signal, ir[i])
    np.save(savePath, audio)

    return np.max(audio)


def fig_freqz(signal, fs=48000, title="Frequency Characteristic", label="signal", legend=False):
    """
    Plot a frequency charasteristic of a signal
    No window processing

    Parameters
    ----------------------
    signal: a signal to plot (numpy array(float64))
    fs: sampling frequency (int or float)
    title: title of figure (string)
    """
    signalF = fft.fft(signal)
    N = signalF.shape[0]
    f = fft.fftfreq(signalF.shape[0], d=1/fs)
    plt.plot(f[:N//2], 20*np.log10(np.abs(signalF[:N//2])/np.max(np.abs(signalF[:N//2]))), label=label)
    plt.xscale('log')
    plt.xlim(20, fs//2)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Level [dB]")
    plt.title(title)
    if legend:
        plt.legend()

def fig_freqz_thirdBand(signal, fs=48000, title="Band-Frequency Charasteristic", legend=False, label="signal",):
    signalF = np.abs(fft.fft(signal))
    N = signalF.shape[0]
    f = fft.fftfreq(N, d=1/fs)[:N//2]
    startFreq = 20
    freq = startFreq * ((np.ones(32) * np.power(2, 1/3)) ** np.arange(32))
    bandFreqList = []
    for i in freq:
        lowerFreq = i / np.power(2, 1/6)
        bandFreqList.append(np.max(np.where(f < lowerFreq)))
    bandLevel = np.zeros([31,])
    for i in range(31):
        for j in range(bandFreqList[i], bandFreqList[i+1]+1, 1):
            bandLevel[i] += signalF[j] ** 2 / (0.23 * freq[i])
    plt.plot(freq[:31], 10*np.log10(bandLevel / np.max(bandLevel)) ,'s-', label=label)
    # plt.bar(bandFreqList, bandLevel)
    plt.xscale('log')
    plt.xlim(18, fs//2)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Level [dB]")
    plt.title(title)
    if legend:
        plt.legend()

def fig_time(signal, fs=48000, title="Signal", xaxis="time", label="signal", legend=False):
    """
    plot a figure of a signal

    Parameters
    ----------------------
    signal: a signal to plot (numpy array(float64))
    fs: sampling frequency (int or float)
    title: title of figure (string)
    xaxis: axis x ... tap or time (string)
    """
    if xaxis == "tap":
        plt.plot(signal)
        plt.xlabel("tap")
    elif xaxis == "time":
        time = np.linspace(0, signal.shape[0]/fs, signal.shape[0])
        plt.plot(time, signal, label=label)
        plt.xlabel("time [s]")
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

# tspからIRを作成
def tsp2ir(itspPath, signalPath, irLength=8192):
    """
    convert responces of tsp signal to IR

    Parameters
    -------------------------
    itspPath: fullpath of itsp(.wav) (string)
    signal1Path: fullpath of first responce(.wav) (string)
    signal2Path: fullpath of second responce(.wav) (string)

    Returns
    -------------------------
    ir: Impulse Responce (numpy array)
    """
    signal = readWave(signalPath)
    itsp = readWave(itspPath)
    itspF = fft.fft(itsp)

    N = itsp.shape[0]
    len = signal.shape[0] - N
    signal = signal[0:N] + np.concatenate((signal[N:], np.zeros((N-len))))
    signalF = fft.fft(signal)

    irF = signalF * itspF
    ir = np.real(fft.ifft(irF))[0:irLength]
    # 書きだし
    return ir

def mkwhite(t, fs):
    """
    This program makes white noise.

    prameters
    ---------------------
    t: time[s] (int or float)
    fs: sampling frequency[Hz] (int or float)

    return
    ---------------------
    white: white noise (numpy array float64)
    """
    tap = int(t*fs)
    white = np.random.randn(tap)
    white /= np.max(np.abs(white))
    return white

def mkpink(t, fs):
    """
    This program makes pink noise.

    prameters
    ---------------------
    t: time[s] (int or float)
    fs: sampling frequency[Hz] (int or float)

    return
    ---------------------
    pink: pink noise (numpy array float64)
    """
    tap = int(t*fs)
    white = mkwhite(t, fs)
    WHITE = fft.fft(white)
    pink_filter = np.concatenate((np.array([1]), 1/np.sqrt(np.arange(start=fs/tap, stop=fs, step=fs/tap))))
    PINK = WHITE * pink_filter
    pink = np.real(fft.ifft(PINK))
    pink /= np.max(np.abs(pink))
    return pink

def wav2npy(channel, wavPath, npyPath):
	"""
	convert .wav to .npy, only for multi channel

	"""
	for i in range(channel):
		if i == 0:
			source = np.load("%s/%02d.wav"%(wavPath, i+1))
		else:
			source = np.concatenate((source, np.load("%s/%02d.wav"%(wavPath, i+1))), axis=1)
	np.save("%s/sound.npy"%(npyPath))
