# import modules

import numpy as np
import wave
import matplotlib.pyplot as plt
import wavePlot
import scipy.fftpack as fft


def readWave(filename):

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

def readGetWave(filename):

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
    return params, data

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

def getInfo(filename):

    wr = wave.open(filename)
    params = wr.getparams()
    wr.close()

    return params


def aweightings(f):
    if f[0] == 0:
        f[0] = 1e-6
    else:
        pass
    ra = (np.power(12194, 2) * np.power(f, 4)) / \
         ((np.power(f, 2) + np.power(20.6, 2)) * \
          np.sqrt((np.power(f, 2) + np.power(107.7, 2)) * \
                  (np.power(f, 2) + np.power(737.9, 2))) * \
          (np.power(f, 2) + np.power(12194, 2)))

    return ra




def fig_time(signal, fs=48000, title="wave form", xaxis="time", label="signal", legend=False):

    if signal.ndim != 1:
        error = "dim of signal must be 1."
        return print(error)

    if xaxis == "tap":
        plt.plot(signal, label=label)
        plt.xlabel("tap")
    elif xaxis == "time":
        time = np.linspace(0, signal.shape[0]/fs, signal.shape[0])
        plt.plot(time, signal, label=label)
        plt.xlabel("time [s]")
    else:
        error = "xaxis must be \"tap\" or \"time\""
        print (error)
        return

    plt.title(title)
    if legend:
        legend()


def fig_freqz(signal, fs=48000, title="Frequency Characteristic", label="signal", legend=False):

    if signal.ndim != 1:
        error = "dim of signal must be 1."
        print(error)
        return

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


def fig_all(signal, fs=48000, num=1, time_title="Signal wave", time_xaxis="time", freqz_title="Frequency Responce", suptitle="Signal", label="signal"):
    plt.figure(num, figsize=[14,5])
    plt.subplot(121)
    fig_time(signal, fs, title=time_title, xaxis=time_xaxis, label=label)
    plt.subplot(122)
    fig_freqz(signal, fs, title=freqz_title, label=label)
    plt.ylim(-70, 0)
    plt.suptitle(suptitle)


def aweight_dB(data, fs):

    data_len = data.size
    f = np.linspace(0, 48000, data_len)
    a_weight = aweightings(f)

    data_f = np.fft.fft(data)

    out_f = data_f * a_weight

    out = np.real(np.fft.ifft(out_f))

    ave2 = 20*np.log10(np.sqrt(np.mean(np.square(out))))

    return ave2

audio_path = r"/Users/watakabe/Documents/python/wave/white_1000ms.wav"
audio_data = readWave(audio_path)
audio_data.shape
audio_params = getInfo(audio_path)
audio_params
fs = audio_params[2]
nframes = audio_params[3]


test_db = aweight_dB(audio_data, fs)

test_db



fig_all(audio_data, fs)



list_db = [10, 20, 30, 40]

import csv

with open("/Users/watakabe/Documents/python/wave/test.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(list_db)
