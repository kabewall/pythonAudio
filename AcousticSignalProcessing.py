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
    if nchannels == params[0]:
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
    else:
        print("Input channels are wrong.")

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
        print("IRのチャンネル数が%d chです。"%(ir.shape[0]))
        print("IRのチャンネル数を見直してください。")
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
def tsp2ir(itspPath, signal1Path, signal2Path, irLength=8192):
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
    signal1 = readWave(signal1Path)
    signal2 = readWave(signal2Path)
    signal = (signal1 + signal2) / 2
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

def binary2float(frames, length, sampwidth):
    # binary -> int
    if sampwidth == 1:
        data = np.frombuffer(frames, dtype=np.uint8)
        data = data - 128
    elif sampwidth == 2:
        data = np.frombuffer(frames, dtype=np.int16)
    elif sampwidth == 3:
        a8 = np.fromstring(frames, dtype=np.uint8)
        tmp = np.empty([length, 4], dtype=np.uint8)
        tmp[:, :sampwidth] = a8.reshape(-1, sampwidth)
        tmp[:, sampwidth:] = (tmp[:, sampwidth - 1:sampwidth] >> 7) * 255
        data = tmp.view('int32')[:, 0]
    elif sampwidth == 4:
        data = np.frombuffer(frames, dtype=np.int32)
    # Normalize to -1.0 ≤ sample < 1.0
    data = data.astype(float) / 2 ** (8 * sampwidth - 1)

    return data

def float2binary(data, sampwidth):
    data = (data * (2 ** (8 * sampwidth - 1) - 1)).reshape(data.size, 1)

    if sampwidth == 1:
        data = data + 128
        frames = data.astype(np.uint8).tostring()
    elif sampwidth == 2:
        frames = data.astype(np.int16).tostring()
    elif sampwidth == 3:
        a32 = np.asarray(data, dtype = np.int32)
        a8 = (a32.reshape(a32.shape + (1,)) >> np.array([0, 8, 16])) & 255
        frames = a8.astype(np.uint8).tostring()
    elif sampwidth == 4:
        frames = data.astype(np.int32).tostring()

    return frames

def max_24ch(data, fs=48000, cor=10):
    if data.shape[0] == 24:
        print("24chの音源を[24, :]のndarray配列で入力してください。")
    else:
        max_list = np.zeros([3, 8])
        for i in range(24):
            max_list[(i+1)//9 , ((i+1)%8 + 2)%8] = np.max(data[i])
        plt.matshow(maxlist)

def nextpow2(n):
    l = np.ceil(np.log2(n))
    m = int(np.log2(2 ** l))
    return m

def progressbar(percent, end=1, bar_length=40, slug='#', space='-', tail=''):
    percent = percent / end # float
    slugs = slug * int( round( percent * bar_length ) )
    spaces = space * ( bar_length - len( slugs ) )
    bar = slugs + spaces
    sys.stdout.write("\r[{bar}] {percent:.1f}% {tail}".format(
    	bar=bar, percent=percent*100., tail=tail
    ))
    sys.stdout.flush()
    if percent == 1:
        print()

def matmul_ver1(A, B):
    C = np.empty([A.shape[0], A.shape[2]], dtype=np.complex128)
    for ch in range(A.shape[0]):
        C[ch, :] = np.sum(A[ch, :, :] * B, axis=0)
    return C

def matmul_ver2(A, B):
    C = np.empty([A.shape[0], A.shape[2]], dtype=np.complex128)
    for fbin in range(A.shape[2]):
        C[:, fbin] = np.dot(A[:, :, fbin], B[:, fbin])
    return C

def convInvFast(filename_input, filename_output, filename_filter, channels=[24, 24]):
    nchannels_output = channels[0]
    nchannels_input = channels[1]
    # len_fir = 8192
    gain_dB = -13

    startRealTime = time.perf_counter()
    startClockTime = time.process_time()

    Gain = 10 ** (gain_dB / 20)

    # fir 畳み込み
    # print('Loading FIR', end='')
    # sys.stdout.flush()
    # cnt = 0
    # err = 0
    # fir = np.zeros([nchannels_output, nchannels_input, len_fir])
    # for index_lsp in range(nchannels_output):
    #     ch_lsp = index_lsp + 1
    #     for index_mic in range(nchannels_input):
    #         ch_mic = index_mic + 1
    #
    #         try:
    #             filename = "%s/inv_s%02d_m%02d.npy" % (path2inv, ch_lsp, ch_mic)
    #             fir[index_lsp, index_mic, :] = np.load(filename)
    #         except:
    #             err += 1
    #         finally:
    #             cnt += 1
    # print(' [%d]'%(cnt - err))

    fir = np.load(filename_filter)
    # invfilter [sp, mic, len]
    len_fir = fir.shape[2]

    # 入力ファイル マルチチャンネルファイル
    w_read = wave.open(filename_input, 'r')
    params = w_read.getparams()

    ws_input = params[1]
    fs = params[2]
    nframes = params[3]

    # 出力ファイル マルチチャンネルファイル
    w_write = wave.open(filename_output, 'wb')
    w_write.setparams((24, ws_input, fs, 0, 'NONE', 'not compressed'))



    # ブロック長の決定
    M = len_fir
    N = 2 ** nextpow2(M + M * 4 - 1)
    L = N - M + 1
    nblocks = int(np.ceil((nframes + M - 1) / L))

    fir_f = np.fft.rfft(fir, axis=2, n=N)
    del fir

    block = np.empty([nchannels_input, N])
    block.shape
    block[:, :-L] = 0.

    flag_continue = False
    convStartTime = time.time()


    for l in range(nblocks - 1):
        progressbar(l, nblocks)
        remain = w_read.getparams()[3] - w_read.tell()


        if remain >= L:
            frames = w_read.readframes(L)
            b = binary2float(frames, L, ws_input)
            block[:, -L:] = np.reshape(b, [b.shape[0]//24, 24]).T
        else:
            frames = w_read.readframes(remain)
            b = binary2float(frames, remain, ws_input)
            block[:, -L:-L+remain] = np.reshape(b, [b.shape[0]//24, 24]).T
            block[:, -L+remain:] = 0.

        block_f = np.fft.rfft(block, n=N)
        out_f = matmul_ver1(fir_f, block_f)
        out = Gain * np.fft.irfft(out_f, axis=1)[:, -L:]

        if not flag_continue:
            if np.max(np.abs(out)) > 1:
                saturation_dB = 20 * np.log10(np.max(np.abs(out)))
                print('Saturation detected. (%.1f dB) Continue? (y/n):'
                      % saturation_dB, end='')
                yesno = input()
                if yesno == 'y':
                    flag_continue = True
                else:
                    print('中断したぜ！')
                    break

        f = np.reshape(out.T, [out.size,])
        frames = float2binary(f, ws_input)
        w_write.writeframes(frames)

        block[:, :-L] = block[:, L:]

        elapsedTime = time.time() - convStartTime
        remainTime = elapsedTime / (l + 1) * (nblocks - l - 1)
        s_remain = '%02d:%02d:%02d' % (remainTime // 3600,
                (remainTime % 3600) // 60, np.ceil(remainTime % 60))
        progressbar(l + 1, nblocks, tail='Remain %s' % (s_remain))

    else:
        #progressbar(l + 1, nblocks)
        # input
        remain = w_read[0].getparams()[3] - w_read[0].tell()
        if remain >= L:
            for ch in range(nchannels_input):
                frames = w_read[ch].readframes(L)
                block[ch, -L:] = binary2float(frames, L, ws_input)
        else:
            for ch in range(nchannels_input):
                frames = w_read[ch].readframes(remain)
                block[ch, -L:-L + remain] = binary2float(frames, remain, ws_input)
            block[:, -L + remain:] = 0.

        # convolution & write
        block_f = np.fft.rfft(block, n=N)
        out_f = matmul_ver1(fir_f, block_f)
        out = Gain * np.fft.irfft(out_f, axis=1)[:, -L:]

        if not flag_continue:
            if np.max(np.abs(out)) > 1:
                print('Saturation detected. Continue? (y/n): ', end='')
                yesno = input()
                if yesno == 'y':
                    flag_continue = True
                else:
                    print('中断')

        # convolution & write
        block_f = np.fft.rfft(block, n=N)
        nDiscard = L * nblocks - (nframes + M - 1)
        out_f = matmul_ver1(fir_f, block_f)
        out = Gain * np.fft.irfft(out_f, axis=1)[:, -L:-nDiscard]
        for ch in range(nchannels_output):
            frames = float2binary(out[ch, :], ws_output)
            w_write[ch].writeframes(frames)

    progressbar(1, tail='Remain 00:00:00')
    w_read.close()
    w_write.close()



if __name__ == "__main__":
    fs = 48000
    pink = mkpink(1, fs)
    writeWave("/Users/kabe/Desktop/pink2.wav", pink)
