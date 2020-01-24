# import modules

import numpy as np
import wave

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
