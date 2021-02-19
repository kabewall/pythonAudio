import numpy as np
import scipy.signal as sg


def mk_highshelf(fs=48000, fc=1500, q=0.707, gain_db=4.0):
    
    omega = 2 * np.pi * fc / fs
    amp = np.sqrt(10.0 ** (gain_db/20.0))
    alpha = np.sin(omega) / q * 0.5
    
    a = np.zeros(3)
    b = np.zeros(3)
    
    a[0] = (amp + 1.0) - (amp - 1.0) * np.cos(omega) + 2.0 * np.sqrt(amp) * alpha
    a[1] = 2.0 * ((amp - 1.0) - (amp + 1.0) * np.cos(omega))
    a[2] = (amp + 1.0) - (amp - 1.0) * np.cos(omega) - 2.0 * np.sqrt(amp) * alpha
    b[0] = amp * ((amp + 1.0) + (amp - 1.0) * np.cos(omega) + 2.0 * np.sqrt(amp) * alpha)
    b[1] = -2.0 * amp * ((amp - 1.0) + (amp + 1.0) * np.cos(omega))
    b[2] = amp * ((amp + 1.0) + (amp - 1.0) * np.cos(omega) - 2.0 * np.sqrt(amp) * alpha)
    
    b /= a[0]
    a /= a[0]
    
    return b, a
    
def mk_k_weight_highpass(fs=48000, fc=37.5, q=0.5):
    omega = 2 * np.pi * fc / fs
    alpha = np.sin(omega) / q * 0.5

    a, b = np.zeros(3), np.zeros(3)

    a[0] = 1.0 + alpha
    a[1] = -2.0 * np.cos(omega)
    a[2] = 1.0 - alpha

    b[0] = (1.0 + np.cos(omega)) * 0.5
    b[1] = (1.0 + np.cos(omega)) * (-1.0)
    b[2] = (1.0 + np.cos(omega)) * 0.5

    b /= a[0]
    a /= a[0]

    return b, a

def show_loudness(data, fs):
    
    gating_time = 400 #ms
    gating_overlap_rate = 75 #%

    gating_tap = int(gating_time * 0.001 * fs)
    overlap_tap = int(gating_tap*gating_overlap_rate/100)
    slide_tap = gating_tap - overlap_tap

    wav_len = data.size
    block_nums = wav_len // slide_tap - overlap_tap // slide_tap

    pre_b, pre_a = mk_highshelf(fs=fs)
    k_b, k_a = mk_k_weight_highpass(fs=fs)

    loudness = 0
    for i in range(block_nums):

        tmp_wav = data[i*slide_tap:i*slide_tap+gating_tap]
        tmp_wav = sg.lfilter(pre_b, pre_a, tmp_wav)
        tmp_wav = sg.lfilter(k_b, k_a, tmp_wav)

        tmp_rms = np.mean(np.square(tmp_wav)) / block_nums

        loudness += tmp_rms
    
    loudness = 10 * np.log10(loudness) - 0.691
    
    return loudness

def adjust_loudness(data1, data2, fs, target_db=-40):
    
    data1_db = show_loudness(data1, fs)
    data2_db = show_loudness(data2, fs)
    
    data1_db_diff = target_db - data1_db
    data2_db_diff = target_db - data2_db
    
    data1_scale = np.power(10, data1_db_diff/20)
    data2_scale = np.power(10, data2_db_diff/20)
    
    return data1_scale, data2_scale