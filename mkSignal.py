# import modules

import numpy as np
import scipy.fftpack as fft

def mkwhite(t, fs=48000):

    tap = int(t*fs)
    white = np.random.randn(tap)
    white /= np.max(np.abs(white))
    return white


def mkpink(t, fs=48000):

    tap = int(t*fs)
    white = mkwhite(t, fs)
    whiteF = fft.fft(white)

    pink_filter = np.concatenate((np.array([1]), 1/np.sqrt(np.arange(start=fs/tap, stop=fs, step=fs/tap))))
    PINK = WHITE * pink_filter
    pink = np.real(fft.ifft(PINK))
    pink /= np.max(np.abs(pink))

    return pink


# TODO mkTSP
