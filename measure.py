# import modules

import numpy as np
import matplotlib.pyplot as plt
import io
import scipy.fftpack as fft

def tsp2ir(signalPath, itspPath, repeat=1, irlen=8192):

    params, signal = io.readGetWave(signalPath)
    itspParams, itsp = io.readGetWave(itspPath)

    if params[2] != itspParams[2]:
        error = "not match framerate between signal and itsp"
        print(error)
        return

    itspF = fft.fft(itsp)

    N = itspParams[3]
    len = params[3] - N
    signal = signal[0:N] + np.concatenate((signal[N:], np.zeros((N-len))))
    signalF = fft.fft(signal)

    irF = signalF * itspF
    ir = np.real(fft.ifft(irF))[0:irlen]

    return ir
