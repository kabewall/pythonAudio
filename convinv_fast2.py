'''
規定

## Input PCM stream
・モノラル WAVE ファイルであること
・ファイル名 'input%02d.wav' % (チャンネル番号)
・チャンネル番号は 01 始まり
'''
import sys
import wave
import time
import numpy as np
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
fs = 48000
filename_input = '/Volumes/Seagate Backup Plus Drive/invFilter/forFigure/distance050cm_s%02d.wav'
filename_output = '/Volumes/Seagate Backup Plus Drive/invFilter/forFigure/bosc_distance050cm_s%02d.wav'
path2inv = '/Volumes/Seagate Backup Plus Drive/invFilter/inv576b1.5'
nchannels_output = 1
nchannels_input = 1
len_fir = 8192
gain_dB = -5
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

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
        tmp[:, :sampwidth] = a8.reshape(-1, sampwidth) # バグる
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


# スピーカに対してループ
#@numba.jit('c8[:, :](c8[:, :, :], c8[:, :])')
def matmul_ver1(A, B):
    C = np.empty([A.shape[0], A.shape[2]], dtype=np.complex128)
    for ch in range(A.shape[0]):
        C[ch, :] = np.sum(A[ch, :, :] * B, axis=0)
    return C

# 周波数ビンに対してループ
#@numba.jit('c8[:, :](c8[:, :, :], c8[:, :])')
def matmul_ver2(A, B):
    C = np.empty([A.shape[0], A.shape[2]], dtype=np.complex128)
    for fbin in range(A.shape[2]):
        C[:, fbin] = np.dot(A[:, :, fbin], B[:, fbin])
    return C

def exeConv(filename_input=filename_input, filename_output=filename_output, path2inv=path2inv, channels=[nchannels_input, nchannels_output]):
    nchannels_output = channels[0]
    nchannels_input = channels[1]
    len_fir = 8192
    gain_dB = -5


    startRealTime = time.perf_counter()
    startClockTime = time.process_time()


    Gain = 10 ** (gain_dB / 20)
    ws_output = 3 # sample width (3: 24bit)


    # fir フィルタ読み込み
    print('Loading FIR', end='')
    sys.stdout.flush()
    cnt = 0
    err = 0
    fir = np.zeros([nchannels_output, nchannels_input, len_fir])
    for index_lsp in range(nchannels_output):
        ch_lsp = index_lsp + 1
        for index_mic in range(nchannels_input):
            ch_mic = index_mic + 1

            try:
                filename = '%s/inv_s%02d_m%02d.npy' % (path2inv, ch_lsp, ch_mic)
                fir[index_lsp, index_mic, :] = np.load(filename)
            except:
                err += 1
            finally:
                cnt += 1
    print(' [%d]' % (cnt - err))
    # fir = np.load("/Volumes/Seagate Backup Plus Drive/190116Bosc/inverse_filter/25ch.npy")


    # 入力ファイル
    w_read = []
    for ch in range(nchannels_input):
        w_read.append(wave.open(filename_input % (ch + 1), 'r'))

    # チェック
    # * モノラルファイル？
    flag_notmono = False
    for ch in range(nchannels_input):
        params = w_read[ch].getparams()
        if params[0] != 1:
            print('file %02d is not monoural wav' % (ch + 1))
            flag_notmono = True

    if flag_notmono:
        sys.exit()

    # * fs は設定値か？


    # * 量子化ビット数, フレーム数 は揃っているか？

    ws_input = params[1]
    fs = params[2]
    nframes =  params[3]


    # 出力ファイル
    w_write = []
    for ch in range(nchannels_output):
        w_write.append(wave.open(filename_output % (ch + 1), 'wb'))
        w_write[ch].setparams((1, ws_output, fs, 0, 'NONE', 'not compressed'))


    # ブロック長の決定（今回は L > 10M になるように設定）
    M = len_fir # fir length
    N = 2 ** nextpow2(M + M * 4 - 1) # FFT point
    L = N - M + 1 # Block length
    nblocks = int(np.ceil((nframes + M - 1) / L))
    text = '======================================\n'
    text += 'FIR length (M): %d tap\n' % (M)
    text += 'Input: %d ch, %d tap\n' % (nchannels_input, nframes)
    text += 'Output: %d ch, %d tap\n' % (nchannels_output, nframes + M - 1)
    text += '--------------------------------------\n'
    text += 'Block length (L): %d\n' % (L)
    text += 'FFT point (N): %d (=2^%d)\n' % (N, np.log2(N))
    text += 'Overlap (M - 1): %d\n' % (M - 1)
    text += 'nBlocks: %d\n' % nblocks
    text += '======================================'
    print(text)

    print('FFT FIRs...', end=''); sys.stdout.flush()
    fir_f = np.fft.rfft(fir, axis=2, n=N)
    print('End')
    del fir

    print('Convoluting')
    block = np.empty([nchannels_input, N])
    block[:, :-L] = 0.

    flag_continue = False # サチっても続けるか？
    convStartTime = time.time()
    for l in range(nblocks - 1):
        progressbar(l, nblocks)

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
                saturation_dB = 20 * np.log10(np.max(np.abs(out)))
                print('Saturation detected. (%.1f dB) Continue? (y/n):'
                      % saturation_dB, end='')
                yesno = input()
                if yesno == 'y':
                    flag_continue = True
                else:
                    print('中断したぜ！')
                    break

        for ch in range(nchannels_output):
            frames = float2binary(out[ch, :], ws_output)
            w_write[ch].writeframes(frames)

        # overlap M-1 taps
        block[:, :-L] = block[:, L:]

        # 時間の予想
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


    # print (' Real Time: %.2f sec' % (time.perf_counter() - startRealTime))
    # print ('Clock Time: %.2f sec' % (time.process_time() - startClockTime))


    # print('%d taps written' % w_write[0].tell())


    for ch in range(nchannels_input):
        w_read[ch].close()
    for ch in range(nchannels_output):
        w_write[ch].close()
