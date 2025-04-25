# 滤波去噪

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

def ECG_Filter(ecg_signal, fs, line_freq=50.0, q_factor=30.0):
    """
    功能：
        滤除ECG信号中的基线漂移和工频干扰。
    参数：
    ecg_signal : numpy.ndarray(一维ECG信号)
    fs : 信号的采样频率(Hz)
    line_freq : 工频干扰的频率(默认为50Hz)
    q_factor :  陷波滤波器的质量因数(默认为30.0)
    返回：
    filtered_signal : numpy.ndarray(滤波后的ECG信号)
    """
    # 高通滤波去除基线漂移（截止频率0.5Hz）
    highpass_cutoff = 0.5
    hp_order = 2  # 二阶滤波器
    b_high, a_high = butter(hp_order, highpass_cutoff, btype='highpass', fs=fs)
    filtered = filtfilt(b_high, a_high, ecg_signal)

    # 陷波滤波去除工频干扰
    notch_freq = line_freq
    b_notch, a_notch = iirnotch(notch_freq, q_factor, fs)
    filtered = filtfilt(b_notch, a_notch, filtered)

    return filtered