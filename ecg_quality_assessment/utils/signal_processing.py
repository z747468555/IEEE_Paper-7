"""
信号处理模块：包含滤波、归一化、导数计算等功能
"""

import numpy as np
from scipy import signal
from scipy.signal import resample
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TARGET_SAMPLING_RATE, SEGMENT_LENGTH,
    HIGHPASS_CUTOFF, LOWPASS_CUTOFF,
    HIGHPASS_ORDER, LOWPASS_ORDER
)


def resample_signal(ecg_signal, original_fs, target_fs=TARGET_SAMPLING_RATE):
    """
    重采样信号到目标采样率
    
    Args:
        ecg_signal: 输入ECG信号
        original_fs: 原始采样率
        target_fs: 目标采样率
    
    Returns:
        重采样后的信号
    """
    if original_fs == target_fs:
        return ecg_signal
    
    num_samples = int(len(ecg_signal) * target_fs / original_fs)
    resampled_signal = resample(ecg_signal, num_samples)
    return resampled_signal


def bandpass_filter(ecg_signal, fs=TARGET_SAMPLING_RATE):
    """
    双向带通滤波：二阶高通切比雪夫滤波器 + 四阶低通切比雪夫滤波器
    
    Args:
        ecg_signal: 输入ECG信号
        fs: 采样率
    
    Returns:
        滤波后的信号
    """
    # 二阶高通切比雪夫滤波器（截止频率0.8Hz）
    nyquist = fs / 2.0
    high_cutoff = HIGHPASS_CUTOFF / nyquist
    
    # 设计高通滤波器
    sos_high = signal.cheby2(HIGHPASS_ORDER, 20, high_cutoff, btype='highpass', output='sos')
    
    # 双向滤波（去除基线漂移）
    filtered_signal = signal.sosfiltfilt(sos_high, ecg_signal)
    
    # 四阶低通切比雪夫滤波器（截止频率40Hz）
    low_cutoff = LOWPASS_CUTOFF / nyquist
    
    # 设计低通滤波器
    sos_low = signal.cheby2(LOWPASS_ORDER, 20, low_cutoff, btype='lowpass', output='sos')
    
    # 双向滤波（去除高频噪声）
    filtered_signal = signal.sosfiltfilt(sos_low, filtered_signal)
    
    return filtered_signal


def normalize_signal(ecg_signal):
    """
    归一化信号到[-1, 1]范围
    公式: y[n] = (x[n] - μ_x) / max(|x[n] - μ_x|)
    
    Args:
        ecg_signal: 输入ECG信号
    
    Returns:
        归一化后的信号
    """
    # 计算均值
    mean_val = np.mean(ecg_signal)
    
    # 去除均值
    centered_signal = ecg_signal - mean_val
    
    # 计算最大绝对值
    max_abs = np.max(np.abs(centered_signal))
    
    # 避免除以零
    if max_abs < 1e-10:
        return np.zeros_like(ecg_signal)
    
    # 归一化
    normalized_signal = centered_signal / max_abs
    
    return normalized_signal


def compute_derivative(ecg_signal):
    """
    计算一阶导数（dECG）
    公式: d[n] = y[n+1] - y[n]
    
    Args:
        ecg_signal: 输入ECG信号
    
    Returns:
        dECG信号（长度为原信号长度-1）
    """
    derivative = np.diff(ecg_signal)
    return derivative


def segment_signal(ecg_signal, segment_length=SEGMENT_LENGTH, overlap=0):
    """
    将信号分割为固定长度的片段
    
    Args:
        ecg_signal: 输入ECG信号
        segment_length: 片段长度（采样点数）
        overlap: 重叠采样点数（默认0，不重叠）
    
    Returns:
        信号片段列表
    """
    segments = []
    step = segment_length - overlap
    
    for start in range(0, len(ecg_signal) - segment_length + 1, step):
        segment = ecg_signal[start:start + segment_length]
        segments.append(segment)
    
    return segments


def preprocess_ecg_signal(ecg_signal, original_fs):
    """
    完整的ECG信号预处理流程
    
    Args:
        ecg_signal: 输入ECG信号
        original_fs: 原始采样率
    
    Returns:
        预处理后的信号
    """
    # 1. 重采样到360Hz
    resampled = resample_signal(ecg_signal, original_fs, TARGET_SAMPLING_RATE)
    
    # 2. 带通滤波
    filtered = bandpass_filter(resampled)
    
    # 3. 归一化
    normalized = normalize_signal(filtered)
    
    return normalized


def preprocess_and_segment(ecg_signal, original_fs, return_derivative=True):
    """
    预处理并分割ECG信号，可选计算导数
    
    Args:
        ecg_signal: 输入ECG信号
        original_fs: 原始采样率
        return_derivative: 是否返回导数信号
    
    Returns:
        处理后的信号片段列表
    """
    # 预处理
    processed = preprocess_ecg_signal(ecg_signal, original_fs)
    
    # 分割
    segments = segment_signal(processed)
    
    # 计算导数（如果需要）
    if return_derivative:
        segments = [compute_derivative(seg) for seg in segments]
    
    return segments


def add_noise_to_signal(clean_signal, noise_signal, attenuation_factor):
    """
    将噪声添加到干净信号
    公式: NECG = NFECG + a × X_n
    
    Args:
        clean_signal: 干净的ECG信号
        noise_signal: 噪声信号
        attenuation_factor: 衰减因子
    
    Returns:
        含噪信号
    """
    # 确保噪声和信号长度相同
    if len(noise_signal) != len(clean_signal):
        if len(noise_signal) > len(clean_signal):
            noise_signal = noise_signal[:len(clean_signal)]
        else:
            # 如果噪声太短，重复使用
            repeats = int(np.ceil(len(clean_signal) / len(noise_signal)))
            noise_signal = np.tile(noise_signal, repeats)[:len(clean_signal)]
    
    noisy_signal = clean_signal + attenuation_factor * noise_signal
    
    # 重新归一化
    noisy_signal = normalize_signal(noisy_signal)
    
    return noisy_signal

