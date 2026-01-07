"""
ECG信号质量评估系统
基于卷积神经网络和导数ECG信号
"""

__version__ = '1.0.0'
__author__ = 'Your Name'

from .model.cnn_model import create_model, load_model
from .utils.signal_processing import (
    preprocess_ecg_signal,
    compute_derivative,
    normalize_signal
)










