"""
噪声生成模块：生成含噪ECG信号和数据增强
"""

import numpy as np
from scipy.signal import resample
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    NOISE_ATTENUATION_FACTORS,
    AUGMENTATION_FACTORS,
    TARGET_SAMPLING_RATE,
    SEGMENT_LENGTH
)
from utils.signal_processing import (
    preprocess_ecg_signal,
    normalize_signal,
    add_noise_to_signal
)


class NoiseGenerator:
    """噪声生成器类"""
    
    def __init__(self, noise_signals_dict):
        """
        初始化噪声生成器
        
        Args:
            noise_signals_dict: 噪声信号字典，格式为 {噪声类型: (信号, 采样率)}
        """
        self.noise_signals = {}
        
        # 预处理噪声信号
        print("正在预处理噪声信号...")
        for noise_type, (noise_signal, fs) in noise_signals_dict.items():
            # 将噪声信号重采样并归一化
            processed_noise = preprocess_ecg_signal(noise_signal, fs)
            self.noise_signals[noise_type] = processed_noise
    
    def get_noise_segment(self, noise_type, length):
        """
        获取指定长度的噪声片段
        
        Args:
            noise_type: 噪声类型（'baseline_wander', 'muscle_artifact', 'electrode_motion'）
            length: 需要的噪声长度
        
        Returns:
            noise_segment: 噪声片段
        """
        if noise_type not in self.noise_signals:
            raise ValueError(f"未知的噪声类型: {noise_type}")
        
        noise_signal = self.noise_signals[noise_type]
        
        # 如果噪声信号比所需长度短，循环使用
        if len(noise_signal) < length:
            repeats = int(np.ceil(length / len(noise_signal)))
            noise_signal = np.tile(noise_signal, repeats)
        
        # 随机选择起始位置
        if len(noise_signal) > length:
            start_idx = np.random.randint(0, len(noise_signal) - length + 1)
            noise_segment = noise_signal[start_idx:start_idx + length]
        else:
            noise_segment = noise_signal[:length]
        
        return noise_segment
    
    def generate_noisy_signal(self, clean_signal, noise_type=None, 
                             attenuation_factor=None):
        """
        生成含噪信号
        
        Args:
            clean_signal: 干净的ECG信号
            noise_type: 噪声类型（如果为None，随机选择）
            attenuation_factor: 衰减因子（如果为None，随机选择）
        
        Returns:
            noisy_signal: 含噪信号
            noise_type: 使用的噪声类型
            attenuation_factor: 使用的衰减因子
        """
        # 随机选择噪声类型
        if noise_type is None:
            noise_type = np.random.choice(list(self.noise_signals.keys()))
        
        # 随机选择衰减因子
        if attenuation_factor is None:
            attenuation_factor = np.random.choice(NOISE_ATTENUATION_FACTORS)
        
        # 获取噪声片段
        noise_segment = self.get_noise_segment(noise_type, len(clean_signal))
        
        # 添加噪声
        noisy_signal = add_noise_to_signal(clean_signal, noise_segment, 
                                          attenuation_factor)
        
        return noisy_signal, noise_type, attenuation_factor
    
    def augment_clean_signal(self, clean_signal):
        """
        对干净信号进行数据增强（添加微小随机噪声）
        
        Args:
            clean_signal: 干净的ECG信号
        
        Returns:
            augmented_signals: 增强后的信号列表
        """
        augmented_signals = []
        
        for aug_factor in AUGMENTATION_FACTORS:
            # 生成随机噪声
            random_noise = np.random.randn(len(clean_signal))
            random_noise = normalize_signal(random_noise)
            
            # 添加微小噪声
            augmented = add_noise_to_signal(clean_signal, random_noise, aug_factor)
            augmented_signals.append(augmented)
        
        return augmented_signals
    
    def generate_training_data(self, clean_segments, use_augmentation=True):
        """
        生成训练数据（干净信号 + 含噪信号）
        
        Args:
            clean_segments: 干净信号片段列表
            use_augmentation: 是否使用数据增强
        
        Returns:
            X: 信号数据
            y: 标签（0=干净，1=含噪）
            metadata: 元数据（包含噪声类型、衰减因子等）
        """
        X = []
        y = []
        metadata = []
        
        print("正在生成训练数据...")
        
        # 处理干净信号
        for clean_seg in clean_segments:
            # 添加原始干净信号
            X.append(clean_seg)
            y.append(0)  # 干净信号标签
            metadata.append({'type': 'clean', 'noise_type': None, 
                           'attenuation': None})
            
            # 数据增强
            if use_augmentation:
                augmented_segs = self.augment_clean_signal(clean_seg)
                for aug_seg in augmented_segs:
                    X.append(aug_seg)
                    y.append(0)
                    metadata.append({'type': 'augmented', 'noise_type': None,
                                   'attenuation': None})
        
        # 生成含噪信号
        print("正在生成含噪信号...")
        for clean_seg in clean_segments:
            # 对每个干净信号，使用所有噪声类型和衰减因子组合
            for noise_type in self.noise_signals.keys():
                for attenuation in NOISE_ATTENUATION_FACTORS:
                    noisy_seg, _, _ = self.generate_noisy_signal(
                        clean_seg, noise_type, attenuation
                    )
                    X.append(noisy_seg)
                    y.append(1)  # 含噪信号标签
                    metadata.append({'type': 'noisy', 'noise_type': noise_type,
                                   'attenuation': attenuation})
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"生成的训练数据: 干净={np.sum(y==0)}, 含噪={np.sum(y==1)}")
        
        return X, y, metadata


class RealNoiseLabeler:
    """真实噪声标注器（用于测试集）"""
    
    @staticmethod
    def label_signal_by_quality(signal, annotation=None):
        """
        根据信号质量标注信号
        这是一个简化版本，实际应用中可能需要更复杂的质量评估
        
        Args:
            signal: ECG信号
            annotation: 可选的标注信息
        
        Returns:
            label: 0（可接受）或 1（不可接受）
        """
        # 这里可以实现基于规则的质量评估
        # 例如：检查信号的信噪比、基线稳定性等
        
        # 简化实现：使用信号的统计特性
        signal_std = np.std(signal)
        signal_range = np.max(signal) - np.min(signal)
        
        # 如果信号变化太小或太大，可能是噪声
        if signal_std < 0.01 or signal_range > 2.5:
            return 1  # 不可接受
        
        return 0  # 可接受
    
    @staticmethod
    def label_from_annotation_file(annotation_file):
        """
        从标注文件读取信号质量标签
        
        Args:
            annotation_file: 标注文件路径
        
        Returns:
            labels: 标签字典
        """
        # 这里需要根据实际的标注格式实现
        # Challenge 2011 提供了真实的质量标注
        pass



