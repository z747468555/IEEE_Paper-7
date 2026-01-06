"""
信号质量指标（SQI）模块：用于自动评估ECG信号质量
包含多种SQI算法，可用于自动标注INCART等数据集
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy.stats import kurtosis, skewness


def calculate_snr(ecg_signal):
    """
    计算信噪比（Signal-to-Noise Ratio, SNR）
    
    原理：高质量信号的SNR应该较高
    
    Args:
        ecg_signal: ECG信号
    
    Returns:
        snr: 信噪比（dB）
    """
    # 估计信号功率（使用信号的方差）
    signal_power = np.var(ecg_signal)
    
    # 估计噪声功率（使用高频成分）
    # 使用高通滤波器提取高频噪声
    sos = scipy_signal.butter(4, 0.5, 'highpass', fs=360, output='sos')
    noise = scipy_signal.sosfiltfilt(sos, ecg_signal)
    noise_power = np.var(noise)
    
    # 避免除以零
    if noise_power < 1e-10:
        return 100.0  # 非常高的SNR
    
    # 计算SNR（dB）
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def detect_baseline_wander(ecg_signal, fs=360):
    """
    检测基线漂移程度
    
    原理：基线漂移会导致低频成分能量过大
    
    Args:
        ecg_signal: ECG信号
        fs: 采样率
    
    Returns:
        baseline_power: 基线漂移功率（归一化）
    """
    # 使用低通滤波器提取基线
    sos = scipy_signal.butter(2, 0.5, 'lowpass', fs=fs, output='sos')
    baseline = scipy_signal.sosfiltfilt(sos, ecg_signal)
    
    # 计算基线功率相对于总信号功率的比例
    baseline_power = np.var(baseline)
    total_power = np.var(ecg_signal)
    
    if total_power < 1e-10:
        return 0.0
    
    return baseline_power / total_power


def detect_flat_line(ecg_signal, threshold=0.01):
    """
    检测平坦线段（信号丢失或电极脱落）
    
    Args:
        ecg_signal: ECG信号
        threshold: 判断为平坦的阈值
    
    Returns:
        flat_ratio: 平坦线段占比
    """
    # 计算信号的局部方差
    window_size = 50  # 约0.14秒
    flat_count = 0
    
    for i in range(0, len(ecg_signal) - window_size, window_size):
        segment = ecg_signal[i:i + window_size]
        if np.std(segment) < threshold:
            flat_count += 1
    
    total_windows = (len(ecg_signal) - window_size) // window_size
    flat_ratio = flat_count / total_windows if total_windows > 0 else 0
    
    return flat_ratio


def detect_saturation(ecg_signal, threshold=0.95):
    """
    检测信号饱和（超出量程）
    
    Args:
        ecg_signal: ECG信号（归一化后）
        threshold: 饱和阈值
    
    Returns:
        saturation_ratio: 饱和点占比
    """
    saturated = np.sum(np.abs(ecg_signal) > threshold)
    saturation_ratio = saturated / len(ecg_signal)
    return saturation_ratio


def calculate_statistical_features(ecg_signal):
    """
    计算统计特征
    
    Returns:
        features: 字典，包含各种统计特征
    """
    features = {
        'mean': np.mean(ecg_signal),
        'std': np.std(ecg_signal),
        'kurtosis': kurtosis(ecg_signal),
        'skewness': skewness(ecg_signal),
        'range': np.ptp(ecg_signal),  # peak-to-peak
    }
    return features


def detect_powerline_interference(ecg_signal, fs=360, powerline_freq=50):
    """
    检测工频干扰（50Hz或60Hz）
    
    Args:
        ecg_signal: ECG信号
        fs: 采样率
        powerline_freq: 工频频率（50Hz或60Hz）
    
    Returns:
        interference_power: 工频干扰功率（归一化）
    """
    # 计算功率谱
    freqs, psd = scipy_signal.welch(ecg_signal, fs=fs, nperseg=min(256, len(ecg_signal)))
    
    # 找到工频附近的功率
    freq_range = 2  # Hz
    mask = (freqs >= powerline_freq - freq_range) & (freqs <= powerline_freq + freq_range)
    
    if np.sum(mask) == 0:
        return 0.0
    
    interference_power = np.mean(psd[mask])
    total_power = np.mean(psd)
    
    if total_power < 1e-10:
        return 0.0
    
    return interference_power / total_power


def detect_high_frequency_noise(ecg_signal, fs=360, cutoff=40):
    """
    检测高频噪声（肌电干扰等）
    
    Args:
        ecg_signal: ECG信号
        fs: 采样率
        cutoff: 高频截止频率
    
    Returns:
        hf_noise_power: 高频噪声功率（归一化）
    """
    # 提取高频成分
    sos = scipy_signal.butter(4, cutoff, 'highpass', fs=fs, output='sos')
    hf_noise = scipy_signal.sosfiltfilt(sos, ecg_signal)
    
    # 计算高频噪声功率
    hf_power = np.var(hf_noise)
    total_power = np.var(ecg_signal)
    
    if total_power < 1e-10:
        return 0.0
    
    return hf_power / total_power


class ECGQualityAssessor:
    """
    ECG信号质量自动评估器
    综合多个SQI指标进行质量判断
    """
    
    def __init__(self, thresholds=None):
        """
        初始化评估器
        
        Args:
            thresholds: 阈值字典，如果为None则使用默认值
        """
        # 默认阈值（根据经验设置，可调整）
        self.thresholds = thresholds or {
            'snr_min': 5.0,              # SNR最小值（dB）
            'baseline_max': 0.3,          # 基线漂移最大占比
            'flat_max': 0.2,              # 平坦线段最大占比
            'saturation_max': 0.05,       # 饱和点最大占比
            'powerline_max': 0.1,         # 工频干扰最大占比
            'hf_noise_max': 0.3,          # 高频噪声最大占比
            'kurtosis_range': (-2, 10),   # 峰度范围
            'std_min': 0.01,              # 标准差最小值
        }
    
    def calculate_all_sqi(self, ecg_signal, fs=360):
        """
        计算所有SQI指标
        
        Args:
            ecg_signal: ECG信号
            fs: 采样率
        
        Returns:
            sqi_dict: 所有SQI指标的字典
        """
        sqi = {
            'snr': calculate_snr(ecg_signal),
            'baseline_wander': detect_baseline_wander(ecg_signal, fs),
            'flat_line': detect_flat_line(ecg_signal),
            'saturation': detect_saturation(ecg_signal),
            'powerline_interference': detect_powerline_interference(ecg_signal, fs),
            'hf_noise': detect_high_frequency_noise(ecg_signal, fs),
        }
        
        # 添加统计特征
        stat_features = calculate_statistical_features(ecg_signal)
        sqi.update(stat_features)
        
        return sqi
    
    def assess_quality(self, ecg_signal, fs=360, return_details=False):
        """
        评估信号质量
        
        Args:
            ecg_signal: ECG信号
            fs: 采样率
            return_details: 是否返回详细的SQI值
        
        Returns:
            quality_label: 0=可接受，1=不可接受
            sqi_dict: SQI指标字典（仅当return_details=True时返回）
        """
        # 计算所有SQI
        sqi = self.calculate_all_sqi(ecg_signal, fs)
        
        # 质量判断规则（多条件）
        is_acceptable = True
        failure_reasons = []
        
        # 规则1: SNR过低
        if sqi['snr'] < self.thresholds['snr_min']:
            is_acceptable = False
            failure_reasons.append(f"SNR过低 ({sqi['snr']:.2f} dB)")
        
        # 规则2: 基线漂移过大
        if sqi['baseline_wander'] > self.thresholds['baseline_max']:
            is_acceptable = False
            failure_reasons.append(f"基线漂移过大 ({sqi['baseline_wander']:.3f})")
        
        # 规则3: 平坦线段过多
        if sqi['flat_line'] > self.thresholds['flat_max']:
            is_acceptable = False
            failure_reasons.append(f"平坦线段过多 ({sqi['flat_line']:.3f})")
        
        # 规则4: 饱和点过多
        if sqi['saturation'] > self.thresholds['saturation_max']:
            is_acceptable = False
            failure_reasons.append(f"饱和点过多 ({sqi['saturation']:.3f})")
        
        # 规则5: 工频干扰过大
        if sqi['powerline_interference'] > self.thresholds['powerline_max']:
            is_acceptable = False
            failure_reasons.append(f"工频干扰过大 ({sqi['powerline_interference']:.3f})")
        
        # 规则6: 高频噪声过大
        if sqi['hf_noise'] > self.thresholds['hf_noise_max']:
            is_acceptable = False
            failure_reasons.append(f"高频噪声过大 ({sqi['hf_noise']:.3f})")
        
        # 规则7: 标准差过小（可能是平坦信号）
        if sqi['std'] < self.thresholds['std_min']:
            is_acceptable = False
            failure_reasons.append(f"标准差过小 ({sqi['std']:.4f})")
        
        # 规则8: 峰度异常
        kurt_min, kurt_max = self.thresholds['kurtosis_range']
        if not (kurt_min <= sqi['kurtosis'] <= kurt_max):
            is_acceptable = False
            failure_reasons.append(f"峰度异常 ({sqi['kurtosis']:.2f})")
        
        # 转换为标签
        quality_label = 0 if is_acceptable else 1
        
        # 添加质量评估结果到SQI字典
        sqi['quality_label'] = quality_label
        sqi['is_acceptable'] = is_acceptable
        sqi['failure_reasons'] = failure_reasons
        
        if return_details:
            return quality_label, sqi
        else:
            return quality_label
    
    def assess_segments(self, segments, fs=360, verbose=False):
        """
        批量评估多个信号片段
        
        Args:
            segments: 信号片段列表
            fs: 采样率
            verbose: 是否显示进度
        
        Returns:
            labels: 质量标签数组
            sqi_list: SQI指标列表
        """
        labels = []
        sqi_list = []
        
        iterator = enumerate(segments)
        if verbose:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=len(segments), desc="SQI评估")
        
        for i, segment in iterator:
            label, sqi = self.assess_quality(segment, fs, return_details=True)
            labels.append(label)
            sqi_list.append(sqi)
        
        return np.array(labels), sqi_list


def auto_label_incart_quality(incart_segments, fs=360, thresholds=None, verbose=True):
    """
    自动为INCART数据片段标注质量
    
    Args:
        incart_segments: INCART信号片段列表
        fs: 采样率
        thresholds: 自定义阈值字典（可选）
        verbose: 是否显示详细信息
    
    Returns:
        labels: 质量标签数组（0=可接受，1=不可接受）
        sqi_results: 详细的SQI评估结果
    """
    print("\n" + "="*60)
    print("使用SQI算法自动标注INCART数据质量")
    print("="*60)
    
    # 创建评估器
    assessor = ECGQualityAssessor(thresholds)
    
    # 批量评估
    labels, sqi_results = assessor.assess_segments(incart_segments, fs, verbose)
    
    # 统计结果
    n_acceptable = np.sum(labels == 0)
    n_unacceptable = np.sum(labels == 1)
    
    if verbose:
        print(f"\n自动标注完成:")
        print(f"  可接受（干净）: {n_acceptable} ({n_acceptable/len(labels)*100:.1f}%)")
        print(f"  不可接受（含噪）: {n_unacceptable} ({n_unacceptable/len(labels)*100:.1f}%)")
        
        # 统计不可接受的主要原因
        if n_unacceptable > 0:
            all_reasons = []
            for sqi in sqi_results:
                if sqi['quality_label'] == 1:
                    all_reasons.extend(sqi['failure_reasons'])
            
            from collections import Counter
            reason_counts = Counter(all_reasons)
            
            print(f"\n不可接受信号的主要原因:")
            for reason, count in reason_counts.most_common(5):
                print(f"  - {reason}: {count}次")
    
    return labels, sqi_results

