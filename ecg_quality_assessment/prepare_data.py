"""
数据准备脚本：下载、预处理和生成训练/测试数据
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import (
    DATA_PATHS, OUTPUT_PATHS, TRAIN_TEST_SPLIT,
    SEGMENT_LENGTH, LABEL_CLEAN, LABEL_NOISY
)
from utils.data_loader import create_data_loaders
from utils.signal_processing import preprocess_and_segment
from utils.noise_generator import NoiseGenerator
from utils.signal_quality_indices import auto_label_incart_quality


def check_data_availability():
    """
    检查数据库是否存在
    
    Returns:
        available: 可用数据库字典
        missing: 缺失数据库列表
    """
    print("="*60)
    print("检查数据库可用性...")
    print("="*60)
    
    available = {}
    missing = []
    
    for db_name, db_path in DATA_PATHS.items():
        if os.path.exists(db_path):
            print(f"✓ {db_name}: {db_path}")
            available[db_name] = db_path
        else:
            print(f"✗ {db_name}: {db_path} (未找到)")
            missing.append(db_name)
    
    print("="*60 + "\n")
    
    return available, missing


def download_data_instructions():
    """打印数据下载说明"""
    print("\n" + "="*60)
    print("数据下载说明")
    print("="*60)
    print("\n请从PhysioNet下载以下数据库：\n")
    
    print("1. MIT-BIH心律失常数据库:")
    print("   https://physionet.org/content/mitdb/1.0.0/")
    print(f"   解压到: {DATA_PATHS['mitdb']}\n")
    
    print("2. PTB诊断ECG数据库:")
    print("   https://physionet.org/content/ptbdb/1.0.0/")
    print(f"   解压到: {DATA_PATHS['ptbdb']}\n")
    
    print("3. INCART数据库:")
    print("   https://physionet.org/content/incartdb/1.0.0/")
    print(f"   解压到: {DATA_PATHS['incart']}\n")
    
    print("4. PhysioNet/CinC Challenge 2011:")
    print("   https://physionet.org/content/challenge-2011/1.0.0/")
    print(f"   解压到: {DATA_PATHS['pccc2011']}\n")
    
    print("5. MIT-BIH噪声压力测试数据库:")
    print("   https://physionet.org/content/nstdb/1.0.0/")
    print(f"   解压到: {DATA_PATHS['nstdb']}\n")
    
    print("="*60)
    print("\n也可以使用wfdb工具自动下载：")
    print("  pip install wfdb")
    print("  python download_datasets.py")
    print("="*60 + "\n")


def load_training_databases():
    """
    加载训练数据库（MIT-BIH和PTB）
    
    Returns:
        clean_segments: 干净信号片段列表
    """
    loaders = create_data_loaders()
    clean_segments = []
    
    # 加载MIT-BIH数据库
    if os.path.exists(DATA_PATHS['mitdb']):
        print("\n" + "="*60)
        print("处理MIT-BIH数据库...")
        print("="*60)
        
        mitdb_records = loaders['mitdb'].load_all_records()
        
        for record_name, (signal, fs) in tqdm(mitdb_records.items(), 
                                             desc="MIT-BIH"):
            segments = preprocess_and_segment(signal, fs, return_derivative=False)
            clean_segments.extend(segments)
        
        print(f"MIT-BIH: 提取了 {len(clean_segments)} 个信号片段")
    
    # 加载PTB数据库
    if os.path.exists(DATA_PATHS['ptbdb']):
        print("\n" + "="*60)
        print("处理PTB数据库...")
        print("="*60)
        
        ptbdb_records = loaders['ptbdb'].load_all_records()
        ptb_segments = []
        
        for record_name, (signals, fs) in tqdm(ptbdb_records.items(), 
                                              desc="PTB"):
            # PTB有15个导联，处理所有导联
            for lead_idx in range(signals.shape[1]):
                lead_signal = signals[:, lead_idx]
                segments = preprocess_and_segment(lead_signal, fs, 
                                                 return_derivative=False)
                ptb_segments.extend(segments)
        
        clean_segments.extend(ptb_segments)
        print(f"PTB: 提取了 {len(ptb_segments)} 个信号片段")
        print(f"总共: {len(clean_segments)} 个干净信号片段")
    
    return clean_segments


def load_test_databases():
    """
    加载测试数据库（INCART和PCCC2011）
    
    Returns:
        test_data: 字典，键为数据库名称，值为(X, y)元组
    """
    loaders = create_data_loaders()
    test_data = {}
    
    # 加载INCART数据库
    if os.path.exists(DATA_PATHS['incart']):
        print("\n" + "="*60)
        print("处理INCART数据库...")
        print("="*60)
        
        incart_records = loaders['incart'].load_all_records()
        incart_segments = []
        
        for record_name, (signals, fs) in tqdm(incart_records.items(), 
                                              desc="INCART"):
            # INCART有12导联
            for lead_idx in range(signals.shape[1]):
                lead_signal = signals[:, lead_idx]
                segments = preprocess_and_segment(lead_signal, fs, 
                                                 return_derivative=False)
                incart_segments.extend(segments)
        
        # 使用SQI算法自动标注质量（不再使用模拟标签）
        print("使用信号质量指标（SQI）自动标注INCART数据...")
        
        X_incart = np.array(incart_segments)
        y_incart, sqi_results = auto_label_incart_quality(X_incart, fs=360, verbose=True)
        
        test_data['incart'] = (X_incart, y_incart)
        
        # 保存SQI评估详情（可选）
        import json
        sqi_summary_path = os.path.join(OUTPUT_PATHS['processed_data'], 
                                        'incart_sqi_summary.json')
        os.makedirs(OUTPUT_PATHS['processed_data'], exist_ok=True)
        
        # 保存统计摘要
        sqi_summary = {
            'total_segments': int(len(y_incart)),
            'acceptable': int(np.sum(y_incart == LABEL_CLEAN)),
            'unacceptable': int(np.sum(y_incart == LABEL_NOISY)),
            'acceptable_percentage': float(np.sum(y_incart == LABEL_CLEAN) / len(y_incart) * 100),
        }
        
        with open(sqi_summary_path, 'w', encoding='utf-8') as f:
            json.dump(sqi_summary, f, indent=2, ensure_ascii=False)
        
        print(f"\nSQI评估摘要已保存到: {sqi_summary_path}")
    
    # 加载PCCC2011数据库（使用真实标注）
    if os.path.exists(DATA_PATHS['pccc2011']):
        print("\n" + "="*60)
        print("处理PhysioNet Challenge 2011数据库（Set B，使用真实标注）...")
        print("="*60)
        
        pccc_records = loaders['pccc2011'].load_all_records_with_labels()
        pccc_segments = []
        pccc_labels = []
        
        for record_name, (signals, fs, quality_labels) in tqdm(pccc_records.items(), 
                                                               desc="PCCC2011"):
            # Challenge 2011的记录是10秒长，每秒有一个质量标签
            # 我们需要将其分割为5秒片段，并为每个片段分配标签
            
            # 处理所有导联
            if len(signals.shape) == 1:
                # 单导联
                segments = preprocess_and_segment(signals, fs, return_derivative=False)
                
                # 为每个5秒片段分配标签
                # 如果5秒内有任何1秒被标记为不可接受，则整个片段标记为不可接受
                for seg_idx, segment in enumerate(segments):
                    # 计算这个片段对应的时间范围（秒）
                    start_sec = seg_idx * 5
                    end_sec = start_sec + 5
                    
                    # 获取这个时间范围内的质量标签
                    relevant_labels = quality_labels[start_sec:end_sec]
                    
                    # 如果有任何不可接受的标签，整个片段标记为不可接受
                    segment_label = LABEL_NOISY if (1 in relevant_labels) else LABEL_CLEAN
                    
                    pccc_segments.append(segment)
                    pccc_labels.append(segment_label)
            else:
                # 多导联
                for lead_idx in range(signals.shape[1]):
                    lead_signal = signals[:, lead_idx]
                    segments = preprocess_and_segment(lead_signal, fs, 
                                                     return_derivative=False)
                    
                    # 为每个片段分配标签（同上）
                    for seg_idx, segment in enumerate(segments):
                        start_sec = seg_idx * 5
                        end_sec = start_sec + 5
                        relevant_labels = quality_labels[start_sec:end_sec]
                        segment_label = LABEL_NOISY if (1 in relevant_labels) else LABEL_CLEAN
                        
                        pccc_segments.append(segment)
                        pccc_labels.append(segment_label)
        
        X_pccc = np.array(pccc_segments)
        y_pccc = np.array(pccc_labels)
        
        # 统计信息
        n_clean = np.sum(y_pccc == LABEL_CLEAN)
        n_noisy = np.sum(y_pccc == LABEL_NOISY)
        
        test_data['pccc2011'] = (X_pccc, y_pccc)
        print(f"PCCC2011: {len(pccc_segments)} 个信号片段")
        print(f"  - 可接受（干净）: {n_clean} ({n_clean/len(pccc_segments)*100:.1f}%)")
        print(f"  - 不可接受（含噪）: {n_noisy} ({n_noisy/len(pccc_segments)*100:.1f}%)")
    
    return test_data


def prepare_training_data():
    """
    准备训练数据
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    # 检查数据可用性
    available, missing = check_data_availability()
    
    # 检查必需的数据库
    required = ['mitdb', 'ptbdb', 'nstdb']
    missing_required = [db for db in required if db in missing]
    
    if missing_required:
        print(f"\n错误：缺少必需的数据库: {missing_required}")
        download_data_instructions()
        return None, None, None, None
    
    # 加载训练数据库
    print("\n" + "="*60)
    print("第1步：加载训练数据库")
    print("="*60)
    clean_segments = load_training_databases()
    
    if len(clean_segments) == 0:
        print("\n错误：未能提取任何干净信号片段！")
        return None, None, None, None
    
    # 加载噪声数据库
    print("\n" + "="*60)
    print("第2步：加载噪声数据库")
    print("="*60)
    
    loaders = create_data_loaders()
    noise_signals = loaders['nstdb'].load_noise_signals()
    
    if len(noise_signals) == 0:
        print("\n错误：未能加载噪声数据！")
        return None, None, None, None
    
    # 生成训练数据（干净 + 含噪）
    print("\n" + "="*60)
    print("第3步：生成训练数据")
    print("="*60)
    
    noise_gen = NoiseGenerator(noise_signals)
    
    # 使用部分数据生成训练集（避免数据量过大）
    # 可以根据实际情况调整
    max_segments = 5000  # 限制最大片段数
    if len(clean_segments) > max_segments:
        print(f"注意：干净信号片段过多，随机采样 {max_segments} 个")
        indices = np.random.choice(len(clean_segments), max_segments, replace=False)
        clean_segments = [clean_segments[i] for i in indices]
    
    X, y, metadata = noise_gen.generate_training_data(clean_segments, 
                                                       use_augmentation=True)
    
    # 划分训练集和测试集
    print("\n" + "="*60)
    print("第4步：划分训练集和测试集")
    print("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TRAIN_TEST_SPLIT['test_size'],
        random_state=TRAIN_TEST_SPLIT['random_state'],
        stratify=y
    )
    
    print(f"训练集: {len(X_train)} (干净: {np.sum(y_train==LABEL_CLEAN)}, "
          f"含噪: {np.sum(y_train==LABEL_NOISY)})")
    print(f"测试集: {len(X_test)} (干净: {np.sum(y_test==LABEL_CLEAN)}, "
          f"含噪: {np.sum(y_test==LABEL_NOISY)})")
    
    return X_train, X_test, y_train, y_test


def save_data(X_train, X_test, y_train, y_test):
    """
    保存处理好的数据
    
    Args:
        X_train, X_test, y_train, y_test: 训练和测试数据
    """
    # 创建输出目录
    os.makedirs(OUTPUT_PATHS['processed_data'], exist_ok=True)
    
    # 保存训练数据
    train_path = os.path.join(OUTPUT_PATHS['processed_data'], 'training_data.npz')
    np.savez_compressed(train_path, X=X_train, y=y_train)
    print(f"\n训练数据已保存到: {train_path}")
    
    # 保存测试数据（来自训练数据库）
    test_path = os.path.join(OUTPUT_PATHS['processed_data'], 'test_data.npz')
    np.savez_compressed(test_path, X=X_test, y=y_test)
    print(f"测试数据已保存到: {test_path}")


def prepare_unseen_test_data():
    """准备未见测试数据（INCART和PCCC2011）"""
    
    print("\n" + "="*60)
    print("准备未见测试数据")
    print("="*60)
    
    test_data = load_test_databases()
    
    # 保存未见测试数据
    os.makedirs(OUTPUT_PATHS['processed_data'], exist_ok=True)
    
    for db_name, (X, y) in test_data.items():
        save_path = os.path.join(OUTPUT_PATHS['processed_data'], 
                                f'{db_name}_test_data.npz')
        np.savez_compressed(save_path, X=X, y=y)
        print(f"{db_name} 测试数据已保存到: {save_path}")


def main():
    """主函数"""
    
    print("\n" + "="*60)
    print("ECG信号质量评估 - 数据准备")
    print("="*60 + "\n")
    
    # 准备训练数据
    X_train, X_test, y_train, y_test = prepare_training_data()
    
    if X_train is not None:
        # 保存数据
        print("\n" + "="*60)
        print("第5步：保存处理后的数据")
        print("="*60)
        save_data(X_train, X_test, y_train, y_test)
        
        # 准备未见测试数据（如果可用）
        prepare_unseen_test_data()
        
        print("\n" + "="*60)
        print("数据准备完成！")
        print("="*60)
        print("\n可以开始训练了：")
        print("  python train.py")
        print("="*60 + "\n")
    else:
        print("\n数据准备失败，请检查数据库路径和配置。")


if __name__ == '__main__':
    main()








