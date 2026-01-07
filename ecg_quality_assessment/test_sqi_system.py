"""
测试SQI自动标注系统
用于验证SQI算法是否正常工作
"""

import numpy as np
import matplotlib.pyplot as plt
from utils.signal_quality_indices import ECGQualityAssessor


def generate_clean_ecg(length=1800, fs=360):
    """生成模拟的干净ECG信号"""
    t = np.arange(length) / fs
    
    # 简单的心电信号模拟（正弦波组合）
    hr = 72  # 心率 72 bpm
    freq = hr / 60  # Hz
    
    # P波、QRS波、T波的模拟
    ecg = (
        0.1 * np.sin(2 * np.pi * freq * t) +  # P波
        0.8 * np.sin(2 * np.pi * freq * t + np.pi/4) +  # R波
        0.2 * np.sin(2 * np.pi * freq * t + np.pi/2)   # T波
    )
    
    # 添加微小噪声
    ecg += 0.01 * np.random.randn(length)
    
    return ecg


def generate_noisy_ecg(length=1800, fs=360, noise_type='baseline'):
    """生成模拟的含噪ECG信号"""
    clean = generate_clean_ecg(length, fs)
    
    if noise_type == 'baseline':
        # 基线漂移
        baseline = 0.5 * np.sin(2 * np.pi * 0.2 * np.arange(length) / fs)
        return clean + baseline
    
    elif noise_type == 'hf_noise':
        # 高频噪声（肌电干扰）
        hf_noise = 0.3 * np.random.randn(length)
        return clean + hf_noise
    
    elif noise_type == 'powerline':
        # 工频干扰
        powerline = 0.2 * np.sin(2 * np.pi * 50 * np.arange(length) / fs)
        return clean + powerline
    
    elif noise_type == 'saturation':
        # 饱和
        noisy = clean.copy()
        noisy[noisy > 0.8] = 1.0
        noisy[noisy < -0.8] = -1.0
        return noisy
    
    elif noise_type == 'flat':
        # 平坦线段（信号丢失）
        noisy = clean.copy()
        noisy[500:700] = 0  # 200个点的平坦线段
        return noisy
    
    else:
        return clean


def test_sqi_on_simulated_signals():
    """测试SQI系统在模拟信号上的表现"""
    
    print("="*60)
    print("测试SQI自动标注系统")
    print("="*60 + "\n")
    
    # 创建评估器
    assessor = ECGQualityAssessor()
    
    # 测试不同类型的信号
    test_cases = [
        ('干净信号', 'clean', None),
        ('基线漂移', 'noisy', 'baseline'),
        ('高频噪声', 'noisy', 'hf_noise'),
        ('工频干扰', 'noisy', 'powerline'),
        ('信号饱和', 'noisy', 'saturation'),
        ('平坦线段', 'noisy', 'flat'),
    ]
    
    results = []
    
    for name, signal_type, noise_type in test_cases:
        # 生成信号
        if signal_type == 'clean':
            signal = generate_clean_ecg()
        else:
            signal = generate_noisy_ecg(noise_type=noise_type)
        
        # 评估质量
        label, sqi = assessor.assess_quality(signal, fs=360, return_details=True)
        
        # 记录结果
        result = {
            'name': name,
            'label': label,
            'is_acceptable': sqi['is_acceptable'],
            'snr': sqi['snr'],
            'baseline': sqi['baseline_wander'],
            'flat': sqi['flat_line'],
            'saturation': sqi['saturation'],
            'hf_noise': sqi['hf_noise'],
            'reasons': sqi['failure_reasons']
        }
        results.append(result)
        
        # 打印结果
        quality_str = "✓ 可接受" if label == 0 else "✗ 不可接受"
        print(f"{name:15} | {quality_str}")
        print(f"  SNR: {sqi['snr']:6.2f} dB")
        print(f"  基线漂移: {sqi['baseline_wander']:.3f}")
        print(f"  高频噪声: {sqi['hf_noise']:.3f}")
        
        if not sqi['is_acceptable']:
            print(f"  失败原因: {', '.join(sqi['failure_reasons'])}")
        print()
    
    # 统计
    n_correct = sum(1 for r in results if 
                   (r['name'] == '干净信号' and r['is_acceptable']) or
                   (r['name'] != '干净信号' and not r['is_acceptable']))
    
    print("="*60)
    print(f"测试结果: {n_correct}/{len(results)} 正确")
    print("="*60)
    
    return results


def visualize_sqi_assessment():
    """可视化SQI评估过程"""
    
    print("\n生成可视化...")
    
    # 创建评估器
    assessor = ECGQualityAssessor()
    
    # 生成不同质量的信号
    signals = {
        '干净信号': generate_clean_ecg(),
        '基线漂移': generate_noisy_ecg(noise_type='baseline'),
        '高频噪声': generate_noisy_ecg(noise_type='hf_noise'),
        '平坦线段': generate_noisy_ecg(noise_type='flat'),
    }
    
    # 创建图形
    fig, axes = plt.subplots(len(signals), 1, figsize=(12, 10))
    
    for idx, (name, signal) in enumerate(signals.items()):
        # 评估质量
        label, sqi = assessor.assess_quality(signal, fs=360, return_details=True)
        
        # 绘制信号
        axes[idx].plot(signal, linewidth=0.8)
        
        # 标题（包含质量判断）
        quality_str = "可接受" if label == 0 else "不可接受"
        color = 'green' if label == 0 else 'red'
        
        title = f"{name} - {quality_str} (SNR: {sqi['snr']:.1f} dB, "
        title += f"基线: {sqi['baseline_wander']:.2f}, "
        title += f"高频噪声: {sqi['hf_noise']:.2f})"
        
        axes[idx].set_title(title, color=color, fontweight='bold')
        axes[idx].set_ylabel('幅度')
        axes[idx].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('采样点')
    
    plt.tight_layout()
    
    # 保存图形
    output_path = 'results/sqi_test_visualization.png'
    import os
    os.makedirs('results', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"可视化已保存到: {output_path}")
    
    # 显示图形（如果在交互环境中）
    try:
        plt.show()
    except:
        pass


def main():
    """主测试函数"""
    
    print("\n" + "="*60)
    print("SQI自动标注系统测试")
    print("="*60 + "\n")
    
    # 测试1: 在模拟信号上测试
    print("测试1: 在模拟信号上测试SQI算法")
    print("-"*60)
    results = test_sqi_on_simulated_signals()
    
    # 测试2: 可视化
    print("\n测试2: 生成可视化")
    print("-"*60)
    try:
        visualize_sqi_assessment()
    except Exception as e:
        print(f"可视化失败（可能缺少matplotlib或显示环境）: {e}")
    
    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)
    print("\n提示：")
    print("1. 如果测试通过，说明SQI系统工作正常")
    print("2. 可以运行 'python prepare_data.py' 开始处理真实数据")
    print("3. 如果需要调整阈值，请编辑 utils/signal_quality_indices.py")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()



