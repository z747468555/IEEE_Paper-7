"""
主程序：ECG信号质量评估系统
提供完整的端到端流程
"""

import os
import sys
import argparse
import numpy as np
from datetime import datetime

from config import OUTPUT_PATHS
from model.cnn_model import load_model
from utils.signal_processing import preprocess_ecg_signal, compute_derivative


class ECGQualityAssessment:
    """ECG质量评估系统"""
    
    def __init__(self, model_path):
        """
        初始化评估系统
        
        Args:
            model_path: 训练好的模型路径
        """
        print("="*60)
        print("ECG信号质量评估系统")
        print("="*60)
        
        # 加载模型
        print("\n加载模型...")
        self.model = load_model(model_path)
        print("模型加载成功！\n")
    
    def assess_signal(self, ecg_signal, sampling_rate):
        """
        评估单个ECG信号的质量
        
        Args:
            ecg_signal: ECG信号数据
            sampling_rate: 采样率
        
        Returns:
            quality: 质量评估结果（'可接受' 或 '不可接受'）
            confidence: 置信度（0-1）
        """
        # 预处理信号
        processed = preprocess_ecg_signal(ecg_signal, sampling_rate)
        
        # 计算导数
        derivative = compute_derivative(processed)
        
        # 重塑为模型输入格式
        X = derivative.reshape(1, -1, 1)
        
        # 预测
        prediction = self.model.predict(X, verbose=0)[0][0]
        
        # 解释结果
        if prediction < 0.5:
            quality = "可接受"
            confidence = 1 - prediction
        else:
            quality = "不可接受"
            confidence = prediction
        
        return quality, float(confidence)
    
    def assess_signal_segments(self, ecg_signal, sampling_rate):
        """
        评估ECG信号的多个片段
        
        Args:
            ecg_signal: ECG信号数据
            sampling_rate: 采样率
        
        Returns:
            results: 每个片段的评估结果列表
        """
        from utils.signal_processing import preprocess_and_segment
        
        # 预处理并分割
        segments = preprocess_and_segment(ecg_signal, sampling_rate, 
                                         return_derivative=True)
        
        if len(segments) == 0:
            return []
        
        # 批量预测
        X = np.array(segments).reshape(len(segments), -1, 1)
        predictions = self.model.predict(X, verbose=0).flatten()
        
        # 整理结果
        results = []
        for i, pred in enumerate(predictions):
            if pred < 0.5:
                quality = "可接受"
                confidence = 1 - pred
            else:
                quality = "不可接受"
                confidence = pred
            
            results.append({
                'segment_index': i,
                'quality': quality,
                'confidence': float(confidence),
                'prediction_score': float(pred)
            })
        
        return results
    
    def assess_file(self, file_path, file_format='wfdb'):
        """
        评估文件中的ECG信号
        
        Args:
            file_path: 文件路径
            file_format: 文件格式（'wfdb', 'csv', 'txt'）
        
        Returns:
            results: 评估结果
        """
        # 加载文件
        if file_format == 'wfdb':
            import wfdb
            record = wfdb.rdrecord(file_path)
            signal = record.p_signal[:, 0]  # 取第一个导联
            fs = record.fs
        elif file_format == 'csv':
            import pandas as pd
            data = pd.read_csv(file_path)
            signal = data.iloc[:, 0].values
            fs = 360  # 默认采样率，应该从文件或参数获取
        elif file_format == 'txt':
            signal = np.loadtxt(file_path)
            fs = 360  # 默认采样率
        else:
            raise ValueError(f"不支持的文件格式: {file_format}")
        
        # 评估信号
        results = self.assess_signal_segments(signal, fs)
        
        return results


def run_pipeline():
    """运行完整的流程"""
    
    print("\n" + "="*60)
    print("ECG信号质量评估 - 完整流程")
    print("="*60 + "\n")
    
    # 步骤1：下载数据
    print("步骤1：下载数据集")
    print("-" * 60)
    print("运行命令: python download_datasets.py --mode essential")
    print("或手动从PhysioNet下载数据集")
    print()
    
    # 步骤2：准备数据
    print("步骤2：准备训练数据")
    print("-" * 60)
    print("运行命令: python prepare_data.py")
    print()
    
    # 步骤3：训练模型
    print("步骤3：训练模型")
    print("-" * 60)
    print("运行命令: python train.py")
    print()
    
    # 步骤4：评估模型
    print("步骤4：评估模型")
    print("-" * 60)
    print("运行命令: python evaluate.py --model models/ecg_quality_model_*.h5 --test_data processed_data/test_data.npz")
    print()
    
    print("="*60 + "\n")


def demo_assessment(model_path):
    """演示评估功能"""
    
    # 创建评估系统
    system = ECGQualityAssessment(model_path)
    
    # 生成模拟ECG信号进行演示
    print("\n" + "="*60)
    print("演示：评估模拟ECG信号")
    print("="*60 + "\n")
    
    # 模拟干净的ECG信号（简单的正弦波）
    t = np.linspace(0, 5, 1800)  # 5秒，360Hz采样率
    clean_signal = np.sin(2 * np.pi * 1.2 * t)  # 1.2 Hz，模拟心率72 bpm
    
    print("1. 评估干净信号...")
    quality, confidence = system.assess_signal(clean_signal, 360)
    print(f"   结果: {quality}")
    print(f"   置信度: {confidence:.2%}\n")
    
    # 模拟含噪信号
    noise = np.random.randn(1800) * 0.5
    noisy_signal = clean_signal + noise
    
    print("2. 评估含噪信号...")
    quality, confidence = system.assess_signal(noisy_signal, 360)
    print(f"   结果: {quality}")
    print(f"   置信度: {confidence:.2%}\n")
    
    print("="*60 + "\n")


def main():
    """主函数"""
    
    parser = argparse.ArgumentParser(
        description='ECG信号质量评估系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 查看完整流程
  python main.py --pipeline
  
  # 评估单个文件
  python main.py --assess --model models/model.h5 --input data/record.dat --format wfdb
  
  # 运行演示
  python main.py --demo --model models/model.h5
        """
    )
    
    parser.add_argument('--pipeline', action='store_true',
                       help='显示完整的流程步骤')
    parser.add_argument('--demo', action='store_true',
                       help='运行演示评估')
    parser.add_argument('--assess', action='store_true',
                       help='评估ECG信号文件')
    parser.add_argument('--model', type=str,
                       help='模型文件路径')
    parser.add_argument('--input', type=str,
                       help='输入文件路径')
    parser.add_argument('--format', type=str, 
                       choices=['wfdb', 'csv', 'txt'],
                       default='wfdb',
                       help='输入文件格式')
    parser.add_argument('--output', type=str,
                       help='输出结果文件路径（可选）')
    
    args = parser.parse_args()
    
    # 如果没有参数，显示帮助
    if len(sys.argv) == 1:
        parser.print_help()
        print("\n提示：首次使用请运行 'python main.py --pipeline' 查看完整流程")
        return
    
    # 显示流程
    if args.pipeline:
        run_pipeline()
        return
    
    # 运行演示
    if args.demo:
        if not args.model:
            print("错误：请指定模型路径 --model")
            return
        if not os.path.exists(args.model):
            print(f"错误：模型文件不存在: {args.model}")
            return
        demo_assessment(args.model)
        return
    
    # 评估文件
    if args.assess:
        if not args.model:
            print("错误：请指定模型路径 --model")
            return
        if not args.input:
            print("错误：请指定输入文件 --input")
            return
        
        if not os.path.exists(args.model):
            print(f"错误：模型文件不存在: {args.model}")
            return
        if not os.path.exists(args.input):
            print(f"错误：输入文件不存在: {args.input}")
            return
        
        # 创建评估系统
        system = ECGQualityAssessment(args.model)
        
        # 评估文件
        print(f"\n评估文件: {args.input}")
        print("="*60)
        
        results = system.assess_file(args.input, args.format)
        
        # 显示结果
        print(f"\n找到 {len(results)} 个信号片段\n")
        
        acceptable_count = sum(1 for r in results if r['quality'] == '可接受')
        unacceptable_count = len(results) - acceptable_count
        
        print(f"可接受片段: {acceptable_count} ({acceptable_count/len(results)*100:.1f}%)")
        print(f"不可接受片段: {unacceptable_count} ({unacceptable_count/len(results)*100:.1f}%)")
        
        print("\n详细结果:")
        print("-"*60)
        for r in results[:10]:  # 只显示前10个
            print(f"片段 {r['segment_index']}: {r['quality']} "
                  f"(置信度: {r['confidence']:.2%})")
        
        if len(results) > 10:
            print(f"... (省略 {len(results)-10} 个片段)")
        
        # 保存结果
        if args.output:
            import json
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n结果已保存到: {args.output}")
        
        print("="*60 + "\n")


if __name__ == '__main__':
    main()

