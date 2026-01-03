"""
评估脚本：评估ECG质量评估模型性能
"""

import os
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from config import OUTPUT_PATHS, LABEL_CLEAN, LABEL_NOISY
from model.cnn_model import load_model
from utils.signal_processing import compute_derivative


def calculate_metrics(y_true, y_pred):
    """
    计算评估指标
    
    根据论文，主要指标为：
    - 准确率 (Accuracy, AC)
    - 敏感度 (Sensitivity, SE) - 即召回率，检测含噪信号的能力
    - 特异度 (Specificity, SP) - 检测干净信号的能力
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
    
    Returns:
        metrics: 指标字典
    """
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 对于二分类：
    # TN (True Negative): 正确预测为干净
    # FP (False Positive): 错误预测为含噪（假阳性）
    # FN (False Negative): 错误预测为干净（假阴性）
    # TP (True Positive): 正确预测为含噪
    
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]
    
    # 计算指标
    accuracy = accuracy_score(y_true, y_pred)
    
    # 敏感度 (Sensitivity) = TP / (TP + FN)
    # 检测含噪信号的能力（召回率）
    sensitivity = recall_score(y_true, y_pred, pos_label=LABEL_NOISY)
    
    # 特异度 (Specificity) = TN / (TN + FP)
    # 检测干净信号的能力
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    
    # 精确率
    precision = precision_score(y_true, y_pred, pos_label=LABEL_NOISY)
    
    # F1分数
    f1 = f1_score(y_true, y_pred, pos_label=LABEL_NOISY)
    
    metrics = {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1,
        'confusion_matrix': cm,
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN
    }
    
    return metrics


def print_metrics(metrics, dataset_name="测试集"):
    """
    打印评估指标
    
    Args:
        metrics: 指标字典
        dataset_name: 数据集名称
    """
    print("\n" + "="*60)
    print(f"{dataset_name} 评估结果")
    print("="*60)
    print(f"准确率 (Accuracy):     {metrics['accuracy']*100:.2f}%")
    print(f"敏感度 (Sensitivity):  {metrics['sensitivity']*100:.2f}%")
    print(f"特异度 (Specificity):  {metrics['specificity']*100:.2f}%")
    print(f"精确率 (Precision):    {metrics['precision']*100:.2f}%")
    print(f"F1分数:                {metrics['f1_score']:.4f}")
    print("-"*60)
    print("混淆矩阵:")
    print(f"  真实干净 -> 预测干净 (TN): {metrics['TN']}")
    print(f"  真实干净 -> 预测含噪 (FP): {metrics['FP']}")
    print(f"  真实含噪 -> 预测干净 (FN): {metrics['FN']}")
    print(f"  真实含噪 -> 预测含噪 (TP): {metrics['TP']}")
    print("="*60 + "\n")


def plot_confusion_matrix(cm, save_path, dataset_name="测试集"):
    """
    绘制混淆矩阵
    
    Args:
        cm: 混淆矩阵
        save_path: 保存路径
        dataset_name: 数据集名称
    """
    plt.figure(figsize=(8, 6))
    
    # 使用seaborn绘制热图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['干净', '含噪'],
                yticklabels=['干净', '含噪'])
    
    plt.title(f'{dataset_name} - 混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"混淆矩阵已保存到: {save_path}")


def plot_roc_curve(y_true, y_pred_proba, save_path, dataset_name="测试集"):
    """
    绘制ROC曲线
    
    Args:
        y_true: 真实标签
        y_pred_proba: 预测概率
        save_path: 保存路径
        dataset_name: 数据集名称
    """
    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # 绘制
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC曲线 (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='随机猜测')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率 (False Positive Rate)')
    plt.ylabel('真阳性率 (True Positive Rate)')
    plt.title(f'{dataset_name} - ROC曲线')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ROC曲线已保存到: {save_path}")


def evaluate_model(model, X_test, y_test, dataset_name="测试集", 
                   save_results=True):
    """
    评估模型
    
    Args:
        model: 训练好的模型
        X_test: 测试数据
        y_test: 测试标签
        dataset_name: 数据集名称
        save_results: 是否保存结果
    
    Returns:
        metrics: 评估指标字典
    """
    print(f"\n正在评估 {dataset_name}...")
    
    # 计算导数
    print("计算信号导数...")
    X_test_derivative = np.array([compute_derivative(signal) for signal in X_test])
    X_test_derivative = X_test_derivative.reshape(X_test_derivative.shape[0], 
                                                  X_test_derivative.shape[1], 1)
    
    # 预测
    print("进行预测...")
    y_pred_proba = model.predict(X_test_derivative, verbose=0).flatten()
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # 计算指标
    metrics = calculate_metrics(y_test, y_pred)
    
    # 打印结果
    print_metrics(metrics, dataset_name)
    
    # 保存结果
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(OUTPUT_PATHS['results'], 
                                   f'evaluation_{timestamp}')
        os.makedirs(results_dir, exist_ok=True)
        
        # 保存混淆矩阵图
        cm_path = os.path.join(results_dir, 
                              f'confusion_matrix_{dataset_name}.png')
        plot_confusion_matrix(metrics['confusion_matrix'], cm_path, dataset_name)
        
        # 保存ROC曲线
        roc_path = os.path.join(results_dir, f'roc_curve_{dataset_name}.png')
        plot_roc_curve(y_test, y_pred_proba, roc_path, dataset_name)
        
        # 保存详细报告
        report = classification_report(y_test, y_pred, 
                                      target_names=['干净', '含噪'])
        report_path = os.path.join(results_dir, 
                                   f'classification_report_{dataset_name}.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"{dataset_name} 分类报告\n")
            f.write("="*60 + "\n")
            f.write(report)
            f.write("\n\n详细指标:\n")
            f.write(f"准确率 (Accuracy):     {metrics['accuracy']*100:.2f}%\n")
            f.write(f"敏感度 (Sensitivity):  {metrics['sensitivity']*100:.2f}%\n")
            f.write(f"特异度 (Specificity):  {metrics['specificity']*100:.2f}%\n")
            f.write(f"精确率 (Precision):    {metrics['precision']*100:.2f}%\n")
            f.write(f"F1分数:                {metrics['f1_score']:.4f}\n")
        
        print(f"评估报告已保存到: {report_path}")
    
    return metrics


def main():
    """主函数：执行评估流程"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='评估ECG质量评估模型')
    parser.add_argument('--model', type=str, required=True,
                       help='模型文件路径')
    parser.add_argument('--test_data', type=str, required=True,
                       help='测试数据文件路径（.npz格式）')
    parser.add_argument('--dataset_name', type=str, default='测试集',
                       help='数据集名称')
    
    args = parser.parse_args()
    
    # 加载模型
    print("="*60)
    print("加载模型...")
    print("="*60)
    model = load_model(args.model)
    
    # 加载测试数据
    print("\n" + "="*60)
    print("加载测试数据...")
    print("="*60)
    data = np.load(args.test_data, allow_pickle=True)
    X_test = data['X']
    y_test = data['y']
    
    print(f"测试数据形状: X={X_test.shape}, y={y_test.shape}")
    print(f"干净信号: {np.sum(y_test==LABEL_CLEAN)}, "
          f"含噪信号: {np.sum(y_test==LABEL_NOISY)}")
    
    # 评估模型
    metrics = evaluate_model(model, X_test, y_test, args.dataset_name)
    
    print("\n评估完成！")


if __name__ == '__main__':
    main()

