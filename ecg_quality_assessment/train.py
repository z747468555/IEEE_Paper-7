"""
训练脚本：训练ECG质量评估模型
"""

import os
import numpy as np
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import platform

# 配置matplotlib中文字体显示
def setup_chinese_font():
    """配置matplotlib以正确显示中文"""
    system = platform.system()
    
    if system == 'Windows':
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi']
    elif system == 'Darwin':  # macOS
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC']
    else:  # Linux
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Droid Sans Fallback', 'DejaVu Sans']
    
    plt.rcParams['axes.unicode_minus'] = False
    print("已配置中文字体显示")

setup_chinese_font()

from config import (
    TRAINING_CONFIG, OUTPUT_PATHS, TRAIN_TEST_SPLIT,
    LABEL_CLEAN, LABEL_NOISY
)
from model.cnn_model import (
    create_model, save_model, 
    EarlyStoppingWithBestModel
)
from utils.signal_processing import compute_derivative


def prepare_training_data(X, y, validation_split=None):
    """
    准备训练数据
    
    Args:
        X: 信号数据 (n_samples, signal_length)
        y: 标签
        validation_split: 验证集比例
    
    Returns:
        X_train, X_val, y_train, y_val
    """
    print("\n准备训练数据...")
    
    # 计算导数
    print("计算信号导数...")
    X_derivative = np.array([compute_derivative(signal) for signal in X])
    
    # 重塑数据为 (n_samples, signal_length, 1)
    X_derivative = X_derivative.reshape(X_derivative.shape[0], 
                                       X_derivative.shape[1], 1)
    
    # 划分训练集和验证集
    if validation_split is None:
        validation_split = TRAINING_CONFIG['validation_split']
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_derivative, y, 
        test_size=validation_split,
        random_state=TRAIN_TEST_SPLIT['random_state'],
        stratify=y
    )
    
    print(f"训练集大小: {len(X_train)} (干净: {np.sum(y_train==LABEL_CLEAN)}, "
          f"含噪: {np.sum(y_train==LABEL_NOISY)})")
    print(f"验证集大小: {len(X_val)} (干净: {np.sum(y_val==LABEL_CLEAN)}, "
          f"含噪: {np.sum(y_val==LABEL_NOISY)})")
    
    return X_train, X_val, y_train, y_val


def create_callbacks(model_save_path, log_dir):
    """
    创建训练回调
    
    Args:
        model_save_path: 模型保存路径
        log_dir: 日志目录
    
    Returns:
        callbacks: 回调列表
    """
    callbacks = [
        # ModelCheckpoint：保存最佳模型
        keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        
        # 早停
        EarlyStoppingWithBestModel(
            monitor='val_loss',
            patience=15,
            mode='min',
            restore_best_weights=True
        ),
        
        # 学习率衰减
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard
        keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        ),
        
        # CSV日志
        keras.callbacks.CSVLogger(
            os.path.join(log_dir, 'training_log.csv')
        )
    ]
    
    return callbacks


def plot_training_history(history, save_path):
    """
    绘制训练历史曲线
    
    Args:
        history: 训练历史对象
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 损失曲线
    axes[0, 0].plot(history.history['loss'], label='训练损失')
    axes[0, 0].plot(history.history['val_loss'], label='验证损失')
    axes[0, 0].set_title('损失曲线')
    axes[0, 0].set_xlabel('轮次')
    axes[0, 0].set_ylabel('损失')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 准确率曲线
    axes[0, 1].plot(history.history['accuracy'], label='训练准确率')
    axes[0, 1].plot(history.history['val_accuracy'], label='验证准确率')
    axes[0, 1].set_title('准确率曲线')
    axes[0, 1].set_xlabel('轮次')
    axes[0, 1].set_ylabel('准确率')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 精确率曲线
    axes[1, 0].plot(history.history['precision'], label='训练精确率')
    axes[1, 0].plot(history.history['val_precision'], label='验证精确率')
    axes[1, 0].set_title('精确率曲线')
    axes[1, 0].set_xlabel('轮次')
    axes[1, 0].set_ylabel('精确率')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 召回率曲线
    axes[1, 1].plot(history.history['recall'], label='训练召回率')
    axes[1, 1].plot(history.history['val_recall'], label='验证召回率')
    axes[1, 1].set_title('召回率曲线')
    axes[1, 1].set_xlabel('轮次')
    axes[1, 1].set_ylabel('召回率')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"训练历史曲线已保存到: {save_path}")


def train_model(X_train, y_train, X_val, y_val, 
                batch_size=None, epochs=None, learning_rate=None):
    """
    训练模型
    
    Args:
        X_train: 训练数据
        y_train: 训练标签
        X_val: 验证数据
        y_val: 验证标签
        batch_size: 批大小
        epochs: 训练轮数
        learning_rate: 学习率
    
    Returns:
        model: 训练好的模型
        history: 训练历史
    """
    # 使用配置文件中的默认值
    if batch_size is None:
        batch_size = TRAINING_CONFIG['batch_size']
    if epochs is None:
        epochs = TRAINING_CONFIG['epochs']
    
    # 创建输出目录
    os.makedirs(OUTPUT_PATHS['models'], exist_ok=True)
    os.makedirs(OUTPUT_PATHS['logs'], exist_ok=True)
    os.makedirs(OUTPUT_PATHS['results'], exist_ok=True)
    
    # 创建时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 模型保存路径
    model_save_path = os.path.join(OUTPUT_PATHS['models'], 
                                   f'ecg_quality_model_{timestamp}.h5')
    
    # 日志目录
    log_dir = os.path.join(OUTPUT_PATHS['logs'], f'training_{timestamp}')
    
    # 创建模型
    print("\n" + "="*60)
    print("创建模型...")
    print("="*60)
    model = create_model(
        input_shape=X_train.shape[1:],
        learning_rate=learning_rate,
        summary=True
    )
    
    # 创建回调
    callbacks = create_callbacks(model_save_path, log_dir)
    
    # 训练模型
    print("\n" + "="*60)
    print("开始训练...")
    print("="*60)
    print(f"批大小: {batch_size}")
    print(f"训练轮数: {epochs}")
    print(f"学习率: {learning_rate if learning_rate else TRAINING_CONFIG['learning_rate']}")
    print("="*60 + "\n")
    
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # 保存训练历史
    history_path = os.path.join(OUTPUT_PATHS['results'], 
                                f'training_history_{timestamp}.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    
    print(f"\n训练历史已保存到: {history_path}")
    
    # 绘制训练曲线
    plot_path = os.path.join(OUTPUT_PATHS['results'], 
                            f'training_curves_{timestamp}.png')
    plot_training_history(history, plot_path)
    
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print(f"最佳模型保存路径: {model_save_path}")
    
    return model, history


def main():
    """主函数：执行完整的训练流程"""
    
    # 检查是否有预处理好的数据
    processed_data_path = os.path.join(OUTPUT_PATHS['processed_data'], 
                                       'training_data.npz')
    
    if not os.path.exists(processed_data_path):
        print("错误：未找到预处理数据！")
        print(f"请先运行数据准备脚本生成训练数据: {processed_data_path}")
        print("或运行: python prepare_data.py")
        return
    
    # 加载数据
    print("="*60)
    print("加载训练数据...")
    print("="*60)
    data = np.load(processed_data_path, allow_pickle=True)
    X = data['X']
    y = data['y']
    
    print(f"加载的数据形状: X={X.shape}, y={y.shape}")
    print(f"干净信号: {np.sum(y==LABEL_CLEAN)}, 含噪信号: {np.sum(y==LABEL_NOISY)}")
    
    # 准备训练数据
    X_train, X_val, y_train, y_val = prepare_training_data(X, y)
    
    # 训练模型
    model, history = train_model(X_train, y_train, X_val, y_val)
    
    print("\n训练流程全部完成！")


if __name__ == '__main__':
    # 设置GPU内存增长（避免占用所有GPU内存）
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"找到 {len(gpus)} 个GPU，已启用内存增长模式")
        except RuntimeError as e:
            print(f"GPU配置错误: {e}")
    
    main()










