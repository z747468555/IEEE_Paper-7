"""
CNN模型定义：用于ECG信号质量评估的1D卷积神经网络
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_CONFIG, TRAINING_CONFIG


def build_ecg_quality_cnn(input_shape=None):
    """
    构建论文中描述的1D CNN模型
    
    模型架构：
    - 输入: dECG信号 (1799, 1)
    - 4层卷积层: 滤波器数量 [8, 16, 32, 64]
    - 卷积核大小: 6×1
    - 激活函数: ELU
    - 池化: 最大池化 (3×1, stride=2)
    - Dropout: 0.3
    - 全连接层: [32, 16, 8, 1]
    - 输出: Sigmoid激活（二分类）
    
    Args:
        input_shape: 输入形状，默认使用配置文件中的设置
    
    Returns:
        model: Keras模型
    """
    if input_shape is None:
        input_shape = MODEL_CONFIG['input_shape']
    
    # 输入层
    inputs = keras.Input(shape=input_shape, name='dECG_input')
    
    x = inputs
    
    # 卷积层块
    conv_filters = MODEL_CONFIG['conv_filters']
    kernel_size = MODEL_CONFIG['kernel_size']
    pool_size = MODEL_CONFIG['pool_size']
    pool_stride = MODEL_CONFIG['pool_stride']
    dropout_rate = MODEL_CONFIG['dropout_rate']
    
    for i, filters in enumerate(conv_filters):
        # 卷积层
        x = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            activation='elu',  # ELU激活函数
            name=f'conv1d_{i+1}'
        )(x)
        
        # 最大池化
        x = layers.MaxPooling1D(
            pool_size=pool_size,
            strides=pool_stride,
            padding='same',
            name=f'maxpool_{i+1}'
        )(x)
        
        # Dropout
        x = layers.Dropout(dropout_rate, name=f'dropout_{i+1}')(x)
    
    # 展平
    x = layers.Flatten(name='flatten')(x)
    
    # 全连接层
    dense_units = MODEL_CONFIG['dense_units']
    
    for i, units in enumerate(dense_units[:-1]):
        x = layers.Dense(
            units=units,
            activation='elu',
            name=f'dense_{i+1}'
        )(x)
        x = layers.Dropout(dropout_rate, name=f'dropout_dense_{i+1}')(x)
    
    # 输出层
    outputs = layers.Dense(
        units=1,
        activation='sigmoid',
        name='output'
    )(x)
    
    # 创建模型
    model = models.Model(inputs=inputs, outputs=outputs, name='ECG_Quality_CNN')
    
    return model


def compile_model(model, learning_rate=None):
    """
    编译模型
    
    Args:
        model: Keras模型
        learning_rate: 学习率，默认使用配置文件中的设置
    
    Returns:
        model: 编译后的模型
    """
    if learning_rate is None:
        learning_rate = TRAINING_CONFIG['learning_rate']
    
    # 优化器
    optimizer = Adam(learning_rate=learning_rate)
    
    # 编译模型
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    return model


def create_model(input_shape=None, learning_rate=None, summary=True):
    """
    创建并编译完整的模型
    
    Args:
        input_shape: 输入形状
        learning_rate: 学习率
        summary: 是否打印模型摘要
    
    Returns:
        model: 编译后的Keras模型
    """
    # 构建模型
    model = build_ecg_quality_cnn(input_shape)
    
    # 编译模型
    model = compile_model(model, learning_rate)
    
    # 打印模型摘要
    if summary:
        print("\n" + "="*60)
        print("模型架构摘要")
        print("="*60)
        model.summary()
        print("="*60 + "\n")
        
        # 计算模型大小
        param_count = model.count_params()
        model_size_mb = param_count * 4 / (1024 * 1024)  # 假设每个参数4字节
        print(f"模型参数数量: {param_count:,}")
        print(f"估计模型大小: {model_size_mb:.2f} MB")
        print("="*60 + "\n")
    
    return model


def save_model(model, model_path):
    """
    保存模型
    
    Args:
        model: Keras模型
        model_path: 保存路径
    """
    model.save(model_path)
    print(f"模型已保存到: {model_path}")


def load_model(model_path):
    """
    加载模型
    
    Args:
        model_path: 模型路径
    
    Returns:
        model: 加载的Keras模型
    """
    model = keras.models.load_model(model_path)
    print(f"模型已从 {model_path} 加载")
    return model


class EarlyStoppingWithBestModel(keras.callbacks.Callback):
    """自定义早停回调，保存最佳模型"""
    
    def __init__(self, monitor='val_loss', patience=10, mode='min', 
                 restore_best_weights=True):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.best = None
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        
    def on_train_begin(self, logs=None):
        self.wait = 0
        if self.mode == 'min':
            self.best = float('inf')
        else:
            self.best = float('-inf')
    
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return
        
        if self.mode == 'min':
            if current < self.best:
                self.best = current
                self.wait = 0
                if self.restore_best_weights:
                    self.best_weights = self.model.get_weights()
            else:
                self.wait += 1
        else:
            if current > self.best:
                self.best = current
                self.wait = 0
                if self.restore_best_weights:
                    self.best_weights = self.model.get_weights()
            else:
                self.wait += 1
        
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            if self.restore_best_weights and self.best_weights is not None:
                print(f"\n恢复第 {epoch - self.patience + 1} 轮的最佳权重")
                self.model.set_weights(self.best_weights)
    
    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print(f"\n早停：在第 {self.stopped_epoch + 1} 轮停止训练")

