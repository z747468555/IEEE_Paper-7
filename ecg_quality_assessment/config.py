"""
配置文件：ECG信号质量评估实验参数
"""

# 数据参数
TARGET_SAMPLING_RATE = 360  # 目标采样率（Hz）
SEGMENT_DURATION = 5  # 信号分段长度（秒）
SEGMENT_LENGTH = TARGET_SAMPLING_RATE * SEGMENT_DURATION  # 1800个采样点

# 滤波器参数
HIGHPASS_CUTOFF = 0.8  # 高通滤波器截止频率（Hz）
LOWPASS_CUTOFF = 40  # 低通滤波器截止频率（Hz）
HIGHPASS_ORDER = 2  # 高通滤波器阶数
LOWPASS_ORDER = 4  # 低通滤波器阶数

# 噪声生成参数
NOISE_ATTENUATION_FACTORS = [0.3, 0.6, 0.9]  # 噪声衰减因子
AUGMENTATION_FACTORS = [0.001, 0.002, 0.003, 0.004]  # 数据增强噪声因子

# 标签设置
LABEL_CLEAN = 0  # 干净信号标签
LABEL_NOISY = 1  # 含噪信号标签

# 模型参数
MODEL_CONFIG = {
    'input_shape': (SEGMENT_LENGTH - 1, 1),  # dECG比原信号少一个采样点
    'conv_filters': [8, 16, 32, 64],  # 卷积层滤波器数量
    'kernel_size': 6,  # 卷积核大小
    'pool_size': 3,  # 池化大小
    'pool_stride': 2,  # 池化步长
    'dropout_rate': 0.3,  # Dropout比率
    'dense_units': [32, 16, 8, 1],  # 全连接层神经元数量
}

# 训练参数
TRAINING_CONFIG = {
    'batch_size': 10,
    'epochs': 100,
    'learning_rate': 0.001,
    'validation_split': 0.2,
    'optimizer': 'adam',
    'loss': 'binary_crossentropy',
}

# 数据库路径（需要用户根据实际情况配置）
DATA_PATHS = {
    'mitdb': './data/raw/mitdb',
    'ptbdb': './data/raw/ptbdb',
    'incart': './data/raw/incartdb/files',
    'pccc2011': './data/raw/challenge-2011/set-b',  # 使用Set B作为测试集
    'nstdb': './data/raw/nstdb',
}

# 输出路径
OUTPUT_PATHS = {
    'processed_data': './processed_data',
    'models': './models',
    'results': './results',
    'logs': './logs',
}

# 数据集划分（记录级别）
TRAIN_TEST_SPLIT = {
    'test_size': 0.2,
    'random_state': 42,
}

