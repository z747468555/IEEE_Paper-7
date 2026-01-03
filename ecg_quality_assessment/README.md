# ECG信号质量评估系统

基于卷积神经网络和导数ECG信号的自动心电信号质量评估系统，用于降低可穿戴生命体征监测设备中的误报率。

## 📄 论文信息

**标题**: Automatic ECG signal quality assessment using convolutional neural networks and derivative ECG signal for false alarm reduction in wearable vital signs monitoring devices

**核心方法**:
- 使用一阶导数ECG信号（dECG）作为输入特征
- 轻量级一维卷积神经网络（1D CNN）
- 二分类任务：可接受（干净）vs 不可接受（含噪）

## 🏗️ 项目结构

```
ecg_quality_assessment/
├── config.py                   # 配置文件
├── requirements.txt            # 依赖包
├── README.md                   # 项目说明
├── main.py                     # 主程序
├── download_datasets.py        # 数据下载脚本
├── prepare_data.py             # 数据准备脚本
├── train.py                    # 训练脚本
├── evaluate.py                 # 评估脚本
├── utils/                      # 工具模块
│   ├── __init__.py
│   ├── signal_processing.py   # 信号处理（滤波、归一化、导数计算）
│   ├── data_loader.py          # 数据加载器
│   └── noise_generator.py      # 噪声生成器
├── model/                      # 模型模块
│   ├── __init__.py
│   └── cnn_model.py            # CNN模型定义
├── data/                       # 原始数据目录（需要下载）
│   ├── mit-bih-arrhythmia-database/
│   ├── ptb-diagnostic-ecg-database/
│   ├── incart-database/
│   ├── challenge-2011/
│   └── mit-bih-noise-stress-test-database/
├── processed_data/             # 处理后的数据
├── models/                     # 训练好的模型
├── results/                    # 评估结果
└── logs/                       # 训练日志
```

## 📊 数据集

### 训练数据
1. **MIT-BIH心律失常数据库** (48条记录, 2通道)
   - https://physionet.org/content/mitdb/1.0.0/

2. **PTB诊断ECG数据库** (549条记录, 15通道)
   - https://physionet.org/content/ptbdb/1.0.0/

3. **MIT-BIH噪声压力测试数据库** (噪声源)
   - https://physionet.org/content/nstdb/1.0.0/
   - 包含：基线漂移、肌电干扰、电极移动

### 测试数据（未见数据）
4. **INCART数据库** (75条记录, 12导联)
   - https://physionet.org/content/incartdb/1.0.0/

5. **PhysioNet/CinC Challenge 2011** (1000条记录, 12导联)
   - https://physionet.org/content/challenge-2011/1.0.0/

## 🚀 快速开始

### 1. 安装依赖

```bash
cd ecg_quality_assessment
pip install -r requirements.txt
```

### 2. 下载数据集

**选项A：自动下载（推荐）**
```bash
# 下载必需的数据集（用于快速测试）
python download_datasets.py --mode essential

# 或下载所有数据集（完整复现）
python download_datasets.py --mode all
```

**选项B：手动下载**
从PhysioNet手动下载数据集并解压到 `data/` 目录相应位置。

### 3. 准备训练数据

```bash
python prepare_data.py
```

这个脚本会：
- 加载并预处理ECG信号
- 生成含噪信号
- 划分训练集和测试集
- 保存处理后的数据到 `processed_data/`

### 4. 训练模型

```bash
python train.py
```

训练参数（在 `config.py` 中配置）：
- 批大小：10
- 训练轮数：100
- 学习率：0.001
- 优化器：Adam

训练完成后，模型会保存到 `models/` 目录。

### 5. 评估模型

```bash
# 评估测试集
python evaluate.py \
    --model models/ecg_quality_model_XXXXXX.h5 \
    --test_data processed_data/test_data.npz \
    --dataset_name "测试集"

# 评估未见数据（INCART）
python evaluate.py \
    --model models/ecg_quality_model_XXXXXX.h5 \
    --test_data processed_data/incart_test_data.npz \
    --dataset_name "INCART"

# 评估未见数据（PCCC2011）
python evaluate.py \
    --model models/ecg_quality_model_XXXXXX.h5 \
    --test_data processed_data/pccc2011_test_data.npz \
    --dataset_name "PCCC2011"
```

## 🔧 使用方法

### 查看完整流程
```bash
python main.py --pipeline
```

### 运行演示
```bash
python main.py --demo --model models/ecg_quality_model_XXXXXX.h5
```

### 评估单个ECG文件
```bash
# WFDB格式
python main.py --assess \
    --model models/ecg_quality_model_XXXXXX.h5 \
    --input data/mitdb/100 \
    --format wfdb \
    --output results/assessment_result.json

# CSV格式
python main.py --assess \
    --model models/ecg_quality_model_XXXXXX.h5 \
    --input data/ecg_signal.csv \
    --format csv
```

## 🧠 模型架构

### 输入
- **dECG信号**: (1799, 1) - 原信号计算一阶导数后

### 网络结构
1. **卷积层**: 4层
   - 滤波器数量: [8, 16, 32, 64]
   - 卷积核大小: 6×1
   - 激活函数: ELU

2. **池化层**: 最大池化
   - 池化大小: 3×1
   - 步长: 2

3. **Dropout**: 0.3

4. **全连接层**: 4层
   - 神经元数量: [32, 16, 8, 1]

5. **输出层**: Sigmoid激活（二分类）

### 模型大小
- 参数量: ~30,000
- 模型大小: ~2,989 KB

## 📈 预期结果

### PhysioNet Challenge 2011
- **准确率 (AC)**: 97.59%
- **敏感度 (SE)**: 98.78%
- **特异度 (SP)**: 89.23%

### INCART
- **敏感度 (SE)**: ~100%
- **特异度 (SP)**: 88.66%

## ⚙️ 信号处理流程

1. **重采样**: 统一到360 Hz
2. **分段**: 5秒片段（1800个采样点）
3. **滤波**:
   - 高通：二阶切比雪夫滤波器（0.8 Hz）
   - 低通：四阶切比雪夫滤波器（40 Hz）
4. **归一化**: 
   ```
   y[n] = (x[n] - μ_x) / max(|x[n] - μ_x|)
   ```
5. **导数计算**:
   ```
   d[n] = y[n+1] - y[n]
   ```

## 📝 噪声生成

### 含噪信号生成
```
NECG = NFECG + a × X_n
```
- 衰减因子 a: [0.3, 0.6, 0.9]
- 噪声类型: 基线漂移、肌电干扰、电极移动

### 数据增强
- 微小随机噪声: 衰减因子 [0.001, 0.002, 0.003, 0.004]

## 🔍 配置说明

编辑 `config.py` 来自定义参数：

```python
# 数据参数
TARGET_SAMPLING_RATE = 360  # 目标采样率（Hz）
SEGMENT_DURATION = 5        # 信号分段长度（秒）

# 滤波器参数
HIGHPASS_CUTOFF = 0.8       # 高通滤波器截止频率（Hz）
LOWPASS_CUTOFF = 40         # 低通滤波器截止频率（Hz）

# 模型参数
MODEL_CONFIG = {
    'conv_filters': [8, 16, 32, 64],
    'kernel_size': 6,
    'dropout_rate': 0.3,
    # ...
}

# 训练参数
TRAINING_CONFIG = {
    'batch_size': 10,
    'epochs': 100,
    'learning_rate': 0.001,
    # ...
}
```

## 📊 评估指标

- **准确率 (Accuracy)**: 整体分类准确度
- **敏感度 (Sensitivity)**: 检测含噪信号的能力（召回率）
- **特异度 (Specificity)**: 检测干净信号的能力
- **精确率 (Precision)**: 预测为含噪的准确度
- **F1分数**: 精确率和召回率的调和平均

## 🐛 故障排除

### 问题1: 数据下载失败
```bash
# 检查网络连接
# 尝试手动从PhysioNet下载

# 或使用代理
export HTTP_PROXY=http://your_proxy:port
export HTTPS_PROXY=http://your_proxy:port
```

### 问题2: 内存不足
```python
# 在 prepare_data.py 中调整最大片段数
max_segments = 2000  # 减少这个值
```

### 问题3: GPU内存不足
```python
# 在 train.py 中减小批大小
batch_size = 5  # 默认是10
```

## 📚 参考文献

Mondal, S., et al. (2025). "Automatic ECG signal quality assessment using convolutional neural networks and derivative ECG signal for false alarm reduction in wearable vital signs monitoring devices."

## 📧 联系方式

如有问题或建议，请提交Issue。

## 📄 许可证

本项目仅用于学术研究和学习目的。

---

**注意**: 
1. 本项目为论文复现实现，实际性能可能与论文有所差异
2. 测试数据的质量标注需要根据实际情况进行调整
3. 建议在真实临床应用前进行充分验证

