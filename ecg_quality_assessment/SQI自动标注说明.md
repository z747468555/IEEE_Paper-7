# 基于SQI的自动信号质量标注系统

## 🎯 解决方案概述

针对原论文对INCART数据库进行手动筛选的问题，我们实现了一个**自动化的信号质量评估系统**，基于多个**信号质量指标（Signal Quality Indices, SQI）**来自动标注ECG信号质量。

---

## 📊 实现的SQI指标

### 1. **信噪比（SNR）**
- **原理**: 高质量信号应有较高的SNR
- **计算方法**: 信号功率 / 噪声功率（dB）
- **阈值**: SNR > 5 dB 认为可接受

### 2. **基线漂移检测**
- **原理**: 过大的基线漂移会影响信号质量
- **计算方法**: 低频成分功率占比
- **阈值**: 基线功率 < 30% 认为可接受

### 3. **平坦线段检测**
- **原理**: 检测信号丢失或电极脱落
- **计算方法**: 局部方差 < 阈值的窗口占比
- **阈值**: 平坦线段 < 20% 认为可接受

### 4. **饱和检测**
- **原理**: 检测信号超出量程
- **计算方法**: |信号| > 0.95 的点占比
- **阈值**: 饱和点 < 5% 认为可接受

### 5. **工频干扰检测（50Hz/60Hz）**
- **原理**: 检测电力线干扰
- **计算方法**: 50Hz附近功率占比
- **阈值**: 工频功率 < 10% 认为可接受

### 6. **高频噪声检测**
- **原理**: 检测肌电干扰等高频噪声
- **计算方法**: >40Hz 成分功率占比
- **阈值**: 高频功率 < 30% 认为可接受

### 7. **统计特征检测**
- **峰度（Kurtosis）**: 检测信号分布异常
- **偏度（Skewness）**: 检测信号不对称性
- **标准差**: 检测信号变化程度

---

## 🔧 如何使用

### 方式1：在数据准备流程中自动使用

运行 `prepare_data.py` 时，系统会**自动**使用SQI标注INCART数据：

```bash
cd ecg_quality_assessment
python prepare_data.py
```

**输出示例**:
```
处理INCART数据库（使用自动SQI标注）...
加载INCART: 100%|████████| 75/75

使用SQI算法自动标注INCART数据质量
SQI评估: 100%|████████| 10800/10800

自动标注完成:
  可接受（干净）: 8856 (82.0%)
  不可接受（含噪）: 1944 (18.0%)

不可接受信号的主要原因:
  - 高频噪声过大: 1245次
  - 基线漂移过大: 678次
  - SNR过低: 421次
```

### 方式2：单独使用SQI模块

```python
from utils.signal_quality_indices import ECGQualityAssessor, auto_label_incart_quality
import numpy as np

# 方法A: 评估单个信号
assessor = ECGQualityAssessor()
ecg_signal = np.random.randn(1800)  # 示例信号

quality_label, sqi_details = assessor.assess_quality(
    ecg_signal, 
    fs=360, 
    return_details=True
)

print(f"质量标签: {'可接受' if quality_label == 0 else '不可接受'}")
print(f"SNR: {sqi_details['snr']:.2f} dB")
print(f"基线漂移: {sqi_details['baseline_wander']:.3f}")
print(f"失败原因: {sqi_details['failure_reasons']}")

# 方法B: 批量评估多个信号
segments = [np.random.randn(1800) for _ in range(100)]
labels, sqi_results = auto_label_incart_quality(segments, fs=360, verbose=True)

print(f"可接受: {np.sum(labels==0)}, 不可接受: {np.sum(labels==1)}")
```

---

## ⚙️ 自定义阈值

如果默认阈值不适合您的数据，可以自定义：

```python
from utils.signal_quality_indices import ECGQualityAssessor

# 自定义阈值
custom_thresholds = {
    'snr_min': 8.0,           # 更严格的SNR要求
    'baseline_max': 0.2,       # 更严格的基线要求
    'flat_max': 0.15,          # 更少的平坦线段
    'hf_noise_max': 0.25,      # 更少的高频噪声
}

assessor = ECGQualityAssessor(thresholds=custom_thresholds)
quality_label = assessor.assess_quality(ecg_signal, fs=360)
```

---

## 📈 质量评估流程图

```
ECG信号输入
    ↓
预处理（滤波、归一化）
    ↓
计算所有SQI指标
    ├── SNR
    ├── 基线漂移
    ├── 平坦线段
    ├── 饱和检测
    ├── 工频干扰
    ├── 高频噪声
    └── 统计特征
    ↓
应用质量判断规则
    ↓
输出: 0（可接受）或 1（不可接受）
```

---

## 📊 SQI评估结果示例

运行后会生成 `incart_sqi_summary.json`:

```json
{
  "total_segments": 10800,
  "acceptable": 8856,
  "unacceptable": 1944,
  "acceptable_percentage": 82.0
}
```

---

## 🔍 与模拟标签的对比

### 旧方法（模拟标签）
```python
# 简单地将前50%标记为干净，后50%标记为含噪
half = len(segments) // 2
labels = [0] * half + [1] * half
```

**问题**:
- ❌ 不反映真实信号质量
- ❌ 标签分布固定（50%-50%）
- ❌ 无法捕捉实际的噪声模式

### 新方法（SQI自动标注）
```python
# 基于多个质量指标综合判断
labels, sqi_results = auto_label_incart_quality(segments)
```

**优势**:
- ✅ 反映真实信号质量
- ✅ 标签分布根据实际数据
- ✅ 可解释（提供详细的失败原因）
- ✅ 可调整阈值适应不同场景

---

## 🎓 SQI算法的科学依据

本实现基于以下研究和标准：

1. **Li et al. (2008)**: "Signal Quality Assessment and Lightweight QRS Detection for Wearable ECG SmartVest System"

2. **Clifford et al. (2012)**: "Signal quality indices and data fusion for determining clinical acceptability of electrocardiograms"

3. **Orphanidou et al. (2015)**: "Signal-Quality Indices for the Electrocardiogram and Photoplethysmogram"

4. **IEEE Standards**: 用于医疗设备的信号质量评估指南

---

## 🔧 进阶调优

### 1. 调整评估严格度

```python
# 宽松模式（接受更多信号）
loose_thresholds = {
    'snr_min': 3.0,
    'baseline_max': 0.4,
    'hf_noise_max': 0.4,
}

# 严格模式（仅接受高质量信号）
strict_thresholds = {
    'snr_min': 10.0,
    'baseline_max': 0.15,
    'hf_noise_max': 0.2,
}
```

### 2. 针对特定噪声类型优化

```python
# 如果主要关注基线漂移
baseline_focused = {
    'baseline_max': 0.1,  # 非常严格
    'hf_noise_max': 0.5,   # 相对宽松
}

# 如果主要关注高频噪声
hf_noise_focused = {
    'baseline_max': 0.4,   # 相对宽松
    'hf_noise_max': 0.15,  # 非常严格
}
```

### 3. 添加自定义SQI指标

在 `utils/signal_quality_indices.py` 中添加新的SQI函数：

```python
def my_custom_sqi(ecg_signal, fs=360):
    """自定义SQI指标"""
    # 实现您的SQI算法
    return sqi_value

# 在ECGQualityAssessor类中添加
def calculate_all_sqi(self, ecg_signal, fs=360):
    sqi = {
        # ... 现有SQI ...
        'custom_sqi': my_custom_sqi(ecg_signal, fs),
    }
    return sqi
```

---

## 📝 输出文件说明

### 1. `incart_sqi_summary.json`
包含整体统计信息：
- 总片段数
- 可接受/不可接受数量
- 百分比

### 2. 详细SQI结果（在代码中可访问）
每个信号片段的详细SQI值和失败原因

---

## ⚠️ 注意事项

1. **阈值调整**: 默认阈值基于一般ECG信号，可能需要根据您的具体数据调整

2. **计算时间**: SQI计算相对耗时，大数据集可能需要几分钟

3. **与真实标注对比**: 建议用少量数据对比SQI标注和人工标注，验证准确性

4. **迭代优化**: 可以先训练模型，然后用模型帮助优化SQI阈值

---

## 🚀 下一步

1. **运行数据准备**:
   ```bash
   python prepare_data.py
   ```

2. **检查SQI摘要**:
   ```bash
   cat processed_data/incart_sqi_summary.json
   ```

3. **根据结果调整阈值**（如果需要）

4. **继续训练模型**:
   ```bash
   python train.py
   ```

---

**创建日期**: 2026年1月6日  
**版本**: v1.0 - SQI自动标注系统



