"""
测试脚本：验证环境配置和依赖安装
"""

import sys
import importlib


def test_imports():
    """测试必需的包是否已安装"""
    
    print("="*60)
    print("测试环境配置")
    print("="*60 + "\n")
    
    required_packages = [
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('tensorflow', 'TensorFlow'),
        ('keras', 'Keras'),
        ('wfdb', 'WFDB'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib'),
        ('sklearn', 'Scikit-learn'),
        ('h5py', 'H5Py'),
        ('tqdm', 'TQDM'),
    ]
    
    print("检查依赖包安装情况：\n")
    
    all_installed = True
    
    for package_name, display_name in required_packages:
        try:
            module = importlib.import_module(package_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {display_name:<15} {version}")
        except ImportError:
            print(f"✗ {display_name:<15} 未安装")
            all_installed = False
    
    print("\n" + "-"*60 + "\n")
    
    if all_installed:
        print("✓ 所有必需的包都已正确安装！\n")
    else:
        print("✗ 某些包未安装，请运行：")
        print("  pip install -r requirements.txt\n")
        return False
    
    return True


def test_tensorflow():
    """测试TensorFlow配置"""
    
    print("检查TensorFlow配置：\n")
    
    try:
        import tensorflow as tf
        
        print(f"TensorFlow版本: {tf.__version__}")
        print(f"Keras版本: {tf.keras.__version__}")
        
        # 检查GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"\n✓ 找到 {len(gpus)} 个GPU:")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
            print("\n提示：训练将使用GPU加速（快速）")
        else:
            print("\n✗ 未找到GPU")
            print("提示：训练将使用CPU（较慢，约3-5小时）")
        
        # 测试基本操作
        print("\n测试TensorFlow基本操作...")
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
        c = tf.matmul(a, b)
        print("✓ TensorFlow基本操作正常")
        
    except Exception as e:
        print(f"✗ TensorFlow测试失败: {e}")
        return False
    
    print("\n" + "-"*60 + "\n")
    
    return True


def test_signal_processing():
    """测试信号处理功能"""
    
    print("测试信号处理功能：\n")
    
    try:
        import numpy as np
        from scipy import signal
        
        # 测试滤波器设计
        print("测试滤波器设计...")
        sos = signal.cheby2(2, 20, 0.8/180, btype='highpass', output='sos')
        print("✓ 切比雪夫滤波器设计正常")
        
        # 测试信号生成
        print("测试信号生成...")
        t = np.linspace(0, 1, 360)
        test_signal = np.sin(2 * np.pi * 1.2 * t)
        print("✓ 信号生成正常")
        
        # 测试滤波
        print("测试信号滤波...")
        filtered = signal.sosfiltfilt(sos, test_signal)
        print("✓ 信号滤波正常")
        
        # 测试导数
        print("测试导数计算...")
        derivative = np.diff(test_signal)
        print("✓ 导数计算正常")
        
    except Exception as e:
        print(f"✗ 信号处理测试失败: {e}")
        return False
    
    print("\n" + "-"*60 + "\n")
    
    return True


def test_project_structure():
    """测试项目结构"""
    
    print("检查项目结构：\n")
    
    import os
    
    required_files = [
        'config.py',
        'main.py',
        'train.py',
        'evaluate.py',
        'prepare_data.py',
        'download_datasets.py',
        'utils/__init__.py',
        'utils/signal_processing.py',
        'utils/data_loader.py',
        'utils/noise_generator.py',
        'model/__init__.py',
        'model/cnn_model.py',
    ]
    
    all_exist = True
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} (缺失)")
            all_exist = False
    
    print("\n" + "-"*60 + "\n")
    
    if all_exist:
        print("✓ 所有必需文件都存在！\n")
    else:
        print("✗ 某些文件缺失\n")
        return False
    
    return True


def test_module_imports():
    """测试模块导入"""
    
    print("测试模块导入：\n")
    
    try:
        # 测试配置导入
        print("导入配置模块...")
        from config import MODEL_CONFIG, TRAINING_CONFIG
        print("✓ 配置模块导入成功")
        
        # 测试工具模块导入
        print("导入工具模块...")
        from utils.signal_processing import (
            preprocess_ecg_signal,
            compute_derivative,
            normalize_signal
        )
        print("✓ 信号处理模块导入成功")
        
        # 测试模型模块导入
        print("导入模型模块...")
        from model.cnn_model import create_model
        print("✓ 模型模块导入成功")
        
    except Exception as e:
        print(f"✗ 模块导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "-"*60 + "\n")
    
    return True


def test_model_creation():
    """测试模型创建"""
    
    print("测试模型创建：\n")
    
    try:
        from model.cnn_model import create_model
        
        print("创建CNN模型...")
        model = create_model(summary=False)
        print("✓ 模型创建成功")
        
        print(f"✓ 模型参数数量: {model.count_params():,}")
        
        # 测试模型预测
        import numpy as np
        print("测试模型推理...")
        test_input = np.random.randn(1, 1799, 1)
        prediction = model.predict(test_input, verbose=0)
        print(f"✓ 模型推理成功，输出形状: {prediction.shape}")
        
    except Exception as e:
        print(f"✗ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "-"*60 + "\n")
    
    return True


def main():
    """主测试函数"""
    
    print("\n" + "="*60)
    print("ECG信号质量评估系统 - 环境测试")
    print("="*60 + "\n")
    
    print(f"Python版本: {sys.version}\n")
    print("="*60 + "\n")
    
    tests = [
        ("依赖包安装", test_imports),
        ("TensorFlow配置", test_tensorflow),
        ("信号处理", test_signal_processing),
        ("项目结构", test_project_structure),
        ("模块导入", test_module_imports),
        ("模型创建", test_model_creation),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name}测试出错: {e}\n")
            results.append((test_name, False))
    
    # 总结
    print("="*60)
    print("测试总结")
    print("="*60 + "\n")
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name:<20} {status}")
    
    print("\n" + "="*60 + "\n")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    if passed == total:
        print(f"✓ 所有测试通过！({passed}/{total})")
        print("\n您可以开始使用系统了：")
        print("  1. 下载数据: python download_datasets.py --mode essential")
        print("  2. 准备数据: python prepare_data.py")
        print("  3. 训练模型: python train.py")
        print("  4. 评估模型: python evaluate.py --model <模型路径> --test_data <测试数据>")
        print("\n或查看完整流程: python main.py --pipeline")
    else:
        print(f"✗ {total - passed} 个测试失败")
        print("\n请检查：")
        print("  1. 是否安装了所有依赖: pip install -r requirements.txt")
        print("  2. Python版本是否>=3.7")
        print("  3. 项目文件是否完整")
    
    print("\n" + "="*60 + "\n")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)










