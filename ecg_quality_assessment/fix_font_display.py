"""
修复matplotlib中文显示问题的工具脚本
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import os


def list_available_fonts():
    """列出系统中所有可用的字体"""
    print("="*60)
    print("系统可用字体列表")
    print("="*60)
    
    fonts = sorted([f.name for f in fm.fontManager.ttflist])
    
    # 筛选中文字体（包含常见中文字体名称）
    chinese_keywords = ['sim', 'hei', 'song', 'kai', 'fang', 'yuan', 
                       'microsoft', 'pingfang', 'hiragino', 'wenquanyi',
                       'noto', 'droid', 'arial unicode']
    
    chinese_fonts = []
    for font in fonts:
        if any(keyword in font.lower() for keyword in chinese_keywords):
            chinese_fonts.append(font)
    
    print("\n可能支持中文的字体:")
    for i, font in enumerate(chinese_fonts, 1):
        print(f"{i:3d}. {font}")
    
    if not chinese_fonts:
        print("警告: 未找到明显的中文字体！")
        print("\n所有字体 (前20个):")
        for i, font in enumerate(fonts[:20], 1):
            print(f"{i:3d}. {font}")
    
    return chinese_fonts


def test_chinese_display(font_name=None):
    """测试中文显示效果"""
    if font_name:
        plt.rcParams['font.sans-serif'] = [font_name]
    else:
        # 自动配置
        system = platform.system()
        if system == 'Windows':
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
        elif system == 'Darwin':
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC']
        else:
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Droid Sans Fallback']
    
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建测试图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    test_text = [
        '中文标题测试',
        '准确率 (Accuracy)',
        '敏感度 (Sensitivity)',
        '特异度 (Specificity)',
        '训练损失',
        '验证准确率',
    ]
    
    for i, text in enumerate(test_text):
        ax.text(0.5, 0.9 - i*0.15, text, 
               ha='center', va='center',
               fontsize=14, transform=ax.transAxes)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    title = f"中文显示测试 - {font_name if font_name else '自动配置'}"
    ax.set_title(title, fontsize=16, pad=20)
    
    output_path = 'results/font_test.png'
    os.makedirs('results', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n测试图已保存到: {output_path}")
    print("请打开图片检查中文是否正常显示")
    
    try:
        plt.show()
    except:
        pass


def install_font_instructions():
    """显示安装字体的说明"""
    system = platform.system()
    
    print("\n" + "="*60)
    print("字体安装指南")
    print("="*60)
    
    if system == 'Windows':
        print("\nWindows系统:")
        print("1. 通常已经安装了中文字体（SimHei, SimSun等）")
        print("2. 如果显示不正常，可以:")
        print("   - 下载思源黑体: https://github.com/adobe-fonts/source-han-sans")
        print("   - 双击字体文件安装")
        print("   - 重启Python程序")
        
    elif system == 'Darwin':
        print("\nmacOS系统:")
        print("1. 通常已安装中文字体")
        print("2. 如果显示不正常:")
        print("   brew install font-source-han-sans")
        
    else:  # Linux
        print("\nLinux系统:")
        print("1. Ubuntu/Debian:")
        print("   sudo apt-get install fonts-wqy-microhei fonts-wqy-zenhei")
        print("\n2. CentOS/RHEL:")
        print("   sudo yum install wqy-microhei-fonts wqy-zenhei-fonts")
        print("\n3. 通用方法 - 安装思源黑体:")
        print("   sudo apt-get install fonts-noto-cjk")
        print("\n4. 安装后需要:")
        print("   - 清除matplotlib缓存: rm -rf ~/.cache/matplotlib")
        print("   - 重启Python")
    
    print("\n" + "="*60)


def fix_existing_plots():
    """重新生成已有的图表（修复中文显示）"""
    print("\n" + "="*60)
    print("重新生成图表以修复中文显示")
    print("="*60)
    
    # 配置中文字体
    system = platform.system()
    if system == 'Windows':
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
    elif system == 'Darwin':
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC']
    else:
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Droid Sans Fallback']
    
    plt.rcParams['axes.unicode_minus'] = False
    
    print("字体已配置，请重新运行评估脚本:")
    print("  python evaluate.py --model <模型路径> --test_data <数据路径> --dataset_name <名称>")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("matplotlib中文显示修复工具")
    print("="*60)
    
    print(f"\n当前系统: {platform.system()}")
    print(f"Python版本: {platform.python_version()}")
    
    # 列出可用字体
    chinese_fonts = list_available_fonts()
    
    # 安装说明
    if not chinese_fonts or len(chinese_fonts) < 3:
        print("\n警告: 系统中中文字体较少！")
        install_font_instructions()
    
    # 测试显示
    print("\n" + "="*60)
    print("测试中文显示")
    print("="*60)
    
    if chinese_fonts:
        print(f"\n使用字体: {chinese_fonts[0]}")
        test_chinese_display(chinese_fonts[0])
    else:
        print("\n使用自动配置")
        test_chinese_display()
    
    # 修复建议
    print("\n" + "="*60)
    print("修复建议")
    print("="*60)
    print("\n如果测试图中文显示正常:")
    print("  ✓ 说明字体配置成功")
    print("  ✓ 重新运行 evaluate.py 即可")
    print("\n如果测试图中文显示异常:")
    print("  1. 检查上面的字体列表，选择一个支持中文的字体")
    print("  2. 根据上面的'字体安装指南'安装中文字体")
    print("  3. 安装后清除matplotlib缓存:")
    if platform.system() == 'Windows':
        print("     del /F /Q %USERPROFILE%\\.matplotlib\\*")
    else:
        print("     rm -rf ~/.cache/matplotlib")
    print("  4. 重启Python，重新运行此脚本")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    main()

