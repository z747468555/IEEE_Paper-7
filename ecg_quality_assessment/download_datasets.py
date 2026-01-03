"""
自动下载数据集脚本
使用wfdb库从PhysioNet下载所需的数据库
"""

import os
import wfdb
from config import DATA_PATHS


def download_database(db_name, db_path, records=None):
    """
    下载数据库
    
    Args:
        db_name: PhysioNet数据库名称
        db_path: 本地保存路径
        records: 要下载的记录列表（如果为None，下载所有）
    """
    print(f"\n{'='*60}")
    print(f"下载数据库: {db_name}")
    print(f"保存路径: {db_path}")
    print(f"{'='*60}")
    
    # 创建目录
    os.makedirs(db_path, exist_ok=True)
    
    try:
        if records is None:
            # 获取数据库中的所有记录
            print("获取记录列表...")
            records = wfdb.get_record_list(db_name)
            print(f"找到 {len(records)} 条记录")
        
        # 下载记录
        print("开始下载...")
        for i, record in enumerate(records, 1):
            try:
                print(f"[{i}/{len(records)}] 下载 {record}...", end=' ')
                wfdb.dl_database(db_name, db_path, [record])
                print("✓")
            except Exception as e:
                print(f"✗ 错误: {e}")
        
        print(f"\n数据库 {db_name} 下载完成！")
        
    except Exception as e:
        print(f"\n下载失败: {e}")


def download_all_datasets():
    """下载所有需要的数据集"""
    
    print("\n" + "="*60)
    print("ECG信号质量评估 - 数据集下载工具")
    print("="*60)
    print("\n注意：完整下载可能需要较长时间和较大存储空间")
    print("建议使用稳定的网络连接")
    print("="*60 + "\n")
    
    # 1. MIT-BIH心律失常数据库
    print("\n1. MIT-BIH心律失常数据库")
    mitdb_records = [
        '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
        '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
        '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
        '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
        '222', '223', '228', '230', '231', '232', '233', '234'
    ]
    download_database('mitdb', DATA_PATHS['mitdb'], mitdb_records)
    
    # 2. PTB诊断ECG数据库
    print("\n2. PTB诊断ECG数据库")
    download_database('ptbdb', DATA_PATHS['ptbdb'])
    
    # 3. MIT-BIH噪声压力测试数据库
    print("\n3. MIT-BIH噪声压力测试数据库")
    nstdb_records = ['bw', 'ma', 'em']  # 基线漂移、肌电干扰、电极移动
    download_database('nstdb', DATA_PATHS['nstdb'], nstdb_records)
    
    # 4. INCART数据库
    print("\n4. INCART数据库")
    incart_records = [f'I{i:02d}' for i in range(1, 76)]
    download_database('incartdb', DATA_PATHS['incart'], incart_records)
    
    # 5. PhysioNet/CinC Challenge 2011
    print("\n5. PhysioNet/CinC Challenge 2011数据库")
    download_database('challenge-2011', DATA_PATHS['pccc2011'])
    
    print("\n" + "="*60)
    print("所有数据集下载完成！")
    print("="*60)
    print("\n接下来可以运行数据准备脚本：")
    print("  python prepare_data.py")
    print("="*60 + "\n")


def download_essential_only():
    """仅下载必需的数据集（用于快速测试）"""
    
    print("\n" + "="*60)
    print("下载必需的数据集（训练所需）")
    print("="*60 + "\n")
    
    # 1. MIT-BIH（部分记录用于测试）
    print("\n1. MIT-BIH心律失常数据库（部分记录）")
    test_records = ['100', '101', '102', '103', '104']
    download_database('mitdb', DATA_PATHS['mitdb'], test_records)
    
    # 2. PTB（部分记录）
    print("\n2. PTB诊断ECG数据库（部分记录）")
    # PTB下载会自动获取所有记录
    print("注意：PTB数据库较大，建议下载完整数据库")
    
    # 3. 噪声数据库（必需）
    print("\n3. MIT-BIH噪声压力测试数据库")
    nstdb_records = ['bw', 'ma', 'em']
    download_database('nstdb', DATA_PATHS['nstdb'], nstdb_records)
    
    print("\n" + "="*60)
    print("必需数据集下载完成！")
    print("="*60 + "\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='下载ECG数据集')
    parser.add_argument('--mode', type=str, choices=['all', 'essential'], 
                       default='essential',
                       help='下载模式: all=所有数据集, essential=仅必需数据集')
    
    args = parser.parse_args()
    
    if args.mode == 'all':
        download_all_datasets()
    else:
        download_essential_only()

