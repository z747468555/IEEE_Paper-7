"""
数据加载模块：从各个数据库加载ECG信号
"""

import os
import numpy as np
import wfdb
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_PATHS, TARGET_SAMPLING_RATE


class ECGDataLoader:
    """ECG数据加载器基类"""
    
    def __init__(self, database_path):
        """
        初始化数据加载器
        
        Args:
            database_path: 数据库路径
        """
        self.database_path = database_path
        
    def load_record(self, record_name):
        """
        加载单条记录
        
        Args:
            record_name: 记录名称
        
        Returns:
            signal: 信号数据，shape (n_samples, n_channels)
            fields: 记录的元数据
        """
        record = wfdb.rdrecord(os.path.join(self.database_path, record_name))
        return record.p_signal, record.__dict__
    
    def get_record_list(self):
        """获取数据库中的所有记录列表"""
        raise NotImplementedError("子类必须实现此方法")


class MITBIHLoader(ECGDataLoader):
    """MIT-BIH心律失常数据库加载器"""
    
    # MIT-BIH数据库的48条记录
    RECORD_LIST = [
        '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
        '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
        '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
        '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
        '222', '223', '228', '230', '231', '232', '233', '234'
    ]
    
    def get_record_list(self):
        """返回MIT-BIH数据库记录列表"""
        return self.RECORD_LIST
    
    def load_all_records(self):
        """
        加载所有MIT-BIH记录
        
        Returns:
            records: 字典，键为记录名，值为(signal, sampling_rate)元组
        """
        records = {}
        print("正在加载MIT-BIH数据库...")
        
        for record_name in tqdm(self.get_record_list()):
            try:
                signal, fields = self.load_record(record_name)
                fs = fields['fs']
                # MIT-BIH有2个导联，取MLII导联（通常是第0个）
                if signal.shape[1] >= 1:
                    records[record_name] = (signal[:, 0], fs)
            except Exception as e:
                print(f"加载记录 {record_name} 失败: {e}")
        
        return records


class PTBDBLoader(ECGDataLoader):
    """PTB诊断数据库加载器"""
    
    def get_record_list(self):
        """
        获取PTB数据库的所有记录
        注意：PTB有549条记录，分布在不同的子目录中
        """
        record_list = []
        
        # PTB数据库的目录结构
        patient_dirs = []
        if os.path.exists(self.database_path):
            for item in os.listdir(self.database_path):
                item_path = os.path.join(self.database_path, item)
                if os.path.isdir(item_path) and item.startswith('patient'):
                    patient_dirs.append(item)
        
        # 遍历每个患者目录
        for patient_dir in patient_dirs:
            patient_path = os.path.join(self.database_path, patient_dir)
            for file in os.listdir(patient_path):
                if file.endswith('.hea'):
                    record_name = os.path.join(patient_dir, file[:-4])
                    record_list.append(record_name)
        
        return record_list
    
    def load_all_records(self):
        """
        加载所有PTB记录
        
        Returns:
            records: 字典，键为记录名，值为(signals, sampling_rate)元组
                    signals是所有导联的列表
        """
        records = {}
        print("正在加载PTB数据库...")
        
        for record_name in tqdm(self.get_record_list()):
            try:
                signal, fields = self.load_record(record_name)
                fs = fields['fs']
                # PTB有15个导联，保存所有导联
                records[record_name] = (signal, fs)
            except Exception as e:
                print(f"加载记录 {record_name} 失败: {e}")
        
        return records


class INCARTLoader(ECGDataLoader):
    """INCART数据库加载器"""
    
    # INCART有75条记录，命名为I01-I75
    def get_record_list(self):
        """返回INCART数据库记录列表"""
        return [f'I{i:02d}' for i in range(1, 76)]
    
    def load_all_records(self):
        """加载所有INCART记录"""
        records = {}
        print("正在加载INCART数据库...")
        
        for record_name in tqdm(self.get_record_list()):
            try:
                signal, fields = self.load_record(record_name)
                fs = fields['fs']
                # INCART有12导联
                records[record_name] = (signal, fs)
            except Exception as e:
                print(f"加载记录 {record_name} 失败: {e}")
        
        return records


class PCCC2011Loader(ECGDataLoader):
    """PhysioNet/CinC Challenge 2011数据库加载器"""
    
    def get_record_list(self):
        """获取Challenge 2011的所有记录"""
        record_list = []
        
        # 查找所有.hea文件
        if os.path.exists(self.database_path):
            for file in os.listdir(self.database_path):
                if file.endswith('.hea'):
                    record_list.append(file[:-4])
        
        return record_list
    
    def load_all_records(self):
        """加载所有Challenge 2011记录"""
        records = {}
        print("正在加载PhysioNet Challenge 2011数据库...")
        
        for record_name in tqdm(self.get_record_list()):
            try:
                signal, fields = self.load_record(record_name)
                fs = fields['fs']
                records[record_name] = (signal, fs)
            except Exception as e:
                print(f"加载记录 {record_name} 失败: {e}")
        
        return records


class NSTDBLoader(ECGDataLoader):
    """MIT-BIH噪声压力测试数据库加载器"""
    
    # NSTDB包含3种噪声类型
    NOISE_TYPES = {
        'baseline_wander': 'bw',  # 基线漂移
        'muscle_artifact': 'ma',  # 肌电干扰
        'electrode_motion': 'em',  # 电极移动
    }
    
    def get_noise_records(self):
        """返回噪声记录列表"""
        return list(self.NOISE_TYPES.values())
    
    def load_noise_signals(self):
        """
        加载所有噪声信号
        
        Returns:
            noise_dict: 字典，键为噪声类型名称，值为(noise_signal, fs)
        """
        noise_dict = {}
        print("正在加载MIT-BIH NSTDB噪声数据...")
        
        for noise_name, noise_code in self.NOISE_TYPES.items():
            try:
                signal, fields = self.load_record(noise_code)
                fs = fields['fs']
                # 噪声数据库通常有2个通道
                if signal.shape[1] >= 1:
                    noise_dict[noise_name] = (signal[:, 0], fs)
            except Exception as e:
                print(f"加载噪声 {noise_name} 失败: {e}")
        
        return noise_dict


def create_data_loaders():
    """
    创建所有数据加载器
    
    Returns:
        loaders: 字典，包含所有数据库的加载器
    """
    loaders = {
        'mitdb': MITBIHLoader(DATA_PATHS['mitdb']),
        'ptbdb': PTBDBLoader(DATA_PATHS['ptbdb']),
        'incart': INCARTLoader(DATA_PATHS['incart']),
        'pccc2011': PCCC2011Loader(DATA_PATHS['pccc2011']),
        'nstdb': NSTDBLoader(DATA_PATHS['nstdb']),
    }
    
    return loaders

