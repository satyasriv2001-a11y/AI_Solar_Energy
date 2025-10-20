#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速一致性测试脚本 - 只测试单个厂，实时显示进度
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
from datetime import datetime

# 抑制所有警告
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['LIGHTGBM_VERBOSE'] = '0'

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sensitivity_analysis.common_utils import (
    create_base_config, run_single_experiment, load_all_plant_configs
)
from data.data_utils import load_raw_data, get_weather_features_by_category


def quick_test_single_plant(plant_id, data_path, model='LSTM'):
    """
    快速测试单个厂的一致性
    """
    print(f"🚀 快速一致性测试 - Plant {plant_id}")
    print(f"=" * 60)
    
    # 加载数据
    print(f"📂 加载数据...")
    try:
        df = load_raw_data(data_path)
        df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
        print(f"✅ 数据加载成功: {df.shape[0]} 行, {df.shape[1]} 列")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return None
    
    # 创建配置
    print(f"⚙️  创建实验配置...")
    plant_config = {
        'plant_id': plant_id,
        'data_path': data_path,
        'future_hours': 24,
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        'shuffle_split': True,
        'random_seed': 42,
        'weather_category': 'medium_weather',
        'start_date': '2022-01-01',
        'end_date': '2024-09-28'
    }
    
    # 检查天气特征
    print(f"🌤️  检查天气特征...")
    features = get_weather_features_by_category('medium_weather')
    print(f"   特征数量: {len(features)}")
    print(f"   特征列表: {features}")
    
    # 创建两个相同的配置
    config_h_m = create_base_config(plant_config, model, complexity='high', 
                                  lookback=24, use_te=False)
    config_h_m['weather_category'] = 'medium_weather'
    
    config_lookback = create_base_config(plant_config, model, complexity='high', 
                                       lookback=24, use_te=False)
    config_lookback['weather_category'] = 'medium_weather'
    
    print(f"✅ 配置创建完成")
    
    # 运行实验1: Weather H+M
    print(f"\n🧪 实验1: Weather H+M (medium_weather)")
    print(f"   模型: {config_h_m['model']}")
    print(f"   复杂度: {config_h_m['model_complexity']}")
    print(f"   特征: {config_h_m['weather_category']} ({len(features)}个)")
    print(f"   回看窗口: {config_h_m['past_hours']}h")
    print(f"   ⏳ 开始训练...")
    
    try:
        result_h_m = run_single_experiment(config_h_m, df.copy(), use_sliding_windows=False)
        
        if result_h_m['status'] == 'SUCCESS':
            print(f"   ✅ 训练完成!")
            print(f"   📊 RMSE: {result_h_m['rmse']:.4f}")
            print(f"   📊 MAE: {result_h_m['mae']:.4f}")
            print(f"   📊 样本数: {result_h_m['test_samples']}")
        else:
            print(f"   ❌ 训练失败: {result_h_m.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"   ❌ 实验1错误: {e}")
        return None
    
    # 运行实验2: Lookback 24h
    print(f"\n🧪 实验2: Lookback 24h (medium_weather)")
    print(f"   模型: {config_lookback['model']}")
    print(f"   复杂度: {config_lookback['model_complexity']}")
    print(f"   特征: {config_lookback['weather_category']} ({len(features)}个)")
    print(f"   回看窗口: {config_lookback['past_hours']}h")
    print(f"   ⏳ 开始训练...")
    
    try:
        result_lookback = run_single_experiment(config_lookback, df.copy(), use_sliding_windows=False)
        
        if result_lookback['status'] == 'SUCCESS':
            print(f"   ✅ 训练完成!")
            print(f"   📊 RMSE: {result_lookback['rmse']:.4f}")
            print(f"   📊 MAE: {result_lookback['mae']:.4f}")
            print(f"   📊 样本数: {result_lookback['test_samples']}")
        else:
            print(f"   ❌ 训练失败: {result_lookback.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"   ❌ 实验2错误: {e}")
        return None
    
    # 比较结果
    print(f"\n📊 结果对比:")
    print(f"   Weather H+M: RMSE={result_h_m['rmse']:.6f}, MAE={result_h_m['mae']:.6f}")
    print(f"   Lookback 24h: RMSE={result_lookback['rmse']:.6f}, MAE={result_lookback['mae']:.6f}")
    
    # 计算差异
    rmse_diff = abs(result_h_m['rmse'] - result_lookback['rmse'])
    mae_diff = abs(result_h_m['mae'] - result_lookback['mae'])
    samples_diff = abs(result_h_m['test_samples'] - result_lookback['test_samples'])
    
    print(f"\n🔍 差异分析:")
    print(f"   RMSE差异: {rmse_diff:.8f}")
    print(f"   MAE差异: {mae_diff:.8f}")
    print(f"   样本数差异: {samples_diff}")
    
    # 判断一致性
    if rmse_diff < 1e-6 and mae_diff < 1e-6 and samples_diff == 0:
        print(f"\n✅ 结果完全一致! 实验配置正确!")
    else:
        print(f"\n❌ 结果不一致!")
        if rmse_diff >= 1e-6:
            print(f"   - RMSE差异过大: {rmse_diff:.8f}")
        if mae_diff >= 1e-6:
            print(f"   - MAE差异过大: {mae_diff:.8f}")
        if samples_diff != 0:
            print(f"   - 样本数不同: {samples_diff}")
    
    return {
        'weather_h_m': {
            'rmse': result_h_m['rmse'],
            'mae': result_h_m['mae'],
            'samples': result_h_m['test_samples']
        },
        'lookback_24h': {
            'rmse': result_lookback['rmse'],
            'mae': result_lookback['mae'],
            'samples': result_lookback['test_samples']
        },
        'differences': {
            'rmse_diff': rmse_diff,
            'mae_diff': mae_diff,
            'samples_diff': samples_diff
        }
    }


def main():
    """主函数"""
    print("=" * 80)
    print("🚀 快速Sensitivity一致性测试 - LSTM模型")
    print("=" * 80)
    
    # 查找CSV文件
    data_dir = 'data'
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if len(csv_files) == 0:
        print(f"❌ 未找到CSV文件")
        return
    
    print(f"📂 找到 {len(csv_files)} 个CSV文件:")
    for csv_file in csv_files:
        print(f"   - {csv_file}")
    
    # 测试第一个厂
    plant_id = csv_files[0].replace('Project', '').replace('.csv', '')
    data_path = os.path.join(data_dir, csv_files[0])
    
    print(f"\n🎯 测试厂: {plant_id}")
    
    # 运行测试
    results = quick_test_single_plant(plant_id, data_path, model='LSTM')
    
    if results is not None:
        print(f"\n" + "=" * 80)
        print("✅ 测试完成!")
        print("=" * 80)
    else:
        print(f"\n" + "=" * 80)
        print("❌ 测试失败!")
        print("=" * 80)


if __name__ == '__main__':
    main()
