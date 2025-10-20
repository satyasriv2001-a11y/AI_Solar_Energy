#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensitivity实验一致性测试脚本

测试不同sensitivity实验的base结果是否一致：
- Weather H+M (experiment 3) vs Lookback 24h (experiment 4)
- 理论上应该使用相同的配置：PV+NWP, 24h lookback, medium_weather (7 features), no TE, high complexity
"""

import os
import sys
import pandas as pd
import numpy as np
import tempfile
import shutil
import warnings
from datetime import datetime

# 抑制所有警告
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['LIGHTGBM_VERBOSE'] = '0'

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sensitivity_analysis.common_utils import (
    create_base_config, run_single_experiment
)
from data.data_utils import load_raw_data
from config_manager import PlantConfigManager


def test_single_plant_consistency(plant_id, data_path, model='LSTM'):
    """
    测试单个厂的一致性
    
    Args:
        plant_id: 厂ID
        data_path: 数据文件路径
        model: 模型名称，默认LSTM
    
    Returns:
        dict: 包含两个实验结果的字典
    """
    print(f"============================================================")
    print(f"测试单厂: {plant_id}")
    print(f"============================================================")
    
    # 加载数据
    try:
        df = load_raw_data(data_path)
        df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
        print(f"数据形状: {df.shape}")
        print(f"数据时间范围: {df['Datetime'].min()} 到 {df['Datetime'].max()}")
    except Exception as e:
        print(f"错误加载数据: {e}")
        return None
    
    # 创建基础配置
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
    
    results = {}
    
    # 实验1: Weather H+M (experiment 3)
    print(f"\n运行Weather H+M实验...")
    try:
        config_h_m = create_base_config(plant_config, model, complexity='high', 
                                      lookback=24, use_te=False)
        config_h_m['weather_category'] = 'medium_weather'  # 7个特征
        
        result_h_m = run_single_experiment(config_h_m, df.copy(), use_sliding_windows=False)
        
        if result_h_m['status'] == 'SUCCESS':
            results['weather_h_m'] = {
                'rmse': result_h_m['rmse'],
                'mae': result_h_m['mae'],
                'samples': result_h_m['test_samples']
            }
            print(f"  Weather H+M: RMSE={result_h_m['rmse']:.4f}, MAE={result_h_m['mae']:.4f}, Samples={result_h_m['test_samples']}")
        else:
            print(f"  Weather H+M实验失败: {result_h_m.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"  Weather H+M实验错误: {e}")
        return None
    
    # 实验2: Lookback 24h (experiment 4)
    print(f"运行Lookback 24h实验...")
    try:
        config_lookback = create_base_config(plant_config, model, complexity='high', 
                                           lookback=24, use_te=False)
        config_lookback['weather_category'] = 'medium_weather'  # 7个特征
        
        result_lookback = run_single_experiment(config_lookback, df.copy(), use_sliding_windows=False)
        
        if result_lookback['status'] == 'SUCCESS':
            results['lookback_24h'] = {
                'rmse': result_lookback['rmse'],
                'mae': result_lookback['mae'],
                'samples': result_lookback['test_samples']
            }
            print(f"  Lookback 24h: RMSE={result_lookback['rmse']:.4f}, MAE={result_lookback['mae']:.4f}, Samples={result_lookback['test_samples']}")
        else:
            print(f"  Lookback 24h实验失败: {result_lookback.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"  Lookback 24h实验错误: {e}")
        return None
    
    # 比较结果
    print(f"\n结果:")
    print(f"  Weather H+M: RMSE={results['weather_h_m']['rmse']:.4f}, MAE={results['weather_h_m']['mae']:.4f}, Samples={results['weather_h_m']['samples']}")
    print(f"  Lookback 24h: RMSE={results['lookback_24h']['rmse']:.4f}, MAE={results['lookback_24h']['mae']:.4f}, Samples={results['lookback_24h']['samples']}")
    
    return results


def test_multi_plant_consistency(data_dir, model='LSTM'):
    """
    测试多厂的一致性
    
    Args:
        data_dir: 数据目录
        model: 模型名称，默认LSTM
    
    Returns:
        dict: 包含汇总结果的字典
    """
    print(f"============================================================")
    print(f"测试多厂: 171 + 1140")
    print(f"============================================================")
    
    # 查找CSV文件
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    print(f"Found {len(csv_files)} CSV files in {data_dir}")
    
    if len(csv_files) == 0:
        print("错误: 未找到CSV文件")
        return None
    
    # 获取厂ID列表
    plant_ids = []
    for csv_file in csv_files:
        if 'Project' in csv_file:
            plant_id = csv_file.replace('Project', '').replace('.csv', '')
            plant_ids.append(plant_id)
    
    print(f"总厂数: {len(plant_ids)}")
    print(f"厂ID列表: {plant_ids}")
    
    all_results = []
    
    # 测试每个厂
    for i, plant_id in enumerate(plant_ids, 1):
        data_path = os.path.join(data_dir, f'Project{plant_id}.csv')
        
        print(f"\n处理厂 {i}/{len(plant_ids)}: {plant_id}")
        
        # 创建临时目录用于单厂测试
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_data_dir = os.path.join(temp_dir, 'data')
            os.makedirs(temp_data_dir, exist_ok=True)
            
            # 复制数据文件到临时目录
            shutil.copy2(data_path, os.path.join(temp_data_dir, f'Project{plant_id}.csv'))
            
            # 运行单厂测试
            result = test_single_plant_consistency(plant_id, data_path, model)
            
            if result is not None:
                all_results.append({
                    'plant_id': plant_id,
                    'weather_h_m_rmse': result['weather_h_m']['rmse'],
                    'weather_h_m_mae': result['weather_h_m']['mae'],
                    'lookback_24h_rmse': result['lookback_24h']['rmse'],
                    'lookback_24h_mae': result['lookback_24h']['mae'],
                    'samples': result['weather_h_m']['samples']
                })
                print(f"  运行Weather H+M实验...")
                print(f"    RMSE: {result['weather_h_m']['rmse']:.4f}, MAE: {result['weather_h_m']['mae']:.4f}")
                print(f"  运行Lookback 24h实验...")
                print(f"    RMSE: {result['lookback_24h']['rmse']:.4f}, MAE: {result['lookback_24h']['mae']:.4f}")
            else:
                print(f"  厂 {plant_id} 测试失败")
    
    if len(all_results) == 0:
        print("错误: 没有成功的测试结果")
        return None
    
    # 计算汇总结果
    print(f"\n汇总结果:")
    
    # Weather H+M 汇总
    weather_rmse_values = [r['weather_h_m_rmse'] for r in all_results]
    weather_mae_values = [r['weather_h_m_mae'] for r in all_results]
    
    weather_rmse_mean = np.mean(weather_rmse_values)
    weather_rmse_std = np.std(weather_rmse_values)
    weather_mae_mean = np.mean(weather_mae_values)
    weather_mae_std = np.std(weather_mae_values)
    
    print(f"  Weather H+M: RMSE={weather_rmse_mean:.4f}±{weather_rmse_std:.4f}, MAE={weather_mae_mean:.4f}±{weather_mae_std:.4f}")
    print(f"  参与厂数: {len(all_results)}")
    
    # Lookback 24h 汇总
    lookback_rmse_values = [r['lookback_24h_rmse'] for r in all_results]
    lookback_mae_values = [r['lookback_24h_mae'] for r in all_results]
    
    lookback_rmse_mean = np.mean(lookback_rmse_values)
    lookback_rmse_std = np.std(lookback_rmse_values)
    lookback_mae_mean = np.mean(lookback_mae_values)
    lookback_mae_std = np.std(lookback_mae_values)
    
    print(f"  Lookback 24h: RMSE={lookback_rmse_mean:.4f}±{lookback_rmse_std:.4f}, MAE={lookback_mae_mean:.4f}±{lookback_mae_std:.4f}")
    print(f"  参与厂数: {len(all_results)}")
    
    # 加权平均计算
    print(f"\n============================================================")
    print(f"加权平均计算")
    print(f"============================================================")
    
    # 使用样本数作为权重
    total_samples = sum(r['samples'] for r in all_results)
    weighted_weather_rmse = sum(r['weather_h_m_rmse'] * r['samples'] for r in all_results) / total_samples
    weighted_weather_mae = sum(r['weather_h_m_mae'] * r['samples'] for r in all_results) / total_samples
    weighted_lookback_rmse = sum(r['lookback_24h_rmse'] * r['samples'] for r in all_results) / total_samples
    weighted_lookback_mae = sum(r['lookback_24h_mae'] * r['samples'] for r in all_results) / total_samples
    
    print(f"单厂加权平均: RMSE={weighted_weather_rmse:.4f}, MAE={weighted_weather_mae:.4f}")
    
    # 结果对比
    print(f"\n============================================================")
    print(f"结果对比")
    print(f"============================================================")
    
    multi_plant_diff = abs(weather_rmse_mean - lookback_rmse_mean)
    single_vs_multi_diff = abs(weighted_weather_rmse - weather_rmse_mean)
    
    print(f"多厂Weather H+M RMSE: {weather_rmse_mean:.4f}")
    print(f"多厂Lookback 24h RMSE: {lookback_rmse_mean:.4f}")
    print(f"多厂差异: {multi_plant_diff:.4f}")
    print(f"单厂加权平均RMSE: {weighted_weather_rmse:.4f}")
    print(f"多厂vs单厂差异: {single_vs_multi_diff:.4f}")
    
    return {
        'weather_h_m': {
            'rmse_mean': weather_rmse_mean,
            'rmse_std': weather_rmse_std,
            'mae_mean': weather_mae_mean,
            'mae_std': weather_mae_std
        },
        'lookback_24h': {
            'rmse_mean': lookback_rmse_mean,
            'rmse_std': lookback_rmse_std,
            'mae_mean': lookback_mae_mean,
            'mae_std': lookback_mae_std
        },
        'weighted_average': {
            'rmse': weighted_weather_rmse,
            'mae': weighted_weather_mae
        },
        'differences': {
            'multi_plant_diff': multi_plant_diff,
            'single_vs_multi_diff': single_vs_multi_diff
        }
    }


def main():
    """主函数"""
    print("=" * 80)
    print("Sensitivity一致性测试 - LSTM模型")
    print("=" * 80)
    
    # 设置数据目录
    data_dir = 'data'
    
    if not os.path.exists(data_dir):
        print(f"错误: 数据目录不存在: {data_dir}")
        return
    
    # 运行多厂一致性测试
    results = test_multi_plant_consistency(data_dir, model='LSTM')
    
    if results is not None:
        print(f"\n" + "=" * 80)
        print("测试完成!")
        print("=" * 80)
        
        # 判断一致性
        if results['differences']['multi_plant_diff'] < 1e-6:
            print("✅ 一致性测试通过: Weather H+M 和 Lookback 24h 结果完全一致")
        else:
            print(f"❌ 一致性测试失败: 差异为 {results['differences']['multi_plant_diff']:.6f}")
    else:
        print("❌ 测试失败")


if __name__ == '__main__':
    main()
