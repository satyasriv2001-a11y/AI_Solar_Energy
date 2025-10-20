#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensitivity实验一致性测试脚本

直接调用lookback_window.py和weather_feature_adoption.py模拟实际运行情况
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
import subprocess
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
from config_manager import PlantConfigManager


def debug_config_differences(plant_config, model='LSTM'):
    """
    调试配置差异
    
    Args:
        plant_config: 厂配置
        model: 模型名称
    """
    print(f"\n🔍 调试配置差异:")
    print(f"=" * 60)
    
    # Weather H+M 配置
    print(f"\n📋 Weather H+M 配置 (experiment 3):")
    config_h_m = create_base_config(plant_config, model, complexity='high', 
                                  lookback=24, use_te=False)
    config_h_m['weather_category'] = 'medium_weather'
    
    print(f"  model: {config_h_m['model']}")
    print(f"  model_complexity: {config_h_m['model_complexity']}")
    print(f"  use_pv: {config_h_m['use_pv']}")
    print(f"  use_hist_weather: {config_h_m['use_hist_weather']}")
    print(f"  use_forecast: {config_h_m['use_forecast']}")
    print(f"  use_ideal_nwp: {config_h_m['use_ideal_nwp']}")
    print(f"  use_time_encoding: {config_h_m['use_time_encoding']}")
    print(f"  past_hours: {config_h_m['past_hours']}")
    print(f"  weather_category: {config_h_m['weather_category']}")
    print(f"  shuffle_split: {config_h_m['shuffle_split']}")
    print(f"  random_seed: {config_h_m['random_seed']}")
    
    # Lookback 24h 配置
    print(f"\n📋 Lookback 24h 配置 (experiment 4):")
    config_lookback = create_base_config(plant_config, model, complexity='high', 
                                       lookback=24, use_te=False)
    config_lookback['weather_category'] = 'medium_weather'
    
    print(f"  model: {config_lookback['model']}")
    print(f"  model_complexity: {config_lookback['model_complexity']}")
    print(f"  use_pv: {config_lookback['use_pv']}")
    print(f"  use_hist_weather: {config_lookback['use_hist_weather']}")
    print(f"  use_forecast: {config_lookback['use_forecast']}")
    print(f"  use_ideal_nwp: {config_lookback['use_ideal_nwp']}")
    print(f"  use_time_encoding: {config_lookback['use_time_encoding']}")
    print(f"  past_hours: {config_lookback['past_hours']}")
    print(f"  weather_category: {config_lookback['weather_category']}")
    print(f"  shuffle_split: {config_lookback['shuffle_split']}")
    print(f"  random_seed: {config_lookback['random_seed']}")
    
    # 检查配置是否相同
    print(f"\n🔍 配置比较:")
    configs_equal = True
    for key in config_h_m:
        if key in config_lookback:
            if config_h_m[key] != config_lookback[key]:
                print(f"  ❌ {key}: H+M={config_h_m[key]} vs Lookback={config_lookback[key]}")
                configs_equal = False
            else:
                print(f"  ✅ {key}: {config_h_m[key]}")
        else:
            print(f"  ⚠️  {key}: 只在H+M中存在")
            configs_equal = False
    
    if configs_equal:
        print(f"\n✅ 配置完全一致!")
    else:
        print(f"\n❌ 配置存在差异!")
    
    # 检查天气特征
    print(f"\n🌤️  天气特征检查:")
    h_m_features = get_weather_features_by_category('medium_weather')
    lookback_features = get_weather_features_by_category('medium_weather')
    
    print(f"  H+M features ({len(h_m_features)}): {h_m_features}")
    print(f"  Lookback features ({len(lookback_features)}): {lookback_features}")
    print(f"  特征相同: {h_m_features == lookback_features}")
    
    return config_h_m, config_lookback


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
    
    # 调试配置差异
    config_h_m, config_lookback = debug_config_differences(plant_config, model)
    
    results = {}
    
    # 实验1: Weather H+M (experiment 3)
    print(f"\n🧪 运行Weather H+M实验...")
    try:
        result_h_m = run_single_experiment(config_h_m, df.copy(), use_sliding_windows=False)
        
        if result_h_m['status'] == 'SUCCESS':
            results['weather_h_m'] = {
                'rmse': result_h_m['rmse'],
                'mae': result_h_m['mae'],
                'samples': result_h_m['test_samples']
            }
            print(f"  ✅ Weather H+M: RMSE={result_h_m['rmse']:.4f}, MAE={result_h_m['mae']:.4f}, Samples={result_h_m['test_samples']}")
        else:
            print(f"  ❌ Weather H+M实验失败: {result_h_m.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"  ❌ Weather H+M实验错误: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 实验2: Lookback 24h (experiment 4)
    print(f"\n🧪 运行Lookback 24h实验...")
    try:
        result_lookback = run_single_experiment(config_lookback, df.copy(), use_sliding_windows=False)
        
        if result_lookback['status'] == 'SUCCESS':
            results['lookback_24h'] = {
                'rmse': result_lookback['rmse'],
                'mae': result_lookback['mae'],
                'samples': result_lookback['test_samples']
            }
            print(f"  ✅ Lookback 24h: RMSE={result_lookback['rmse']:.4f}, MAE={result_lookback['mae']:.4f}, Samples={result_lookback['test_samples']}")
        else:
            print(f"  ❌ Lookback 24h实验失败: {result_lookback.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"  ❌ Lookback 24h实验错误: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 比较结果
    print(f"\n📊 结果对比:")
    print(f"  Weather H+M: RMSE={results['weather_h_m']['rmse']:.4f}, MAE={results['weather_h_m']['mae']:.4f}, Samples={results['weather_h_m']['samples']}")
    print(f"  Lookback 24h: RMSE={results['lookback_24h']['rmse']:.4f}, MAE={results['lookback_24h']['mae']:.4f}, Samples={results['lookback_24h']['samples']}")
    
    # 计算差异
    rmse_diff = abs(results['weather_h_m']['rmse'] - results['lookback_24h']['rmse'])
    mae_diff = abs(results['weather_h_m']['mae'] - results['lookback_24h']['mae'])
    samples_diff = abs(results['weather_h_m']['samples'] - results['lookback_24h']['samples'])
    
    print(f"\n🔍 差异分析:")
    print(f"  RMSE差异: {rmse_diff:.6f}")
    print(f"  MAE差异: {mae_diff:.6f}")
    print(f"  样本数差异: {samples_diff}")
    
    if rmse_diff < 1e-6 and mae_diff < 1e-6 and samples_diff == 0:
        print(f"  ✅ 结果完全一致!")
    else:
        print(f"  ❌ 结果不一致!")
        if rmse_diff >= 1e-6:
            print(f"    - RMSE差异过大: {rmse_diff:.6f}")
        if mae_diff >= 1e-6:
            print(f"    - MAE差异过大: {mae_diff:.6f}")
        if samples_diff != 0:
            print(f"    - 样本数不同: {samples_diff}")
    
    return results


def run_actual_experiments(data_dir, output_dir, model='LSTM'):
    """
    直接运行实际的实验脚本
    
    Args:
        data_dir: 数据目录
        output_dir: 输出目录
        model: 模型名称
    
    Returns:
        dict: 实验结果
    """
    print(f"\n🚀 运行实际实验脚本...")
    print(f"=" * 80)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # 运行 Weather Feature Adoption 实验 (experiment 3)
    print(f"\n📊 运行 Weather Feature Adoption 实验...")
    try:
        cmd = [
            'python', 'sensitivity_analysis/weather_feature_adoption.py',
            '--data-dir', data_dir,
            '--output-dir', output_dir,
            '--local-output', output_dir
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        print(f"⏳ 正在运行Weather Feature Adoption实验...")
        print(f"📊 预计运行: 2厂 × 8模型 × 4特征层级 = 64个实验")
        print(f"🕐 预计时间: 约5-10分钟")
        print(f"💡 提示: 实验正在后台运行，请耐心等待...")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print(f"✅ Weather Feature Adoption 实验成功")
            # 读取结果文件
            result_file = os.path.join(output_dir, 'weather_feature_adoption_aggregated.csv')
            if os.path.exists(result_file):
                df = pd.read_csv(result_file)
                h_m_results = df[df['feature_tier'] == 'H+M']
                if len(h_m_results) > 0:
                    lstm_result = h_m_results[h_m_results['model'] == model]
                    if len(lstm_result) > 0:
                        results['weather_h_m'] = {
                            'rmse': lstm_result['rmse_mean'].iloc[0],
                            'mae': lstm_result['mae_mean'].iloc[0],
                            'samples': lstm_result['n_plants'].iloc[0]
                        }
                        print(f"  H+M 结果: RMSE={results['weather_h_m']['rmse']:.4f}, MAE={results['weather_h_m']['mae']:.4f}")
                    else:
                        print(f"  ❌ 未找到 {model} 模型结果")
                else:
                    print(f"  ❌ 未找到 H+M 结果")
            else:
                print(f"  ❌ 结果文件不存在: {result_file}")
        else:
            print(f"❌ Weather Feature Adoption 实验失败")
            print(f"错误输出: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print(f"❌ Weather Feature Adoption 实验超时")
    except Exception as e:
        print(f"❌ Weather Feature Adoption 实验错误: {e}")
    
    # 运行 Lookback Window 实验 (experiment 4)
    print(f"\n📊 运行 Lookback Window 实验...")
    try:
        cmd = [
            'python', 'sensitivity_analysis/lookback_window.py',
            '--data-dir', data_dir,
            '--output-dir', output_dir,
            '--local-output', output_dir
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        print(f"⏳ 正在运行Lookback Window实验...")
        print(f"📊 预计运行: 2厂 × 7模型 × 4时间窗口 = 56个实验")
        print(f"🕐 预计时间: 约5-10分钟")
        print(f"💡 提示: 实验正在后台运行，请耐心等待...")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print(f"✅ Lookback Window 实验成功")
            # 读取结果文件
            result_file = os.path.join(output_dir, 'lookback_window_aggregated.csv')
            if os.path.exists(result_file):
                df = pd.read_csv(result_file)
                lookback_24_results = df[df['lookback_hours'] == 24]
                if len(lookback_24_results) > 0:
                    lstm_result = lookback_24_results[lookback_24_results['model'] == model]
                    if len(lstm_result) > 0:
                        results['lookback_24h'] = {
                            'rmse': lstm_result['rmse_mean'].iloc[0],
                            'mae': lstm_result['mae_mean'].iloc[0],
                            'samples': lstm_result['n_plants'].iloc[0]
                        }
                        print(f"  Lookback 24h 结果: RMSE={results['lookback_24h']['rmse']:.4f}, MAE={results['lookback_24h']['mae']:.4f}")
                    else:
                        print(f"  ❌ 未找到 {model} 模型结果")
                else:
                    print(f"  ❌ 未找到 Lookback 24h 结果")
            else:
                print(f"  ❌ 结果文件不存在: {result_file}")
        else:
            print(f"❌ Lookback Window 实验失败")
            print(f"错误输出: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print(f"❌ Lookback Window 实验超时")
    except Exception as e:
        print(f"❌ Lookback Window 实验错误: {e}")
    
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
    output_dir = 'test_consistency_results'
    
    if not os.path.exists(data_dir):
        print(f"错误: 数据目录不存在: {data_dir}")
        return
    
    # 方法1: 直接调用实际实验脚本
    print(f"\n🔬 方法1: 直接调用实际实验脚本")
    print(f"=" * 80)
    
    actual_results = run_actual_experiments(data_dir, output_dir, model='LSTM')
    
    if len(actual_results) == 2:
        print(f"\n📊 实际实验结果对比:")
        print(f"  Weather H+M: RMSE={actual_results['weather_h_m']['rmse']:.4f}, MAE={actual_results['weather_h_m']['mae']:.4f}")
        print(f"  Lookback 24h: RMSE={actual_results['lookback_24h']['rmse']:.4f}, MAE={actual_results['lookback_24h']['mae']:.4f}")
        
        rmse_diff = abs(actual_results['weather_h_m']['rmse'] - actual_results['lookback_24h']['rmse'])
        mae_diff = abs(actual_results['weather_h_m']['mae'] - actual_results['lookback_24h']['mae'])
        
        print(f"\n🔍 实际实验差异分析:")
        print(f"  RMSE差异: {rmse_diff:.6f}")
        print(f"  MAE差异: {mae_diff:.6f}")
        
        if rmse_diff < 1e-6 and mae_diff < 1e-6:
            print(f"  ✅ 实际实验结果完全一致!")
        else:
            print(f"  ❌ 实际实验结果不一致!")
    else:
        print(f"❌ 实际实验失败，只获得 {len(actual_results)} 个结果")
    
    # 方法2: 直接测试单厂配置
    print(f"\n🔬 方法2: 直接测试单厂配置")
    print(f"=" * 80)
    
    # 查找CSV文件
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if len(csv_files) > 0:
        # 测试第一个厂
        plant_id = csv_files[0].replace('Project', '').replace('.csv', '')
        data_path = os.path.join(data_dir, csv_files[0])
        
        single_results = test_single_plant_consistency(plant_id, data_path, model='LSTM')
        
        if single_results is not None:
            print(f"\n📊 单厂配置测试结果:")
            print(f"  Weather H+M: RMSE={single_results['weather_h_m']['rmse']:.4f}, MAE={single_results['weather_h_m']['mae']:.4f}")
            print(f"  Lookback 24h: RMSE={single_results['lookback_24h']['rmse']:.4f}, MAE={single_results['lookback_24h']['mae']:.4f}")
        else:
            print(f"❌ 单厂配置测试失败")
    else:
        print(f"❌ 未找到CSV文件")
    
    print(f"\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)


if __name__ == '__main__':
    main()
