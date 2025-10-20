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

# 设置全局随机种子确保完全可重复性
def set_global_seed(seed=42):
    """设置所有随机种子确保完全可重复性"""
    import random
    import torch
    
    # Python随机种子
    random.seed(seed)
    
    # NumPy随机种子
    np.random.seed(seed)
    
    # PyTorch随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 确保PyTorch的确定性行为
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"🎲 已设置全局随机种子: {seed}")

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sensitivity_analysis.common_utils import (
    create_base_config, run_single_experiment, load_all_plant_configs
)
from data.data_utils import load_raw_data, get_weather_features_by_category, preprocess_features, create_daily_windows, split_data


def debug_data_processing(config, df, experiment_name):
    """
    调试数据处理过程
    """
    print(f"\n🔍 调试 {experiment_name} 数据处理过程:")
    
    # 数据预处理
    print(f"   1. 数据预处理...")
    df_clean, hist_feats, fcst_feats, scaler_hist, scaler_fcst, scaler_target, no_hist_power = preprocess_features(df, config)
    print(f"     历史特征: {len(hist_feats)} 个 - {hist_feats}")
    print(f"     预测特征: {len(fcst_feats)} 个 - {fcst_feats}")
    print(f"     无历史功率: {no_hist_power}")
    
    # 创建窗口
    print(f"   2. 创建窗口...")
    past_hours = config.get('past_hours', 24)
    X_hist, X_fcst, y, hours, dates = create_daily_windows(
        df_clean, config['future_hours'], hist_feats, fcst_feats, no_hist_power, past_hours
    )
    print(f"     历史数据形状: {X_hist.shape}")
    print(f"     预测数据形状: {X_fcst.shape if X_fcst is not None else 'None'}")
    print(f"     目标数据形状: {y.shape}")
    print(f"     总样本数: {len(X_hist)}")
    
    # 数据分割
    print(f"   3. 数据分割...")
    total_samples = len(X_hist)
    indices = np.arange(total_samples)
    
    shuffle_split = config.get('shuffle_split', True)
    random_seed = config.get('random_seed', 42)
    
    if shuffle_split:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        print(f"     随机种子: {random_seed}, 已打乱数据")
    else:
        print(f"     未打乱数据")
    
    train_ratio = config.get('train_ratio', 0.8)
    val_ratio = config.get('val_ratio', 0.1)
    
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    print(f"     训练集: {len(train_idx)} 样本")
    print(f"     验证集: {len(val_idx)} 样本")
    print(f"     测试集: {len(test_idx)} 样本")
    
    # 检查测试集数据
    y_test = y[test_idx]
    print(f"     测试集目标值范围: {y_test.min():.4f} - {y_test.max():.4f}")
    print(f"     测试集目标值均值: {y_test.mean():.4f}")
    print(f"     测试集目标值标准差: {y_test.std():.4f}")
    
    return {
        'X_hist': X_hist,
        'X_fcst': X_fcst,
        'y': y,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
        'y_test': y_test
    }


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
    
    # 详细比较配置
    print(f"\n🔍 详细配置比较:")
    print(f"   Weather H+M 配置:")
    for key, value in config_h_m.items():
        print(f"     {key}: {value}")
    
    print(f"\n   Lookback 24h 配置:")
    for key, value in config_lookback.items():
        print(f"     {key}: {value}")
    
    # 检查配置是否完全相同
    configs_equal = True
    print(f"\n🔍 配置差异检查:")
    for key in config_h_m:
        if key in config_lookback:
            if config_h_m[key] != config_lookback[key]:
                print(f"   ❌ {key}: H+M={config_h_m[key]} vs Lookback={config_lookback[key]}")
                configs_equal = False
            else:
                print(f"   ✅ {key}: {config_h_m[key]}")
        else:
            print(f"   ⚠️  {key}: 只在H+M中存在")
            configs_equal = False
    
    if configs_equal:
        print(f"\n✅ 配置完全相同!")
    else:
        print(f"\n❌ 配置存在差异!")
    
    # 运行实验1: Weather H+M
    print(f"\n🧪 实验1: Weather H+M (medium_weather)")
    print(f"   模型: {config_h_m['model']}")
    print(f"   复杂度: {config_h_m['model_complexity']}")
    print(f"   特征: {config_h_m['weather_category']} ({len(features)}个)")
    print(f"   回看窗口: {config_h_m['past_hours']}h")
    print(f"   随机种子: {config_h_m['random_seed']}")
    print(f"   数据分割: {config_h_m['shuffle_split']}")
    print(f"   ⏳ 开始训练...")
    
    # 重新设置随机种子确保实验间的一致性
    set_global_seed(config_h_m['random_seed'])
    
    # 调试数据处理过程
    debug_data_h_m = debug_data_processing(config_h_m, df.copy(), "Weather H+M")
    
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
    print(f"   随机种子: {config_lookback['random_seed']}")
    print(f"   数据分割: {config_lookback['shuffle_split']}")
    print(f"   ⏳ 开始训练...")
    
    # 重新设置随机种子确保实验间的一致性
    set_global_seed(config_lookback['random_seed'])
    
    # 调试数据处理过程
    debug_data_lookback = debug_data_processing(config_lookback, df.copy(), "Lookback 24h")
    
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
    
    # 比较测试集数据
    print(f"\n🔍 测试集数据比较:")
    y_test_h_m = debug_data_h_m['y_test']
    y_test_lookback = debug_data_lookback['y_test']
    
    print(f"   Weather H+M 测试集:")
    print(f"     样本数: {len(y_test_h_m)}")
    print(f"     范围: {y_test_h_m.min():.6f} - {y_test_h_m.max():.6f}")
    print(f"     均值: {y_test_h_m.mean():.6f}")
    print(f"     标准差: {y_test_h_m.std():.6f}")
    
    print(f"   Lookback 24h 测试集:")
    print(f"     样本数: {len(y_test_lookback)}")
    print(f"     范围: {y_test_lookback.min():.6f} - {y_test_lookback.max():.6f}")
    print(f"     均值: {y_test_lookback.mean():.6f}")
    print(f"     标准差: {y_test_lookback.std():.6f}")
    
    # 检查测试集是否相同
    test_data_same = np.array_equal(y_test_h_m, y_test_lookback)
    print(f"   测试集数据相同: {test_data_same}")
    
    if not test_data_same:
        diff_count = np.sum(y_test_h_m != y_test_lookback)
        print(f"   不同样本数: {diff_count} / {len(y_test_h_m)}")
        if diff_count < 10:  # 只显示前几个不同的值
            diff_indices = np.where(y_test_h_m != y_test_lookback)[0][:5]
            for idx in diff_indices:
                print(f"     样本 {idx}: H+M={y_test_h_m[idx]:.6f}, Lookback={y_test_lookback[idx]:.6f}")
    
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
    
    # 设置全局随机种子确保完全可重复性
    set_global_seed(42)
    
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
