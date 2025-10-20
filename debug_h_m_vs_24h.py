#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试H+M和24h lookback结果不一致问题
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
    create_base_config, run_single_experiment, load_all_plant_configs,
    set_global_seed
)
from data.data_utils import load_raw_data, get_weather_features_by_category, preprocess_features, create_daily_windows, split_data


def debug_h_m_vs_24h(plant_id, data_path, model='LSTM'):
    """
    调试H+M和24h lookback的差异
    """
    print(f"🔍 调试H+M vs 24h Lookback - Plant {plant_id}")
    print(f"=" * 80)
    
    # 加载数据
    print(f"📂 加载数据...")
    df = load_raw_data(data_path)
    df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
    print(f"✅ 数据加载成功: {df.shape[0]} 行, {df.shape[1]} 列")
    
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
    
    # 实验1: Weather H+M (experiment 3)
    print(f"\n🧪 实验1: Weather H+M (experiment 3)")
    print(f"=" * 60)
    
    # 设置随机种子
    set_global_seed(42)
    
    config_h_m = create_base_config(plant_config, model, complexity='high', 
                                  lookback=24, use_te=False)
    config_h_m['weather_category'] = 'medium_weather'
    
    print(f"📋 H+M 配置:")
    for key, value in config_h_m.items():
        print(f"  {key}: {value}")
    
    # 调试数据处理
    print(f"\n🔍 H+M 数据处理调试:")
    df_clean_h_m, hist_feats_h_m, fcst_feats_h_m, scaler_hist_h_m, scaler_fcst_h_m, scaler_target_h_m, no_hist_power_h_m = preprocess_features(df.copy(), config_h_m)
    print(f"  历史特征: {hist_feats_h_m}")
    print(f"  预测特征: {fcst_feats_h_m}")
    print(f"  无历史功率: {no_hist_power_h_m}")
    
    # 创建窗口
    X_hist_h_m, X_fcst_h_m, y_h_m, hours_h_m, dates_h_m = create_daily_windows(
        df_clean_h_m, config_h_m['future_hours'], hist_feats_h_m, fcst_feats_h_m, no_hist_power_h_m, config_h_m['past_hours']
    )
    print(f"  历史数据形状: {X_hist_h_m.shape}")
    print(f"  预测数据形状: {X_fcst_h_m.shape if X_fcst_h_m is not None else 'None'}")
    print(f"  目标数据形状: {y_h_m.shape}")
    
    # 数据分割
    total_samples = len(X_hist_h_m)
    indices = np.arange(total_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_size = int(total_samples * 0.8)
    val_size = int(total_samples * 0.1)
    
    train_idx_h_m = indices[:train_size]
    val_idx_h_m = indices[train_size:train_size + val_size]
    test_idx_h_m = indices[train_size + val_size:]
    
    print(f"  训练集: {len(train_idx_h_m)} 样本")
    print(f"  验证集: {len(val_idx_h_m)} 样本")
    print(f"  测试集: {len(test_idx_h_m)} 样本")
    
    # 运行H+M实验
    print(f"\n⏳ 运行H+M实验...")
    result_h_m = run_single_experiment(config_h_m, df.copy(), use_sliding_windows=False)
    
    if result_h_m['status'] == 'SUCCESS':
        print(f"✅ H+M 结果: RMSE={result_h_m['rmse']:.4f}, MAE={result_h_m['mae']:.4f}")
    else:
        print(f"❌ H+M 失败: {result_h_m.get('error', 'Unknown error')}")
        return
    
    # 实验2: Lookback 24h (experiment 4)
    print(f"\n🧪 实验2: Lookback 24h (experiment 4)")
    print(f"=" * 60)
    
    # 重新设置随机种子
    set_global_seed(42)
    
    config_lookback = create_base_config(plant_config, model, complexity='high', 
                                       lookback=24, use_te=False)
    config_lookback['weather_category'] = 'medium_weather'
    
    print(f"📋 Lookback 24h 配置:")
    for key, value in config_lookback.items():
        print(f"  {key}: {value}")
    
    # 调试数据处理
    print(f"\n🔍 Lookback 24h 数据处理调试:")
    df_clean_lookback, hist_feats_lookback, fcst_feats_lookback, scaler_hist_lookback, scaler_fcst_lookback, scaler_target_lookback, no_hist_power_lookback = preprocess_features(df.copy(), config_lookback)
    print(f"  历史特征: {hist_feats_lookback}")
    print(f"  预测特征: {fcst_feats_lookback}")
    print(f"  无历史功率: {no_hist_power_lookback}")
    
    # 创建窗口
    X_hist_lookback, X_fcst_lookback, y_lookback, hours_lookback, dates_lookback = create_daily_windows(
        df_clean_lookback, config_lookback['future_hours'], hist_feats_lookback, fcst_feats_lookback, no_hist_power_lookback, config_lookback['past_hours']
    )
    print(f"  历史数据形状: {X_hist_lookback.shape}")
    print(f"  预测数据形状: {X_fcst_lookback.shape if X_fcst_lookback is not None else 'None'}")
    print(f"  目标数据形状: {y_lookback.shape}")
    
    # 数据分割
    total_samples = len(X_hist_lookback)
    indices = np.arange(total_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_size = int(total_samples * 0.8)
    val_size = int(total_samples * 0.1)
    
    train_idx_lookback = indices[:train_size]
    val_idx_lookback = indices[train_size:train_size + val_size]
    test_idx_lookback = indices[train_size + val_size:]
    
    print(f"  训练集: {len(train_idx_lookback)} 样本")
    print(f"  验证集: {len(val_idx_lookback)} 样本")
    print(f"  测试集: {len(test_idx_lookback)} 样本")
    
    # 运行Lookback实验
    print(f"\n⏳ 运行Lookback 24h实验...")
    result_lookback = run_single_experiment(config_lookback, df.copy(), use_sliding_windows=False)
    
    if result_lookback['status'] == 'SUCCESS':
        print(f"✅ Lookback 24h 结果: RMSE={result_lookback['rmse']:.4f}, MAE={result_lookback['mae']:.4f}")
    else:
        print(f"❌ Lookback 24h 失败: {result_lookback.get('error', 'Unknown error')}")
        return
    
    # 比较结果
    print(f"\n📊 结果对比:")
    print(f"  H+M: RMSE={result_h_m['rmse']:.4f}, MAE={result_h_m['mae']:.4f}")
    print(f"  Lookback 24h: RMSE={result_lookback['rmse']:.4f}, MAE={result_lookback['mae']:.4f}")
    
    rmse_diff = abs(result_h_m['rmse'] - result_lookback['rmse'])
    mae_diff = abs(result_h_m['mae'] - result_lookback['mae'])
    
    print(f"\n🔍 差异分析:")
    print(f"  RMSE差异: {rmse_diff:.6f}")
    print(f"  MAE差异: {mae_diff:.6f}")
    
    # 比较配置
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
        print(f"\n✅ 配置完全相同!")
    else:
        print(f"\n❌ 配置存在差异!")
    
    # 比较特征
    print(f"\n🔍 特征比较:")
    print(f"  H+M 历史特征: {hist_feats_h_m}")
    print(f"  Lookback 历史特征: {hist_feats_lookback}")
    print(f"  历史特征相同: {hist_feats_h_m == hist_feats_lookback}")
    
    print(f"  H+M 预测特征: {fcst_feats_h_m}")
    print(f"  Lookback 预测特征: {fcst_feats_lookback}")
    print(f"  预测特征相同: {fcst_feats_h_m == fcst_feats_lookback}")
    
    # 比较数据形状
    print(f"\n🔍 数据形状比较:")
    print(f"  H+M 历史数据: {X_hist_h_m.shape}")
    print(f"  Lookback 历史数据: {X_hist_lookback.shape}")
    print(f"  历史数据形状相同: {X_hist_h_m.shape == X_hist_lookback.shape}")
    
    print(f"  H+M 预测数据: {X_fcst_h_m.shape if X_fcst_h_m is not None else 'None'}")
    print(f"  Lookback 预测数据: {X_fcst_lookback.shape if X_fcst_lookback is not None else 'None'}")
    print(f"  预测数据形状相同: {X_fcst_h_m.shape == X_fcst_lookback.shape if X_fcst_h_m is not None and X_fcst_lookback is not None else X_fcst_h_m is X_fcst_lookback}")
    
    # 比较测试集索引
    print(f"\n🔍 测试集索引比较:")
    print(f"  H+M 测试集索引: {test_idx_h_m[:10]}...")
    print(f"  Lookback 测试集索引: {test_idx_lookback[:10]}...")
    print(f"  测试集索引相同: {np.array_equal(test_idx_h_m, test_idx_lookback)}")
    
    if not np.array_equal(test_idx_h_m, test_idx_lookback):
        print(f"  不同索引数量: {np.sum(test_idx_h_m != test_idx_lookback)}")
        diff_indices = np.where(test_idx_h_m != test_idx_lookback)[0][:5]
        for idx in diff_indices:
            print(f"    索引 {idx}: H+M={test_idx_h_m[idx]}, Lookback={test_idx_lookback[idx]}")


def main():
    """主函数"""
    print("=" * 80)
    print("🔍 H+M vs 24h Lookback 调试工具")
    print("=" * 80)
    
    # 查找CSV文件
    data_dir = 'data'
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if len(csv_files) == 0:
        print(f"❌ 未找到CSV文件")
        return
    
    # 测试第一个厂
    plant_id = csv_files[0].replace('Project', '').replace('.csv', '')
    data_path = os.path.join(data_dir, csv_files[0])
    
    print(f"🎯 测试厂: {plant_id}")
    
    # 运行调试
    debug_h_m_vs_24h(plant_id, data_path, model='LSTM')
    
    print(f"\n" + "=" * 80)
    print("✅ 调试完成!")
    print("=" * 80)


if __name__ == '__main__':
    main()
