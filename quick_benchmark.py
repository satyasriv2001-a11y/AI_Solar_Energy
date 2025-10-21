#!/usr/bin/env python3
"""
快速XGB vs LGBM基准测试
专注于核心配置的性能对比
"""
import time
import pandas as pd
import numpy as np
from sensitivity_analysis.common_utils import create_base_config, run_single_experiment, set_global_seed

def quick_benchmark():
    """快速比较XGB和LGBM性能"""
    print("🚀 快速XGB vs LGBM基准测试")
    print("=" * 50)
    
    # 设置随机种子
    set_global_seed(42)
    
    # 加载数据
    data_path = "data/Project1140.csv"
    df = pd.read_csv(data_path)
    
    # 检查并修复列名
    if 'DateTime' in df.columns and 'Datetime' not in df.columns:
        df = df.rename(columns={'DateTime': 'Datetime'})
        print("📝 已重命名 DateTime -> Datetime")
    
    # 转换Datetime列为datetime类型
    if 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        print("📝 已转换 Datetime 列为 datetime 类型")
    
    print(f"📊 数据形状: {df.shape}")
    
    # 核心测试配置
    test_configs = [
        # (model, complexity, lookback, weather_category, description)
        ('XGB', 'low', 24, 'medium_weather', 'XGB-Low-24h-Medium'),
        ('LGBM', 'low', 24, 'medium_weather', 'LGBM-Low-24h-Medium'),
        ('XGB', 'high', 24, 'medium_weather', 'XGB-High-24h-Medium'),
        ('LGBM', 'high', 24, 'medium_weather', 'LGBM-High-24h-Medium'),
        ('XGB', 'low', 72, 'medium_weather', 'XGB-Low-72h-Medium'),
        ('LGBM', 'low', 72, 'medium_weather', 'LGBM-Low-72h-Medium'),
        ('XGB', 'high', 72, 'medium_weather', 'XGB-High-72h-Medium'),
        ('LGBM', 'high', 72, 'medium_weather', 'LGBM-High-72h-Medium'),
    ]
    
    results = []
    
    print(f"\n🧪 开始快速基准测试...")
    print(f"📋 测试配置: {len(test_configs)} 个实验")
    
    for i, (model, complexity, lookback, weather_category, description) in enumerate(test_configs, 1):
        print(f"\n{'='*50}")
        print(f"🧪 实验 {i}/{len(test_configs)}: {description}")
        print(f"{'='*50}")
        
        try:
            # 创建配置
            config = create_base_config(
                plant_config={'plant_id': '1140', 'data_path': data_path},
                model=model,
                complexity=complexity,
                lookback=lookback,
                use_te=False
            )
            
            # 设置天气特征
            config['weather_category'] = weather_category
            
            # 记录开始时间
            start_time = time.time()
            
            # 运行实验
            result = run_single_experiment(config, df.copy(), use_sliding_windows=False)
            
            # 记录结束时间
            end_time = time.time()
            total_time = end_time - start_time
            
            if result['status'] == 'SUCCESS':
                print(f"  ✅ 成功: {total_time:.2f}s")
                print(f"     RMSE: {result['rmse']:.4f}")
                print(f"     MAE: {result['mae']:.4f}")
                print(f"     训练时间: {result['train_time']:.2f}s")
                
                results.append({
                    'description': description,
                    'model': model,
                    'complexity': complexity,
                    'lookback': lookback,
                    'weather_category': weather_category,
                    'total_time': total_time,
                    'train_time': result['train_time'],
                    'rmse': result['rmse'],
                    'mae': result['mae'],
                    'test_samples': result['test_samples'],
                    'status': 'SUCCESS'
                })
            else:
                print(f"  ❌ 失败: {result.get('error', 'Unknown error')}")
                results.append({
                    'description': description,
                    'model': model,
                    'complexity': complexity,
                    'lookback': lookback,
                    'weather_category': weather_category,
                    'total_time': total_time,
                    'train_time': 0,
                    'rmse': np.nan,
                    'mae': np.nan,
                    'test_samples': 0,
                    'status': 'FAILED'
                })
                
        except Exception as e:
            print(f"  ❌ 异常: {str(e)}")
            results.append({
                'description': description,
                'model': model,
                'complexity': complexity,
                'lookback': lookback,
                'weather_category': weather_category,
                'total_time': 0,
                'train_time': 0,
                'rmse': np.nan,
                'mae': np.nan,
                'test_samples': 0,
                'status': 'EXCEPTION'
            })
    
    # 分析结果
    print(f"\n📊 快速基准测试结果")
    print("=" * 50)
    
    df_results = pd.DataFrame(results)
    
    # 显示结果表格
    print(f"\n📋 结果表格:")
    print("-" * 80)
    print(f"{'描述':<25} {'总时间':<8} {'训练时间':<8} {'RMSE':<8} {'状态':<8}")
    print("-" * 80)
    
    for _, row in df_results.iterrows():
        if row['status'] == 'SUCCESS':
            print(f"{row['description']:<25} {row['total_time']:<8.2f} {row['train_time']:<8.2f} {row['rmse']:<8.4f} {row['status']:<8}")
        else:
            print(f"{row['description']:<25} {'N/A':<8} {'N/A':<8} {'N/A':<8} {row['status']:<8}")
    
    # 性能对比
    print(f"\n🏆 性能对比:")
    print("-" * 50)
    
    successful_results = df_results[df_results['status'] == 'SUCCESS']
    if len(successful_results) > 0:
        # 按模型分组
        xgb_results = successful_results[successful_results['model'] == 'XGB']
        lgbm_results = successful_results[successful_results['model'] == 'LGBM']
        
        if len(xgb_results) > 0:
            print(f"XGB 结果:")
            print(f"  平均总时间: {xgb_results['total_time'].mean():.2f}s")
            print(f"  平均训练时间: {xgb_results['train_time'].mean():.2f}s")
            print(f"  平均RMSE: {xgb_results['rmse'].mean():.4f}")
        
        if len(lgbm_results) > 0:
            print(f"LGBM 结果:")
            print(f"  平均总时间: {lgbm_results['total_time'].mean():.2f}s")
            print(f"  平均训练时间: {lgbm_results['train_time'].mean():.2f}s")
            print(f"  平均RMSE: {lgbm_results['rmse'].mean():.4f}")
        
        # 直接对比
        if len(xgb_results) > 0 and len(lgbm_results) > 0:
            print(f"\n📊 直接对比:")
            xgb_avg = xgb_results['total_time'].mean()
            lgbm_avg = lgbm_results['total_time'].mean()
            
            print(f"  XGB 平均总时间: {xgb_avg:.2f}s")
            print(f"  LGBM 平均总时间: {lgbm_avg:.2f}s")
            
            if xgb_avg < lgbm_avg:
                speedup = lgbm_avg / xgb_avg
                print(f"  🥇 XGB 比 LGBM 快 {speedup:.2f}x")
            else:
                speedup = xgb_avg / lgbm_avg
                print(f"  🥇 LGBM 比 XGB 快 {speedup:.2f}x")
    
    # 保存结果
    output_file = "quick_benchmark_results.csv"
    df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n💾 结果已保存到: {output_file}")
    
    return df_results

if __name__ == "__main__":
    quick_benchmark()
