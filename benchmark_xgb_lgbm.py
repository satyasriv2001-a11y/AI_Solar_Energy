#!/usr/bin/env python3
"""
XGB vs LGBM 性能基准测试
在GPU环境下比较XGB和LGBM在不同配置下的训练时间
"""
import time
import pandas as pd
import numpy as np
from sensitivity_analysis.common_utils import create_base_config, run_single_experiment, set_global_seed

def benchmark_xgb_lgbm():
    """比较XGB和LGBM在不同配置下的性能"""
    print("🚀 XGB vs LGBM 性能基准测试")
    print("=" * 60)
    
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
    print(f"📊 时间范围: {df['Datetime'].min()} 到 {df['Datetime'].max()}")
    
    # 测试配置
    models_to_test = ['XGB', 'LGBM']
    complexities = ['low', 'high']
    lookbacks = [24, 72, 120, 168]  # 不同lookback窗口
    weather_categories = ['solar_irradiance_only', 'high_weather', 'medium_weather', 'low_weather']
    
    results = []
    
    print(f"\n🧪 开始基准测试...")
    print(f"📋 测试配置:")
    print(f"   - 模型: {models_to_test}")
    print(f"   - 复杂度: {complexities}")
    print(f"   - Lookback: {lookbacks}")
    print(f"   - 天气特征: {weather_categories}")
    
    total_experiments = len(models_to_test) * len(complexities) * len(lookbacks) * len(weather_categories)
    current_exp = 0
    
    for model in models_to_test:
        for complexity in complexities:
            for lookback in lookbacks:
                for weather_category in weather_categories:
                    current_exp += 1
                    exp_name = f"{model}_{complexity}_{lookback}h_{weather_category}"
                    
                    print(f"\n{'='*60}")
                    print(f"🧪 实验 {current_exp}/{total_experiments}: {exp_name}")
                    print(f"{'='*60}")
                    
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
                            print(f"     测试样本: {result['test_samples']}")
                            
                            results.append({
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
    print(f"\n📊 基准测试结果分析")
    print("=" * 60)
    
    df_results = pd.DataFrame(results)
    
    # 按模型分组显示
    for model in models_to_test:
        model_results = df_results[df_results['model'] == model]
        successful_results = model_results[model_results['status'] == 'SUCCESS']
        
        print(f"\n🔍 {model} 结果:")
        if len(successful_results) > 0:
            print(f"  成功实验: {len(successful_results)}/{len(model_results)}")
            print(f"  平均总时间: {successful_results['total_time'].mean():.2f}s")
            print(f"  平均训练时间: {successful_results['train_time'].mean():.2f}s")
            print(f"  平均RMSE: {successful_results['rmse'].mean():.4f}")
            print(f"  平均MAE: {successful_results['mae'].mean():.4f}")
        else:
            print(f"  成功实验: 0/{len(model_results)}")
    
    # 详细结果表格
    print(f"\n📋 详细结果表格:")
    print("-" * 100)
    print(f"{'模型':<6} {'复杂度':<6} {'Lookback':<8} {'天气特征':<20} {'总时间':<8} {'训练时间':<8} {'RMSE':<8} {'状态':<8}")
    print("-" * 100)
    
    for _, row in df_results.iterrows():
        if row['status'] == 'SUCCESS':
            print(f"{row['model']:<6} {row['complexity']:<6} {row['lookback']:<8} {row['weather_category']:<20} {row['total_time']:<8.2f} {row['train_time']:<8.2f} {row['rmse']:<8.4f} {row['status']:<8}")
        else:
            print(f"{row['model']:<6} {row['complexity']:<6} {row['lookback']:<8} {row['weather_category']:<20} {'N/A':<8} {'N/A':<8} {'N/A':<8} {row['status']:<8}")
    
    # 性能对比
    print(f"\n🏆 性能对比分析:")
    print("-" * 60)
    
    successful_results = df_results[df_results['status'] == 'SUCCESS']
    if len(successful_results) > 0:
        # 按模型分组比较
        for model in models_to_test:
            model_data = successful_results[successful_results['model'] == model]
            if len(model_data) > 0:
                print(f"\n{model}:")
                print(f"  平均总时间: {model_data['total_time'].mean():.2f}s")
                print(f"  最快实验: {model_data['total_time'].min():.2f}s")
                print(f"  最慢实验: {model_data['total_time'].max():.2f}s")
                print(f"  平均训练时间: {model_data['train_time'].mean():.2f}s")
                print(f"  平均RMSE: {model_data['rmse'].mean():.4f}")
        
        # 直接对比
        xgb_results = successful_results[successful_results['model'] == 'XGB']
        lgbm_results = successful_results[successful_results['model'] == 'LGBM']
        
        if len(xgb_results) > 0 and len(lgbm_results) > 0:
            print(f"\n📊 直接对比:")
            print(f"  XGB 平均总时间: {xgb_results['total_time'].mean():.2f}s")
            print(f"  LGBM 平均总时间: {lgbm_results['total_time'].mean():.2f}s")
            
            if xgb_results['total_time'].mean() < lgbm_results['total_time'].mean():
                speedup = lgbm_results['total_time'].mean() / xgb_results['total_time'].mean()
                print(f"  🥇 XGB 比 LGBM 快 {speedup:.2f}x")
            else:
                speedup = xgb_results['total_time'].mean() / lgbm_results['total_time'].mean()
                print(f"  🥇 LGBM 比 XGB 快 {speedup:.2f}x")
    
    # 保存结果到CSV
    output_file = "xgb_lgbm_benchmark_results.csv"
    df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n💾 结果已保存到: {output_file}")
    
    return df_results

if __name__ == "__main__":
    benchmark_xgb_lgbm()
