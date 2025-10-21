#!/usr/bin/env python3
"""
大参数配置XGB vs LGBM基准测试
测试在非常大参数配置下的性能表现
"""
import time
import pandas as pd
import numpy as np
from sensitivity_analysis.common_utils import create_base_config, run_single_experiment, set_global_seed

def large_config_benchmark():
    """测试大参数配置下的性能"""
    print("🚀 大参数配置XGB vs LGBM基准测试")
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
    
    # 大参数配置测试
    large_test_configs = [
        # (model, complexity, lookback, weather_category, description, custom_params)
        ('XGB', 'high', 24, 'medium_weather', 'XGB-High-24h-Medium', {}),
        ('LGBM', 'high', 24, 'medium_weather', 'LGBM-High-24h-Medium', {}),
        
        # 超大参数配置
        ('XGB', 'high', 24, 'medium_weather', 'XGB-Extra-Large', {
            'n_estimators': 1000,  # 1000棵树
            'max_depth': 10,       # 深度10
            'learning_rate': 0.01, # 小学习率
            'subsample': 0.8,      # 子采样
            'colsample_bytree': 0.8, # 特征采样
        }),
        ('LGBM', 'high', 24, 'medium_weather', 'LGBM-Extra-Large', {
            'n_estimators': 1000,  # 1000棵树
            'max_depth': 10,       # 深度10
            'learning_rate': 0.01, # 小学习率
            'subsample': 0.8,      # 子采样
            'colsample_bytree': 0.8, # 特征采样
        }),
        
        # 极大参数配置
        ('XGB', 'high', 24, 'medium_weather', 'XGB-Mega-Large', {
            'n_estimators': 2000,  # 2000棵树
            'max_depth': 15,       # 深度15
            'learning_rate': 0.005, # 更小学习率
            'subsample': 0.7,      # 子采样
            'colsample_bytree': 0.7, # 特征采样
            'reg_alpha': 0.1,      # L1正则化
            'reg_lambda': 0.1,     # L2正则化
        }),
        ('LGBM', 'high', 24, 'medium_weather', 'LGBM-Mega-Large', {
            'n_estimators': 2000,  # 2000棵树
            'max_depth': 15,       # 深度15
            'learning_rate': 0.005, # 更小学习率
            'subsample': 0.7,      # 子采样
            'colsample_bytree': 0.7, # 特征采样
            'reg_alpha': 0.1,   # L1正则化
            'reg_lambda': 0.1,     # L2正则化
        }),
        
        # 超大数据集测试
        ('XGB', 'high', 168, 'low_weather', 'XGB-Large-Data', {
            'n_estimators': 500,
            'max_depth': 8,
            'learning_rate': 0.02,
        }),
        ('LGBM', 'high', 168, 'low_weather', 'LGBM-Large-Data', {
            'n_estimators': 500,
            'max_depth': 8,
            'learning_rate': 0.02,
        }),
    ]
    
    results = []
    
    print(f"\n🧪 开始大参数配置基准测试...")
    print(f"📋 测试配置: {len(large_test_configs)} 个实验")
    print(f"⚠️  注意: 大参数配置可能需要更长时间")
    
    for i, (model, complexity, lookback, weather_category, description, custom_params) in enumerate(large_test_configs, 1):
        print(f"\n{'='*60}")
        print(f"🧪 实验 {i}/{len(large_test_configs)}: {description}")
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
            
            # 应用自定义参数
            if custom_params:
                config['model_params'].update(custom_params)
                print(f"  🔧 自定义参数: {custom_params}")
            
            # 记录各个阶段的时间
            print(f"  📊 配置详情:")
            print(f"     - 模型: {model}")
            print(f"     - 复杂度: {complexity}")
            print(f"     - Lookback: {lookback}h")
            print(f"     - 天气特征: {weather_category}")
            print(f"     - 最终参数: {config.get('model_params', {})}")
            
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
                
                # 计算效率指标
                efficiency = result['test_samples'] / total_time if total_time > 0 else 0
                print(f"     效率: {efficiency:.1f} 样本/秒")
                
                results.append({
                    'description': description,
                    'model': model,
                    'complexity': complexity,
                    'lookback': lookback,
                    'weather_category': weather_category,
                    'custom_params': str(custom_params),
                    'total_time': total_time,
                    'train_time': result['train_time'],
                    'rmse': result['rmse'],
                    'mae': result['mae'],
                    'test_samples': result['test_samples'],
                    'efficiency': efficiency,
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
                    'custom_params': str(custom_params),
                    'total_time': total_time,
                    'train_time': 0,
                    'rmse': np.nan,
                    'mae': np.nan,
                    'test_samples': 0,
                    'efficiency': 0,
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
                'custom_params': str(custom_params),
                'total_time': 0,
                'train_time': 0,
                'rmse': np.nan,
                'mae': np.nan,
                'test_samples': 0,
                'efficiency': 0,
                'status': 'EXCEPTION'
            })
    
    # 分析结果
    print(f"\n📊 大参数配置基准测试结果")
    print("=" * 60)
    
    df_results = pd.DataFrame(results)
    
    # 显示结果表格
    print(f"\n📋 结果表格:")
    print("-" * 100)
    print(f"{'描述':<20} {'模型':<6} {'总时间':<8} {'训练时间':<8} {'RMSE':<8} {'效率':<10} {'状态':<8}")
    print("-" * 100)
    
    for _, row in df_results.iterrows():
        if row['status'] == 'SUCCESS':
            print(f"{row['description']:<20} {row['model']:<6} {row['total_time']:<8.2f} {row['train_time']:<8.2f} {row['rmse']:<8.4f} {row['efficiency']:<10.1f} {row['status']:<8}")
        else:
            print(f"{row['description']:<20} {row['model']:<6} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<10} {row['status']:<8}")
    
    # 性能对比
    print(f"\n🏆 大参数配置性能对比:")
    print("-" * 60)
    
    successful_results = df_results[df_results['status'] == 'SUCCESS']
    if len(successful_results) > 0:
        # 按模型分组
        xgb_results = successful_results[successful_results['model'] == 'XGB']
        lgbm_results = successful_results[successful_results['model'] == 'LGBM']
        
        if len(xgb_results) > 0:
            print(f"XGB 大参数配置结果:")
            print(f"  平均总时间: {xgb_results['total_time'].mean():.2f}s")
            print(f"  平均训练时间: {xgb_results['train_time'].mean():.2f}s")
            print(f"  平均RMSE: {xgb_results['rmse'].mean():.4f}")
            print(f"  平均效率: {xgb_results['efficiency'].mean():.1f} 样本/秒")
        
        if len(lgbm_results) > 0:
            print(f"LGBM 大参数配置结果:")
            print(f"  平均总时间: {lgbm_results['total_time'].mean():.2f}s")
            print(f"  平均训练时间: {lgbm_results['train_time'].mean():.2f}s")
            print(f"  平均RMSE: {lgbm_results['rmse'].mean():.4f}")
            print(f"  平均效率: {lgbm_results['efficiency'].mean():.1f} 样本/秒")
        
        # 直接对比
        if len(xgb_results) > 0 and len(lgbm_results) > 0:
            print(f"\n📊 大参数配置直接对比:")
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
            
            # 效率对比
            xgb_efficiency = xgb_results['efficiency'].mean()
            lgbm_efficiency = lgbm_results['efficiency'].mean()
            print(f"  XGB 平均效率: {xgb_efficiency:.1f} 样本/秒")
            print(f"  LGBM 平均效率: {lgbm_efficiency:.1f} 样本/秒")
    
    # 参数规模分析
    print(f"\n📈 参数规模影响分析:")
    print("-" * 60)
    
    # 按参数规模分组
    small_params = successful_results[successful_results['description'].str.contains('High-24h-Medium')]
    large_params = successful_results[successful_results['description'].str.contains('Extra-Large')]
    mega_params = successful_results[successful_results['description'].str.contains('Mega-Large')]
    
    for group_name, group_data in [("标准参数", small_params), ("超大参数", large_params), ("极大参数", mega_params)]:
        if len(group_data) > 0:
            print(f"\n{group_name}:")
            print(f"  平均总时间: {group_data['total_time'].mean():.2f}s")
            print(f"  平均训练时间: {group_data['train_time'].mean():.2f}s")
            print(f"  平均效率: {group_data['efficiency'].mean():.1f} 样本/秒")
    
    # 保存结果
    output_file = "large_config_benchmark_results.csv"
    df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n💾 结果已保存到: {output_file}")
    
    return df_results

if __name__ == "__main__":
    large_config_benchmark()
