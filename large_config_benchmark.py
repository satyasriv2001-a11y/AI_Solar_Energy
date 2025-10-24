#!/usr/bin/env python3
"""
大参数配置XGB vs LGBM基准测试
测试在非常大参数配置下的性能表现
使用所有数据集进行200个实验
"""
import time
import pandas as pd
import numpy as np
import os
import glob
from sensitivity_analysis.common_utils import create_base_config, run_single_experiment, set_global_seed

def large_config_benchmark():
    """测试大参数配置下的性能"""
    print("🚀 大参数配置XGB vs LGBM基准测试")
    print("=" * 60)
    
    # 设置随机种子
    set_global_seed(42)
    
    # 获取所有数据集文件
    data_files = glob.glob("data/Project*.csv")
    print(f"📁 发现数据集文件: {len(data_files)} 个")
    for file in data_files:
        print(f"   - {file}")
    
    # 超大参数配置 - 每个厂进行24hour lookback noTE medium_weather Extra-Large实验
    extra_large_configs = [
        # (model, complexity, lookback, weather_category, description, custom_params)
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
    ]
    
    results = []
    
    # 计算总实验数：每个数据集 × 每个模型配置 × 100次重复
    total_experiments = len(data_files) * len(extra_large_configs) * 100
    print(f"\n🧪 开始大参数配置基准测试...")
    print(f"📋 总实验数: {total_experiments} 个")
    print(f"📊 数据集: {len(data_files)} 个")
    print(f"🔧 模型配置: {len(extra_large_configs)} 个")
    print(f"🔄 每个配置重复: 100 次")
    print(f"⚠️  注意: 大参数配置可能需要更长时间")
    
    experiment_count = 0
    
    # 对每个数据集进行实验
    for data_file in data_files:
        print(f"\n{'='*80}")
        print(f"📁 处理数据集: {data_file}")
        print(f"{'='*80}")
        
        # 加载数据
        df = pd.read_csv(data_file)
        
        # 检查并修复列名
        if 'DateTime' in df.columns and 'Datetime' not in df.columns:
            df = df.rename(columns={'DateTime': 'Datetime'})
            print("📝 已重命名 DateTime -> Datetime")
        
        # 转换Datetime列为datetime类型
        if 'Datetime' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            print("📝 已转换 Datetime 列为 datetime 类型")
        
        print(f"📊 数据形状: {df.shape}")
        
        # 从文件名提取plant_id
        plant_id = os.path.basename(data_file).replace('Project', '').replace('.csv', '')
        
        # 对每个模型配置进行100次重复实验
        for model, complexity, lookback, weather_category, description, custom_params in extra_large_configs:
            print(f"\n🔧 模型配置: {description}")
            print(f"   数据集: {plant_id}")
            
            for repeat in range(100):  # 每个配置重复100次
                experiment_count += 1
                
                print(f"\n{'='*60}")
                print(f"🧪 实验 {experiment_count}/{total_experiments}: {description} - {plant_id} - 重复{repeat+1}")
                print(f"{'='*60}")
                
                try:
                    # 创建配置
                    config = create_base_config(
                        plant_config={'plant_id': plant_id, 'data_path': data_file},
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
                    print(f"     - 数据集: {plant_id}")
                    print(f"     - 模型: {model}")
                    print(f"     - 复杂度: {complexity}")
                    print(f"     - Lookback: {lookback}h")
                    print(f"     - 天气特征: {weather_category}")
                    print(f"     - 重复次数: {repeat+1}/100")
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
                            'plant_id': plant_id,
                            'data_file': data_file,
                            'description': description,
                            'model': model,
                            'complexity': complexity,
                            'lookback': lookback,
                            'weather_category': weather_category,
                            'custom_params': str(custom_params),
                            'repeat': repeat + 1,
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
                            'plant_id': plant_id,
                            'data_file': data_file,
                            'description': description,
                            'model': model,
                            'complexity': complexity,
                            'lookback': lookback,
                            'weather_category': weather_category,
                            'custom_params': str(custom_params),
                            'repeat': repeat + 1,
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
                        'plant_id': plant_id,
                        'data_file': data_file,
                        'description': description,
                        'model': model,
                        'complexity': complexity,
                        'lookback': lookback,
                        'weather_category': weather_category,
                        'custom_params': str(custom_params),
                        'repeat': repeat + 1,
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
    print(f"📊 总实验数: {len(df_results)}")
    print(f"✅ 成功实验: {len(df_results[df_results['status'] == 'SUCCESS'])}")
    print(f"❌ 失败实验: {len(df_results[df_results['status'] != 'SUCCESS'])}")
    
    # 按数据集和模型分组统计
    print(f"\n📋 按数据集和模型分组统计:")
    print("-" * 80)
    print(f"{'数据集':<10} {'模型':<6} {'成功数':<8} {'失败数':<8} {'平均RMSE':<10} {'平均时间':<10}")
    print("-" * 80)
    
    for plant_id in df_results['plant_id'].unique():
        for model in df_results['model'].unique():
            subset = df_results[(df_results['plant_id'] == plant_id) & (df_results['model'] == model)]
            success_count = len(subset[subset['status'] == 'SUCCESS'])
            fail_count = len(subset[subset['status'] != 'SUCCESS'])
            avg_rmse = subset[subset['status'] == 'SUCCESS']['rmse'].mean() if success_count > 0 else np.nan
            avg_time = subset[subset['status'] == 'SUCCESS']['total_time'].mean() if success_count > 0 else np.nan
            
            print(f"{plant_id:<10} {model:<6} {success_count:<8} {fail_count:<8} {avg_rmse:<10.4f} {avg_time:<10.2f}")
    
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
            print(f"  实验总数: {len(xgb_results)}")
            print(f"  平均总时间: {xgb_results['total_time'].mean():.2f}s")
            print(f"  平均训练时间: {xgb_results['train_time'].mean():.2f}s")
            print(f"  平均RMSE: {xgb_results['rmse'].mean():.4f}")
            print(f"  平均效率: {xgb_results['efficiency'].mean():.1f} 样本/秒")
            print(f"  RMSE标准差: {xgb_results['rmse'].std():.4f}")
        
        if len(lgbm_results) > 0:
            print(f"LGBM 大参数配置结果:")
            print(f"  实验总数: {len(lgbm_results)}")
            print(f"  平均总时间: {lgbm_results['total_time'].mean():.2f}s")
            print(f"  平均训练时间: {lgbm_results['train_time'].mean():.2f}s")
            print(f"  平均RMSE: {lgbm_results['rmse'].mean():.4f}")
            print(f"  平均效率: {lgbm_results['efficiency'].mean():.1f} 样本/秒")
            print(f"  RMSE标准差: {lgbm_results['rmse'].std():.4f}")
        
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
            
            # RMSE对比
            xgb_rmse = xgb_results['rmse'].mean()
            lgbm_rmse = lgbm_results['rmse'].mean()
            print(f"  XGB 平均RMSE: {xgb_rmse:.4f}")
            print(f"  LGBM 平均RMSE: {lgbm_rmse:.4f}")
            
            if xgb_rmse < lgbm_rmse:
                print(f"  🥇 XGB 比 LGBM 准确 {lgbm_rmse/xgb_rmse:.2f}x")
            else:
                print(f"  🥇 LGBM 比 XGB 准确 {xgb_rmse/lgbm_rmse:.2f}x")
    
    # 按数据集分析
    print(f"\n📈 按数据集分析:")
    print("-" * 60)
    
    for plant_id in df_results['plant_id'].unique():
        plant_data = successful_results[successful_results['plant_id'] == plant_id]
        if len(plant_data) > 0:
            print(f"\n数据集 {plant_id}:")
            print(f"  成功实验数: {len(plant_data)}")
            print(f"  平均RMSE: {plant_data['rmse'].mean():.4f}")
            print(f"  平均时间: {plant_data['total_time'].mean():.2f}s")
            print(f"  RMSE标准差: {plant_data['rmse'].std():.4f}")
    
    # 保存结果
    output_file = "large_config_benchmark_results.csv"
    df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n💾 结果已保存到: {output_file}")
    print(f"📊 保存了 {len(df_results)} 行实验结果")
    
    return df_results

if __name__ == "__main__":
    large_config_benchmark()

