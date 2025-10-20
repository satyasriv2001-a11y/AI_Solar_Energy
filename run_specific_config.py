#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run specific configuration experiments for PV forecasting
Only runs: 24h lookback, no TE, high complexity, PV+NWP for all models
"""

import pandas as pd
import numpy as np
import os
import sys
import time
import warnings
from datetime import datetime
from typing import List, Dict

warnings.filterwarnings('ignore')

# Suppress warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append(script_dir)

from config_manager import PlantConfigManager
from data.data_utils import preprocess_features, create_daily_windows
from train.train_dl import train_dl_model
from train.train_ml import train_ml_model


def run_single_experiment(config: Dict, df: pd.DataFrame) -> Dict:
    """
    Run a single experiment
    
    Args:
        config: Experiment configuration
        df: Data DataFrame
        
    Returns:
        Experiment result dictionary
    """
    try:
        # Data preprocessing
        df_clean, hist_feats, fcst_feats, scaler_hist, scaler_fcst, scaler_target, no_hist_power = preprocess_features(df, config)
        
        # Create daily windows (one prediction per day)
        past_hours = config.get('past_hours', 24)
        X_hist, X_fcst, y, hours, dates = create_daily_windows(
            df_clean, config['future_hours'], hist_feats, fcst_feats, no_hist_power, past_hours
        )
        
        # Data splitting: Random shuffle for robust evaluation
        total_samples = len(X_hist)
        indices = np.arange(total_samples)
        
        shuffle_split = config.get('shuffle_split', True)
        random_seed = config.get('random_seed', 42)
        
        if shuffle_split:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        
        train_ratio = config.get('train_ratio', 0.8)
        val_ratio = config.get('val_ratio', 0.1)
        
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
        
        print(f"  Data split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
        print(f"  Test period: {dates[test_idx[0]]} to {dates[test_idx[-1]]}")
        
        X_hist_train, y_train = X_hist[train_idx], y[train_idx]
        X_hist_val, y_val = X_hist[val_idx], y[val_idx]
        X_hist_test, y_test = X_hist[test_idx], y[test_idx]
        
        if X_fcst is not None:
            X_fcst_train, X_fcst_val, X_fcst_test = X_fcst[train_idx], X_fcst[val_idx], X_fcst[test_idx]
        else:
            X_fcst_train = X_fcst_val = X_fcst_test = None
        
        # Split hours and dates
        train_hours = np.array([hours[i] for i in train_idx])
        val_hours = np.array([hours[i] for i in val_idx])
        test_hours = np.array([hours[i] for i in test_idx])
        test_dates = [dates[i] for i in test_idx]
        
        train_data = (X_hist_train, X_fcst_train, y_train, train_hours, [])
        val_data = (X_hist_val, X_fcst_val, y_val, val_hours, [])
        test_data = (X_hist_test, X_fcst_test, y_test, test_hours, test_dates)
        scalers = (scaler_hist, scaler_fcst, scaler_target)
        
        # Train model
        start_time = time.time()
        if config['model'] in ['LSTM', 'GRU', 'Transformer', 'TCN']:
            model, metrics = train_dl_model(config, train_data, val_data, test_data, scalers)
        else:
            model, metrics = train_ml_model(config, train_data, val_data, test_data, scalers)
        training_time = time.time() - start_time
        
        # Parse scenario name
        use_pv = config.get('use_pv', False)
        use_hist_weather = config.get('use_hist_weather', False)
        use_forecast = config.get('use_forecast', False)
        use_ideal_nwp = config.get('use_ideal_nwp', False)
        
        if use_pv and use_hist_weather:
            scenario = 'PV+HW'
        elif use_pv and use_forecast and use_ideal_nwp:
            scenario = 'PV+NWP+'
        elif use_pv and use_forecast:
            scenario = 'PV+NWP'
        elif use_pv:
            scenario = 'PV'
        elif use_forecast and use_ideal_nwp:
            scenario = 'NWP+'
        elif use_forecast:
            scenario = 'NWP'
        else:
            scenario = 'Unknown'
        
        # Return result
        result = {
            'plant_id': config['plant_id'],
            'experiment_name': config['experiment_name'],
            'model': config['model'],
            'complexity': config.get('model_complexity', 'N/A'),
            'scenario': scenario,
            'lookback_hours': config['past_hours'],
            'use_time_encoding': config['use_time_encoding'],
            'mae': metrics.get('mae', 0.0),
            'rmse': metrics.get('rmse', 0.0),
            'r2': metrics.get('r2', 0.0),
            'nrmse': metrics.get('nrmse', 0.0),
            'train_time_sec': metrics.get('train_time_sec', round(training_time, 2)),
            'test_samples': metrics.get('samples_count', 0),
            'best_epoch': int(metrics.get('best_epoch', 0)) if not pd.isna(metrics.get('best_epoch', 0)) else 0,
            'param_count': int(metrics.get('param_count', 0)),
            'status': 'SUCCESS'
        }
        
        return result
        
    except Exception as e:
        print(f"  [ERROR] Experiment failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'plant_id': config.get('plant_id', 'Unknown'),
            'experiment_name': config.get('experiment_name', 'Unknown'),
            'model': config.get('model', 'Unknown'),
            'complexity': config.get('model_complexity', 'N/A'),
            'scenario': 'FAILED',
            'lookback_hours': config.get('past_hours', 0),
            'use_time_encoding': config.get('use_time_encoding', False),
            'mae': np.nan,
            'rmse': np.nan,
            'r2': np.nan,
            'train_time_sec': 0,
            'test_samples': 0,
            'best_epoch': 0,
            'param_count': 0,
            'status': 'FAILED',
            'error': str(e)
        }


def create_specific_config(plant_config: Dict, model: str) -> Dict:
    """
    Create configuration for specific experiment: 24h lookback, no TE, high complexity, PV+NWP
    
    Args:
        plant_config: Plant configuration
        model: Model name
        
    Returns:
        Experiment configuration
    """
    config = {
        'plant_id': plant_config['plant_id'],
        'data_path': plant_config['data_path'],
        'model': model,
        'model_complexity': 'high',
        'use_pv': True,
        'use_hist_weather': False,
        'use_forecast': True,
        'use_ideal_nwp': False,
        'use_time_encoding': False,
        'past_hours': 24,
        'past_days': 1,
        'future_hours': plant_config.get('future_hours', 24),
        'train_ratio': plant_config.get('train_ratio', 0.8),
        'val_ratio': plant_config.get('val_ratio', 0.1),
        'test_ratio': plant_config.get('test_ratio', 0.1),
        'shuffle_split': True,
        'random_seed': plant_config.get('random_seed', 42),
        'weather_category': plant_config.get('weather_category', 'medium_weather'),
        'start_date': plant_config.get('start_date', '2022-01-01'),
        'end_date': plant_config.get('end_date', '2024-09-28'),
        'save_options': {
            'save_model': False,
            'save_predictions': False,
            'save_training_log': False,
            'save_excel_results': False
        }
    }
    
    # Add model parameters
    if model in ['LSTM', 'GRU', 'Transformer', 'TCN']:
        # Get DL parameters from plant config
        dl_params = plant_config.get('dl_params', {}).get('high', {})
        config['train_params'] = {
            'epochs': dl_params.get('epochs', 50),
            'batch_size': dl_params.get('batch_size', 64),
            'learning_rate': dl_params.get('learning_rate', 0.001),
            'patience': dl_params.get('patience', 10),
            'min_delta': dl_params.get('min_delta', 0.001),
            'weight_decay': dl_params.get('weight_decay', 0.0001)
        }
        config['model_params'] = {
            'd_model': dl_params.get('d_model', 32),
            'hidden_dim': dl_params.get('hidden_dim', 16),
            'num_heads': dl_params.get('num_heads', 2),
            'num_layers': dl_params.get('num_layers', 2),
            'dropout': dl_params.get('dropout', 0.1),
            'tcn_channels': dl_params.get('tcn_channels', [16, 32]),
            'kernel_size': dl_params.get('kernel_size', 3)
        }
    elif model in ['RF', 'XGB', 'LGBM']:
        # Get ML parameters from plant config
        ml_params = plant_config.get('ml_params', {}).get('high', {})
        config['model_params'] = ml_params
    elif model == 'Linear':
        config['model_params'] = {}
    
    # Experiment name
    config['experiment_name'] = f"{model}_high_PV+NWP_noTE"
    config['save_dir'] = f"{config['plant_id']}_results/{config['experiment_name']}"
    
    return config


def run_specific_experiments(output_dir: str = None, max_plants: int = None):
    """
    Run specific configuration experiments: 24h lookback, no TE, high complexity, PV+NWP
    
    Args:
        output_dir: Directory to save results
        max_plants: Maximum number of plants to process
    """
    print("=" * 80)
    print("Specific Configuration Experiments")
    print("Configuration: 24h lookback, no TE, high complexity, PV+NWP")
    print("=" * 80)
    
    # Get all plant configurations
    manager = PlantConfigManager()
    all_plants = manager.get_all_plants()
    
    if not all_plants:
        print("[ERROR] No plant configurations found in config/plants/")
        print("Please run: python batch_create_configs.py")
        return
    
    # Filter plants if max_plants is specified
    if max_plants:
        all_plants = all_plants[:max_plants]
    
    # Set output directory
    if output_dir is None:
        output_dir = script_dir
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Plants to process: {len(all_plants)}")
    
    # Models to test
    models = ['LSTM', 'GRU', 'Transformer', 'TCN', 'RF', 'XGB', 'LGBM', 'Linear']
    print(f"Models to test: {', '.join(models)}")
    print(f"Total experiments: {len(all_plants)} plants × {len(models)} models = {len(all_plants) * len(models)}")
    
    # Store all results
    all_results = []
    
    # Run experiments for each plant
    for plant_idx, plant_config in enumerate(all_plants, 1):
        plant_id = plant_config['plant_id']
        data_path = plant_config['data_path']
        
        print(f"\n{'=' * 80}")
        print(f"Plant {plant_idx}/{len(all_plants)}: {plant_id}")
        print(f"{'=' * 80}")
        
        # Load data
        try:
            df = pd.read_csv(data_path)
            df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
            print(f"  Loaded data: {len(df)} rows")
        except Exception as e:
            print(f"  [ERROR] Failed to load data: {e}")
            continue
        
        # Run experiments for each model
        for model in models:
            print(f"\n  Running {model}...")
            
            try:
                # Create configuration
                config = create_specific_config(plant_config, model)
                
                # Run experiment
                result = run_single_experiment(config, df.copy())
                
                if result['status'] == 'SUCCESS':
                    print(f"    ✓ {model}: MAE={result['mae']:.4f}, RMSE={result['rmse']:.4f}, R²={result['r2']:.4f}")
                else:
                    print(f"    ✗ {model}: FAILED - {result.get('error', 'Unknown error')}")
                
                all_results.append(result)
                
            except Exception as e:
                print(f"    ✗ {model}: ERROR - {e}")
                continue
    
    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"specific_config_results_{timestamp}.csv")
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n{'=' * 80}")
        print(f"Results saved to: {output_file}")
        
        # Print summary
        success_results = results_df[results_df['status'] == 'SUCCESS']
        print(f"Successful experiments: {len(success_results)}/{len(all_results)}")
        
        if len(success_results) > 0:
            print("\nModel Performance Summary (MAE):")
            summary = success_results.groupby('model')['mae'].agg(['mean', 'std', 'count']).round(4)
            print(summary)
    else:
        print("\n[ERROR] No results to save")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run specific configuration experiments')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for results')
    parser.add_argument('--max-plants', type=int, default=None, help='Maximum number of plants to process')
    
    args = parser.parse_args()
    
    run_specific_experiments(output_dir=args.output_dir, max_plants=args.max_plants)
