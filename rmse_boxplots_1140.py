#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RMSE Box and Whisker Plots for Project1140

Creates box plots showing RMSE distribution for each prediction start time.
For each 24-hour sliding window prediction, calculates RMSE across all forecasted points
and displays as box plots grouped by datetime.

Usage:
    python rmse_boxplots_1140.py --data-path data/Project1140.csv --model XGB --complexity high --scenario PV+NWP
"""

import pandas as pd
import numpy as np
import os
import sys
import time
from datetime import datetime, timedelta
import warnings
import argparse
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.dates import HourLocator, MinuteLocator, DateFormatter
import seaborn as sns
warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append(script_dir)

from data.data_utils import preprocess_features, create_sliding_windows
from train.train_dl import train_dl_model
from train.train_ml import train_ml_model


# Import config creation functions from multi_resolution_predictions_1140
sys.path.insert(0, script_dir)
from multi_resolution_predictions_1140 import create_config_from_args, make_prediction_at_time


def calculate_rmse_for_prediction(preds, gt):
    """
    Calculate RMSE for a single prediction (24-hour forecast).
    
    Args:
        preds: Predicted values (array)
        gt: Ground truth values (array)
    
    Returns:
        RMSE value (float) or np.nan if insufficient valid data
    """
    # Filter out NaN values
    valid_mask = ~(np.isnan(preds) | np.isnan(gt))
    
    if np.sum(valid_mask) < 2:  # Need at least 2 points for meaningful RMSE
        return np.nan
    
    preds_valid = preds[valid_mask]
    gt_valid = gt[valid_mask]
    
    mse = np.mean((preds_valid - gt_valid) ** 2)
    rmse = np.sqrt(mse)
    
    return rmse


def run_predictions_and_calculate_rmse(data_path, config, resolution_minutes, test_intervals):
    """
    Run predictions and calculate RMSE for each prediction start time.
    
    Args:
        data_path: Path to data CSV file
        config: Configuration dictionary
        resolution_minutes: Resolution in minutes (60, 30, or 15)
        test_intervals: Number of intervals to test
    
    Returns:
        List of (prediction_start_datetime, rmse_value) tuples
    """
    print("=" * 80)
    resolution_name = f"{resolution_minutes}-minute" if resolution_minutes < 60 else "hourly"
    print(f"Running {resolution_name.upper()} Predictions for RMSE Box Plots")
    print("=" * 80)
    print(f"Data file: {data_path}")
    print(f"Model: {config['experiment_name']}")
    print(f"Resolution: {resolution_minutes} minutes")
    print(f"Test intervals: {test_intervals}")
    print("=" * 80)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    has_minute_column = 'Minute' in df.columns
    
    if has_minute_column:
        df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
        print(f"  Detected {resolution_minutes}-minute resolution data (Minute column found)")
    else:
        df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
        print(f"  Detected hourly resolution data (no Minute column)")
        print(f"  Interpolating to {resolution_minutes}-minute resolution...")
        
        df = df.drop_duplicates(subset='Datetime', keep='first')
        df = df.sort_values('Datetime').reset_index(drop=True)
        df_indexed = df.set_index('Datetime')
        
        # Resample to target resolution
        freq_map = {60: 'H', 30: '30T', 15: '15T'}
        freq = freq_map.get(resolution_minutes, 'H')
        df_resampled = df_indexed.resample(freq).asfreq()
        
        numeric_cols = df_resampled.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_resampled[col] = df_resampled[col].interpolate(method='pchip', limit_direction='both')
        
        non_numeric_cols = df_resampled.select_dtypes(exclude=[np.number]).columns
        for col in non_numeric_cols:
            if col not in ['Datetime']:
                df_resampled[col] = df_resampled[col].ffill()
        
        df = df_resampled.reset_index()
        df['Year'] = df['Datetime'].dt.year
        df['Month'] = df['Datetime'].dt.month
        df['Day'] = df['Datetime'].dt.day
        df['Hour'] = df['Datetime'].dt.hour
        df['Minute'] = df['Datetime'].dt.minute
        
        has_minute_column = True
        print(f"  Interpolated to {len(df)} {resolution_minutes}-minute points")
    
    print("\n[1/4] Preprocessing data...")
    df_clean, hist_feats, fcst_feats, scaler_hist, scaler_fcst, scaler_target, no_hist_power = preprocess_features(df, config)
    
    print("\n[2/4] Creating sliding windows and splitting data...")
    past_hours = int(config.get('past_hours', 24))
    future_hours = int(config.get('future_hours', 24))
    
    # Convert hours to intervals based on resolution
    intervals_per_hour = 60 // resolution_minutes
    past_intervals = past_hours * intervals_per_hour
    future_intervals = future_hours * intervals_per_hour
    
    X_hist, X_fcst, y, hours, dates = create_sliding_windows(
        df_clean, past_intervals, future_intervals, hist_feats, fcst_feats, no_hist_power
    )
    
    total_samples = len(X_hist)
    indices = np.arange(total_samples)
    
    if config.get('shuffle_split', True):
        np.random.seed(config.get('random_seed', 42))
        np.random.shuffle(indices)
    
    train_size = int(total_samples * config['train_ratio'])
    val_size = int(total_samples * config['val_ratio'])
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    print(f"  Train samples: {len(train_idx)}")
    print(f"  Val samples: {len(val_idx)}")
    print(f"  Test samples: {len(test_idx)}")
    
    X_hist_train, y_train = X_hist[train_idx], y[train_idx]
    X_hist_val, y_val = X_hist[val_idx], y[val_idx]
    X_hist_test, y_test = X_hist[test_idx], y[test_idx]
    
    if X_fcst is not None:
        X_fcst_train, X_fcst_val, X_fcst_test = X_fcst[train_idx], X_fcst[val_idx], X_fcst[test_idx]
    else:
        X_fcst_train = X_fcst_val = X_fcst_test = None
    
    train_hours = np.array([hours[int(i)] for i in train_idx])
    val_hours = np.array([hours[int(i)] for i in val_idx])
    test_hours_array = np.array([hours[int(i)] for i in test_idx])
    
    train_data = (X_hist_train, X_fcst_train, y_train, train_hours, [])
    val_data = (X_hist_val, X_fcst_val, y_val, val_hours, [])
    test_data = (X_hist_test, X_fcst_test, y_test, test_hours_array, [])
    scalers = (scaler_hist, scaler_fcst, scaler_target)
    
    print("\n[3/4] Training model...")
    import torch
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    start_time = time.time()
    if config['model'] in ['LSTM', 'GRU', 'Transformer', 'TCN']:
        model, metrics = train_dl_model(config, train_data, val_data, test_data, scalers)
    else:
        model, metrics = train_ml_model(config, train_data, val_data, test_data, scalers)
    
    training_time = time.time() - start_time
    print(f"  Training completed in {training_time:.1f} seconds")
    print(f"  MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}")
    
    print("\n[4/4] Making predictions and calculating RMSE...")
    
    if len(test_idx) == 0:
        raise ValueError("No test samples available.")
    
    if isinstance(test_idx, np.ndarray):
        test_idx_list = test_idx.tolist()
    else:
        test_idx_list = list(test_idx)
    
    first_test_sample_idx = int(test_idx_list[0])
    intervals_per_hour = 60 // resolution_minutes
    past_intervals = past_hours * intervals_per_hour
    first_test_start_in_df = int(past_intervals) + first_test_sample_idx
    
    first_test_datetime = df_clean.iloc[first_test_start_in_df]['Datetime']
    target_year = first_test_datetime.year
    
    target_date = pd.Timestamp(year=target_year, month=6, day=20, hour=0, minute=0)
    
    start_idx = None
    for idx in range(len(df_clean)):
        if df_clean.iloc[idx]['Datetime'] >= target_date:
            start_idx = idx
            break
    
    if start_idx is None or start_idx < first_test_start_in_df:
        print(f"  Warning: Could not find June 20, {target_year} 00:00 in test data.")
        print(f"  Using first test sample start instead: {df_clean.iloc[first_test_start_in_df]['Datetime']}")
        start_idx = first_test_start_in_df
    else:
        actual_start_date = df_clean.iloc[start_idx]['Datetime']
        print(f"  Starting predictions from: {actual_start_date.strftime('%Y-%m-%d %H:%M')}")
    
    test_time_indices = []
    for i in range(test_intervals):
        time_idx = int(start_idx + i)
        if time_idx >= 0 and time_idx < len(df_clean) - future_intervals:
            test_time_indices.append(time_idx)
        else:
            break
    
    if len(test_time_indices) < test_intervals:
        print(f"  Warning: Only found {len(test_time_indices)} valid intervals (requested {test_intervals})")
    
    print(f"  Making predictions for {len(test_time_indices)} intervals...")
    
    rmse_by_datetime = []
    
    for pred_num, time_idx in enumerate(test_time_indices, 1):
        try:
            preds, gt, pred_datetime, future_dt = make_prediction_at_time(
                model, config, df_clean, hist_feats, fcst_feats, 
                scaler_hist, scaler_fcst, scaler_target, no_hist_power,
                time_idx, past_intervals, future_intervals, resolution_minutes
            )
            
            # Calculate RMSE for this prediction
            rmse = calculate_rmse_for_prediction(preds, gt)
            
            if not np.isnan(rmse):
                rmse_by_datetime.append((pred_datetime, rmse))
            
            if pred_num % 10 == 0 or pred_num == len(test_time_indices):
                print(f"  [{pred_num}/{len(test_time_indices)}] Completed")
        
        except Exception as e:
            print(f"  [ERROR] Prediction {pred_num} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"  Calculated RMSE for {len(rmse_by_datetime)} predictions")
    
    return rmse_by_datetime, resolution_name


def create_rmse_boxplots(rmse_by_datetime_list, output_dir, model_name, resolution_name, group_by='hour'):
    """
    Create box and whisker plots of RMSE values grouped by datetime.
    
    Args:
        rmse_by_datetime_list: List of (datetime, rmse) tuples
        output_dir: Output directory for plots
        model_name: Name of the model
        resolution_name: Resolution name (e.g., "Hourly", "30-minute", "15-minute")
        group_by: How to group data ('hour', 'day', or '6hours')
    """
    if len(rmse_by_datetime_list) == 0:
        print(f"  [WARNING] No RMSE data to plot for {resolution_name}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame(rmse_by_datetime_list, columns=['Datetime', 'RMSE'])
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    
    # Group by specified time interval
    if group_by == 'hour':
        df['Time_Group'] = df['Datetime'].dt.floor('H')
        x_label = 'Hour'
    elif group_by == 'day':
        df['Time_Group'] = df['Datetime'].dt.floor('D')
        x_label = 'Day'
    elif group_by == '6hours':
        df['Time_Group'] = df['Datetime'].dt.floor('6H')
        x_label = '6-Hour Period'
    else:
        raise ValueError(f"Unknown group_by option: {group_by}. Use 'hour', 'day', or '6hours'")
    
    # Remove groups with too few data points (less than 2)
    group_counts = df.groupby('Time_Group').size()
    valid_groups = group_counts[group_counts >= 2].index
    df_filtered = df[df['Time_Group'].isin(valid_groups)]
    
    if len(df_filtered) == 0:
        print(f"  [WARNING] No valid groups with sufficient data points for {resolution_name}")
        return
    
    # Create box plot
    plt.figure(figsize=(20, 8))
    
    # Prepare data for box plot
    groups = sorted(df_filtered['Time_Group'].unique())
    rmse_data_by_group = [df_filtered[df_filtered['Time_Group'] == g]['RMSE'].values for g in groups]
    
    # Create box plot
    bp = plt.boxplot(rmse_data_by_group, labels=groups, patch_artist=True, widths=0.6)
    
    # Color the boxes
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(bp['boxes'])))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Format x-axis
    ax = plt.gca()
    if group_by == 'hour':
        ax.xaxis.set_major_locator(HourLocator(interval=6))
        ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H:00'))
        ax.xaxis.set_minor_locator(HourLocator(interval=1))
    elif group_by == 'day':
        ax.xaxis.set_major_locator(HourLocator(interval=24))
        ax.xaxis.set_major_formatter(DateFormatter('%m-%d'))
    elif group_by == '6hours':
        ax.xaxis.set_major_locator(HourLocator(interval=6))
        ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H:00'))
    
    plt.xlabel(f'Prediction Start Time ({x_label})', fontsize=14, fontweight='bold')
    plt.ylabel('RMSE (%)', fontsize=14, fontweight='bold')
    plt.title(f'RMSE Distribution by Prediction Start Time - {model_name} ({resolution_name})\n'
              f'Box plot shows RMSE distribution for 24-hour forecasts starting at each time',
              fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"rmse_boxplot_{resolution_name.lower().replace('-', '_')}_{group_by}.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  RMSE box plot saved: {output_path}")


def run_rmse_boxplots(data_path, config, output_dir, resolutions=None, group_by='hour'):
    """
    Run RMSE box plots for multiple resolutions.
    
    Args:
        data_path: Path to data CSV file
        config: Configuration dictionary
        output_dir: Output directory for plots
        resolutions: List of (resolution_minutes, test_intervals) tuples. If None, uses defaults
        group_by: How to group data ('hour', 'day', or '6hours')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if resolutions is None:
        resolutions = [
            (60, 24, "Hourly"),      # 60 minutes, 24 intervals, "Hourly"
            (30, 48, "30-minute"),   # 30 minutes, 48 intervals, "30-minute"
            (15, 96, "15-minute")    # 15 minutes, 96 intervals, "15-minute"
        ]
    
    for resolution_minutes, test_intervals, resolution_name in resolutions:
        print(f"\n{'='*80}")
        print(f"PROCESSING {resolution_name.upper()} RESOLUTION")
        print(f"{'='*80}")
        
        try:
            rmse_by_datetime, _ = run_predictions_and_calculate_rmse(
                data_path, config, resolution_minutes, test_intervals
            )
            
            if len(rmse_by_datetime) > 0:
                # Create box plots with different grouping options
                for group_option in [group_by] if isinstance(group_by, str) else group_by:
                    create_rmse_boxplots(rmse_by_datetime, output_dir, 
                                       config.get('experiment_name', 'Model'), 
                                       resolution_name, group_by=group_option)
                
                print(f"\n[SUCCESS] {resolution_name} resolution RMSE box plots completed")
            else:
                print(f"\n[WARNING] No RMSE data generated for {resolution_name} resolution")
        
        except Exception as e:
            print(f"\n[ERROR] Failed {resolution_name} resolution: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print("[SUCCESS] RMSE Box Plot Generation Completed!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}")


# =============================================================================
# MAIN ENTRY
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create RMSE box plots for multi-resolution predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to data CSV file (e.g., data/Project1140.csv)')
    parser.add_argument('--model', type=str, default='LSTM',
                       choices=['LSTM', 'GRU', 'Transformer', 'TCN', 'RF', 'XGB', 'LGBM', 'Linear'],
                       help='Model to use (default: LSTM)')
    parser.add_argument('--complexity', type=str, default='high',
                       choices=['low', 'high'],
                       help='Model complexity (default: high)')
    parser.add_argument('--scenario', type=str, default='PV+NWP',
                       choices=['PV', 'PV+HW', 'PV+NWP', 'PV+NWP+', 'NWP', 'NWP+'],
                       help='Feature scenario (default: PV+NWP)')
    parser.add_argument('--lookback', type=int, default=24,
                       choices=[24, 72],
                       help='Lookback window in hours (default: 24)')
    parser.add_argument('--use-time-encoding', action='store_true', default=True,
                       help='Use time encoding features (default: True)')
    parser.add_argument('--no-time-encoding', dest='use_time_encoding', action='store_false',
                       help='Disable time encoding features')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for plots (default: ./rmse_boxplots_<model>_<scenario>)')
    parser.add_argument('--group-by', type=str, default='hour',
                       choices=['hour', 'day', '6hours'],
                       help='How to group RMSE values by time (default: hour)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("MODE: RMSE Box and Whisker Plots")
    print(f"Algorithm: {args.model} {args.complexity} {args.scenario}")
    print(f"Group by: {args.group_by}")
    print("=" * 80 + "\n")
    
    config = create_config_from_args(
        args.data_path, args.model, args.complexity, args.scenario,
        args.lookback, args.use_time_encoding
    )
    
    if args.output_dir is None:
        output_dir = os.path.join(script_dir, f"rmse_boxplots_{args.model}_{args.scenario}")
    else:
        output_dir = args.output_dir
    
    try:
        run_rmse_boxplots(
            args.data_path, config, output_dir, group_by=args.group_by
        )
    except Exception as e:
        print(f"\n[ERROR] Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

