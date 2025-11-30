#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hourly Predictions for 48 Hours of Test Data

For each hour in 48 hours of testing data, predict the next 24 hours' capacity factor.
Each hour's prediction is saved as a separate CSV file.

Usage:
    python hourly_predictions_48h.py --data-path data/Project1140.csv --model-config <config_name>
    python hourly_predictions_48h.py --data-path data/Project1140.csv --model LSTM --complexity high --scenario PV+NWP
"""

import pandas as pd
import numpy as np
import os
import sys
import time
from datetime import datetime, timedelta
import warnings
import argparse
warnings.filterwarnings('ignore')

# Suppress warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append(script_dir)

from data.data_utils import preprocess_features, create_sliding_windows
from train.train_dl import train_dl_model
from train.train_ml import train_ml_model


# =============================================================================
# CONFIG CREATION
# =============================================================================
def create_config(data_path, model, complexity, lookback, feat_combo, use_te, is_nwp_only):
    """Create a single experiment configuration"""
    config = {
        'data_path': data_path,
        'model': model,
        'model_complexity': complexity,
        'use_pv': feat_combo['use_pv'],
        'use_hist_weather': feat_combo['use_hist_weather'],
        'use_forecast': feat_combo['use_forecast'],
        'use_ideal_nwp': feat_combo['use_ideal_nwp'],
        'use_time_encoding': use_te,
        'weather_category': 'medium_weather',
        'future_hours': 24,
        'start_date': '2022-01-01',
        'end_date': '2024-09-28',
        'save_options': {
            'save_model': False,
            'save_predictions': False,
            'save_excel_results': False,
            'save_training_log': False
        }
    }

    if is_nwp_only:
        config['past_hours'] = 0
        config['past_days'] = 0
        config['no_hist_power'] = True
        feat_name = feat_combo['name']
    else:
        config['past_hours'] = lookback
        config['past_days'] = lookback // 24
        config['no_hist_power'] = False
        feat_name = f"{feat_combo['name']}_{lookback}h"

    config.update({
        'train_ratio': 0.8, 
        'val_ratio': 0.1, 
        'test_ratio': 0.1,
        'shuffle_split': False,   # Sequential split for temporal evaluation
        'random_seed': 42         # Fixed seed for reproducibility
    })

    # Model-specific hyperparams
    if model in ['LSTM', 'GRU', 'Transformer', 'TCN']:
        if complexity == 'low':
            config.update({
                'train_params': {'epochs': 20, 'batch_size': 64, 'learning_rate': 0.001,
                                 'patience': 10, 'min_delta': 0.001, 'weight_decay': 1e-4},
                'model_params': {'d_model': 16, 'hidden_dim': 8, 'num_heads': 2, 'num_layers': 1,
                                 'dropout': 0.1, 'tcn_channels': [8, 16], 'kernel_size': 3}
            })
        else:
            config.update({
                'train_params': {'epochs': 50, 'batch_size': 64, 'learning_rate': 0.001,
                                 'patience': 10, 'min_delta': 0.001, 'weight_decay': 1e-4},
                'model_params': {'d_model': 32, 'hidden_dim': 16, 'num_heads': 2, 'num_layers': 2,
                                 'dropout': 0.1, 'tcn_channels': [16, 32], 'kernel_size': 3}
            })
    elif model == 'Linear':
        config['model_params'] = {}
    else:
        if complexity == 'low':
            config['model_params'] = {'n_estimators': 10, 'max_depth': 1, 'learning_rate': 0.2,
                                      'random_state': 42, 'verbosity': -1}
        else:
            config['model_params'] = {'n_estimators': 30, 'max_depth': 3, 'learning_rate': 0.1,
                                      'random_state': 42, 'verbosity': -1}

    te_suffix = 'TE' if use_te else 'noTE'
    config['experiment_name'] = f"{model}_{feat_name}_{te_suffix}" if model == 'Linear' else f"{model}_{complexity}_{feat_name}_{te_suffix}"
    config['save_dir'] = f'results/{config["experiment_name"]}'
    return config


def create_config_from_args(data_path, model, complexity, scenario, lookback, use_time_encoding):
    """Create config from command line arguments"""
    feature_combos_pv = [
        {'name': 'PV', 'use_pv': True, 'use_hist_weather': False, 'use_forecast': False, 'use_ideal_nwp': False},
        {'name': 'PV+HW', 'use_pv': True, 'use_hist_weather': True, 'use_forecast': False, 'use_ideal_nwp': False},
        {'name': 'PV+NWP', 'use_pv': True, 'use_hist_weather': False, 'use_forecast': True, 'use_ideal_nwp': False},
        {'name': 'PV+NWP+', 'use_pv': True, 'use_hist_weather': False, 'use_forecast': True, 'use_ideal_nwp': True},
    ]

    feature_combos_nwp = [
        {'name': 'NWP', 'use_pv': False, 'use_hist_weather': False, 'use_forecast': True, 'use_ideal_nwp': False},
        {'name': 'NWP+', 'use_pv': False, 'use_hist_weather': False, 'use_forecast': True, 'use_ideal_nwp': True},
    ]

    # Find matching feature combo
    feat_combo = None
    is_nwp_only = False
    
    for combo in feature_combos_pv:
        if combo['name'] == scenario:
            feat_combo = combo
            break
    
    if feat_combo is None:
        for combo in feature_combos_nwp:
            if combo['name'] == scenario:
                feat_combo = combo
                is_nwp_only = True
                break
    
    if feat_combo is None:
        raise ValueError(f"Unknown scenario: {scenario}. Must be one of: PV, PV+HW, PV+NWP, PV+NWP+, NWP, NWP+")
    
    return create_config(data_path, model, complexity, lookback, feat_combo, use_time_encoding, is_nwp_only)


# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================
def make_prediction_at_hour(model, config, df_clean, hist_feats, fcst_feats, scaler_hist, scaler_fcst, 
                            scaler_target, no_hist_power, hour_idx, past_hours, future_hours):
    """
    Make a prediction for the next 24 hours starting from a specific hour index.
    
    Args:
        model: Trained model
        config: Model configuration
        df_clean: Preprocessed dataframe
        hist_feats: Historical feature columns
        fcst_feats: Forecast feature columns
        scaler_hist: Historical features scaler
        scaler_fcst: Forecast features scaler
        scaler_target: Target scaler
        no_hist_power: Whether to use historical power
        hour_idx: Index of the hour in df_clean to start prediction from
        past_hours: Lookback window size
        future_hours: Number of hours to predict (24)
    
    Returns:
        predictions: Array of predicted capacity factors for next 24 hours
        ground_truth: Array of actual capacity factors for next 24 hours
        prediction_datetime: Datetime of the prediction hour
        future_datetimes: Array of datetimes for the 24 predicted hours
    """
    import torch
    
    # Ensure hour_idx is a scalar integer
    hour_idx = int(hour_idx)
    past_hours = int(past_hours)
    future_hours = int(future_hours)
    
    # Get historical data (past_hours before hour_idx)
    hist_start = max(0, hour_idx - past_hours)
    hist_end = hour_idx
    hist_data = df_clean.iloc[hist_start:hist_end].copy()
    
    # Get future data (24 hours starting from hour_idx)
    fut_start = hour_idx
    fut_end = min(len(df_clean), hour_idx + future_hours)
    fut_data = df_clean.iloc[fut_start:fut_end].copy()
    
    # Check if we have enough data
    if len(hist_data) < past_hours and not no_hist_power:
        # Pad with zeros if needed
        if len(hist_data) > 0:
            padding = np.zeros((past_hours - len(hist_data), len(hist_feats)))
            hist_array = np.vstack([padding, hist_data[hist_feats].values])
        else:
            hist_array = np.zeros((past_hours, len(hist_feats)))
    elif no_hist_power:
        hist_array = np.zeros((past_hours if past_hours > 0 else 1, len(hist_feats) if hist_feats else 0))
    else:
        hist_array = hist_data[hist_feats].values
    
    # Get forecast features for future period
    if fcst_feats:
        if len(fut_data) < future_hours:
            # Pad with last available values
            last_row = fut_data[fcst_feats].iloc[-1:].values if len(fut_data) > 0 else np.zeros((1, len(fcst_feats)))
            padding = np.tile(last_row, (future_hours - len(fut_data), 1))
            fcst_array = np.vstack([fut_data[fcst_feats].values, padding])
        else:
            fcst_array = fut_data[fcst_feats].values[:future_hours]
    else:
        fcst_array = None
    
    # Get ground truth
    if len(fut_data) < future_hours:
        # Pad with NaN if not enough data
        gt = np.full(future_hours, np.nan)
        gt[:len(fut_data)] = fut_data['Capacity Factor'].values
    else:
        gt = fut_data['Capacity Factor'].values[:future_hours]
    
    # Prepare input for model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Reshape for model input
    X_hist = hist_array.reshape(1, past_hours if past_hours > 0 else 1, -1)
    if fcst_array is not None:
        X_fcst = fcst_array.reshape(1, future_hours, -1)
    else:
        X_fcst = None
    
    # Get prediction datetime
    prediction_datetime = df_clean.iloc[hour_idx]['Datetime']
    
    # Make prediction
    if config['model'] in ['LSTM', 'GRU', 'Transformer', 'TCN']:
        # Deep learning model
        model.eval()
        with torch.no_grad():
            X_hist_tensor = torch.tensor(X_hist, dtype=torch.float32).to(device)
            
            if X_fcst is not None:
                X_fcst_tensor = torch.tensor(X_fcst, dtype=torch.float32).to(device)
                preds = model(X_hist_tensor, X_fcst_tensor)
            else:
                preds = model(X_hist_tensor)
            
            preds_np = preds.cpu().numpy().flatten()
    else:
        # Machine learning model
        # Flatten features
        if X_fcst is not None:
            X_flat = np.concatenate([X_hist.reshape(1, -1), X_fcst.reshape(1, -1)], axis=1)
        else:
            X_flat = X_hist.reshape(1, -1)
        
        preds_np = model.predict(X_flat).flatten()
    
    # Inverse transform predictions
    if scaler_target is not None:
        preds_inv = scaler_target.inverse_transform(preds_np.reshape(-1, 1)).flatten()
        gt_inv = scaler_target.inverse_transform(gt.reshape(-1, 1)).flatten() if not np.isnan(gt).all() else gt
    else:
        preds_inv = preds_np
        gt_inv = gt
    
    # Clip predictions to reasonable range [0, 100]
    preds_inv = np.clip(preds_inv, 0, 100)
    
    # Create future datetimes
    future_datetimes = pd.date_range(start=prediction_datetime, periods=future_hours, freq='H')
    
    return preds_inv, gt_inv, prediction_datetime, future_datetimes


# =============================================================================
# MAIN FUNCTION
# =============================================================================
def run_hourly_predictions(data_path, config, output_dir, test_hours=48):
    """
    Run hourly predictions for test_hours consecutive hours from test data.
    Each hour's prediction (next 24 hours) is saved as a separate CSV file.
    
    Args:
        data_path: Path to data CSV file
        config: Model configuration
        output_dir: Directory to save prediction CSV files
        test_hours: Number of consecutive hours from test data to use (default: 48)
    """
    print("=" * 80)
    print("Hourly Predictions for Test Data")
    print("=" * 80)
    print(f"Data file: {data_path}")
    print(f"Model: {config['experiment_name']}")
    print(f"Test hours: {test_hours}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    # Load and preprocess data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
    
    print("\n[1/4] Preprocessing data...")
    df_clean, hist_feats, fcst_feats, scaler_hist, scaler_fcst, scaler_target, no_hist_power = preprocess_features(df, config)
    
    # Create sliding windows for train/val/test split
    print("\n[2/4] Creating sliding windows and splitting data...")
    past_hours = config.get('past_hours', 24)
    future_hours = config.get('future_hours', 24)
    
    X_hist, X_fcst, y, hours, dates = create_sliding_windows(
        df_clean, past_hours, future_hours, hist_feats, fcst_feats, no_hist_power
    )
    
    total_samples = len(X_hist)
    indices = np.arange(total_samples)
    
    # Sequential split (no shuffle)
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
    
    # Prepare training data
    X_hist_train, y_train = X_hist[train_idx], y[train_idx]
    X_hist_val, y_val = X_hist[val_idx], y[val_idx]
    X_hist_test, y_test = X_hist[test_idx], y[test_idx]
    
    if X_fcst is not None:
        X_fcst_train, X_fcst_val, X_fcst_test = X_fcst[train_idx], X_fcst[val_idx], X_fcst[test_idx]
    else:
        X_fcst_train = X_fcst_val = X_fcst_test = None
    
    train_hours = np.array([hours[i] for i in train_idx])
    val_hours = np.array([hours[i] for i in val_idx])
    test_hours = np.array([hours[i] for i in test_idx])
    
    train_data = (X_hist_train, X_fcst_train, y_train, train_hours, [])
    val_data = (X_hist_val, X_fcst_val, y_val, val_hours, [])
    test_data = (X_hist_test, X_fcst_test, y_test, test_hours, [])
    scalers = (scaler_hist, scaler_fcst, scaler_target)
    
    # Train model
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
    
    # Get test data indices in original dataframe
    # The test_idx refers to sliding window samples, we need to map back to df_clean indices
    print("\n[4/4] Making hourly predictions...")
    
    # The sliding window creation starts from index 'past_hours' in df_clean
    # So test sample i corresponds to starting index: past_hours + test_idx[i] in df_clean
    # But we want to predict from each hour, not from each sliding window start
    
    # Get the first test sample's starting index in df_clean
    first_test_sample_idx = int(test_idx[0])  # Ensure it's a scalar integer
    first_test_start_in_df = past_hours + first_test_sample_idx
    
    # Use 48 consecutive hours starting from the first test sample's start
    # Each hour will be used to predict the next 24 hours
    test_hour_indices = []
    for i in range(test_hours):
        hour_idx = int(first_test_start_in_df + i)  # Ensure it's a scalar integer
        # Make sure we have enough data after this hour for prediction
        if hour_idx >= 0 and hour_idx < len(df_clean) - future_hours:
            test_hour_indices.append(hour_idx)
        else:
            break
    
    if len(test_hour_indices) < test_hours:
        print(f"  Warning: Only found {len(test_hour_indices)} valid test hours (requested {test_hours})")
        print(f"  Using available hours: {len(test_hour_indices)}")
    
    print(f"  Making predictions for {len(test_hour_indices)} hours...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Make prediction for each hour
    for hour_num, hour_idx in enumerate(test_hour_indices, 1):
        try:
            preds, gt, pred_datetime, future_dt = make_prediction_at_hour(
                model, config, df_clean, hist_feats, fcst_feats, 
                scaler_hist, scaler_fcst, scaler_target, no_hist_power,
                hour_idx, past_hours, future_hours
            )
            
            # Create DataFrame for this hour's prediction
            pred_df = pd.DataFrame({
                'Datetime': future_dt,
                'Predicted_Capacity_Factor': preds,
                'Ground_Truth_Capacity_Factor': gt
            })
            
            # Save to CSV
            # Format: predictions_hour_001_YYYY-MM-DD_HH.csv
            timestamp_str = pred_datetime.strftime('%Y-%m-%d_%H')
            output_file = os.path.join(output_dir, f"predictions_hour_{hour_num:03d}_{timestamp_str}.csv")
            pred_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            if hour_num % 10 == 0 or hour_num == len(test_hour_indices):
                print(f"  [{hour_num}/{len(test_hour_indices)}] Saved: {output_file}")
        
        except Exception as e:
            print(f"  [ERROR] Hour {hour_num} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print(f"[SUCCESS] Completed predictions for {len(test_hour_indices)} hours")
    print(f"Prediction files saved to: {output_dir}")
    print(f"{'='*80}")


# =============================================================================
# MAIN ENTRY
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Make hourly predictions for 48 hours of test data. Each hour predicts next 24 hours.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default model (LSTM high complexity, PV+NWP scenario)
  python hourly_predictions_48h.py --data-path data/Project1140.csv
  
  # Specify model and scenario
  python hourly_predictions_48h.py --data-path data/Project1140.csv --model LSTM --complexity high --scenario PV+NWP
  
  # Use different model
  python hourly_predictions_48h.py --data-path data/Project1140.csv --model XGB --complexity high --scenario PV+NWP
  
  # Use different number of test hours
  python hourly_predictions_48h.py --data-path data/Project1140.csv --test-hours 24
        """
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
    parser.add_argument('--test-hours', type=int, default=48,
                       help='Number of consecutive hours from test data to use (default: 48)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for prediction CSV files (default: ./hourly_predictions_<model>_<scenario>)')
    
    args = parser.parse_args()
    
    # Create config
    config = create_config_from_args(
        args.data_path, args.model, args.complexity, args.scenario,
        args.lookback, args.use_time_encoding
    )
    
    # Set output directory
    if args.output_dir is None:
        output_dir = os.path.join(script_dir, f"hourly_predictions_{args.model}_{args.scenario}")
    else:
        output_dir = args.output_dir
    
    # Run predictions
    try:
        run_hourly_predictions(args.data_path, config, output_dir, args.test_hours)
    except Exception as e:
        print(f"\n[ERROR] Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

