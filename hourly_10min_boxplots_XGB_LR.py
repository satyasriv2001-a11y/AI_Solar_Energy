#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hourly and 10-Minute RMSE Box Plots for XGB and LR Models

Creates box plots showing RMSE distribution for each prediction start time.
Runs for both XGB and Linear Regression models at hourly and 10-minute resolutions.

Usage:
    python hourly_10min_boxplots_XGB_LR.py --data-path data/Project1140.csv
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
from rmse_boxplots_1140 import calculate_rmse_for_prediction, run_predictions_and_calculate_rmse, create_rmse_boxplots


def run_boxplots_for_model(data_path, model, complexity, scenario, lookback, use_time_encoding, output_base_dir):
    """
    Run box plots for a specific model configuration.
    
    Args:
        data_path: Path to data CSV file
        model: Model name ('XGB' or 'Linear')
        complexity: Model complexity ('low' or 'high')
        scenario: Feature scenario (e.g., 'PV+NWP')
        lookback: Lookback window in hours
        use_time_encoding: Whether to use time encoding
        output_base_dir: Base output directory
    """
    print("\n" + "=" * 80)
    print(f"RUNNING BOX PLOTS FOR {model.upper()} MODEL")
    print("=" * 80)
    
    # Create config
    config = create_config_from_args(
        data_path, model, complexity, scenario,
        lookback, use_time_encoding
    )
    
    # Create output directory for this model
    model_suffix = 'LR' if model == 'Linear' else 'XGB'
    output_dir = os.path.join(output_base_dir, f"{model_suffix}_{complexity}_{scenario}_{lookback}h")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Model: {model} {complexity} {scenario} {lookback}h")
    print(f"Time encoding: {use_time_encoding}")
    print("=" * 80)
    
    # Define resolutions: hourly and 10-minute
    resolutions = [
        (60, 24, "Hourly"),      # 60 minutes, 24 intervals
        (10, 144, "10-minute")   # 10 minutes, 144 intervals (24 hours)
    ]
    
    # Store RMSE data for each resolution for summary table
    all_rmse_data = {}
    
    for resolution_minutes, test_intervals, resolution_name in resolutions:
        print(f"\n{'='*80}")
        print(f"PROCESSING {resolution_name.upper()} RESOLUTION")
        print(f"{'='*80}")
        
        try:
            rmse_by_datetime, _ = run_predictions_and_calculate_rmse(
                data_path, config, resolution_minutes, test_intervals
            )
            
            if len(rmse_by_datetime) > 0:
                # Store RMSE data for summary table
                all_rmse_data[resolution_name] = rmse_by_datetime
                
                # Create box plots grouped by hour
                create_rmse_boxplots(rmse_by_datetime, output_dir, 
                                   config.get('experiment_name', 'Model'), 
                                   resolution_name, group_by='hour')
                
                print(f"\n[SUCCESS] {resolution_name} resolution RMSE box plots completed")
            else:
                print(f"\n[WARNING] No RMSE data generated for {resolution_name} resolution")
        
        except Exception as e:
            print(f"\n[ERROR] Failed {resolution_name} resolution: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create summary table for this model
    print(f"\n{'='*80}")
    print(f"Creating Summary Table for {model.upper()} Model")
    print(f"{'='*80}")
    
    rmse_summary = []
    
    for resolution_name, rmse_by_datetime in all_rmse_data.items():
        if len(rmse_by_datetime) > 0:
            # Extract RMSE values
            rmse_values = [rmse for _, rmse in rmse_by_datetime]
            rmse_values = [r for r in rmse_values if not np.isnan(r)]
            
            if len(rmse_values) > 0:
                avg_rmse = np.mean(rmse_values)
                median_rmse = np.median(rmse_values)
                std_rmse = np.std(rmse_values)
                min_rmse = np.min(rmse_values)
                max_rmse = np.max(rmse_values)
                n_predictions = len(rmse_values)
                
                rmse_summary.append({
                    'Resolution': resolution_name,
                    'Average_RMSE': avg_rmse,
                    'Median_RMSE': median_rmse,
                    'Std_RMSE': std_rmse,
                    'Min_RMSE': min_rmse,
                    'Max_RMSE': max_rmse,
                    'Number_of_Predictions': n_predictions
                })
    
    # Save and display summary table
    if len(rmse_summary) > 0:
        rmse_summary_df = pd.DataFrame(rmse_summary)
        rmse_summary_csv_path = os.path.join(output_dir, f'average_rmse_boxplot_{model_suffix}_by_resolution.csv')
        rmse_summary_df.to_csv(rmse_summary_csv_path, index=False)
        print(f"\n  Average RMSE Summary saved: {rmse_summary_csv_path}")
        print(f"\n  Average RMSE by Resolution ({model}):")
        print(rmse_summary_df.to_string(index=False))
    else:
        print(f"\n  [WARNING] No RMSE data available for summary table")
    
    return all_rmse_data


def main():
    parser = argparse.ArgumentParser(
        description='Create hourly and 10-minute RMSE box plots for XGB and LR models',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to data CSV file (e.g., data/Project1140.csv)')
    parser.add_argument('--complexity', type=str, default='high',
                       choices=['low', 'high'],
                       help='Model complexity (default: high)')
    parser.add_argument('--scenario', type=str, default='PV+NWP',
                       choices=['PV', 'PV+HW', 'PV+NWP', 'PV+NWP+', 'NWP', 'NWP+'],
                       help='Feature scenario (default: PV+NWP)')
    parser.add_argument('--lookback', type=int, default=24,
                       choices=[24, 72],
                       help='Lookback window in hours (default: 24)')
    parser.add_argument('--use-time-encoding', action='store_true', default=False,
                       help='Use time encoding features (default: False)')
    parser.add_argument('--no-time-encoding', dest='use_time_encoding', action='store_false',
                       help='Disable time encoding features')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for plots (default: ./hourly_10min_boxplots_XGB_LR)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("Hourly and 10-Minute RMSE Box Plots for XGB and LR Models")
    print("=" * 80)
    print(f"Data file: {args.data_path}")
    print(f"Complexity: {args.complexity}")
    print(f"Scenario: {args.scenario}")
    print(f"Lookback: {args.lookback}h")
    print(f"Time encoding: {args.use_time_encoding}")
    print("=" * 80 + "\n")
    
    if args.output_dir is None:
        output_base_dir = os.path.join(script_dir, "hourly_10min_boxplots_XGB_LR")
    else:
        output_base_dir = args.output_dir
    
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Run for both models
    models = ['XGB', 'Linear']
    all_results = {}
    
    for model in models:
        try:
            rmse_data = run_boxplots_for_model(
                args.data_path, model, args.complexity, args.scenario,
                args.lookback, args.use_time_encoding, output_base_dir
            )
            all_results[model] = rmse_data
        except Exception as e:
            print(f"\n[ERROR] Failed to run {model} model: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create combined summary table
    print(f"\n{'='*80}")
    print("Creating Combined Summary Table")
    print(f"{'='*80}")
    
    combined_summary = []
    
    for model, rmse_data in all_results.items():
        model_suffix = 'LR' if model == 'Linear' else 'XGB'
        for resolution_name, rmse_by_datetime in rmse_data.items():
            if len(rmse_by_datetime) > 0:
                rmse_values = [rmse for _, rmse in rmse_by_datetime]
                rmse_values = [r for r in rmse_values if not np.isnan(r)]
                
                if len(rmse_values) > 0:
                    avg_rmse = np.mean(rmse_values)
                    median_rmse = np.median(rmse_values)
                    std_rmse = np.std(rmse_values)
                    min_rmse = np.min(rmse_values)
                    max_rmse = np.max(rmse_values)
                    n_predictions = len(rmse_values)
                    
                    combined_summary.append({
                        'Model': model_suffix,
                        'Resolution': resolution_name,
                        'Average_RMSE': avg_rmse,
                        'Median_RMSE': median_rmse,
                        'Std_RMSE': std_rmse,
                        'Min_RMSE': min_rmse,
                        'Max_RMSE': max_rmse,
                        'Number_of_Predictions': n_predictions
                    })
    
    # Save combined summary
    if len(combined_summary) > 0:
        combined_df = pd.DataFrame(combined_summary)
        combined_csv_path = os.path.join(output_base_dir, 'average_rmse_boxplot_combined_XGB_LR.csv')
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"\n  Combined Summary saved: {combined_csv_path}")
        print(f"\n  Combined Average RMSE by Model and Resolution:")
        print(combined_df.to_string(index=False))
    
    print(f"\n{'='*80}")
    print("[SUCCESS] All Box Plot Generation Completed!")
    print(f"Output directory: {output_base_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

