#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RMSE Box and Whisker Plots for All Plants - Multi-Resolution

Creates box plots showing RMSE distribution across all plants for each prediction start hour.
Plots average RMSE values for hourly, 30-min, 15-min, and 10-min resolutions.
The x-axis is the starting hour of the 24-hour sliding window (0-23), and y-axis is RMSE.

Usage:
    python rmse_boxplots_all_plants.py --predictions-dir /path/to/predictions --output-dir outputs/
    python rmse_boxplots_all_plants.py --predictions-dir ./all_plants_predictions --output-dir ./rmse_boxplots_all_plants
"""

import pandas as pd
import numpy as np
import os
import sys
import glob
import argparse
import warnings
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from collections import defaultdict

warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append(script_dir)


def calculate_rmse(preds, gt):
    """
    Calculate RMSE for a prediction window.
    
    Args:
        preds: Predicted values (array)
        gt: Ground truth values (array)
    
    Returns:
        RMSE (float) or np.nan if insufficient valid data
    """
    # Filter out NaN values
    valid_mask = ~(np.isnan(preds) | np.isnan(gt))
    
    if np.sum(valid_mask) < 1:  # Need at least 1 point
        return np.nan
    
    preds_valid = preds[valid_mask]
    gt_valid = gt[valid_mask]
    
    # Calculate RMSE
    mse = np.mean((preds_valid - gt_valid) ** 2)
    rmse = np.sqrt(mse)
    
    return rmse


def detect_resolution(df, file_path):
    """
    Detect the resolution (hourly, 30-min, 15-min, or 10-min) from data or filename.
    
    Args:
        df: DataFrame with Datetime column
        file_path: Path to the file (for filename-based detection)
    
    Returns:
        Resolution name ('Hourly', '30-minute', '15-minute', '10-minute') and resolution_minutes
    """
    # First try filename-based detection
    filename = os.path.basename(file_path).lower()
    if 'hour' in filename or 'hourly' in filename:
        return 'Hourly', 60
    if '30min' in filename or 'halfhour' in filename or '30_min' in filename:
        return '30-minute', 30
    if '15min' in filename or '15_min' in filename:
        return '15-minute', 15
    if '10min' in filename or '10_min' in filename:
        return '10-minute', 10
    
    # Otherwise, detect from data frequency
    if 'Datetime' not in df.columns or len(df) < 2:
        return 'Hourly', 60  # Default
    
    df = df.copy()
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime').reset_index(drop=True)
    
    # Calculate time differences
    time_diffs = df['Datetime'].diff().dropna()
    if len(time_diffs) == 0:
        return 'Hourly', 60  # Default
    
    median_diff_minutes = time_diffs.median().total_seconds() / 60.0
    
    # Determine resolution based on median time difference
    if median_diff_minutes <= 12:  # <= 12 minutes -> 10-minute
        return '10-minute', 10
    elif median_diff_minutes <= 20:  # <= 20 minutes -> 15-minute
        return '15-minute', 15
    elif median_diff_minutes <= 40:  # <= 40 minutes -> 30-minute
        return '30-minute', 30
    else:  # > 40 minutes -> hourly
        return 'Hourly', 60


def load_predictions_from_dir(predictions_dir):
    """
    Load all prediction CSV files from plant directories.
    
    Args:
        predictions_dir: Base directory containing plant prediction folders
    
    Returns:
        Dictionary: {resolution_name: {plant_name: [(start_hour, rmse), ...]}}
    """
    if not os.path.exists(predictions_dir):
        raise FileNotFoundError(f"Predictions directory not found: {predictions_dir}")
    
    # Find all plant folders (directories in predictions_dir)
    plant_folders = []
    for item in os.listdir(predictions_dir):
        item_path = os.path.join(predictions_dir, item)
        if os.path.isdir(item_path):
            plant_folders.append(item_path)
    
    plant_folders.sort()
    
    if len(plant_folders) == 0:
        # Maybe predictions_dir itself contains CSV files directly
        print(f"[INFO] No plant folders found, checking for CSV files directly in {predictions_dir}")
        csv_files = glob.glob(os.path.join(predictions_dir, "*.csv"))
        if len(csv_files) > 0:
            print(f"[INFO] Found {len(csv_files)} CSV file(s) directly in directory")
            # Treat the directory as a single "plant"
            plant_folders = [predictions_dir]
        else:
            raise ValueError(f"No plant folders or CSV files found in {predictions_dir}")
    
    print(f"Found {len(plant_folders)} plant folder(s)")
    
    # Dictionary to store results: {resolution: {plant: [(hour, rmse), ...]}}
    results_by_resolution = defaultdict(lambda: defaultdict(list))
    
    # Process each plant folder
    for plant_folder in plant_folders:
        plant_name = os.path.basename(plant_folder)
        if plant_name == os.path.basename(predictions_dir):
            plant_name = "All_Plants"  # Use generic name if processing directory directly
        
        print(f"\nProcessing plant: {plant_name}")
        
        # Find all prediction CSV files in this plant's folder
        # Try multiple patterns
        prediction_files = []
        patterns = [
            "predictions_*.csv",
            "prediction_*.csv",
            "*predictions*.csv",
            "*.csv"  # Last resort: all CSV files
        ]
        
        for pattern in patterns:
            found_files = glob.glob(os.path.join(plant_folder, pattern))
            if len(found_files) > 0:
                prediction_files.extend(found_files)
        
        # Remove duplicates
        prediction_files = list(set(prediction_files))
        
        # Filter out summary/result files
        prediction_files = [f for f in prediction_files 
                          if not any(x in os.path.basename(f).lower() 
                                   for x in ['summary', 'result', 'rmse', 'error', 'average', 'status'])]
        
        if len(prediction_files) == 0:
            print(f"  [WARNING] No prediction files found in {plant_folder}")
            continue
        
        print(f"  Found {len(prediction_files)} prediction file(s)")
        
        # Process each prediction file
        processed_count = 0
        for pred_file in sorted(prediction_files):
            try:
                df = pd.read_csv(pred_file)
                
                # Check required columns - try different possible column names
                datetime_col = None
                pred_col = None
                gt_col = None
                
                for col in df.columns:
                    col_lower = col.lower()
                    if datetime_col is None and ('datetime' in col_lower or 'date' in col_lower or 'time' in col_lower):
                        datetime_col = col
                    if pred_col is None and ('predicted' in col_lower and 'capacity' in col_lower):
                        pred_col = col
                    if gt_col is None and (('ground' in col_lower and 'truth' in col_lower) or 
                                         ('actual' in col_lower) or
                                         ('true' in col_lower and 'capacity' in col_lower)):
                        gt_col = col
                
                # Use exact matches if available
                if 'Datetime' in df.columns:
                    datetime_col = 'Datetime'
                if 'Predicted_Capacity_Factor' in df.columns:
                    pred_col = 'Predicted_Capacity_Factor'
                if 'Ground_Truth_Capacity_Factor' in df.columns:
                    gt_col = 'Ground_Truth_Capacity_Factor'
                
                if datetime_col is None or pred_col is None or gt_col is None:
                    print(f"    [SKIP] {os.path.basename(pred_file)}: Missing required columns")
                    print(f"      Found columns: {list(df.columns)}")
                    continue
                
                # Convert datetime column to datetime
                df[datetime_col] = pd.to_datetime(df[datetime_col])
                df = df.sort_values(datetime_col).reset_index(drop=True)
                
                # Detect resolution
                resolution_name, resolution_minutes = detect_resolution(df, pred_file)
                
                # Calculate RMSE for this prediction window
                preds = df[pred_col].values
                gt = df[gt_col].values
                
                rmse = calculate_rmse(preds, gt)
                
                if np.isnan(rmse):
                    print(f"    [SKIP] {os.path.basename(pred_file)}: Invalid RMSE (insufficient data)")
                    continue
                
                # Extract starting hour from the first datetime
                start_datetime = df[datetime_col].iloc[0]
                start_hour = start_datetime.hour
                
                # Store result
                results_by_resolution[resolution_name][plant_name].append((start_hour, rmse))
                processed_count += 1
                
                if processed_count <= 3 or processed_count % 10 == 0:
                    print(f"    [{processed_count}/{len(prediction_files)}] {os.path.basename(pred_file)}: {resolution_name}, Hour {start_hour:02d}, RMSE={rmse:.4f}")
                
            except Exception as e:
                print(f"  [ERROR] Error processing {os.path.basename(pred_file)}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"  Processed {processed_count}/{len(prediction_files)} files successfully")
    
    return results_by_resolution


def create_rmse_boxplots(results_by_resolution, output_dir):
    """
    Create RMSE box and whisker plots for each resolution.
    
    Args:
        results_by_resolution: Dictionary {resolution_name: {plant_name: [(hour, rmse), ...]}}
        output_dir: Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define resolution order
    resolution_order = ['Hourly', '30-minute', '15-minute', '10-minute']
    
    # Filter to only resolutions we have data for
    available_resolutions = [r for r in resolution_order if r in results_by_resolution]
    
    if len(available_resolutions) == 0:
        print("[WARNING] No data found for any resolution")
        return
    
    print(f"\n{'='*80}")
    print("Creating RMSE Box Plots")
    print(f"{'='*80}")
    
    # Create separate plot for each resolution
    for resolution_name in available_resolutions:
        print(f"\nProcessing {resolution_name} resolution...")
        
        resolution_data = results_by_resolution[resolution_name]
        
        # Collect all (hour, rmse) pairs for this resolution
        all_data = []
        for plant_name, hour_rmse_pairs in resolution_data.items():
            all_data.extend([(hour, rmse, plant_name) for hour, rmse in hour_rmse_pairs])
        
        if len(all_data) == 0:
            print(f"  [WARNING] No data for {resolution_name}")
            continue
        
        # Convert to DataFrame for easier handling
        df = pd.DataFrame(all_data, columns=['Hour', 'RMSE', 'Plant'])
        
        # Group by hour and prepare data for box plot
        hours = sorted(df['Hour'].unique())
        rmse_data_by_hour = []
        hour_labels = []
        
        for hour in range(24):  # Hours 0-23
            hour_data = df[df['Hour'] == hour]['RMSE'].values
            if len(hour_data) >= 1:  # Need at least 1 data point
                rmse_data_by_hour.append(hour_data)
                hour_labels.append(hour)
        
        if len(rmse_data_by_hour) == 0:
            print(f"  [WARNING] No valid hour data for {resolution_name}")
            continue
        
        # Create box plot
        plt.figure(figsize=(16, 8))
        plt.rcParams.update({'font.size': 12})
        
        # Create box plot
        bp = plt.boxplot(rmse_data_by_hour, labels=hour_labels, patch_artist=True, widths=0.6)
        
        # Color the boxes
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Format x-axis - show hours 0-23
        ax = plt.gca()
        ax.set_xlim(0.5, len(hour_labels) + 0.5)
        
        plt.xlabel('Starting Hour of 24-Hour Sliding Window (0-23)', fontsize=14, fontweight='bold')
        plt.ylabel('RMSE (Capacity Factor)', fontsize=14, fontweight='bold')
        plt.title(f'RMSE Distribution Across All Plants by Prediction Start Hour - {resolution_name}\n'
                  f'Box plot shows RMSE distribution for 24-hour forecasts starting at each hour',
                  fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=0)
        plt.tight_layout()
        
        # Save plot
        resolution_file_name = resolution_name.lower().replace('-', '_')
        output_path = os.path.join(output_dir, f"rmse_boxplot_{resolution_file_name}_all_plants.png")
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Box plot saved: {output_path}")
        
        # Print statistics
        print(f"  Statistics for {resolution_name}:")
        print(f"    Total predictions: {len(df)}")
        print(f"    Unique plants: {df['Plant'].nunique()}")
        print(f"    Hours with data: {len(hour_labels)}")
        print(f"    Mean RMSE: {df['RMSE'].mean():.4f}")
        print(f"    Median RMSE: {df['RMSE'].median():.4f}")
        print(f"    Std RMSE: {df['RMSE'].std():.4f}")
    
    # Create combined plot with all resolutions
    print(f"\nCreating combined plot with all resolutions...")
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('RMSE Distribution Across All Plants by Prediction Start Hour\n'
                 'Comparison Across All Resolutions', 
                 fontsize=18, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, resolution_name in enumerate(available_resolutions[:4]):  # Max 4 resolutions
        ax = axes[idx]
        resolution_data = results_by_resolution[resolution_name]
        
        # Collect all (hour, rmse) pairs
        all_data = []
        for plant_name, hour_rmse_pairs in resolution_data.items():
            all_data.extend([(hour, rmse) for hour, rmse in hour_rmse_pairs])
        
        if len(all_data) == 0:
            ax.text(0.5, 0.5, f'No data for\n{resolution_name}', 
                   ha='center', va='center', fontsize=14)
            ax.set_title(resolution_name, fontsize=14, fontweight='bold')
            continue
        
        df = pd.DataFrame(all_data, columns=['Hour', 'RMSE'])
        
        # Group by hour
        rmse_data_by_hour = []
        hour_labels = []
        
        for hour in range(24):
            hour_data = df[df['Hour'] == hour]['RMSE'].values
            if len(hour_data) >= 1:
                rmse_data_by_hour.append(hour_data)
                hour_labels.append(hour)
        
        if len(rmse_data_by_hour) == 0:
            ax.text(0.5, 0.5, f'No valid data for\n{resolution_name}', 
                   ha='center', va='center', fontsize=14)
            ax.set_title(resolution_name, fontsize=14, fontweight='bold')
            continue
        
        # Create box plot
        bp = ax.boxplot(rmse_data_by_hour, labels=hour_labels, patch_artist=True, widths=0.6)
        
        # Color the boxes
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Starting Hour (0-23)', fontsize=12, fontweight='bold')
        ax.set_ylabel('RMSE', fontsize=12, fontweight='bold')
        ax.set_title(f'{resolution_name} (n={len(df)}, plants={len(resolution_data)})', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xlim(0.5, len(hour_labels) + 0.5)
    
    # Hide unused subplots
    for idx in range(len(available_resolutions), 4):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save combined plot
    combined_output_path = os.path.join(output_dir, "rmse_boxplot_all_resolutions_all_plants.png")
    plt.savefig(combined_output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Combined plot saved: {combined_output_path}")
    
    # Create summary CSV
    print(f"\nCreating summary CSV...")
    summary_data = []
    
    for resolution_name in available_resolutions:
        resolution_data = results_by_resolution[resolution_name]
        
        # Collect all data for this resolution
        all_data = []
        for plant_name, hour_rmse_pairs in resolution_data.items():
            for hour, rmse in hour_rmse_pairs:
                all_data.append({
                    'Resolution': resolution_name,
                    'Plant': plant_name,
                    'Start_Hour': hour,
                    'RMSE': rmse
                })
        
        if len(all_data) > 0:
            summary_data.extend(all_data)
    
    if len(summary_data) > 0:
        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = os.path.join(output_dir, 'rmse_summary_all_plants.csv')
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"  Summary CSV saved: {summary_csv_path}")
        
        # Print summary statistics
        print(f"\nSummary Statistics:")
        for resolution_name in available_resolutions:
            res_df = summary_df[summary_df['Resolution'] == resolution_name]
            if len(res_df) > 0:
                print(f"\n  {resolution_name}:")
                print(f"    Total predictions: {len(res_df)}")
                print(f"    Unique plants: {res_df['Plant'].nunique()}")
                print(f"    Mean RMSE: {res_df['RMSE'].mean():.4f}")
                print(f"    Median RMSE: {res_df['RMSE'].median():.4f}")
                print(f"    Std RMSE: {res_df['RMSE'].std():.4f}")
                print(f"    Min RMSE: {res_df['RMSE'].min():.4f}")
                print(f"    Max RMSE: {res_df['RMSE'].max():.4f}")
    
    print(f"\n{'='*80}")
    print("[SUCCESS] RMSE Box Plot Generation Completed!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description='Create RMSE box and whisker plots for all plants across multiple resolutions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create plots from predictions directory
  python rmse_boxplots_all_plants.py --predictions-dir ./all_plants_predictions --output-dir ./rmse_boxplots
  
  # Use Drive path
  python rmse_boxplots_all_plants.py --predictions-dir /content/drive/MyDrive/.../predictions --output-dir ./outputs
        """
    )
    
    parser.add_argument('--predictions-dir', type=str, required=True,
                       help='Base directory containing plant prediction folders')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for plots (default: ./rmse_boxplots_all_plants)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("RMSE Box and Whisker Plots - All Plants (Multi-Resolution)")
    print("=" * 80)
    print(f"Predictions directory: {os.path.abspath(args.predictions_dir)}")
    
    if args.output_dir is None:
        output_dir = os.path.join(script_dir, "rmse_boxplots_all_plants")
    else:
        output_dir = args.output_dir
    
    print(f"Output directory: {os.path.abspath(output_dir)}")
    print("=" * 80)
    
    try:
        # Load predictions from all plants
        print("\n[1/2] Loading predictions from all plants...")
        results_by_resolution = load_predictions_from_dir(args.predictions_dir)
        
        if len(results_by_resolution) == 0:
            print("\n[ERROR] No prediction data found!")
            sys.exit(1)
        
        print(f"\nFound data for {len(results_by_resolution)} resolution(s): {list(results_by_resolution.keys())}")
        
        # Create box plots
        print("\n[2/2] Creating RMSE box plots...")
        create_rmse_boxplots(results_by_resolution, output_dir)
        
    except Exception as e:
        print(f"\n[ERROR] Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

