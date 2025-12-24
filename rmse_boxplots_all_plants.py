#download dependencies

#!pip install -q torch pyyaml pandas scikit-learn matplotlib seaborn tqdm

# download XGBoost和LightGBM (GPU版本)

#!pip install xgboost lightgbm

#download装cuML GPU (CUDA 12)

#!pip install cuml-cu12 --extra-index-url=https://pypi.nvidia.com

# 1. Clone repository (remove if exists to avoid nesting)
# Navigate to /content first
%cd /content

# Clean up any existing directories using shell command
!rm -rf PV-Forecasting

# Clone repository
!git clone https://github.com/satyasriv2001-a11y/PV-Forecasting

# Navigate into the cloned repository
%cd /content/PV-Forecasting

# 2. Install dependencies

!pip install -q -r requirements.txt

# Install Google Sheets and plotting libraries

!pip install -q gspread google-auth matplotlib seaborn

# 3. Mount Google Drive

from google.colab import drive

drive.mount('/content/drive')

# 4. Run RMSE Box and Whisker Plots for ALL PLANTS
# This script loads prediction data from all plants and creates RMSE box plots
# for hourly, 30-min, 15-min, and 10-min resolutions
# X-axis: Starting hour of 24-hour sliding window (0-23)
# Y-axis: RMSE (Capacity Factor)

import subprocess
import os

# Set paths - UPDATE THESE PATHS TO MATCH YOUR DRIVE STRUCTURE
# This should point to the directory containing all plant prediction folders
PREDICTIONS_DIR = "/content/drive/MyDrive/Solar PV electricity/final_hourly_predictions/all_plants_XGB_high_PV+NWP_24h_noTE"

# Alternative paths you might need (uncomment the one that matches your structure):
# PREDICTIONS_DIR = "/content/drive/MyDrive/Solar PV electricity/all_plants_predictions"
# PREDICTIONS_DIR = "/content/drive/MyDrive/Solar PV electricity/final_hourly_predictions"
# PREDICTIONS_DIR = "/content/drive/MyDrive/predictions"

# Output directory (plots will be saved here)
OUTPUT_DIR = "/content/drive/MyDrive/Solar PV electricity/rmse_boxplots_all_plants"

print("=" * 80)
print("RMSE Box and Whisker Plots - ALL PLANTS (Multi-Resolution)")
print("=" * 80)
print("Description: Creates box plots showing RMSE distribution across all plants")
print("             for each prediction start hour (0-23)")
print("Resolutions: Hourly, 30-minute, 15-minute, 10-minute")
print(f"Predictions directory: {PREDICTIONS_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print("=" * 80)

# Check if predictions directory exists
if not os.path.exists(PREDICTIONS_DIR):
    print(f"\n[ERROR] Predictions directory not found: {PREDICTIONS_DIR}")
    print("\nPlease update PREDICTIONS_DIR in the script to match your Drive structure.")
    print("\nCommon locations:")
    print("  - /content/drive/MyDrive/Solar PV electricity/final_hourly_predictions/all_plants_XGB_high_PV+NWP_24h_noTE")
    print("  - /content/drive/MyDrive/Solar PV electricity/all_plants_predictions")
    print("  - /content/drive/MyDrive/predictions")
    print("\nListing Drive contents to help locate your predictions:")
    try:
        drive_base = "/content/drive/MyDrive"
        if os.path.exists(drive_base):
            items = os.listdir(drive_base)
            print(f"\nItems in {drive_base}:")
            for item in items[:20]:
                print(f"  - {item}")
    except:
        pass
else:
    # List what's in the predictions directory
    print(f"\nPredictions directory exists. Contents:")
    try:
        items = os.listdir(PREDICTIONS_DIR)
        print(f"  Found {len(items)} item(s)")
        for item in items[:10]:
            item_path = os.path.join(PREDICTIONS_DIR, item)
            if os.path.isdir(item_path):
                print(f"  [DIR]  {item}")
            else:
                print(f"  [FILE] {item}")
        if len(items) > 10:
            print(f"  ... and {len(items) - 10} more items")
    except Exception as e:
        print(f"  Could not list contents: {str(e)}")
    
    # Get the script path
    script_path = os.path.join(os.getcwd(), 'rmse_boxplots_all_plants.py')
    
    if not os.path.exists(script_path):
        print(f"\n[ERROR] Script not found: {script_path}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in directory: {os.listdir('.')[:10]}")
    else:
        print(f"\nScript found: {script_path}")
        
        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Run RMSE box plots script
        cmd = [
            'python', script_path,
            '--predictions-dir', PREDICTIONS_DIR,
            '--output-dir', OUTPUT_DIR
        ]
        
        print(f"\nRunning command:")
        print(f"  {' '.join(cmd)}")
        print()
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"\n[SUCCESS] RMSE Box Plot Generation Completed!")
            if result.stdout:
                print("\n--- Script Output ---")
                print(result.stdout)
            
            print(f"\n{'='*80}")
            print("Generated Output Files:")
            print(f"{'='*80}")
            print(f"Output directory: {OUTPUT_DIR}")
            print("\nExpected files:")
            print("  - rmse_boxplot_hourly_all_plants.png (Hourly resolution)")
            print("  - rmse_boxplot_30_minute_all_plants.png (30-minute resolution)")
            print("  - rmse_boxplot_15_minute_all_plants.png (15-minute resolution)")
            print("  - rmse_boxplot_10_minute_all_plants.png (10-minute resolution)")
            print("  - rmse_boxplot_all_resolutions_all_plants.png (Combined comparison)")
            print("  - rmse_summary_all_plants.csv (Summary data)")
            print(f"{'='*80}")
            
            # List generated files
            try:
                if os.path.exists(OUTPUT_DIR):
                    generated_files = os.listdir(OUTPUT_DIR)
                    if len(generated_files) > 0:
                        print(f"\nGenerated files in output directory:")
                        for f in generated_files:
                            print(f"  - {f}")
            except:
                pass
        else:
            print(f"\n[ERROR] Script execution failed:")
            print("--- Standard Error ---")
            print(result.stderr)
            print("\n--- Standard Output ---")
            print(result.stdout)

print(f"\n{'='*80}")
print("Script Execution Complete")
print(f"{'='*80}")

