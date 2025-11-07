# Sensitivity Analysis Experiments

This directory contains 8 sensitivity analysis experiments to evaluate various factors affecting PV forecasting model performance.

## Overview

All experiments use:
- **Base configuration**: PV+NWP features, 24-hour lookback, no time encoding, high model complexity
- **Models**: 7 models (LSTM, GRU, Transformer, TCN, RF, XGB, LGBM) + Linear Regression for specific experiments
- **Metrics**: MAE, RMSE, R², NRMSE (RMSE/mean), training time
- **Aggregation**: Mean and standard deviation across 100 plants

## Experiments

### 1. Seasonal Effect (`seasonal_effect.py`)
Analyzes model performance across four seasons:
- **Seasons**: Spring (Mar-May), Summer (Jun-Aug), Fall (Sep-Nov), Winter (Dec-Feb)
- **Models**: 7 models + Linear (NWP only)
- **Output**: Results grouped by season and model

```bash
python sensitivity_analysis/seasonal_effect.py --data-dir data --output-dir sensitivity_analysis/results
```

### 2. Hourly Effect (`hourly_effect.py`)
Analyzes model performance across 24 hours of the day:
- **Hours**: 0-23
- **Models**: 7 models + Linear (NWP only)
- **Output**: Results grouped by hour and model

```bash
python sensitivity_analysis/hourly_effect.py --data-dir data --output-dir sensitivity_analysis/results
```

### 3. Weather Feature Adoption (`weather_feature_adoption.py`)
Analyzes impact of different weather feature tiers:
- **Feature Tiers**:
  - SI: Solar Irradiance only (1 feature)
  - H: High correlation (3 features: SI + vapour pressure deficit + relative humidity)
  - H+M: High + Medium (7 features: H + temperature + wind gusts + cloud cover + wind speed)
  - H+M+L: All features (11 features: H+M + snow depth + dew point + surface pressure + precipitation)
- **Models**: 7 models + Linear (NWP only)
- **Output**: Results grouped by feature tier and model

```bash
python sensitivity_analysis/weather_feature_adoption.py --data-dir data --output-dir sensitivity_analysis/results
```

### 4. Lookback Window Length (`lookback_window.py`)
Analyzes impact of different lookback window lengths:
- **Lookback Windows**: 24h, 72h, 120h, 168h
- **Models**: 7 models (Linear NOT included - no lookback concept)
- **Output**: Results grouped by lookback window and model

```bash
python sensitivity_analysis/lookback_window.py --data-dir data --output-dir sensitivity_analysis/results
```

### 5. Model Complexity (`model_complexity.py`)
Analyzes impact of model complexity:
- **Complexity Levels**:
  - Low: Small models (epochs=20, d_model=16, hidden=8, layers=1, n_estimators=50)
  - Mid-Low: Medium-small (epochs=35, d_model=24, hidden=12, layers=1, n_estimators=100)
  - Mid-High: Medium-large (epochs=50, d_model=32, hidden=16, layers=2, n_estimators=150)
  - High: Large models (epochs=50, d_model=32, hidden=16, layers=2, n_estimators=200)
- **Models**: 7 models (Linear NOT included)
- **Output**: Results grouped by complexity and model

```bash
python sensitivity_analysis/model_complexity.py --data-dir data --output-dir sensitivity_analysis/results
```

### 6. Training Dataset Scale (`training_scale.py`)
Analyzes impact of training dataset size:
- **Training Scales**: Low (20%), Medium (40%), High (60%), Full (80%)
- **Models**: 7 models + Linear (NWP only)
- **Output**: Results grouped by training scale and model

```bash
python sensitivity_analysis/training_scale.py --data-dir data --output-dir sensitivity_analysis/results
```

### 7. Shuffle Training (`no_shuffle.py`)
Analyzes model performance with shuffled data splitting (random split):
- **Data Split**: Shuffled (random split) - covers all seasons for robust evaluation
- **Models**: 7 models + Linear (NWP only)
- **Output**: Results with shuffled data splitting (compared to default sequential split)

```bash
python sensitivity_analysis/no_shuffle.py --data-dir data --output-dir sensitivity_analysis/results
```

### 8. Dataset Extension (`dataset_extension.py`)
Analyzes impact of using hourly sliding windows (vs. daily windows):
- **Window Type**: Hourly sliding windows (creates ~24x more samples)
- **Current**: Daily windows predict 0-23 hours starting from 23:00 each day
- **New**: Hourly windows can predict from any hour (e.g., 1-0, 2-1, etc.)
- **Models**: 7 models + Linear (NWP only)
- **Output**: Results grouped by model

```bash
python sensitivity_analysis/dataset_extension.py --data-dir data --output-dir sensitivity_analysis/results
```

## Running All Experiments

Use the main script to run all experiments or selected ones:

```bash
# Run all 8 experiments
python sensitivity_analysis/run_all_experiments.py --data-dir data --output-dir sensitivity_analysis/results

# Run specific experiments (e.g., 1, 3, 5)
python sensitivity_analysis/run_all_experiments.py --experiments 1 3 5
```

## Output Files

Each experiment generates three CSV files:

1. **`{experiment}_detailed.csv`**: Raw results for each plant and configuration
2. **`{experiment}_aggregated.csv`**: Mean and std across plants
3. **`{experiment}_pivot.csv`**: Pivot table for easy visualization

## Google Colab Usage

### Setup

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone repository
!git clone https://github.com/zhesun-0209/PV-Forecasting.git
%cd PV-Forecasting

# Install dependencies
!pip install -r requirements.txt
```

### Upload Data

Upload your 100 CSV files to the `data/` directory:

```python
# Option 1: Upload from local
from google.colab import files
uploaded = files.upload()  # Select multiple CSV files

# Move to data directory
!mkdir -p data
!mv *.csv data/

# Option 2: Copy from Google Drive
!cp /content/drive/MyDrive/your_data_folder/*.csv data/
```

### Run Experiments

```python
# Run all experiments with results saved to Google Drive
!python sensitivity_analysis/run_all_experiments.py \
    --data-dir data \
    --output-dir "/content/drive/MyDrive/Solar PV electricity/sensitivity_analysis_results"

# Or run individual experiments
!python sensitivity_analysis/seasonal_effect.py \
    --data-dir data \
    --output-dir "/content/drive/MyDrive/Solar PV electricity/sensitivity_analysis_results"
```

### Check GPU

```python
import torch
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Common Utilities

The `common_utils.py` module provides shared functions:

- **`get_season(month)`**: Get season from month
- **`compute_nrmse(y_true, y_pred)`**: Compute normalized RMSE
- **`create_base_config(...)`**: Create base experiment configuration
- **`run_single_experiment(...)`**: Run a single experiment
- **`aggregate_results(...)`**: Aggregate results across plants
- **`load_all_plant_configs(data_dir)`**: Load all plant configurations from CSV files
- **`save_results(...)`**: Save results to CSV

## Model Definitions

- **DL_MODELS**: `['LSTM', 'GRU', 'Transformer', 'TCN']` (4 models)
- **ML_MODELS**: `['RF', 'XGB', 'LGBM']` (3 models)
- **ALL_MODELS_NO_LINEAR**: DL + ML (7 models, excluding Linear)

## Weather Feature Tiers

Based on correlation with PV output:

- **SI (Solar Irradiance)**: `global_tilted_irradiance`
- **H (High)**: SI + `vapour_pressure_deficit` + `relative_humidity_2m`
- **H+M (High+Medium)**: H + `temperature_2m` + `wind_gusts_10m` + `cloud_cover_low` + `wind_speed_100m`
- **H+M+L (All)**: H+M + `snow_depth` + `dew_point_2m` + `surface_pressure` + `precipitation`

## Notes

1. **Linear Regression**: Only included in experiments 1, 2, 3, 6, 7, 8 (uses NWP only configuration)
2. **Data Requirements**: Each CSV file should contain:
   - Columns: `Year`, `Month`, `Day`, `Hour`, `Capacity Factor`
   - Weather features: 11 meteorological variables (with and without `_pred` suffix)
3. **Computational Cost**: Experiment 8 (hourly sliding windows) creates ~24x more samples and takes significantly longer
4. **Resume Support**: Not implemented - experiments run from scratch each time

## Troubleshooting

### Missing Features Error
If you get errors about missing weather features, check:
- CSV files contain all 11 meteorological variables
- Both actual and predicted versions (with `_pred` suffix)

### Out of Memory
- Reduce batch size in config
- Use fewer plants for testing
- Enable GPU acceleration in Colab (Runtime > Change runtime type > GPU)

### Slow Execution
- Use GPU acceleration
- Reduce number of plants for testing
- Run experiments individually instead of all at once
- For experiment 8, consider reducing the lookback window

## Contact

For questions or issues, please contact the repository maintainer or open an issue on GitHub.

