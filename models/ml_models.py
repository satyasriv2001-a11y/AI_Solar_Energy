"""
Machine learning regressors with config-driven parameters.
Uses GPU-accelerated versions for Random Forest and Gradient Boosting.
"""
import warnings
import numpy as np
import torch

# Suppress warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Suppress library output
import logging
logging.getLogger('lightgbm').setLevel(logging.ERROR)
logging.getLogger('xgboost').setLevel(logging.ERROR)
logging.getLogger('cuml').setLevel(logging.ERROR)

# Try to import cuML for GPU-accelerated Random Forest and Linear Regression
GPU_RF_AVAILABLE = False
GPU_LINEAR_AVAILABLE = False

try:
    import cuml
    from cuml.ensemble import RandomForestRegressor as cuRandomForestRegressor
    from cuml.linear_model import LinearRegression as cuLinearRegression
    GPU_RF_AVAILABLE = True
    GPU_LINEAR_AVAILABLE = True
    # cuML available (GPU-accelerated Random Forest and Linear Regression)
except Exception:
    # Silently fallback to sklearn
    from sklearn.ensemble import RandomForestRegressor as cuRandomForestRegressor
    from sklearn.linear_model import LinearRegression as cuLinearRegression
    GPU_RF_AVAILABLE = False
    GPU_LINEAR_AVAILABLE = False
    # Using sklearn (CPU) for Random Forest and Linear Regression

# Check XGBoost GPU support
XGB_GPU_AVAILABLE = False
try:
    import xgboost as xgb
    # Test XGBoost GPU support with proper GPU method
    try:
        test_model = xgb.XGBRegressor(tree_method='gpu_hist', device='cuda', n_estimators=1)
        XGB_GPU_AVAILABLE = True
        # XGBoost GPU available
    except:
        # XGBoost GPU unavailable, using CPU version
        pass
except ImportError:
    # XGBoost unavailable
    pass

# Check LightGBM GPU support
LGB_GPU_AVAILABLE = False
try:
    # Suppress LightGBM import warnings and output
    import warnings
    import os
    import sys
    from io import StringIO
    
    # Redirect stderr to suppress LightGBM output
    old_stderr = sys.stderr
    sys.stderr = StringIO()
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import lightgbm as lgb
        import numpy as np
        from sklearn.multioutput import MultiOutputRegressor
        
        # Test LightGBM GPU support
        try:
            # Create test data
            X_test = np.random.rand(10, 5)
            y_test = np.random.rand(10, 24)
            
            # Test GPU training
            base = lgb.LGBMRegressor(device='gpu', gpu_platform_id=0, gpu_device_id=0, n_estimators=1, verbose=-1)
            model = MultiOutputRegressor(base)
            model.fit(X_test, y_test)
            LGB_GPU_AVAILABLE = True
            # LightGBM GPU available
        except Exception as e:
            if "GPU Tree Learner was not enabled" in str(e):
                # LightGBM GPU unavailable - needs recompilation with GPU support
                pass
            else:
                # LightGBM GPU unavailable: {e}
                pass
    finally:
        # Restore stderr
        sys.stderr = old_stderr
        
except ImportError:
    # LightGBM unavailable
    pass

from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import threading

# GPU resource lock to prevent conflicts during parallel training
gpu_lock = threading.Lock()

def train_rf(X_train, y_train, params: dict):
    """Train Random Forest regressor with GPU support via cuML."""
    try:
        # Check data validity
        if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
            print("Detected NaN or Inf values, cleaning")
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)
        if np.any(np.isnan(y_train)) or np.any(np.isinf(y_train)):
            print("Detected NaN or Inf values, cleaning")
            y_train = np.nan_to_num(y_train, nan=0.0, posinf=1.0, neginf=-1.0)
        
        n_est = params.get('n_estimators', 30)
        max_d = params.get('max_depth', 3)
        
        # cuML GPU parameters (no n_jobs!)
        cuml_params = {
            'n_estimators': n_est,
            'max_depth': max_d,
            'random_state': 42
            # Note: cuML does NOT support n_jobs parameter
        }
        
        from sklearn.multioutput import MultiOutputRegressor
        
        if GPU_RF_AVAILABLE:
            # cuML works directly with NumPy arrays
            base = cuRandomForestRegressor(**cuml_params)
            model = MultiOutputRegressor(base, n_jobs=1)
            model.fit(X_train, y_train)
        else:
            # Use sklearn fallback with CPU
            base = cuRandomForestRegressor(**cuml_params)  # Already imported as sklearn RF
            model = MultiOutputRegressor(base, n_jobs=-1)  # Use all CPU cores
            model.fit(X_train, y_train)
        
        return model
        
    except Exception as e:
        print(f"[ERROR] Random Forest training completely failed: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Random Forest training failed: {e}")

# GBR removed, use XGBoost and LightGBM instead

def train_xgb(X_train, y_train, params: dict):
    """Train XGBoost regressor with multi-output support - GPU optimized."""
    try:
        import time
        
        # Check data validity
        if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
            print("Detected NaN or Inf values, cleaning")
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)
        if np.any(np.isnan(y_train)) or np.any(np.isinf(y_train)):
            print("Detected NaN or Inf values, cleaning")
            y_train = np.nan_to_num(y_train, nan=0.0, posinf=1.0, neginf=-1.0)
        
        n_est = params.get('n_estimators', 30)
        max_d = params.get('max_depth', 3)
        lr = params.get('learning_rate', 0.1)
        
        # GPU-only version (no CPU fallback)
        if not (torch.cuda.is_available() and XGB_GPU_AVAILABLE):
            raise RuntimeError("XGBoost GPU not available! Cannot train.")
        
        gpu_params = params.copy()
        gpu_params.update({
            'tree_method': 'hist',  # Use histogram algorithm (auto GPU with device='cuda')
            'device': 'cuda',       # Unified GPU control (XGBoost 2.0+ style)
            'verbosity': 0,         # Silent mode (suppress warnings)
        })
        
        # Important: GPU memory is limited!
        # n_jobs=-1 can cause OOM when multiple models compete for GPU memory
        # Use n_jobs=2 for limited parallelism (train 2 models at a time)
        base = XGBRegressor(**gpu_params)
        
        start_time = time.time()
        model = MultiOutputRegressor(base, n_jobs=1)  # Sequential to avoid OOM
        model.fit(X_train, y_train)
        elapsed = time.time() - start_time
        return model
        
    except Exception as e:
        print(f"[ERROR] XGBoost training failed: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"XGBoost training failed: {e}")

def train_lgbm(X_train, y_train, params: dict):
    """Train LightGBM regressor with multi-output support - GPU optimized."""
    try:
        import time
        
        # Check data validity
        if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
            print("Detected NaN or Inf values, cleaning")
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)
        if np.any(np.isnan(y_train)) or np.any(np.isinf(y_train)):
            print("Detected NaN or Inf values, cleaning")
            y_train = np.nan_to_num(y_train, nan=0.0, posinf=1.0, neginf=-1.0)
        
        n_est = params.get('n_estimators', 30)
        max_d = params.get('max_depth', 3)
        lr = params.get('learning_rate', 0.1)
        
        # GPU-only version (no CPU fallback)
        if not (torch.cuda.is_available() and LGB_GPU_AVAILABLE):
            raise RuntimeError("LightGBM GPU not available! Cannot train.")
        
        # Suppress LightGBM warnings
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning)
        
        gpu_params = params.copy()
        gpu_params.update({
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'verbose': -1  # Silent mode: suppress all warnings
        })
        base = LGBMRegressor(**gpu_params)
        
        start_time = time.time()
        model = MultiOutputRegressor(base, n_jobs=1)  # Sequential to avoid OOM
        model.fit(X_train, y_train)
        elapsed = time.time() - start_time
        return model
        
    except Exception as e:
        print(f"[ERROR] LightGBM training failed: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"LightGBM training failed: {e}")

def train_linear(X_train, y_train, params: dict):
    """Train Linear Regression with GPU support if available."""
    with gpu_lock:
        try:
            # Check data validity
            if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
                print("Detected NaN or Inf values, cleaning")
                X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)
            if np.any(np.isnan(y_train)) or np.any(np.isinf(y_train)):
                print("Detected NaN or Inf values, cleaning")
                y_train = np.nan_to_num(y_train, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Try GPU version with direct NumPy arrays (cuML compatibility)
            if GPU_LINEAR_AVAILABLE:
                try:
                    # cuML works directly with NumPy arrays (no cuDF needed!)
                    model = cuLinearRegression()
                    model.fit(X_train, y_train)  # Use NumPy arrays directly
                    return model
                except Exception as e:
                    # Fall through to CPU version
                    pass
            
            # CPU version (sklearn)
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X_train, y_train)
            return model
            
        except Exception as e:
            print(f"Linear Regression training failed: {e}")
            raise RuntimeError(f"Linear Regression training failed: {e}")
