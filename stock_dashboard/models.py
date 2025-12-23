import numpy as np
import pandas as pd
import time
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, precision_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import warnings
warnings.filterwarnings('ignore')

def debug_dataframe(df, stage_name="DataFrame"):
    """Debug helper function to inspect DataFrame structure"""
    print(f"\n=== {stage_name} Debug Info ===")
    if df is None:
        print("DataFrame is None!")
        return df
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    if not df.empty and len(df.columns) > 0:
        print(f"Sample data:\n{df.head(2)}")
        print(f"Null counts:\n{df.isnull().sum()}")
    print("=" * 50)
    return df

def prepare_features_and_target(df_processed, train_split=0.7, debug=True):
    """
    Prepare features and target variables with robust train/test splitting
    
    Args:
        df_processed (pd.DataFrame): DataFrame with processed stock data
        train_split (float): Percentage of data to use for training (0.0 to 1.0)
        debug (bool): Whether to print debug information
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, df)
            - X_train, X_test: Feature matrices for training and testing
            - y_train, y_test: Target variables for training and testing
            - df: Processed DataFrame with Target column added
            
    Raises:
        ValueError: If input data is invalid or cannot be split properly
    """
    # Input validation
    if df_processed is None or df_processed.empty:
        raise ValueError("Input DataFrame is None or empty")
        
    if not isinstance(train_split, (int, float)) or not 0 < train_split < 1:
        raise ValueError(f"train_split must be a number between 0 and 1, got {train_split}")
        
    if debug:
        print(f"\nStarting feature preparation...")
        print(f"Train/Test ratio: {train_split:.1%} training, {(1-train_split):.1%} testing")
        debug_dataframe(df_processed, "Input data")
    
    try:
        # Validate required columns
        required_cols = ['Return']
        missing_cols = [col for col in required_cols if col not in df_processed.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if debug:
            print("\nChecking Return column statistics:")
            print(f"  Total rows: {len(df_processed)}")
            print(f"  Non-null values: {df_processed['Return'].notna().sum()}")
            print(f"  Null values: {df_processed['Return'].isna().sum()}")
            if df_processed['Return'].notna().sum() > 0:
                print(f"  Value range: {df_processed['Return'].min():.4f} to {df_processed['Return'].max():.4f}")
        
        # Create target variable with proper handling
        df = df_processed.copy()
        if 'Ticker' in df.columns:
            if debug:
                print("\nMultiple tickers detected, creating targets by ticker")
            df = df.sort_values(['Ticker', 'Date'])
            df['Target'] = df.groupby('Ticker')['Return'].shift(-1).apply(lambda x: 1 if x > 0 else 0)
            df = df.groupby('Ticker').apply(lambda x: x.iloc[:-1]).reset_index(drop=True)
        else:
            if debug:
                print("\nSingle ticker detected, creating targets directly")
            df = df.sort_values('Date')
            df['Target'] = np.where(df['Return'].shift(-1) > 0, 1, 0)
            df = df.iloc[:-1]
        
        # Clean up and validate
        df = df.dropna(subset=['Target', 'Return'])
        
        if df.empty:
            raise ValueError("No data remaining after target creation and cleaning")
            
        if debug:
            print("\nTarget variable created:")
            print(f"Target distribution:\n{df['Target'].value_counts(normalize=True)}")
        
        # Prepare features
        exclude_cols = ['Date', 'Ticker', 'Return', 'Target']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        if not feature_cols:
            raise ValueError("No feature columns found after excluding metadata")
            
        if debug:
            print(f"\nSelected {len(feature_cols)} features:")
            for col in feature_cols:
                print(f"  • {col}")
        
        X = df[feature_cols]
        y = df['Target']
        
        # Create train/test split
        if 'Date' in df.columns:
            dates = pd.to_datetime(df['Date']).sort_values().unique()
            if len(dates) < 2:
                raise ValueError("Need at least 2 dates for time-based split")
                
            split_idx = max(1, min(int(len(dates) * train_split), len(dates) - 1))
            split_date = dates[split_idx]
            
            train_mask = df['Date'] <= split_date
            test_mask = df['Date'] > split_date
            
            X_train = X[train_mask]
            X_test = X[test_mask]
            y_train = y[train_mask]
            y_test = y[test_mask]
            
            if debug:
                print(f"\nTime-based split at {split_date}:")
                print(f"Train period: {df[train_mask]['Date'].min()} to {df[train_mask]['Date'].max()}")
                print(f"Test period:  {df[test_mask]['Date'].min()} to {df[test_mask]['Date'].max()}")
        else:
            split_idx = int(len(df) * train_split)
            if split_idx < 1 or split_idx >= len(df):
                raise ValueError(f"Split ratio {train_split} would result in empty sets")
                
            X_train = X.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
            y_train = y.iloc[:split_idx]
            y_test = y.iloc[split_idx:]
            
            if debug:
                print("\nIndex-based split:")
                print(f"Train size: {len(X_train)}")
                print(f"Test size:  {len(X_test)}")
        
        # Validate split results
        if len(X_train) == 0 or len(X_test) == 0:
            raise ValueError(f"Split resulted in empty sets. Train: {len(X_train)}, Test: {len(X_test)}")
            
        if debug:
            print(f"\nFinal split summary:")
            print(f"Train set: {len(X_train)} samples ({len(X_train)/len(X):.1%})")
            print(f"Test set:  {len(X_test)} samples ({len(X_test)/len(X):.1%})")
            
        return X_train, X_test, y_train, y_test, df
        
    except Exception as e:
        raise ValueError(f"Error during feature preparation: {str(e)}")


    if 'Ticker' in df.columns:
        df = df.sort_values(['Ticker', 'Date'])
        df['Target'] = df.groupby('Ticker')['Return'].shift(-1).apply(lambda x: np.where(x > 0, 1, 0))
        df = df.groupby('Ticker').apply(lambda x: x.iloc[:-1]).reset_index(drop=True)
    else:
        df = df.sort_values('Date')
        df['Target'] = np.where(df['Return'].shift(-1) > 0, 1, 0)
        df = df.iloc[:-1]
    
    # Remove rows with NaN targets
    df = df.dropna(subset=['Target'])
    
    if debug:
        debug_dataframe(df, "After target creation")
        print(f"Target distribution:\n{df['Target'].value_counts()}")
    
    if df.empty:
        raise ValueError("No data remaining after target creation")
    
    # Define feature columns
    exclude_cols = ['Date', 'Ticker', 'Return', 'Target']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    if debug:
        print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
    
    if len(feature_cols) == 0:
        raise ValueError("No feature columns found after excluding target and metadata columns")
    
    # Prepare features and target - wrap the entire process in a try block
    try:
        # Create target variable
        if 'Date' in df_processed.columns:
            df = df_processed.sort_values('Date')
            df['Target'] = np.where(df['Return'].shift(-1) > 0, 1, 0)
            df = df.iloc[:-1]  # Remove last row since we can't calculate its target
        else:
            df = df_processed.copy()
            df['Target'] = np.where(df['Return'].shift(-1) > 0, 1, 0)
            df = df.iloc[:-1]
        
        # Remove rows with NaN targets
        df = df.dropna(subset=['Target'])
        
        if df.empty:
            raise ValueError("No data remaining after target creation")
        
        if debug:
            debug_dataframe(df, "After target creation")
            print(f"Target distribution:\n{df['Target'].value_counts()}")
        
        # Define feature columns
        exclude_cols = ['Date', 'Ticker', 'Return', 'Target']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        if len(feature_cols) == 0:
            raise ValueError("No feature columns found after excluding target and metadata columns")
            
        if debug:
            print(f"\nFeature columns ({len(feature_cols)}):")
            for col in feature_cols:
                print(f"  • {col}")
        
        # Prepare X and y
        X = df[feature_cols]
        y = df['Target']
        
        # Create train/test split
        if 'Date' in df.columns:
            dates = pd.to_datetime(df['Date']).sort_values().unique()
            
            if len(dates) < 2:
                raise ValueError("Not enough unique dates for train/test split. Need at least 2 dates.")
                
            split_idx = max(1, min(int(len(dates) * train_split), len(dates) - 1))
            split_date = dates[split_idx]
            
            if debug:
                print(f"\nSplit details:")
                print(f"Date range: {dates[0]} to {dates[-1]}")
                print(f"Split date: {split_date}")
                print(f"Training on dates <= {split_date}")
                print(f"Testing on dates > {split_date}")
            
            # Create masks and split the data
            train_mask = df['Date'] <= split_date
            test_mask = df['Date'] > split_date
            
            X_train = X[train_mask]
            X_test = X[test_mask]
            y_train = y[train_mask]
            y_test = y[test_mask]
            
        else:
            if debug:
                print("\nNo date column found, using simple index-based split")
            
            split_idx = int(len(df) * train_split)
            if split_idx == 0 or split_idx >= len(df):
                raise ValueError(f"Split ratio {train_split} would result in empty train or test set")
            
            X_train = X.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
            y_train = y.iloc[:split_idx]
            y_test = y.iloc[split_idx:]
        
        # Validate split results
        if len(X_train) == 0 or len(X_test) == 0:
            raise ValueError(f"Split resulted in empty sets. Train: {len(X_train)}, Test: {len(X_test)}")
        
        if debug:
            print(f"\nSplit results:")
            print(f"Train set: {len(X_train)} samples ({len(X_train)/len(X):.1%})")
            print(f"Test set: {len(X_test)} samples ({len(X_test)/len(X):.1%})")
            if 'Date' in df.columns:
                print(f"Train dates: {df[train_mask]['Date'].min()} to {df[train_mask]['Date'].max()}")
                print(f"Test dates: {df[test_mask]['Date'].min()} to {df[test_mask]['Date'].max()}")
        
        return X_train, X_test, y_train, y_test, df
        
    except Exception as e:
        raise ValueError(f"Error during feature preparation: {str(e)}")

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train and evaluate models with error handling"""
    results = {}
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    }
    
    for name, model in models.items():
        try:
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            y_pred = model.predict(X_test)
            
            results[name] = {
                'model': model,
                'train_time': train_time,
                'mse': mean_squared_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
            }
        except Exception as e:
            print(f"Error training {name}: {e}")
            results[name] = None
    
    return results

def optimize_models(X_train, y_train, debug=True):
    """
    Optimized model training with robust error handling and debugging
    """
    if debug:
        print(f"Starting optimize_models with X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"y_train distribution: {pd.Series(y_train).value_counts().to_dict()}")
    
    results = {}
    
    # Check if we have enough data
    if len(X_train) < 10:
        print("Warning: Very small training set, using simple models")
        # Return simple trained models instead of optimized ones
        lr = LogisticRegression(max_iter=1000, random_state=42)
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        
        try:
            lr.fit(X_train, y_train)
            results["Logistic Regression"] = {
                "best_model": lr,
                "best_params": {"C": 1.0, "penalty": "l2"},
                "best_score": 0.5
            }
        except Exception as e:
            print(f"Error fitting simple Logistic Regression: {e}")
        
        try:
            rf.fit(X_train, y_train)
            results["Random Forest"] = {
                "best_model": rf,
                "best_params": {"n_estimators": 50},
                "best_score": 0.5
            }
        except Exception as e:
            print(f"Error fitting simple Random Forest: {e}")
        
        return results
    
    # Time series cross-validation
    try:
        n_splits = min(5, len(X_train) // 20)  # Adjust splits based on data size
        if n_splits < 2:
            n_splits = 2
        tscv = TimeSeriesSplit(n_splits=n_splits)
        if debug:
            print(f"Using TimeSeriesSplit with {n_splits} splits")
    except Exception as e:
        print(f"Error creating TimeSeriesSplit: {e}")
        from sklearn.model_selection import KFold
        tscv = KFold(n_splits=3, shuffle=False)
        if debug:
            print("Falling back to KFold")
    
    # Optimize Logistic Regression
    if debug:
        print("Optimizing Logistic Regression...")
    
    try:
        log_reg = LogisticRegression(max_iter=1000, random_state=42)
        log_params = {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l2"],
            "solver": ["lbfgs"]
        }
        
        log_search = GridSearchCV(
            estimator=log_reg,
            param_grid=log_params,
            cv=tscv,
            scoring="f1",
            n_jobs=1,  # Use single job to avoid issues
            verbose=0
        )
        
        log_search.fit(X_train, y_train)
        
        results["Logistic Regression"] = {
            "best_model": log_search.best_estimator_,
            "best_params": log_search.best_params_,
            "best_score": log_search.best_score_
        }
        
        if debug:
            print(f"Logistic Regression best score: {log_search.best_score_:.4f}")
            print(f"Logistic Regression best params: {log_search.best_params_}")
        
    except Exception as e:
        print(f"Error optimizing Logistic Regression: {e}")
        # Fallback to simple model
        try:
            simple_lr = LogisticRegression(max_iter=1000, random_state=42)
            simple_lr.fit(X_train, y_train)
            results["Logistic Regression"] = {
                "best_model": simple_lr,
                "best_params": {"C": 1.0},
                "best_score": 0.5
            }
            print("Using fallback simple Logistic Regression")
        except Exception as e2:
            print(f"Even simple Logistic Regression failed: {e2}")
    
    # Optimize Random Forest
    if debug:
        print("Optimizing Random Forest...")
    
    try:
        rf = RandomForestClassifier(random_state=42)
        
        # Simplified parameter grid for faster execution
        rf_params = {
            "n_estimators": [50, 100],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        }
        
        rf_search = GridSearchCV(
            estimator=rf,
            param_grid=rf_params,
            cv=tscv,
            scoring="f1",
            n_jobs=1,  # Use single job to avoid issues
            verbose=0
        )
        
        rf_search.fit(X_train, y_train)
        
        results["Random Forest"] = {
            "best_model": rf_search.best_estimator_,
            "best_params": rf_search.best_params_,
            "best_score": rf_search.best_score_
        }
        
        if debug:
            print(f"Random Forest best score: {rf_search.best_score_:.4f}")
            print(f"Random Forest best params: {rf_search.best_params_}")
        
    except Exception as e:
        print(f"Error optimizing Random Forest: {e}")
        # Fallback to simple model
        try:
            simple_rf = RandomForestClassifier(n_estimators=50, random_state=42)
            simple_rf.fit(X_train, y_train)
            results["Random Forest"] = {
                "best_model": simple_rf,
                "best_params": {"n_estimators": 50},
                "best_score": 0.5
            }
            print("Using fallback simple Random Forest")
        except Exception as e2:
            print(f"Even simple Random Forest failed: {e2}")
    
    if debug:
        print(f"optimize_models returning keys: {list(results.keys())}")
    
    return results

def evaluate_best_models(models, X_test, y_test):
    """Evaluate optimized models with error handling"""
    eval_results = {}
    
    for name, info in models.items():
        if info is None:
            print(f"Skipping {name} - no model available")
            continue
            
        try:
            model = info["best_model"]
            y_pred = model.predict(X_test)
            eval_results[name] = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1": f1_score(y_test, y_pred, zero_division=0)
            }
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            eval_results[name] = {
                "Accuracy": 0.0,
                "Precision": 0.0,
                "Recall": 0.0,
                "F1": 0.0
            }
    
    return eval_results

# Safe wrapper functions for your app.py
def safe_optimize_models(X_train, y_train, model_choice):
    """
    Safe wrapper for optimize_models that handles the specific model choice
    """
    try:
        print(f"Starting optimization for {model_choice}...")
        results = optimize_models(X_train, y_train, debug=True)
        
        # Debug: print what was returned
        print(f"Available models after optimization: {list(results.keys())}")
        
        # Check if the requested model is available
        if model_choice not in results:
            print(f"Warning: {model_choice} not found in results!")
            print(f"Available models: {list(results.keys())}")
            
            # Try to find a close match
            if "Logistic Regression" in results and model_choice == "Logistic Regression":
                return results["Logistic Regression"]["best_model"]
            elif "Random Forest" in results and model_choice == "Random Forest":
                return results["Random Forest"]["best_model"]
            else:
                # Return any available model as fallback
                available_models = [k for k, v in results.items() if v is not None]
                if available_models:
                    fallback_model = available_models[0]
                    print(f"Using fallback model: {fallback_model}")
                    return results[fallback_model]["best_model"]
                else:
                    raise ValueError("No models successfully trained")
        
        return results[model_choice]["best_model"]
        
    except Exception as e:
        print(f"Error in safe_optimize_models: {e}")
        # Last resort: return a simple trained model
        print("Creating simple fallback model...")
        if model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=1000, random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=50, random_state=42)
        
        model.fit(X_train, y_train)
        return model

# Example usage and testing
if __name__ == "__main__":
    print("Testing models module...")
    
    # Create sample data for testing
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    
    X_train = pd.DataFrame(np.random.randn(n_samples, n_features), 
                          columns=[f'feature_{i}' for i in range(n_features)])
    y_train = np.random.choice([0, 1], size=n_samples)
    
    X_test = pd.DataFrame(np.random.randn(200, n_features), 
                         columns=[f'feature_{i}' for i in range(n_features)])
    y_test = np.random.choice([0, 1], size=200)
    
    # Test optimize_models
    results = optimize_models(X_train, y_train)
    print("Test completed successfully!")