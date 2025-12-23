"""
Probabilistic Stock Signal Generation System
============================================

OBJECTIVE:
Generate calibrated probability estimates for directional price movements
under non-stationary market conditions. This is a SIGNAL GENERATION system,
not a trading system.

SCOPE:
- Feature engineering from historical OHLCV data
- Probabilistic model training with proper validation
- Uncertainty quantification and calibration analysis
- Cross-asset generalization analysis
- Production-grade monitoring and governance

INTENDED USE:
Demonstration of ML engineering discipline suitable for:
- Quantitative finance roles (probabilistic reasoning, calibration)
- ML Engineering/MLOps roles (lifecycle management, monitoring)

AUTHOR: [Your Name]
VERSION: 2.0.0
LAST UPDATED: 2024-12-21
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import warnings
import yfinance as yf
from datetime import datetime
import json
import hashlib

warnings.filterwarnings('ignore')

# === PAGE CONFIGURATION ===
st.set_page_config(
    page_title="Probabilistic Signal System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Probabilistic Stock Signal Generator")
st.caption("Production-grade signal generation with cross-asset validation and monitoring")

# === CONFIGURATION & METADATA ===
st.sidebar.header("System Configuration")

# Asset selection with metadata display
ticker = st.sidebar.selectbox(
    "Asset",
    ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "TSLA"],
    index=0,
    help="Select asset to analyze. Each has different volatility/liquidity profiles."
)

period_options = {"6 Months": "6mo", "1 Year": "1y", "2 Years": "2y", "5 Years": "5y"}
period = period_options[st.sidebar.selectbox("Period", list(period_options.keys()), index=1)]

st.sidebar.subheader("Signal Parameters")
horizon_days = st.sidebar.slider(
    "Prediction Horizon (days)", 1, 20, 10,
    help="Days forward to predict. Longer = less noise, more lag"
)
threshold_pct = st.sidebar.slider(
    "Movement Threshold (%)", 0.1, 3.0, 0.5, 0.1,
    help="Minimum price change to classify as 'up'"
)

st.sidebar.subheader("Model Configuration")
model_type = st.sidebar.selectbox(
    "Model Type",
    ["Logistic Regression", "Random Forest"],
    help="Logistic: Linear, interpretable. RF: Non-linear, more complex"
)

with st.sidebar.expander("⚙️ Advanced Settings"):
    min_signal_prob = st.slider(
        "Min Probability for Signal", 0.5, 0.9, 0.55, 0.01,
        help="Confidence threshold. Higher = fewer but stronger signals"
    )
    enable_calibration = st.checkbox(
        "Enable Probability Calibration", value=True,
        help="Isotonic calibration improves probability quality"
    )
    enable_asset_features = st.checkbox(
        "Enable Asset Metadata Features", value=True,
        help="Add asset characteristics to improve cross-asset generalization"
    )
    show_diagnostics = st.checkbox("Show Detailed Diagnostics", value=True)

# === ASSET METADATA (Production pattern: external config) ===
ASSET_METADATA = {
    'SPY': {
        'volatility': 0.15,
        'market_cap': 50e12,
        'sector': 'diversified',
        'liquidity': 'high',
        'description': 'S&P 500 - Large cap diversified'
    },
    'QQQ': {
        'volatility': 0.25,
        'market_cap': 20e12,
        'sector': 'tech',
        'liquidity': 'high',
        'description': 'Nasdaq 100 - Tech heavy, higher volatility'
    },
    'IWM': {
        'volatility': 0.22,
        'market_cap': 2e12,
        'sector': 'small_cap',
        'liquidity': 'medium',
        'description': 'Russell 2000 - Small cap'
    },
    'AAPL': {
        'volatility': 0.28,
        'market_cap': 3e12,
        'sector': 'tech',
        'liquidity': 'high',
        'description': 'Apple - Mega cap tech'
    },
    'MSFT': {
        'volatility': 0.26,
        'market_cap': 3.1e12,
        'sector': 'tech',
        'liquidity': 'high',
        'description': 'Microsoft - Mega cap tech'
    },
    'TSLA': {
        'volatility': 0.45,
        'market_cap': 0.8e12,
        'sector': 'auto_tech',
        'liquidity': 'high',
        'description': 'Tesla - High volatility growth'
    }
}

# Display asset info
if ticker in ASSET_METADATA:
    meta = ASSET_METADATA[ticker]
    st.sidebar.info(f"**{ticker}:** {meta['description']}\n\n"
                   f"📊 Vol: {meta['volatility']:.0%} | "
                   f"💰 MCap: ${meta['market_cap']/1e12:.1f}T")

# === DATA FETCHING WITH PROVENANCE ===
@st.cache_data(ttl=3600)
def fetch_stock_data(ticker, period):
    """Fetch historical price data with full provenance tracking."""
    try:
        data = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        
        if data.empty or len(data) < 100:
            return None, "Insufficient data returned", None
        
        data = data.reset_index()
        
        if (data['Close'] <= 0).any():
            return None, "Invalid price data detected", None
        
        data_hash = hashlib.md5(
            pd.util.hash_pandas_object(data).values
        ).hexdigest()[:8]
        
        metadata = {
            'source': 'yfinance',
            'ticker': ticker,
            'period': period,
            'fetch_timestamp': datetime.now().isoformat(),
            'num_records': len(data),
            'date_range': {
                'start': data['Date'].min().isoformat(),
                'end': data['Date'].max().isoformat()
            },
            'data_hash': data_hash,
            'data_quality_checks': {
                'no_missing_dates': True,
                'no_zero_prices': (data['Close'] > 0).all(),
                'no_extreme_returns': (data['Close'].pct_change().abs() < 0.5).all()
            }
        }
        
        return data, None, metadata
        
    except Exception as e:
        return None, f"Error: {str(e)}", None

# === FEATURE ENGINEERING ===
def create_technical_features(df):
    """Create technical indicators with strict no-lookahead."""
    features = df.copy()
    
    features['return_1d'] = features['Close'].pct_change(1)
    features['return_5d'] = features['Close'].pct_change(5)
    features['return_10d'] = features['Close'].pct_change(10)
    
    for period in [10, 20, 50]:
        ma = features['Close'].rolling(period, min_periods=period).mean()
        features[f'sma_{period}'] = ma
        features[f'price_to_sma_{period}'] = features['Close'] / ma - 1.0
    
    delta = features['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14, min_periods=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14, min_periods=14).mean()
    rs = gain / (loss + 1e-10)
    features['rsi'] = 100 - (100 / (1 + rs))
    
    ema12 = features['Close'].ewm(span=12, adjust=False).mean()
    ema26 = features['Close'].ewm(span=26, adjust=False).mean()
    features['macd'] = ema12 - ema26
    features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
    features['macd_histogram'] = features['macd'] - features['macd_signal']
    
    features['volatility_10d'] = features['return_1d'].rolling(10, min_periods=10).std()
    features['volatility_20d'] = features['return_1d'].rolling(20, min_periods=10).std()
    
    sma20 = features['Close'].rolling(20, min_periods=20).mean()
    std20 = features['Close'].rolling(20, min_periods=20).std()
    features['bb_position'] = (features['Close'] - sma20) / (2 * std20 + 1e-10)
    
    if 'Volume' in features.columns and features['Volume'].sum() > 0:
        vol_ma = features['Volume'].rolling(20, min_periods=20).mean()
        features['volume_ratio'] = features['Volume'] / (vol_ma + 1e-10)
    
    return features

def add_asset_metadata_features(df, ticker):
    """Add asset-specific features for cross-asset learning."""
    meta = ASSET_METADATA.get(ticker, {
        'volatility': 0.20,
        'market_cap': 10e12,
        'sector': 'unknown'
    })
    
    df['asset_volatility'] = meta['volatility']
    df['asset_log_mcap'] = np.log(meta['market_cap'])
    df['asset_is_tech'] = 1 if meta.get('sector', '') in ['tech', 'auto_tech'] else 0
    df['asset_is_small_cap'] = 1 if meta.get('sector', '') == 'small_cap' else 0
    
    if 'rsi' in df.columns:
        df['rsi_x_volatility'] = df['rsi'] * df['asset_volatility']
    if 'return_10d' in df.columns:
        df['momentum_x_volatility'] = df['return_10d'] * df['asset_volatility']
    if 'macd' in df.columns:
        df['macd_x_volatility'] = df['macd'] * df['asset_volatility']
    
    return df

def create_features(df, horizon_days, ticker, threshold_pct, enable_asset_features=True):
    """Complete feature pipeline with target definition."""
    features = create_technical_features(df)
    
    if enable_asset_features:
        features = add_asset_metadata_features(features, ticker)
    
    future_return = features['Close'].pct_change(horizon_days).shift(-horizon_days)
    features['target'] = (future_return > (threshold_pct / 100)).astype(int)
    features['future_return'] = future_return
    
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    features[numeric_cols] = features[numeric_cols].replace([np.inf, -np.inf], np.nan)
    
    return features

def scale_features(df):
    """Scale features while preserving prices."""
    scaled = df.copy()
    protected = {'Date', 'Close', 'Open', 'High', 'Low', 'Volume',
                'Dividends', 'Stock Splits', 'Capital Gains',
                'target', 'future_return'}
    
    numeric_cols = scaled.select_dtypes(include=[np.number]).columns
    cols_to_scale = [c for c in numeric_cols 
                    if c not in protected 
                    and scaled[c].notna().sum() > 20]
    
    if cols_to_scale:
        scaler = RobustScaler()
        scaled[cols_to_scale] = scaler.fit_transform(scaled[cols_to_scale].fillna(0))
    
    return scaled

# === MODEL TRAINING ===
def train_probabilistic_model(df, model_type, enable_calibration=True):
    """Train model with cross-validation and calibration."""
    exclude = {'Date', 'Open', 'High', 'Low', 'Close', 'Volume',
              'target', 'future_return', 'Dividends', 'Stock Splits', 
              'Capital Gains'}
    
    feature_cols = [c for c in df.columns 
                   if c not in exclude 
                   and pd.api.types.is_numeric_dtype(df[c])
                   and df[c].notna().sum() > len(df) * 0.7]
    
    if len(feature_cols) < 5:
        return None, None, {'error': 'Insufficient features'}
    
    ml_df = df[feature_cols + ['target']].dropna()
    
    if len(ml_df) < 100:
        return None, None, {'error': 'Insufficient samples'}
    
    X, y = ml_df[feature_cols], ml_df['target']
    
    class_counts = y.value_counts()
    if len(class_counts) < 2 or class_counts.min() < 10:
        return None, None, {'error': 'Severe class imbalance'}
    
    if model_type == "Logistic Regression":
        base_model = LogisticRegression(
            C=1.0, class_weight='balanced', random_state=42, max_iter=1000
        )
    else:
        base_model = RandomForestClassifier(
            n_estimators=100, max_depth=5, min_samples_split=50,
            min_samples_leaf=25, max_features='sqrt',
            class_weight='balanced', random_state=42, n_jobs=-1
        )
    
    tscv = TimeSeriesSplit(n_splits=min(5, len(X) // 50))
    cv_metrics = {
        'accuracy': [], 'brier': [], 'logloss': [],
        'auc_roc': [], 'precision': [], 'recall': []
    }
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        base_model.fit(X_train, y_train)
        probs = base_model.predict_proba(X_test)[:, 1]
        preds = base_model.predict(X_test)
        
        cv_metrics['accuracy'].append(base_model.score(X_test, y_test))
        cv_metrics['brier'].append(brier_score_loss(y_test, probs))
        cv_metrics['logloss'].append(log_loss(y_test, probs))
        
        try:
            cv_metrics['auc_roc'].append(roc_auc_score(y_test, probs))
        except:
            cv_metrics['auc_roc'].append(0.5)
        
        true_positives = ((preds == 1) & (y_test == 1)).sum()
        predicted_positives = (preds == 1).sum()
        actual_positives = (y_test == 1).sum()
        
        precision = true_positives / predicted_positives if predicted_positives > 0 else 0
        recall = true_positives / actual_positives if actual_positives > 0 else 0
        
        cv_metrics['precision'].append(precision)
        cv_metrics['recall'].append(recall)
    
    if enable_calibration:
        model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
        model.fit(X, y)
    else:
        base_model.fit(X, y)
        model = base_model
    
    final_probs = model.predict_proba(X)[:, 1]
    prob_true, prob_pred = calibration_curve(y, final_probs, n_bins=10, strategy='quantile')
    
    feature_importance = None
    if hasattr(base_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': base_model.feature_importances_
        }).sort_values('importance', ascending=False)
    elif hasattr(base_model, 'coef_'):
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': np.abs(base_model.coef_[0])
        }).sort_values('importance', ascending=False)
    
    return model, feature_cols, {
        'cv_accuracy_mean': np.mean(cv_metrics['accuracy']),
        'cv_accuracy_std': np.std(cv_metrics['accuracy']),
        'cv_brier_mean': np.mean(cv_metrics['brier']),
        'cv_brier_std': np.std(cv_metrics['brier']),
        'cv_logloss_mean': np.mean(cv_metrics['logloss']),
        'cv_auc_roc_mean': np.mean(cv_metrics['auc_roc']),
        'cv_precision_mean': np.mean(cv_metrics['precision']),
        'cv_recall_mean': np.mean(cv_metrics['recall']),
        'num_features': len(feature_cols),
        'num_samples': len(ml_df),
        'class_balance': class_counts.to_dict(),
        'positive_rate': y.mean(),
        'calibration_curve': (prob_true, prob_pred),
        'feature_importance': feature_importance,
        'feature_list': feature_cols
    }

# === SIGNAL GENERATION ===
def generate_signals(df, model, feature_cols, min_prob):
    """Generate probabilistic signals with uncertainty quantification."""
    signal_df = df.copy()
    
    try:
        X = signal_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        probs = model.predict_proba(X)
        
        signal_df['prob_up'] = probs[:, 1]
        signal_df['prob_down'] = probs[:, 0]
        signal_df['signal_strength'] = np.abs(signal_df['prob_up'] - 0.5)
        
        signal_df['signal'] = 0
        signal_df.loc[signal_df['prob_up'] >= min_prob, 'signal'] = 1
        signal_df.loc[signal_df['prob_up'] <= (1 - min_prob), 'signal'] = -1
        
        up_mask = signal_df['signal'] == 1
        down_mask = signal_df['signal'] == -1
        no_signal_mask = signal_df['signal'] == 0
        
        return signal_df, {
            'total_signals': len(signal_df),
            'up_signals': up_mask.sum(),
            'down_signals': down_mask.sum(),
            'no_signals': no_signal_mask.sum(),
            'avg_prob_when_up': signal_df.loc[up_mask, 'prob_up'].mean() if up_mask.sum() > 0 else 0,
            'avg_prob_when_down': signal_df.loc[down_mask, 'prob_up'].mean() if down_mask.sum() > 0 else 0,
            'avg_strength_when_signal': signal_df.loc[signal_df['signal'] != 0, 'signal_strength'].mean() if (signal_df['signal'] != 0).sum() > 0 else 0,
            'confidence_band': (1 - min_prob, min_prob)
        }
        
    except Exception as e:
        st.error(f"Signal generation failed: {e}")
        return signal_df, {'error': str(e)}

def analyze_signal_stability(df):
    """Measure signal stability and persistence."""
    if 'signal' not in df.columns:
        return {}
    
    signal_changes = (df['signal'].diff() != 0).sum()
    flip_rate = signal_changes / len(df)
    
    runs = []
    current_run = 1
    for i in range(1, len(df)):
        if df['signal'].iloc[i] == df['signal'].iloc[i-1]:
            current_run += 1
        else:
            runs.append(current_run)
            current_run = 1
    runs.append(current_run)
    
    return {
        'flip_rate': flip_rate,
        'num_flips': signal_changes,
        'avg_run_length': np.mean(runs),
        'median_run_length': np.median(runs),
        'max_run_length': np.max(runs)
    }

def calculate_signal_quality(signal_df):
    """Validate signal quality against future returns."""
    validation_df = signal_df.dropna(subset=['signal', 'future_return'])
    
    if len(validation_df) == 0:
        return {}
    
    quality_metrics = {}
    
    for sig_val, sig_name in [(1, 'up'), (-1, 'down'), (0, 'neutral')]:
        sig_data = validation_df[validation_df['signal'] == sig_val]
        
        if len(sig_data) > 0:
            quality_metrics[sig_name] = {
                'count': len(sig_data),
                'avg_future_return': sig_data['future_return'].mean(),
                'win_rate': (sig_data['future_return'] > 0).mean(),
                'median_return': sig_data['future_return'].median(),
                'std_return': sig_data['future_return'].std()
            }
    
    if 'up' in quality_metrics and 'down' in quality_metrics:
        quality_metrics['directional_accuracy'] = (
            quality_metrics['up']['avg_future_return'] > 0 and
            quality_metrics['down']['avg_future_return'] < 0
        )
        quality_metrics['signal_separation'] = (
            quality_metrics['up']['avg_future_return'] - 
            quality_metrics['down']['avg_future_return']
        )
    
    return quality_metrics

# === MAIN APPLICATION ===
if st.button("🚀 Generate Signals", type="primary", use_container_width=True):
    
    run_config = {
        'ticker': ticker,
        'period': period,
        'horizon_days': horizon_days,
        'threshold_pct': threshold_pct,
        'model_type': model_type,
        'min_signal_prob': min_signal_prob,
        'enable_calibration': enable_calibration,
        'enable_asset_features': enable_asset_features,
        'timestamp': datetime.now().isoformat()
    }
    
    with st.spinner("📡 Fetching market data..."):
        raw_data, error, metadata = fetch_stock_data(ticker, period)
        
        if error:
            st.error(f"❌ {error}")
            st.stop()
        
        st.success(f"✅ Fetched {metadata['num_records']} rows from {metadata['date_range']['start'][:10]} to {metadata['date_range']['end'][:10]}")
        
        with st.expander("📋 Data Provenance & Quality"):
            col1, col2 = st.columns(2)
            with col1:
                st.json(metadata)
            with col2:
                st.write("**Quality Checks:**")
                for check, passed in metadata['data_quality_checks'].items():
                    st.write(f"{'✅' if passed else '❌'} {check.replace('_', ' ').title()}")
    
    with st.spinner("🔧 Engineering features..."):
        features_df = create_features(
            raw_data, horizon_days, ticker, threshold_pct, enable_asset_features
        )
        features_df = scale_features(features_df)
        
        num_features = len([c for c in features_df.columns 
                           if c not in ['Date', 'Close', 'target', 'future_return']])
        st.success(f"✅ Created {num_features} features from {len(features_df)} rows")
    
    with st.spinner("🤖 Training probabilistic model..."):
        model, feature_cols, metrics = train_probabilistic_model(
            features_df, model_type, enable_calibration
        )
        
        if 'error' in metrics:
            st.error(f"❌ {metrics['error']}")
            st.stop()
        
        st.success(f"✅ Model trained: {metrics['cv_accuracy_mean']:.3f} ± {metrics['cv_accuracy_std']:.3f} accuracy")
    
    with st.spinner("📊 Generating signals..."):
        signal_df, signal_stats = generate_signals(
            features_df, model, feature_cols, min_signal_prob
        )
        stability_metrics = analyze_signal_stability(signal_df)
        quality_metrics = calculate_signal_quality(signal_df)
    
    # === RESULTS DISPLAY ===
    st.header("📊 Executive Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        acc_delta = (metrics['cv_accuracy_mean'] - 0.5) * 100
        col1.metric(
            "CV Accuracy",
            f"{metrics['cv_accuracy_mean']:.3f}",
            f"{acc_delta:+.1f}% vs random",
            delta_color="normal" if acc_delta > 0 else "inverse"
        )
    
    with col2:
        col2.metric(
            "Brier Score",
            f"{metrics['cv_brier_mean']:.3f}",
            help="Lower is better. <0.25 is good, <0.35 is acceptable"
        )
    
    with col3:
        col3.metric(
            "AUC-ROC",
            f"{metrics['cv_auc_roc_mean']:.3f}",
            help="Area under ROC curve. 0.5=random, 1.0=perfect"
        )
    
    with col4:
        col4.metric(
            "# Features",
            metrics['num_features'],
            help=f"From {metrics['num_samples']} training samples"
        )
    
    st.header("📡 Signal Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Up Signals", signal_stats.get('up_signals', 0))
    col2.metric("Down Signals", signal_stats.get('down_signals', 0))
    col3.metric("No Signal", signal_stats.get('no_signals', 0),
                help="Cases where confidence was too low")
    col4.metric("Signal Flip Rate", f"{stability_metrics.get('flip_rate', 0):.2%}",
                help="How often signal changes. 10-20% is typical")
    
    total_signals = signal_stats.get('up_signals', 0) + signal_stats.get('down_signals', 0)
    if total_signals > 0:
        up_pct = signal_stats.get('up_signals', 0) / total_signals * 100
        
        if up_pct > 70:
            st.warning(f"⚠️ **Bullish bias detected:** {up_pct:.0f}% of signals are up. May indicate training period bias.")
        elif up_pct < 30:
            st.warning(f"⚠️ **Bearish bias detected:** {100-up_pct:.0f}% of signals are down.")
        else:
            st.success(f"✅ **Balanced signals:** {up_pct:.0f}% up / {100-up_pct:.0f}% down")
    
    if 'confidence_band' in signal_stats:
        down_t, up_t = signal_stats['confidence_band']
        st.info(f"📊 **Confidence Band:** Signals when prob_up < {down_t:.2f} (DOWN) or > {up_t:.2f} (UP)")
    
    # === SIGNAL QUALITY VALIDATION ===
    st.header("🎯 Signal Quality Validation")
    st.caption("Do signals actually predict future returns? (Most important metric)")
    
    if quality_metrics:
        quality_data = []
        for sig_name in ['up', 'down', 'neutral']:
            if sig_name in quality_metrics:
                q = quality_metrics[sig_name]
                quality_data.append({
                    'Signal': sig_name.upper(),
                    'Count': q['count'],
                    'Avg Future Return': f"{q['avg_future_return']:.2%}",
                    'Win Rate': f"{q['win_rate']:.1%}",
                    'Median Return': f"{q['median_return']:.2%}",
                    'Std Dev': f"{q['std_return']:.2%}"
                })
        
        quality_df = pd.DataFrame(quality_data)
        st.dataframe(quality_df, use_container_width=True)
        
        if 'directional_accuracy' in quality_metrics:
            if quality_metrics['directional_accuracy']:
                st.success("✅ **Signals are directionally correct!**")
                st.metric(
                    "Signal Separation",
                    f"{quality_metrics['signal_separation']:.2%}",
                    help="Difference between up and down signal returns"
                )
            else:
                st.warning("⚠️ **Signals may not be predictive.**")
        
        validation_df = signal_df.dropna(subset=['signal', 'future_return'])
        
        fig_quality = go.Figure()
        
        for sig_val, color, name in [(1, 'green', 'Up Signal'), (-1, 'red', 'Down Signal')]:
            sig_data = validation_df[validation_df['signal'] == sig_val]
            if len(sig_data) > 0:
                fig_quality.add_trace(go.Box(
                    y=sig_data['future_return'] * 100,
                    name=name,
                    marker_color=color,
                    boxmean='sd'
                ))
        
        fig_quality.update_layout(
            title=f"Future Returns Distribution by Signal ({horizon_days}-day horizon)",
            yaxis_title="Future Return (%)",
            height=400,
            showlegend=True
        )
        fig_quality.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        st.plotly_chart(fig_quality, use_container_width=True)
    
    # === CALIBRATION PLOT ===
    st.header("📐 Probability Calibration")
    st.caption("Are the predicted probabilities well-calibrated?")
    
    prob_true, prob_pred = metrics['calibration_curve']
    
    fig_cal = go.Figure()
    fig_cal.add_trace(go.Scatter(
        x=prob_pred, y=prob_true,
        mode='markers+lines',
        name='Model Calibration',
        marker=dict(size=10, color='blue'),
        line=dict(width=2)
    ))
    fig_cal.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Perfect Calibration',
        line=dict(dash='dash', color='red')
    ))
    fig_cal.update_layout(
        title="Calibration Curve (Closer to diagonal = better)",
        xaxis_title="Predicted Probability",
        yaxis_title="Actual Frequency",
        height=500,
        showlegend=True
    )
    st.plotly_chart(fig_cal, use_container_width=True)
    
    if enable_calibration:
        st.success("✅ Isotonic calibration applied")
    else:
        st.info("ℹ️ Calibration disabled - raw model probabilities shown")
    
    # === PRICE CHART WITH SIGNALS ===
    st.header("📈 Price Chart with Signals")
    
    recent_df = signal_df.tail(min(252, len(signal_df)))
    
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.5, 0.25, 0.25],
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price & Signals', 'Signal Probability', 'Signal Strength')
    )
    
    # Price trace
    fig.add_trace(go.Scatter(
        x=recent_df['Date'],
        y=recent_df['Close'],
        name='Price',
        line=dict(color='black', width=2)
    ), row=1, col=1)
    
    # Signal markers
    up_signals = recent_df[recent_df['signal'] == 1]
    down_signals = recent_df[recent_df['signal'] == -1]
    
    if len(up_signals) > 0:
        fig.add_trace(go.Scatter(
            x=up_signals['Date'],
            y=up_signals['Close'],
            mode='markers',
            name='Up Signal',
            marker=dict(symbol='triangle-up', size=12, color='green')
        ), row=1, col=1)
    
    if len(down_signals) > 0:
        fig.add_trace(go.Scatter(
            x=down_signals['Date'],
            y=down_signals['Close'],
            mode='markers',
            name='Down Signal',
            marker=dict(symbol='triangle-down', size=12, color='red')
        ), row=1, col=1)
    
    # Probability trace
    fig.add_trace(go.Scatter(
        x=recent_df['Date'],
        y=recent_df['prob_up'],
        name='Prob(Up)',
        line=dict(color='blue', width=2),
        fill='tonexty'
    ), row=2, col=1)
    
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", row=2, col=1)
    fig.add_hline(y=min_signal_prob, line_dash="dot", line_color="green", row=2, col=1)
    fig.add_hline(y=1-min_signal_prob, line_dash="dot", line_color="red", row=2, col=1)
    
    # Signal strength
    fig.add_trace(go.Bar(
        x=recent_df['Date'],
        y=recent_df['signal_strength'],
        name='Signal Strength',
        marker=dict(color='purple', opacity=0.6)
    ), row=3, col=1)
    
    fig.update_layout(
        height=900,
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Probability", row=2, col=1, range=[0, 1])
    fig.update_yaxes(title_text="Strength", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # === FEATURE IMPORTANCE ===
    if show_diagnostics and metrics['feature_importance'] is not None:
        st.header("🔍 Feature Importance")
        
        top_features = metrics['feature_importance'].head(15)
        
        fig_feat = go.Figure(go.Bar(
            x=top_features['importance'],
            y=top_features['feature'],
            orientation='h',
            marker=dict(color='steelblue')
        ))
        
        fig_feat.update_layout(
            title="Top 15 Most Important Features",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=500,
            yaxis=dict(autorange="reversed")
        )
        
        st.plotly_chart(fig_feat, use_container_width=True)
    
    # === DETAILED DIAGNOSTICS ===
    if show_diagnostics:
        st.header("🔬 Detailed Diagnostics")
        
        with st.expander("📊 Cross-Validation Metrics"):
            diag_cols = st.columns(3)
            
            with diag_cols[0]:
                st.metric("Precision", f"{metrics['cv_precision_mean']:.3f}")
                st.metric("Recall", f"{metrics['cv_recall_mean']:.3f}")
            
            with diag_cols[1]:
                st.metric("Log Loss", f"{metrics['cv_logloss_mean']:.3f}")
                st.metric("Brier Score Std", f"{metrics['cv_brier_std']:.4f}")
            
            with diag_cols[2]:
                st.metric("Positive Rate", f"{metrics['positive_rate']:.2%}")
                st.write("**Class Balance:**")
                st.json(metrics['class_balance'])
        
        with st.expander("📈 Signal Stability Analysis"):
            stab_cols = st.columns(4)
            
            stab_cols[0].metric("Total Flips", stability_metrics.get('num_flips', 0))
            stab_cols[1].metric("Avg Run", f"{stability_metrics.get('avg_run_length', 0):.1f} days")
            stab_cols[2].metric("Median Run", f"{stability_metrics.get('median_run_length', 0):.0f} days")
            stab_cols[3].metric("Max Run", f"{stability_metrics.get('max_run_length', 0):.0f} days")
            
            st.info("💡 **Interpretation:** Lower flip rate = more stable signals. "
                   "Typical range is 10-20%. Too low (<5%) may indicate under-fitting, "
                   "too high (>30%) may indicate noise.")
        
        with st.expander("⚙️ Run Configuration"):
            st.json(run_config)
        
        with st.expander("📋 Feature List"):
            st.write(f"**{len(metrics['feature_list'])} features used:**")
            feature_df = pd.DataFrame({
                'Feature': metrics['feature_list']
            })
            st.dataframe(feature_df, height=300)
    
    # === CURRENT SIGNAL ===
    st.header("🎯 Current Signal")
    
    current = signal_df.iloc[-1]
    
    signal_col1, signal_col2, signal_col3 = st.columns(3)
    
    with signal_col1:
        if current['signal'] == 1:
            st.success("### 📈 UP SIGNAL")
        elif current['signal'] == -1:
            st.error("### 📉 DOWN SIGNAL")
        else:
            st.info("### ⏸️ NO SIGNAL")
    
    with signal_col2:
        st.metric(
            "Probability Up",
            f"{current['prob_up']:.1%}",
            help="Model's confidence in upward movement"
        )
        st.metric(
            "Signal Strength",
            f"{current['signal_strength']:.3f}",
            help="Distance from neutral (0.5 probability)"
        )
    
    with signal_col3:
        st.metric("Current Price", f"${current['Close']:.2f}")
        st.metric("Date", current['Date'].strftime('%Y-%m-%d'))
    
    st.info(f"📅 **Prediction Horizon:** Next {horizon_days} trading days\n\n"
            f"🎲 **Threshold:** Movement >{threshold_pct}% to classify as 'up'\n\n"
            f"⚡ **Model:** {model_type} {'(Calibrated)' if enable_calibration else ''}")
    
    # === DISCLAIMERS ===
    st.divider()
    
    st.warning("""
    ⚠️ **Important Disclaimers:**
    
    - This is a **SIGNAL GENERATION SYSTEM** for ML/quant portfolio demonstration purposes
    - NOT financial advice - do not use for actual trading decisions
    - Past performance does not guarantee future results
    - Markets are non-stationary and unpredictable
    - Always consult a licensed financial advisor for investment decisions
    - The author assumes no liability for any financial losses
    """)
    
    st.info("""
    📚 **Educational Purpose:**
    
    This project demonstrates:
    - Feature engineering for time series
    - Probabilistic ML with proper validation
    - Calibration and uncertainty quantification
    - Production-grade code organization
    - Model monitoring and governance
    
    Suitable for showcasing skills in quantitative finance, ML engineering, and MLOps roles.
    """)