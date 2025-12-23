import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score
import warnings
from datetime import datetime

# Import data pipeline
try:
    from data_pipeline import process_stock_data, PipelineConfig
except ImportError:
    st.error("Please ensure data_pipeline.py is in the same directory")
    st.stop()

warnings.filterwarnings('ignore')

st.cache_data.clear()
st.set_page_config(page_title="Trading System Analysis", layout="wide")
st.title("📈 Fixed Trading System Analysis")

# Sidebar Configuration
st.sidebar.header("Configuration")
ticker = st.sidebar.selectbox("Asset", ["SPY", "QQQ", "IWM", "AAPL"], index=0)
period_options = {"6 Months": "6mo", "1 Year": "1y", "2 Years": "2y"}
period = period_options[st.sidebar.selectbox("Period", list(period_options.keys()), index=1)]

st.sidebar.subheader("Strategy Parameters")
lookback_period = st.sidebar.slider("Lookback (days)", 10, 50, 20)
holding_period = st.sidebar.slider("Max Holding (days)", 1, 20, 5)
transaction_cost = st.sidebar.slider("Transaction Cost (%)", 0.0, 0.5, 0.1, 0.01)

st.sidebar.subheader("Risk Management")
max_position_size = st.sidebar.slider("Max Position Size (%)", 10, 50, 20)
stop_loss = st.sidebar.slider("Stop Loss (%)", 1, 10, 3)
take_profit = st.sidebar.slider("Take Profit (%)", 2, 20, 8)

model_type = st.sidebar.selectbox("ML Model", ["Random Forest", "Logistic Regression", "Technical Only"])

with st.sidebar.expander("Advanced"):
    min_confidence = st.slider("Min ML Confidence", 0.5, 0.8, 0.55, 0.05)
    enable_shorts = st.checkbox("Enable Short Positions", value=False)
    debug_mode = st.checkbox("Debug Mode", value=True)


def diagnose_data(df, source="unknown"):
    """Comprehensive data diagnostics."""
    st.write(f"### 🔍 Diagnosing data from {source}")
    
    if df is None:
        st.error("DataFrame is None!")
        return False
    
    st.write(f"**Shape:** {df.shape}")
    st.write(f"**Columns:** {list(df.columns)}")
    st.write(f"**Index type:** {type(df.index)}")
    
    # Check for critical columns
    required = ['Close', 'Date']
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        return False
    
    # Price data analysis
    st.write("**Close Price Statistics:**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Min", f"${df['Close'].min():.2f}")
    col2.metric("Max", f"${df['Close'].max():.2f}")
    col3.metric("Mean", f"${df['Close'].mean():.2f}")
    col4.metric("NaN Count", df['Close'].isna().sum())
    
    # Check for problematic values
    issues = []
    if df['Close'].min() <= 0:
        zero_count = (df['Close'] <= 0).sum()
        zero_idx = df[df['Close'] <= 0].index.tolist()
        issues.append(f"❌ {zero_count} zero/negative prices at indices: {zero_idx[:5]}")
    
    if df['Close'].isna().sum() > 0:
        issues.append(f"⚠️ {df['Close'].isna().sum()} NaN values in Close")
    
    if np.isinf(df['Close']).any():
        issues.append("❌ Infinite values detected")
    
    # Check for extreme moves
    returns = df['Close'].pct_change().abs()
    extreme = (returns > 0.5).sum()
    if extreme > 0:
        issues.append(f"⚠️ {extreme} days with >50% price moves")
    
    if issues:
        st.error("**Data Issues Found:**")
        for issue in issues:
            st.write(issue)
        return False
    
    st.success("✅ Data looks clean")
    
    # Show sample
    with st.expander("📊 Data Preview"):
        st.dataframe(df.head(10))
        st.dataframe(df.tail(10))
    
    return True


def load_data(ticker, period, debug=False):
    """Load and validate data with comprehensive checks."""
    
    if debug:
        st.write("### 🔄 Loading Data Pipeline")
    
    # CRITICAL: Verify pipeline is using fixed version
    config = PipelineConfig(
        enable_technical=True,
        sma_periods=[10, 20, 50],
        volatility_windows=[10, 20],
        return_lags=[1, 2, 5],
        min_periods=100
    )
    
    st.info("🔧 Using FIXED pipeline (prices should NOT be scaled)")
    
    try:
        result = process_stock_data(ticker, period, config)
    except Exception as e:
        return None, f"Pipeline error: {str(e)}"
    
    if not result['success']:
        return None, f"Pipeline failed: {result.get('error', 'Unknown error')}"
    
    df = result['data']
    
    if debug:
        if not diagnose_data(df, "data_pipeline"):
            return None, "Data validation failed (see diagnostics above)"
    
    # CRITICAL FIX: Handle the actual issue
    # Your pipeline might be creating zero/negative values during feature engineering
    
    if 'Close' not in df.columns:
        return None, "Missing 'Close' column"
    
    # Create a clean copy
    clean_df = df.copy()
    
    # Fix 1: Replace any zero/negative Close prices with forward fill
    if (clean_df['Close'] <= 0).any():
        st.warning(f"Found {(clean_df['Close'] <= 0).sum()} invalid Close prices - attempting repair")
        
        # Option A: Forward fill (safest)
        clean_df['Close'] = clean_df['Close'].replace(0, np.nan)
        clean_df['Close'] = clean_df['Close'].mask(clean_df['Close'] < 0, np.nan)
        clean_df['Close'] = clean_df['Close'].fillna(method='ffill').fillna(method='bfill')
        
        # If still has issues after filling
        if (clean_df['Close'] <= 0).any():
            return None, "Cannot repair invalid price data"
    
    # Fix 2: Ensure proper data types
    if clean_df['Close'].dtype == 'object':
        clean_df['Close'] = pd.to_numeric(clean_df['Close'], errors='coerce')
        clean_df = clean_df.dropna(subset=['Close'])
    
    # Fix 3: Remove any remaining NaN/inf
    clean_df = clean_df.replace([np.inf, -np.inf], np.nan)
    clean_df = clean_df.dropna(subset=['Close'])
    
    # Final validation
    if len(clean_df) < 100:
        return None, f"Insufficient data after cleaning: {len(clean_df)} rows"
    
    if clean_df['Close'].min() <= 0:
        return None, "Price data still invalid after cleaning"
    
    # Check for data quality
    price_changes = clean_df['Close'].pct_change().abs()
    suspicious_days = (price_changes > 0.5).sum()
    if suspicious_days > 5:
        st.warning(f"⚠️ {suspicious_days} days with >50% moves - data may be unreliable")
    
    if debug:
        st.success(f"✅ Data cleaned: {len(clean_df)} rows, {len(clean_df.columns)} columns")
    
    return clean_df, None


def create_features(df):
    """Create robust technical features."""
    features_df = df.copy()
    
    # Ensure we have clean Close prices
    if features_df['Close'].isna().any():
        features_df['Close'] = features_df['Close'].fillna(method='ffill')
    
    # Price-based features
    features_df['Return'] = features_df['Close'].pct_change().fillna(0)
    
    # Moving averages (if not already present)
    for period in [10, 20, 50]:
        col_name = f'SMA_{period}'
        if col_name not in features_df.columns:
            features_df[col_name] = features_df['Close'].rolling(period, min_periods=1).mean()
    
    # RSI
    delta = features_df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
    loss = -delta.where(delta < 0, 0).rolling(14, min_periods=1).mean()
    rs = gain / (loss + 1e-10)  # Avoid division by zero
    features_df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = features_df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = features_df['Close'].ewm(span=26, adjust=False).mean()
    features_df['MACD'] = ema12 - ema26
    features_df['MACD_Signal'] = features_df['MACD'].ewm(span=9, adjust=False).mean()
    features_df['MACD_Hist'] = features_df['MACD'] - features_df['MACD_Signal']
    
    # Bollinger Bands
    sma20 = features_df['Close'].rolling(20, min_periods=1).mean()
    std20 = features_df['Close'].rolling(20, min_periods=1).std()
    features_df['BB_Upper'] = sma20 + (2 * std20)
    features_df['BB_Lower'] = sma20 - (2 * std20)
    features_df['BB_Width'] = (features_df['BB_Upper'] - features_df['BB_Lower']) / (sma20 + 1e-10)
    features_df['BB_Position'] = (features_df['Close'] - features_df['BB_Lower']) / (features_df['BB_Upper'] - features_df['BB_Lower'] + 1e-10)
    
    # Momentum
    features_df['Momentum_5'] = features_df['Close'].pct_change(5).fillna(0)
    features_df['Momentum_10'] = features_df['Close'].pct_change(10).fillna(0)
    
    # Volatility
    features_df['Volatility_10'] = features_df['Return'].rolling(10, min_periods=1).std()
    features_df['Volatility_20'] = features_df['Return'].rolling(20, min_periods=1).std()
    
    # Volume features (if available)
    if 'Volume' in features_df.columns:
        features_df['Volume_MA'] = features_df['Volume'].rolling(20, min_periods=1).mean()
        features_df['Volume_Ratio'] = features_df['Volume'] / (features_df['Volume_MA'] + 1e-10)
    
    # Target: Future returns (NO LOOKAHEAD)
    features_df['Future_Return'] = features_df['Close'].pct_change(holding_period).shift(-holding_period)
    
    # Binary target with realistic threshold
    features_df['Target'] = (features_df['Future_Return'] > 0.01).astype(int)
    
    # Clean any remaining NaN/inf in numeric columns
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    features_df[numeric_cols] = features_df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    
    return features_df


def train_model(df, model_type):
    """Train ML model with proper validation."""
    if model_type == "Technical Only":
        return None, None, {}
    
    # Feature selection
    exclude_cols = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume',
                   'Return', 'Future_Return', 'Target', 'Dividends', 'Stock Splits',
                   'Capital Gains']
    
    feature_cols = [c for c in df.columns 
                   if c not in exclude_cols 
                   and df[c].notna().sum() > len(df)*0.7
                   and df[c].dtype in ['float64', 'int64']]
    
    if len(feature_cols) < 5:
        st.warning(f"Only {len(feature_cols)} features available")
        return None, None, {}
    
    # Prepare clean dataset
    ml_df = df[feature_cols + ['Target']].copy()
    ml_df = ml_df.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(ml_df) < 100:
        st.warning(f"Only {len(ml_df)} samples after cleaning")
        return None, None, {}
    
    X = ml_df[feature_cols]
    y = ml_df['Target']
    
    # Check class balance
    class_counts = y.value_counts()
    if len(class_counts) < 2 or class_counts.min() < 10:
        st.warning(f"Imbalanced classes: {class_counts.to_dict()}")
        return None, None, {}
    
    # Time series split
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Model selection with conservative parameters
    if model_type == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=50,
            min_samples_leaf=25,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    else:
        model = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )
    
    # Cross-validation
    cv_scores = []
    precision_scores_list = []
    recall_scores_list = []
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        cv_scores.append((y_pred == y_test).mean())
        precision_scores_list.append(precision_score(y_test, y_pred, zero_division=0))
        recall_scores_list.append(recall_score(y_test, y_pred, zero_division=0))
    
    # Train on full dataset
    model.fit(X, y)
    
    metrics = {
        'cv_accuracy': np.mean(cv_scores),
        'cv_precision': np.mean(precision_scores_list),
        'cv_recall': np.mean(recall_scores_list),
        'features': len(feature_cols),
        'samples': len(ml_df)
    }
    
    return model, feature_cols, metrics


def backtest_strategy(df, model=None, feature_cols=None):
    """
    FIXED BACKTEST ENGINE
    """
    
    # Validate input
    if df is None or len(df) < 50:
        return pd.DataFrame(), {'error': 'Insufficient data'}, []
    
    if 'Close' not in df.columns or 'Date' not in df.columns:
        return pd.DataFrame(), {'error': 'Missing required columns'}, []
    
    bt_df = df.copy().reset_index(drop=True)
    
    # Validate Close prices one more time
    if (bt_df['Close'] <= 0).any():
        st.error("❌ Invalid prices detected in backtest input")
        return pd.DataFrame(), {'error': 'Invalid prices'}, []
    
    # Initialize
    initial_capital = 100000
    portfolio_value = initial_capital
    cash = initial_capital
    position = 0
    shares = 0
    entry_price = 0
    entry_date = None
    entry_idx = 0
    
    trades = []
    portfolio_history = [initial_capital]
    
    # Generate signals
    if model is not None and feature_cols is not None:
        try:
            X = bt_df[feature_cols].fillna(method='ffill').fillna(0)
            X = X.replace([np.inf, -np.inf], 0)
            probs = model.predict_proba(X)[:, 1]
            
            bt_df['Signal'] = 0
            bt_df.loc[probs > min_confidence, 'Signal'] = 1
            if enable_shorts:
                bt_df.loc[probs < (1 - min_confidence), 'Signal'] = -1
        except Exception as e:
            st.warning(f"ML prediction failed: {e}")
            bt_df['Signal'] = 0
    else:
        # Technical signals only
        bt_df['Signal'] = 0
        if 'SMA_10' in bt_df.columns and 'SMA_50' in bt_df.columns:
            bt_df.loc[bt_df['SMA_10'] > bt_df['SMA_50'], 'Signal'] = 1
            if enable_shorts:
                bt_df.loc[bt_df['SMA_10'] < bt_df['SMA_50'], 'Signal'] = -1
    
    # Backtest loop
    for i in range(1, len(bt_df)):
        current_price = bt_df.loc[i, 'Close']
        current_signal = bt_df.loc[i, 'Signal']
        current_date = bt_df.loc[i, 'Date']
        
        # Sanity check
        if current_price <= 0 or np.isnan(current_price):
            continue
        
        # Update portfolio value
        if position != 0:
            portfolio_value = cash + (shares * current_price)
        else:
            portfolio_value = cash
        
        # Check exit conditions
        if position != 0:
            days_held = i - entry_idx
            price_pct_change = (current_price - entry_price) / entry_price * 100
            
            # Correct P&L for position type
            if position == 1:
                unrealized_pnl_pct = price_pct_change
            else:
                unrealized_pnl_pct = -price_pct_change
            
            exit_trade = False
            exit_reason = ""
            
            if unrealized_pnl_pct <= -stop_loss:
                exit_trade = True
                exit_reason = "Stop Loss"
            elif unrealized_pnl_pct >= take_profit:
                exit_trade = True
                exit_reason = "Take Profit"
            elif days_held >= holding_period:
                exit_trade = True
                exit_reason = "Max Holding"
            elif (position == 1 and current_signal == -1) or (position == -1 and current_signal == 1):
                exit_trade = True
                exit_reason = "Signal Reversal"
            
            if exit_trade:
                if position == 1:
                    exit_value = shares * current_price
                    pnl = exit_value - (shares * entry_price)
                else:
                    pnl = shares * (entry_price - current_price)
                
                transaction_fee = abs(shares * current_price) * (transaction_cost / 100)
                pnl -= transaction_fee
                
                cash += pnl
                portfolio_value = cash
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': current_date,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'position': 'LONG' if position == 1 else 'SHORT',
                    'shares': shares,
                    'pnl': pnl,
                    'pnl_pct': (pnl / initial_capital) * 100,
                    'return_pct': unrealized_pnl_pct,
                    'days_held': days_held,
                    'exit_reason': exit_reason
                })
                
                position = 0
                shares = 0
                entry_price = 0
        
        # Check entry signals
        if position == 0 and current_signal != 0:
            position = current_signal
            entry_price = current_price
            entry_date = current_date
            entry_idx = i
            
            position_value = portfolio_value * (max_position_size / 100)
            shares = position_value / current_price
            
            transaction_fee = position_value * (transaction_cost / 100)
            cash -= transaction_fee
        
        portfolio_history.append(portfolio_value)
    
    # Calculate metrics
    if len(trades) > 0:
        trades_df = pd.DataFrame(trades)
        
        total_return = (portfolio_value - initial_capital) / initial_capital * 100
        buy_hold_return = (bt_df['Close'].iloc[-1] / bt_df['Close'].iloc[0] - 1) * 100
        
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        win_rate = len(winning_trades) / len(trades_df) * 100
        avg_win = winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0
        
        returns = pd.Series(portfolio_history).pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        cummax = pd.Series(portfolio_history).cummax()
        drawdown = (pd.Series(portfolio_history) - cummax) / cummax * 100
        max_drawdown = drawdown.min()
        
        metrics = {
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'excess_return': total_return - buy_hold_return,
            'num_trades': len(trades_df),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'final_value': portfolio_value
        }
        
        return trades_df, metrics, portfolio_history
    
    return pd.DataFrame(), {
        'total_return': 0,
        'buy_hold_return': (bt_df['Close'].iloc[-1] / bt_df['Close'].iloc[0] - 1) * 100,
        'num_trades': 0
    }, portfolio_history


# Main app
if st.button("🚀 Run Analysis", type="primary"):
    with st.spinner("Loading data..."):
        df, error = load_data(ticker, period, debug=debug_mode)
        
        if error:
            st.error(f"❌ Data error: {error}")
            st.stop()
        
        st.success(f"✅ Loaded {len(df)} rows, {len(df.columns)} features")
    
    with st.spinner("Engineering features..."):
        features_df = create_features(df)
        
        if debug_mode:
            diagnose_data(features_df, "after feature engineering")
        
        st.success(f"✅ Created features")
    
    with st.spinner("Training model..."):
        model, feature_cols, ml_metrics = train_model(features_df, model_type)
        
        if model:
            st.success(f"✅ Model: {ml_metrics['cv_accuracy']:.3f} accuracy")
        else:
            st.info("Using technical signals only")
    
    with st.spinner("Running backtest..."):
        trades_df, metrics, portfolio_history = backtest_strategy(features_df, model, feature_cols)
    
    if 'error' in metrics:
        st.error(f"Backtest failed: {metrics['error']}")
        st.stop()
    
    # Results
    st.header("📊 Strategy Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Return", f"{metrics.get('total_return', 0):.2f}%")
    col2.metric("# Trades", metrics.get('num_trades', 0))
    col3.metric("Win Rate", f"{metrics.get('win_rate', 0):.1f}%")
    col4.metric("Sharpe", f"{metrics.get('sharpe_ratio', 0):.2f}")
    
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Avg Win", f"{metrics.get('avg_win', 0):.2f}%")
    col6.metric("Avg Loss", f"{metrics.get('avg_loss', 0):.2f}%")
    col7.metric("Max DD", f"{metrics.get('max_drawdown', 0):.2f}%")
    col8.metric("vs B&H", f"{metrics.get('excess_return', 0):.2f}%")
    
    # Validation
    if metrics.get('total_return', 0) > 100:
        st.warning("⚠️ Returns > 100% may indicate overfitting")
    if metrics.get('num_trades', 0) < 5:
        st.warning("⚠️ Sample size too small")
    if abs(metrics.get('max_drawdown', 0)) > 30:
        st.error("⚠️ Drawdown > 30% - very risky")
    
    # Charts
    if len(trades_df) > 0:
        fig = make_subplots(rows=2, cols=1, subplot_titles=['Price & Signals', 'Portfolio Value'])
        
        fig.add_trace(go.Scatter(x=features_df['Date'], y=features_df['Close'], name='Price'), row=1, col=1)
        
        long_trades = trades_df[trades_df['position'] == 'LONG']
        short_trades = trades_df[trades_df['position'] == 'SHORT']
        
        if len(long_trades) > 0:
            fig.add_trace(go.Scatter(
                x=long_trades['entry_date'], y=long_trades['entry_price'],
                mode='markers', marker=dict(color='green', size=10, symbol='triangle-up'),
                name='Long'
            ), row=1, col=1)
        
        if len(short_trades) > 0:
            fig.add_trace(go.Scatter(
                x=short_trades['entry_date'], y=short_trades['entry_price'],
                mode='markers', marker=dict(color='red', size=10, symbol='triangle-down'),
                name='Short'
            ), row=1, col=1)
        
        port_dates = features_df['Date'].iloc[:len(portfolio_history)]
        fig.add_trace(go.Scatter(x=port_dates, y=portfolio_history, name='Strategy'), row=2, col=1)
        
        bh_values = [100000 * (p / features_df['Close'].iloc[0]) for p in features_df['Close']]
        fig.add_trace(go.Scatter(x=features_df['Date'], y=bh_values, name='Buy & Hold', line=dict(dash='dash')), row=2, col=1)
        
        fig.update_layout(height=800, showlegend=True)
        st.plotly_chart(fig, width='stretch')
        
        st.subheader("📋 Recent Trades")
        st.dataframe(trades_df.tail(10), width='stretch')

st.markdown("---")
st.caption("⚠️ Educational only. Not financial advice.")