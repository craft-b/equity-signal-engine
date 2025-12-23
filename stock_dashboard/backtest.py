import numpy as np
import pandas as pd

def backtest_strategy(model, X_test, y_test, df_test, risk_free_rate=0.0):
    """
    Backtest a long-short strategy based on model predictions.

    Args:
        model: Fitted ML model with .predict method
        X_test: Features for testing
        y_test: True labels (not used for trading, optional)
        df_test: Original test DataFrame containing 'Return'
        risk_free_rate: Daily risk-free rate (annualized 0.0 by default)

    Returns:
        results: Dict with Total Return, Sharpe Ratio, Max Drawdown, Final Equity
        df_bt: DataFrame with positions, strategy returns, cumulative performance
    """
    df_bt = df_test.copy().reset_index(drop=True)
    
    # Generate trading signals
    signals = model.predict(X_test)
    df_bt["Signal"] = signals
    
    # Map signals: 1 -> long, 0 -> short
    df_bt["Position"] = np.where(df_bt["Signal"] == 1, 1, -1)
    
    # Strategy returns
    df_bt["Strategy_Return"] = df_bt["Position"] * df_bt["Return"].fillna(0)
    
    # Cumulative returns
    df_bt["Cumulative_Strategy"] = (1 + df_bt["Strategy_Return"]).cumprod()
    df_bt["Cumulative_BuyHold"] = (1 + df_bt["Return"].fillna(0)).cumprod()
    
    # Total return
    total_return = df_bt["Cumulative_Strategy"].iloc[-1] - 1
    
    # Sharpe ratio (annualized)
    avg_daily_return = df_bt["Strategy_Return"].mean()
    std_daily_return = df_bt["Strategy_Return"].std()
    sharpe_ratio = (avg_daily_return - risk_free_rate / 252) / std_daily_return * np.sqrt(252) if std_daily_return != 0 else np.nan
    
    # Max drawdown
    cum_max = df_bt["Cumulative_Strategy"].cummax()
    drawdown = (df_bt["Cumulative_Strategy"] - cum_max) / cum_max
    max_drawdown = drawdown.min()
    
    results = {
        "Total Return (%)": round(total_return * 100, 2),
        "Sharpe Ratio": round(sharpe_ratio, 3),
        "Max Drawdown (%)": round(max_drawdown * 100, 2),
        "Final Equity": round(df_bt["Cumulative_Strategy"].iloc[-1], 3)
    }
    
    return results, df_bt
