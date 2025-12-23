import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def compare_models(test_results: dict, backtest_results: dict) -> pd.DataFrame:
    """
    Combine model performance metrics and backtest results into a single summary table.
    
    Args:
        test_results: Dict with model metrics (Accuracy, Precision, Recall, F1)
        backtest_results: Dict with backtest metrics (Total Return, Sharpe, etc.)
        
    Returns:
        Pandas DataFrame summarizing all models
    """
    rows = []
    for model_name in test_results.keys():
        row = {
            "Model": model_name,
            "Accuracy": round(test_results[model_name].get("Accuracy", 0), 3),
            "Precision": round(test_results[model_name].get("Precision", 0), 3),
            "Recall": round(test_results[model_name].get("Recall", 0), 3),
            "F1": round(test_results[model_name].get("F1", 0), 3),
            "Total Return (%)": backtest_results[model_name].get("Total Return (%)", 0),
            "Sharpe Ratio": backtest_results[model_name].get("Sharpe Ratio", 0),
            "Max Drawdown (%)": backtest_results[model_name].get("Max Drawdown (%)", 0),
            "Final Equity": backtest_results[model_name].get("Final Equity", 0),
        }
        rows.append(row)
    summary_df = pd.DataFrame(rows)
    return summary_df

def plot_equity_curves(backtest_dfs: dict, title: str = "Equity Curves: Models vs Buy & Hold"):
    """
    Plot cumulative returns of multiple models against Buy & Hold.
    
    Args:
        backtest_dfs: Dict of model name -> backtest DataFrame (must include 'Date', 'Cumulative_Strategy', 'Cumulative_BuyHold')
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot each model's strategy
    for model_name, df_bt in backtest_dfs.items():
        ax.plot(df_bt["Date"], df_bt["Cumulative_Strategy"], label=f"{model_name} Strategy")
    
    # Plot buy & hold from first model as reference
    sample_df = list(backtest_dfs.values())[0]
    ax.plot(sample_df["Date"], sample_df["Cumulative_BuyHold"], label="Buy & Hold", linestyle="--", color="black")
    
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Returns")
    ax.legend()
    ax.grid(True)
    
    st.pyplot(fig)
