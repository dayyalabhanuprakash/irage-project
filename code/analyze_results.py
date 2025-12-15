#!/usr/bin/env python3
"""
Analyze and visualize backtesting results.

Usage:
    python analyze_results.py --results_dir results/
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json


def calculate_metrics(results_df):
    """Calculate comprehensive performance metrics."""
    metrics = {}
    
    # Basic stats
    metrics['total_days'] = len(results_df)
    metrics['total_pnl'] = results_df['final_pnl'].sum()
    metrics['avg_pnl_per_day'] = results_df['final_pnl'].mean()
    metrics['median_pnl_per_day'] = results_df['final_pnl'].median()
    metrics['std_pnl_per_day'] = results_df['final_pnl'].std()
    
    # Win rate
    winning_days = (results_df['final_pnl'] > 0).sum()
    metrics['win_rate'] = winning_days / len(results_df) if len(results_df) > 0 else 0
    
    # Sharpe ratio
    if results_df['final_pnl'].std() > 0:
        daily_sharpe = results_df['final_pnl'].mean() / results_df['final_pnl'].std()
        metrics['daily_sharpe'] = daily_sharpe
        # Annualize using actual trading days (111 days in our dataset)
        metrics['annualized_sharpe'] = daily_sharpe * np.sqrt(111)
    else:
        metrics['daily_sharpe'] = 0
        metrics['annualized_sharpe'] = 0
    
    # Drawdown analysis
    cumulative_pnl = results_df['final_pnl'].cumsum()
    running_max = cumulative_pnl.cummax()
    drawdown = cumulative_pnl - running_max
    metrics['max_drawdown'] = drawdown.min()
    metrics['max_drawdown_pct'] = (drawdown.min() / running_max.max() * 100) if running_max.max() > 0 else 0
    
    # Trading stats
    metrics['total_trades'] = results_df['total_trades'].sum()
    metrics['avg_trades_per_day'] = results_df['total_trades'].mean()
    metrics['total_transaction_costs'] = results_df['total_transaction_costs'].sum()
    
    # Cost analysis
    gross_pnl = metrics['total_pnl'] + metrics['total_transaction_costs']
    metrics['gross_pnl'] = gross_pnl
    metrics['net_pnl'] = metrics['total_pnl']
    metrics['cost_ratio'] = metrics['total_transaction_costs'] / gross_pnl if gross_pnl != 0 else 0
    
    # Best/worst days
    metrics['best_day_pnl'] = results_df['final_pnl'].max()
    metrics['worst_day_pnl'] = results_df['final_pnl'].min()
    metrics['best_day'] = results_df.loc[results_df['final_pnl'].idxmax(), 'day'] if 'day' in results_df.columns else None
    metrics['worst_day'] = results_df.loc[results_df['final_pnl'].idxmin(), 'day'] if 'day' in results_df.columns else None
    
    # Consistency
    metrics['positive_days'] = winning_days
    metrics['negative_days'] = len(results_df) - winning_days
    metrics['avg_win'] = results_df[results_df['final_pnl'] > 0]['final_pnl'].mean() if winning_days > 0 else 0
    metrics['avg_loss'] = results_df[results_df['final_pnl'] < 0]['final_pnl'].mean() if (len(results_df) - winning_days) > 0 else 0
    metrics['profit_factor'] = abs(metrics['avg_win'] / metrics['avg_loss']) if metrics['avg_loss'] != 0 else 0
    
    return metrics


def print_metrics(metrics):
    """Pretty print metrics."""
    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS")
    print("=" * 80)
    
    print("\n--- Overall Performance ---")
    print(f"Total Days: {metrics['total_days']}")
    print(f"Gross PnL: {metrics['gross_pnl']:.6f}")
    print(f"Transaction Costs: {metrics['total_transaction_costs']:.6f}")
    print(f"Net PnL: {metrics['net_pnl']:.6f}")
    print(f"Cost Ratio: {metrics['cost_ratio']*100:.2f}%")
    
    print("\n--- Daily Statistics ---")
    print(f"Average PnL/Day: {metrics['avg_pnl_per_day']:.6f}")
    print(f"Median PnL/Day: {metrics['median_pnl_per_day']:.6f}")
    print(f"Std Dev PnL/Day: {metrics['std_pnl_per_day']:.6f}")
    print(f"Best Day: {metrics['best_day_pnl']:.6f} (Day {metrics['best_day']})")
    print(f"Worst Day: {metrics['worst_day_pnl']:.6f} (Day {metrics['worst_day']})")
    
    print("\n--- Risk-Adjusted Returns ---")
    print(f"Daily Sharpe Ratio: {metrics['daily_sharpe']:.4f}")
    print(f"Annualized Sharpe Ratio: {metrics['annualized_sharpe']:.4f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.6f}")
    print(f"Max Drawdown %: {metrics['max_drawdown_pct']:.2f}%")
    
    print("\n--- Win/Loss Analysis ---")
    print(f"Win Rate: {metrics['win_rate']*100:.2f}%")
    print(f"Positive Days: {metrics['positive_days']}")
    print(f"Negative Days: {metrics['negative_days']}")
    print(f"Average Win: {metrics['avg_win']:.6f}")
    print(f"Average Loss: {metrics['avg_loss']:.6f}")
    print(f"Profit Factor: {metrics['profit_factor']:.4f}")
    
    print("\n--- Trading Activity ---")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Average Trades/Day: {metrics['avg_trades_per_day']:.2f}")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Analyze backtesting results')
    parser.add_argument('--results_dir', type=str, default='results', help='Results directory')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    # Load aggregate results
    results_file = results_dir / "aggregate_results.csv"
    
    if not results_file.exists():
        print(f"ERROR: {results_file} not found!")
        print("Run batch_backtest.py first to generate results.")
        return
    
    print(f"Loading results from {results_file}...")
    results_df = pd.read_csv(results_file)
    
    print(f"Loaded {len(results_df)} days of results")
    
    # Calculate metrics
    metrics = calculate_metrics(results_df)
    
    # Print metrics
    print_metrics(metrics)
    
    # Load and print configuration if available
    summary_file = results_dir / "summary.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        print("\n" + "=" * 80)
        print("CONFIGURATION")
        print("=" * 80)
        if 'configuration' in summary:
            for key, value in summary['configuration'].items():
                print(f"{key}: {value}")
        print("=" * 80)
    
    # Save detailed metrics
    metrics_file = results_dir / "detailed_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nDetailed metrics saved to {metrics_file}")
    
    # Generate equity curve CSV
    if 'day' in results_df.columns:
        equity_curve = pd.DataFrame({
            'day': results_df['day'],
            'daily_pnl': results_df['final_pnl'],
            'cumulative_pnl': results_df['final_pnl'].cumsum()
        })
        equity_file = results_dir / "equity_curve.csv"
        equity_curve.to_csv(equity_file, index=False)
        print(f"Equity curve saved to {equity_file}")


if __name__ == '__main__':
    main()
