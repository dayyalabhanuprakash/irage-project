#!/usr/bin/env python3
"""
Enhanced Batch Backtesting Script for Strategy V2
"""

import argparse
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json

def run_backtest_v2(day_file: str, output_dir: Path, train_dir: str, 
                    train_days: int = 80, confidence_threshold: float = 0.0005,
                    max_position: int = 10) -> dict:
    """Run backtest for a single day using enhanced strategy."""
    day_num = int(Path(day_file).stem)
    
    output_file = output_dir / f"trades_day_{day_num}.csv"
    log_file = output_dir / f"log_day_{day_num}.txt"
    
    cmd = [
        'python3', 'code/strategy_v2.py',
        '--input', day_file,
        '--output', str(output_file),
        '--train_dir', train_dir,
        '--train_days', str(train_days),
        '--confidence_threshold', str(confidence_threshold),
        '--max_position', str(max_position)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        # Save log
        with open(log_file, 'w') as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n\nSTDERR:\n")
                f.write(result.stderr)
        
        # Parse results
        output = result.stdout
        
        metrics = {
            'day': day_num,
            'status': 'success' if result.returncode == 0 else 'failed'
        }
        
        # Extract metrics from output
        for line in output.split('\n'):
            if 'Total Trades:' in line:
                metrics['trades'] = int(line.split(':')[1].strip())
            elif 'Final PnL:' in line:
                metrics['final_pnl'] = float(line.split(':')[1].strip())
            elif 'Sharpe Ratio:' in line:
                metrics['sharpe_ratio'] = float(line.split(':')[1].strip())
            elif 'Max Drawdown:' in line:
                metrics['max_drawdown'] = float(line.split(':')[1].strip())
        
        # Set defaults if not found
        metrics.setdefault('trades', 0)
        metrics.setdefault('final_pnl', 0.0)
        metrics.setdefault('sharpe_ratio', 0.0)
        metrics.setdefault('max_drawdown', 0.0)
        
        return metrics
        
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT on day {day_num}")
        return {
            'day': day_num,
            'trades': 0,
            'final_pnl': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'status': 'timeout'
        }
    except Exception as e:
        print(f"  ERROR on day {day_num}: {e}")
        return {
            'day': day_num,
            'trades': 0,
            'final_pnl': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'status': 'error'
        }


def main():
    parser = argparse.ArgumentParser(description='Batch backtest with enhanced strategy')
    parser.add_argument('--train_dir', type=str, default='train')
    parser.add_argument('--output_dir', type=str, default='results_v2')
    parser.add_argument('--start_day', type=int, default=81)
    parser.add_argument('--end_day', type=int, default=110)
    parser.add_argument('--train_days', type=int, default=80)
    parser.add_argument('--confidence_threshold', type=float, default=0.0005)
    parser.add_argument('--max_position', type=int, default=10)
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 80)
    print("ENHANCED BATCH BACKTEST - STRATEGY V2")
    print("=" * 80)
    print(f"Train Directory: {args.train_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Day Range: {args.start_day} to {args.end_day}")
    print(f"Training Days: {args.train_days}")
    print(f"Confidence Threshold: {args.confidence_threshold}")
    print(f"Max Position: {args.max_position}")
    print("=" * 80)
    
    # Get available days
    train_dir = Path(args.train_dir)
    all_files = sorted(train_dir.glob('*.csv'), key=lambda x: int(x.stem))
    
    test_files = [f for f in all_files 
                  if args.start_day <= int(f.stem) <= args.end_day]
    
    print(f"\nFound {len(test_files)} days to backtest")
    print(f"Starting backtest at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Run backtests
    results = []
    for i, day_file in enumerate(test_files, 1):
        day_num = int(day_file.stem)
        print(f"[{i}/{len(test_files)}] Testing day {day_num}...", end=' ', flush=True)
        
        metrics = run_backtest_v2(
            str(day_file),
            output_dir,
            args.train_dir,
            args.train_days,
            args.confidence_threshold,
            args.max_position
        )
        
        results.append(metrics)
        
        print(f"PnL: {metrics['final_pnl']:>10.6f}, Trades: {metrics['trades']:>3}, "
              f"Sharpe: {metrics['sharpe_ratio']:>7.4f}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_file = output_dir / 'daily_results.csv'
    results_df.to_csv(results_file, index=False)
    
    # Calculate aggregate metrics
    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)
    
    total_days = len(results_df)
    successful_days = (results_df['status'] == 'success').sum()
    
    print(f"Total Days: {total_days}")
    print(f"Successful: {successful_days}")
    print(f"Days with Trades: {(results_df['trades'] > 0).sum()}")
    
    # PnL metrics
    total_pnl = results_df['final_pnl'].sum()
    avg_pnl = results_df['final_pnl'].mean()
    std_pnl = results_df['final_pnl'].std()
    
    print(f"\n--- PnL Metrics ---")
    print(f"Total PnL: {total_pnl:.6f}")
    print(f"Average PnL/Day: {avg_pnl:.6f}")
    print(f"Std Dev: {std_pnl:.6f}")
    print(f"Best Day: {results_df['final_pnl'].max():.6f}")
    print(f"Worst Day: {results_df['final_pnl'].min():.6f}")
    
    # Sharpe ratio
    if std_pnl > 0:
        daily_sharpe = avg_pnl / std_pnl
        # Annualize using actual trading days (111 days in our dataset)
        annualized_sharpe = daily_sharpe * np.sqrt(111)
        print(f"\n--- Risk-Adjusted Returns ---")
        print(f"Daily Sharpe: {daily_sharpe:.4f}")
        print(f"Annualized Sharpe: {annualized_sharpe:.4f}")
    
    # Win rate
    winning_days = (results_df['final_pnl'] > 0).sum()
    win_rate = winning_days / total_days * 100 if total_days > 0 else 0
    print(f"\n--- Win/Loss ---")
    print(f"Winning Days: {winning_days}")
    print(f"Losing Days: {total_days - winning_days}")
    print(f"Win Rate: {win_rate:.2f}%")
    
    # Trading activity
    total_trades = results_df['trades'].sum()
    avg_trades = results_df['trades'].mean()
    print(f"\n--- Trading Activity ---")
    print(f"Total Trades: {total_trades}")
    print(f"Avg Trades/Day: {avg_trades:.2f}")
    
    # Drawdown
    cumulative_pnl = results_df['final_pnl'].cumsum()
    running_max = cumulative_pnl.cummax()
    drawdown = cumulative_pnl - running_max
    max_drawdown = drawdown.min()
    
    print(f"\n--- Drawdown ---")
    print(f"Max Drawdown: {max_drawdown:.6f}")
    
    print("=" * 80)
    
    # Save summary
    summary = {
        'configuration': {
            'train_days': args.train_days,
            'confidence_threshold': args.confidence_threshold,
            'max_position': args.max_position,
            'start_day': args.start_day,
            'end_day': args.end_day
        },
        'metrics': {
            'total_days': int(total_days),
            'successful_days': int(successful_days),
            'total_pnl': float(total_pnl),
            'avg_pnl_per_day': float(avg_pnl),
            'std_pnl': float(std_pnl),
            'daily_sharpe': float(daily_sharpe) if std_pnl > 0 else 0.0,
            'annualized_sharpe': float(annualized_sharpe) if std_pnl > 0 else 0.0,
            'win_rate': float(win_rate),
            'total_trades': int(total_trades),
            'avg_trades_per_day': float(avg_trades),
            'max_drawdown': float(max_drawdown)
        }
    }
    
    summary_file = output_dir / 'summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
    print(f"Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
