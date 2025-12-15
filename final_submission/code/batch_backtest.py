#!/usr/bin/env python3
"""
Batch backtesting script to run strategy on multiple days.

Usage:
    python batch_backtest.py --start_day 51 --end_day 60 --train_days 50 --output_dir results/
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from strategy import IntradayStrategy
import json
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description='Batch backtest across multiple days')
    parser.add_argument('--start_day', type=int, required=True, help='First day to test')
    parser.add_argument('--end_day', type=int, required=True, help='Last day to test')
    parser.add_argument('--train_days', type=int, default=10, help='Days to use for training')
    parser.add_argument('--sample_rate', type=int, default=10, help='Sample every Nth row during training')
    parser.add_argument('--train_dir', type=str, default='train', help='Training data directory')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--horizon', type=int, default=30, help='Prediction horizon')
    parser.add_argument('--transaction_cost', type=float, default=0.00001, help='Transaction cost (0.001% for HFT)')
    parser.add_argument('--confidence_threshold', type=float, default=0.0003, help='Confidence threshold')
    parser.add_argument('--position_multiplier', type=int, default=1, help='Position size multiplier')
    parser.add_argument('--retrain_freq', type=int, default=10, help='Retrain every N days')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize strategy
    strategy = IntradayStrategy(
        horizon=args.horizon,
        transaction_cost=args.transaction_cost
    )
    
    # Get all available training files
    train_dir = Path(args.train_dir)
    all_files = sorted(train_dir.glob('*.csv'), key=lambda x: int(x.stem))
    
    # Results storage
    all_results = []
    
    print("=" * 80)
    print("BATCH BACKTESTING - OPTIMIZED STRATEGY")
    print("=" * 80)
    print(f"Testing days: {args.start_day} to {args.end_day}")
    print(f"Training window: {args.train_days} days")
    print(f"Confidence threshold: {args.confidence_threshold}")
    print(f"Position multiplier: {args.position_multiplier}x")
    print(f"Retrain frequency: Every {args.retrain_freq} days")
    print("=" * 80)
    
    # Initial training
    train_files = [str(f) for f in all_files[:min(args.train_days, args.start_day - 1)]]
    if len(train_files) > 0:
        print(f"\nInitial training on {len(train_files)} days with {args.sample_rate}x sampling...")
        strategy.train_on_historical_data(train_files, sample_rate=args.sample_rate)
        print("Training complete!")
    
    # Backtest loop
    for day_num in range(args.start_day, args.end_day + 1):
        print(f"\n{'='*80}")
        print(f"Testing Day {day_num}")
        print(f"{'='*80}")
        
        # Retrain periodically
        if (day_num - args.start_day) % args.retrain_freq == 0 and day_num > args.start_day:
            print(f"Retraining model with expanding window...")
            train_end = min(day_num - 1, len(all_files))
            train_start = max(0, train_end - args.train_days)
            train_files = [str(all_files[i]) for i in range(train_start, train_end)]
            strategy.train_on_historical_data(train_files, sample_rate=args.sample_rate)
        
        # Load test day
        test_file = train_dir / f"{day_num}.csv"
        if not test_file.exists():
            print(f"WARNING: {test_file} not found, skipping...")
            continue
        
        # Execute strategy
        try:
            df = pd.read_csv(test_file)
            results_df, summary = strategy.execute_day(df, args.confidence_threshold, 
                                                      args.position_multiplier)
            
            # Save trade log
            trade_log = strategy.execution_engine.get_trade_log()
            trade_log_file = output_dir / f"trades_day_{day_num}.csv"
            if len(trade_log) > 0:
                trade_log.to_csv(trade_log_file, index=False)
            
            # Store summary
            summary['day'] = day_num
            all_results.append(summary)
            
            # Print summary
            print(f"  Iterations: {summary['total_iterations']}")
            print(f"  Trades: {summary['total_trades']}")
            print(f"  Final PnL: {summary['final_pnl']:.6f}")
            print(f"  Sharpe: {summary['sharpe_ratio']:.4f}")
            print(f"  Max Drawdown: {summary['max_drawdown']:.6f}")
            
        except Exception as e:
            print(f"ERROR on day {day_num}: {str(e)}")
            continue
    
    # Aggregate results
    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)
    
    results_df = pd.DataFrame(all_results)
    
    if len(results_df) > 0:
        total_pnl = results_df['final_pnl'].sum()
        avg_pnl = results_df['final_pnl'].mean()
        total_trades = results_df['total_trades'].sum()
        avg_sharpe = results_df['sharpe_ratio'].mean()
        total_costs = results_df['total_transaction_costs'].sum()
        
        print(f"Total Days Tested: {len(results_df)}")
        print(f"Total PnL: {total_pnl:.6f}")
        print(f"Average Daily PnL: {avg_pnl:.6f}")
        print(f"Total Trades: {total_trades}")
        print(f"Average Sharpe Ratio: {avg_sharpe:.4f}")
        print(f"Total Transaction Costs: {total_costs:.6f}")
        print(f"Net PnL (after costs): {total_pnl:.6f}")
        
        # Calculate overall Sharpe
        daily_returns = results_df['final_pnl']
        # Annualize using actual trading days (111 days in our dataset)
        overall_sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(111) if daily_returns.std() > 0 else 0
        print(f"Annualized Sharpe Ratio: {overall_sharpe:.4f}")
        
        # Save results
        results_file = output_dir / "aggregate_results.csv"
        results_df.to_csv(results_file, index=False)
        print(f"\nResults saved to {results_file}")
        
        # Save summary JSON
        summary_dict = {
            'total_days': len(results_df),
            'total_pnl': float(total_pnl),
            'avg_daily_pnl': float(avg_pnl),
            'total_trades': int(total_trades),
            'avg_sharpe': float(avg_sharpe),
            'annualized_sharpe': float(overall_sharpe),
            'total_costs': float(total_costs),
            'configuration': {
                'horizon': args.horizon,
                'transaction_cost': args.transaction_cost,
                'confidence_threshold': args.confidence_threshold,
                'position_multiplier': args.position_multiplier,
                'train_days': args.train_days
            }
        }
        
        summary_file = output_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_dict, f, indent=2)
        print(f"Summary saved to {summary_file}")
    else:
        print("No results to aggregate.")
    
    print("=" * 80)


if __name__ == '__main__':
    main()
