#!/usr/bin/env python3
"""
Train model once on historical data and save to disk.
This allows fast backtesting by loading pre-trained model.
"""

import argparse
import pickle
from pathlib import Path
from strategy import IntradayStrategy

def main():
    parser = argparse.ArgumentParser(description='Train and save model')
    parser.add_argument('--train_dir', type=str, default='train')
    parser.add_argument('--train_days', type=int, default=60, help='Days to train on')
    parser.add_argument('--sample_rate', type=int, default=5, help='Sampling rate')
    parser.add_argument('--output', type=str, default='trained_model.pkl')
    parser.add_argument('--horizon', type=int, default=30)
    parser.add_argument('--transaction_cost', type=float, default=0.00001)
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("TRAINING ML MODEL FOR HFT")
    print("=" * 80)
    print(f"\nTraining on {args.train_days} days with {args.sample_rate}x sampling")
    print("This will take 5-10 minutes but only needs to be done ONCE!")
    print()
    
    # Initialize strategy
    strategy = IntradayStrategy(
        horizon=args.horizon,
        transaction_cost=args.transaction_cost
    )
    
    # Get training files
    train_dir = Path(args.train_dir)
    all_files = sorted(train_dir.glob('*.csv'), key=lambda x: int(x.stem))
    train_files = all_files[:args.train_days]
    
    if len(train_files) == 0:
        print("ERROR: No training files found!")
        return
    
    print(f"Training on {len(train_files)} days...")
    strategy.train_on_historical_data([str(f) for f in train_files], 
                                     sample_rate=args.sample_rate)
    
    # Save model
    model_data = {
        'model_direction': strategy.model.model_direction,
        'model_magnitude': strategy.model.model_magnitude,
        'feature_cols': strategy.model.feature_cols,
        'horizon': args.horizon,
        'transaction_cost': args.transaction_cost
    }
    
    with open(args.output, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n✅ Model saved to {args.output}")
    print(f"   Size: {Path(args.output).stat().st_size / 1024 / 1024:.2f} MB")
    print()
    print("Now you can run fast backtests using:")
    print(f"  python3 code/batch_backtest.py --load_model {args.output} ...")
    print()
    print("=" * 80)

if __name__ == '__main__':
    main()
