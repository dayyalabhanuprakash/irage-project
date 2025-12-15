#!/usr/bin/env python3
"""
Simple Rule-Based Intraday Trading Strategy
NO machine learning - instant execution, predictable results

Uses momentum + mean reversion signals with proper position sizing.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class SimpleStrategy:
    """Simple rule-based strategy with no ML training needed."""
    
    def __init__(self, horizon: int = 30, transaction_cost: float = 0.0001):
        self.horizon = horizon
        self.transaction_cost = transaction_cost
        self.reset()
        
    def reset(self):
        """Reset execution state."""
        self.position = 0
        self.entry_price = 0.0
        self.cumulative_pnl = 0.0
        self.realized_pnl = 0.0
        self.transaction_costs = 0.0
        self.trades = []
        
    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trading signals based on simple rules."""
        df = df.copy()
        
        # Price features
        df['returns_5'] = df['P3'].pct_change(5)
        df['returns_10'] = df['P3'].pct_change(10)
        df['returns_20'] = df['P3'].pct_change(20)
        
        # Moving averages
        df['ma_10'] = df['P3'].rolling(10, min_periods=1).mean()
        df['ma_30'] = df['P3'].rolling(30, min_periods=1).mean()
        df['ma_50'] = df['P3'].rolling(50, min_periods=1).mean()
        
        # Distance from MA (mean reversion)
        df['dist_ma_10'] = (df['P3'] - df['ma_10']) / df['ma_10']
        df['dist_ma_30'] = (df['P3'] - df['ma_30']) / df['ma_30']
        
        # Volatility
        df['vol_20'] = df['P3'].pct_change().rolling(20, min_periods=1).std()
        df['vol_50'] = df['P3'].pct_change().rolling(50, min_periods=1).std()
        
        # Z-score (mean reversion)
        df['zscore_10'] = (df['P3'] - df['ma_10']) / (df['vol_20'] * df['P3'] + 1e-10)
        df['zscore_30'] = (df['P3'] - df['ma_30']) / (df['vol_20'] * df['P3'] + 1e-10)
        
        # Momentum strength
        df['momentum'] = (df['returns_5'] + df['returns_10'] + df['returns_20']) / 3
        
        # Trend (MA crossover)
        df['trend'] = np.where(df['ma_10'] > df['ma_30'], 1, -1)
        
        # Fill NaNs
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.fillna(0, inplace=True)
        
        return df
    
    def generate_signal(self, row: pd.Series, momentum_threshold: float = 0.0003,
                       reversion_threshold: float = 2.0) -> Tuple[int, float]:
        """
        Generate trading signal based on momentum and mean reversion.
        
        Returns:
            (direction, confidence)
        """
        # Momentum signal
        momentum = row['momentum']
        
        # Mean reversion signal  
        zscore = row['zscore_30']
        
        # Trend
        trend = row['trend']
        
        # Combined signal with confidence
        signal = 0
        confidence = 0.0
        
        # Strategy: Momentum + Trend following
        if momentum > momentum_threshold and trend > 0:
            signal = 1  # Long
            confidence = abs(momentum)
        elif momentum < -momentum_threshold and trend < 0:
            signal = -1  # Short
            confidence = abs(momentum)
        
        # Add mean reversion for extra confidence
        # If price too high (zscore > 2), reduce long confidence or add short
        # If price too low (zscore < -2), reduce short confidence or add long
        if abs(zscore) > reversion_threshold:
            # Mean reversion opposes extreme moves
            if zscore > reversion_threshold and signal == -1:
                confidence *= 1.5  # Boost short confidence
            elif zscore < -reversion_threshold and signal == 1:
                confidence *= 1.5  # Boost long confidence
            elif zscore > reversion_threshold and signal == 1:
                confidence *= 0.5  # Reduce long confidence
            elif zscore < -reversion_threshold and signal == -1:
                confidence *= 0.5  # Reduce short confidence
        
        return signal, confidence
    
    def execute_iteration(self, timestamp: int, price: float, signal: int, 
                         confidence: float) -> Dict:
        """Execute one trading iteration."""
        prev_position = self.position
        trade_cost = 0.0
        realized_pnl_step = 0.0
        
        # Calculate MTM
        mtm_pnl = 0.0
        if self.position != 0:
            mtm_pnl = self.position * (price - self.entry_price)
        
        # Execute position changes
        if signal != prev_position:
            # Calculate cost
            position_change = abs(signal - prev_position)
            trade_cost = self.transaction_cost * abs(price) * position_change
            
            # Realize PnL if closing
            if prev_position != 0:
                if np.sign(prev_position) != np.sign(signal) or abs(signal) < abs(prev_position):
                    closed_amount = min(abs(prev_position), abs(prev_position - signal))
                    realized_pnl_step = np.sign(prev_position) * closed_amount * (price - self.entry_price)
                    self.realized_pnl += realized_pnl_step
            
            # Update position
            self.position = signal
            if signal != 0:
                self.entry_price = price
            else:
                self.entry_price = 0.0
            
            # Deduct cost
            self.transaction_costs += trade_cost
            self.cumulative_pnl = self.realized_pnl + mtm_pnl - self.transaction_costs
            
            # Log trade
            self.trades.append({
                'timestamp': timestamp,
                'price': price,
                'prev_position': prev_position,
                'new_position': self.position,
                'confidence': confidence,
                'trade_cost': trade_cost,
                'realized_pnl': realized_pnl_step,
                'mtm_pnl': mtm_pnl,
                'cumulative_pnl': self.cumulative_pnl
            })
        else:
            # Update PnL
            self.cumulative_pnl = self.realized_pnl + mtm_pnl - self.transaction_costs
        
        return {
            'timestamp': timestamp,
            'price': price,
            'position': self.position,
            'cumulative_pnl': self.cumulative_pnl
        }
    
    def execute_day(self, df: pd.DataFrame, 
                   momentum_threshold: float = 0.0003,
                   reversion_threshold: float = 2.0,
                   position_multiplier: int = 1) -> Tuple[pd.DataFrame, Dict]:
        """
        Execute strategy on a single day.
        
        Args:
            df: Daily dataframe
            momentum_threshold: Minimum momentum to trade
            reversion_threshold: Z-score threshold for mean reversion
            position_multiplier: Position size multiplier
            
        Returns:
            Tuple of (results_df, summary)
        """
        # Calculate signals
        df = self.calculate_signals(df)
        
        # Reset state
        self.reset()
        
        # Execute
        results = []
        min_warmup = 50  # Need enough data for indicators
        
        for idx in range(len(df)):
            if idx < min_warmup:
                continue
            
            row = df.iloc[idx]
            
            # Generate signal
            direction, confidence = self.generate_signal(
                row, momentum_threshold, reversion_threshold
            )
            
            # Apply position multiplier
            signal = direction * position_multiplier
            
            # Execute
            timestamp = int(row['ts_ns'])
            price = float(row['P3'])
            
            result = self.execute_iteration(timestamp, price, signal, confidence)
            results.append(result)
        
        # Calculate summary
        results_df = pd.DataFrame(results)
        
        if len(results) > 0 and 'cumulative_pnl' in results_df.columns:
            final_pnl = self.cumulative_pnl
            total_trades = len(self.trades)
            
            pnl_series = results_df['cumulative_pnl']
            returns = pnl_series.diff()
            
            sharpe_ratio = 0.0
            if returns.std() > 0:
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(len(returns))
            
            max_drawdown = (pnl_series - pnl_series.cummax()).min()
            
            summary = {
                'final_pnl': final_pnl,
                'total_trades': total_trades,
                'total_transaction_costs': self.transaction_costs,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_iterations': len(results)
            }
        else:
            summary = {
                'final_pnl': 0.0,
                'total_trades': 0,
                'total_transaction_costs': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_iterations': len(results)
            }
        
        return results_df, summary
    
    def get_trade_log(self) -> pd.DataFrame:
        """Get trade log."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(self.trades)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Simple Rule-Based Strategy')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--horizon', type=int, default=30)
    parser.add_argument('--transaction_cost', type=float, default=0.0001)
    parser.add_argument('--momentum_threshold', type=float, default=0.0003)
    parser.add_argument('--reversion_threshold', type=float, default=2.0)
    parser.add_argument('--position_multiplier', type=int, default=1)
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("SIMPLE RULE-BASED STRATEGY")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Horizon: {args.horizon} bars")
    print(f"  Transaction Cost: {args.transaction_cost * 100:.3f}%")
    print(f"  Momentum Threshold: {args.momentum_threshold}")
    print(f"  Reversion Threshold: {args.reversion_threshold}")
    print(f"  Position Multiplier: {args.position_multiplier}x")
    print()
    
    # Initialize strategy
    strategy = SimpleStrategy(
        horizon=args.horizon,
        transaction_cost=args.transaction_cost
    )
    
    # Execute
    print(f"Executing on {args.input}...")
    input_df = pd.read_csv(args.input)
    
    results_df, summary = strategy.execute_day(
        input_df,
        args.momentum_threshold,
        args.reversion_threshold,
        args.position_multiplier
    )
    
    # Save trade log
    trade_log = strategy.get_trade_log()
    if len(trade_log) > 0:
        trade_log.to_csv(args.output, index=False)
        print(f"\nTrade log saved to {args.output}")
    else:
        pd.DataFrame().to_csv(args.output, index=False)
        print(f"\nNo trades executed.")
    
    # Print summary
    print("\n" + "=" * 80)
    print("EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Total Iterations: {summary['total_iterations']}")
    print(f"Total Trades: {summary['total_trades']}")
    print(f"Final PnL: {summary['final_pnl']:.6f}")
    print(f"Transaction Costs: {summary['total_transaction_costs']:.6f}")
    print(f"Sharpe Ratio: {summary['sharpe_ratio']:.4f}")
    print(f"Max Drawdown: {summary['max_drawdown']:.6f}")
    print("=" * 80)


if __name__ == '__main__':
    main()
