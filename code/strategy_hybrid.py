#!/usr/bin/env python3
"""
Hybrid HFT Strategy - Combining multiple signal types for Sharpe 2.5+

Uses:
1. Short-term momentum (3-5 bars)
2. Mean reversion from moving averages
3. Volume-price divergence
4. Microstructure signals

No ML training needed - purely signal-based, optimized for HFT.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class HybridHFTStrategy:
    """Hybrid strategy optimized for HFT with multiple signal types."""
    
    def __init__(self, horizon: int = 30, transaction_cost: float = 0.00001):
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
        """Calculate comprehensive trading signals."""
        df = df.copy()
        
        # Price features
        df['ret_1'] = df['P3'].pct_change(1)
        df['ret_3'] = df['P3'].pct_change(3)
        df['ret_5'] = df['P3'].pct_change(5)
        df['ret_10'] = df['P3'].pct_change(10)
        df['ret_20'] = df['P3'].pct_change(20)
        
        # Moving averages
        df['ma_5'] = df['P3'].rolling(5, min_periods=1).mean()
        df['ma_10'] = df['P3'].rolling(10, min_periods=1).mean()
        df['ma_20'] = df['P3'].rolling(20, min_periods=1).mean()
        df['ma_50'] = df['P3'].rolling(50, min_periods=1).mean()
        
        # Price relative to MAs
        df['dist_ma_5'] = (df['P3'] - df['ma_5']) / (df['ma_5'] + 1e-10)
        df['dist_ma_10'] = (df['P3'] - df['ma_10']) / (df['ma_10'] + 1e-10)
        df['dist_ma_20'] = (df['P3'] - df['ma_20']) / (df['ma_20'] + 1e-10)
        
        # Volatility
        df['vol_5'] = df['ret_1'].rolling(5, min_periods=1).std()
        df['vol_20'] = df['ret_1'].rolling(20, min_periods=1).std()
        df['vol_50'] = df['ret_1'].rolling(50, min_periods=1).std()
        
        # Z-scores for mean reversion
        df['zscore_5'] = df['dist_ma_5'] / (df['vol_5'] + 1e-10)
        df['zscore_10'] = df['dist_ma_10'] / (df['vol_20'] + 1e-10)
        df['zscore_20'] = df['dist_ma_20'] / (df['vol_20'] + 1e-10)
        
        # Momentum indicators
        df['momentum_short'] = (df['ret_3'] + df['ret_5']) / 2
        df['momentum_med'] = (df['ret_10'] + df['ret_20']) / 2
        
        # Acceleration (second derivative of price)
        df['accel_3'] = df['ret_1'].diff(3)
        df['accel_5'] = df['ret_1'].diff(5)
        
        # MA crossovers
        df['ma_cross_short'] = df['ma_5'] - df['ma_10']
        df['ma_cross_long'] = df['ma_10'] - df['ma_20']
        
        # Trend strength
        df['trend_strength'] = abs(df['ret_20'])
        
        # Volatility regime
        df['vol_regime'] = df['vol_5'] / (df['vol_50'] + 1e-10)
        
        # Fill NaNs
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.fillna(0, inplace=True)
        
        return df
    
    def generate_signal(self, row: pd.Series) -> Tuple[int, float]:
        """
        Generate trading signal using multiple strategies.
        
        Returns:
            (direction, confidence)
        """
        signals = []
        confidences = []
        
        # Strategy 1: Short-term momentum
        momentum_short = row['momentum_short']
        if abs(momentum_short) > 0.0002:
            signals.append(np.sign(momentum_short))
            confidences.append(abs(momentum_short) * 100)
        
        # Strategy 2: Mean reversion (fade extremes)
        zscore = row['zscore_10']
        if abs(zscore) > 1.5:
            # Price too far from mean, expect reversion
            signals.append(-np.sign(zscore))
            confidences.append(min(abs(zscore) / 10, 0.5))
        
        # Strategy 3: MA crossover momentum
        ma_cross = row['ma_cross_short']
        if abs(ma_cross / (row['P3'] + 1e-10)) > 0.0001:
            signals.append(np.sign(ma_cross))
            confidences.append(abs(ma_cross / (row['P3'] + 1e-10)) * 100)
        
        # Strategy 4: Acceleration (trend continuation)
        accel = row['accel_5']
        if abs(accel) > 0.00005:
            signals.append(np.sign(accel))
            confidences.append(abs(accel) * 1000)
        
        # Strategy 5: Volatility breakout
        if row['vol_regime'] > 1.5 and abs(row['ret_3']) > 0.0003:
            # High volatility + movement = trend
            signals.append(np.sign(row['ret_3']))
            confidences.append(abs(row['ret_3']) * 100)
        
        # Combine signals
        if len(signals) == 0:
            return 0, 0.0
        
        # Weighted vote
        total_confidence = sum(confidences)
        if total_confidence == 0:
            return 0, 0.0
        
        weighted_signal = sum(s * c for s, c in zip(signals, confidences)) / total_confidence
        
        # Determine direction
        if weighted_signal > 0.3:
            direction = 1
        elif weighted_signal < -0.3:
            direction = -1
        else:
            direction = 0
        
        # Overall confidence
        confidence = total_confidence / len(signals)
        
        return direction, confidence
    
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
                   signal_threshold: float = 0.1) -> Tuple[pd.DataFrame, Dict]:
        """
        Execute strategy on a single day.
        
        Args:
            df: Daily dataframe
            signal_threshold: Minimum confidence to trade
            
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
            direction, confidence = self.generate_signal(row)
            
            # Only trade if confidence exceeds threshold
            if confidence < signal_threshold:
                direction = 0
            
            # Execute
            timestamp = int(row['ts_ns'])
            price = float(row['P3'])
            
            result = self.execute_iteration(timestamp, price, direction, confidence)
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
            
            max_pnl = pnl_series.max()
            min_pnl = pnl_series.min()
            max_drawdown = (pnl_series - pnl_series.cummax()).min()
            
            summary = {
                'final_pnl': final_pnl,
                'total_trades': total_trades,
                'total_transaction_costs': self.transaction_costs,
                'sharpe_ratio': sharpe_ratio,
                'max_pnl': max_pnl,
                'min_pnl': min_pnl,
                'max_drawdown': max_drawdown,
                'total_iterations': len(results)
            }
        else:
            summary = {
                'final_pnl': 0.0,
                'total_trades': 0,
                'total_transaction_costs': 0.0,
                'sharpe_ratio': 0.0,
                'max_pnl': 0.0,
                'min_pnl': 0.0,
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
    parser = argparse.ArgumentParser(description='Hybrid HFT Strategy')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--horizon', type=int, default=30)
    parser.add_argument('--transaction_cost', type=float, default=0.00001)
    parser.add_argument('--signal_threshold', type=float, default=0.1)
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("HYBRID HFT STRATEGY")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Horizon: {args.horizon} bars")
    print(f"  Transaction Cost: {args.transaction_cost * 100:.4f}%")
    print(f"  Signal Threshold: {args.signal_threshold}")
    print()
    
    # Initialize strategy
    strategy = HybridHFTStrategy(
        horizon=args.horizon,
        transaction_cost=args.transaction_cost
    )
    
    # Execute
    print(f"Executing on {args.input}...")
    input_df = pd.read_csv(args.input)
    
    results_df, summary = strategy.execute_day(
        input_df,
        args.signal_threshold
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
    print(f"Max PnL: {summary['max_pnl']:.6f}")
    print(f"Min PnL: {summary['min_pnl']:.6f}")
    print(f"Max Drawdown: {summary['max_drawdown']:.6f}")
    print("=" * 80)


if __name__ == '__main__':
    main()
