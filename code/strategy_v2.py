"""
Enhanced Intraday Predictive Modeling & Execution Framework - V2
Optimized for HFT-level performance with Sharpe 3.0+ target

Key improvements:
1. Multi-horizon ensemble predictions
2. Dynamic position sizing based on confidence
3. Volatility-adaptive thresholds
4. Enhanced feature engineering
5. Better signal generation
6. Transaction cost optimization

Author: Quantitative Research Team
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from typing import Dict, List, Tuple, Optional
import lightgbm as lgb
from sklearn.preprocessing import RobustScaler
import json


class EnhancedFeatureEngine:
    """Enhanced feature engineering with better causal features."""
    
    def __init__(self, horizons: List[int] = [10, 20, 30, 50]):
        self.horizons = horizons
        self.scaler = RobustScaler()
        
    def create_targets(self, df: pd.DataFrame, price_col: str = 'P3') -> pd.DataFrame:
        """Create multiple target horizons."""
        df = df.copy()
        
        for horizon in self.horizons:
            df[f'target_price_{horizon}'] = df[price_col].shift(-horizon)
            df[f'target_return_{horizon}'] = (df[f'target_price_{horizon}'] - df[price_col]) / df[price_col]
            df[f'target_direction_{horizon}'] = np.sign(df[f'target_return_{horizon}'])
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer comprehensive causal features - optimized version."""
        df = df.copy()
        
        # Price momentum and microstructure - reduced windows
        for window in [5, 10, 20, 30, 50]:
            # Returns
            df[f'P3_ret_{window}'] = df['P3'].pct_change(window)
            
            # Volatility (realized)
            df[f'P3_vol_{window}'] = df['P3'].pct_change().rolling(window, min_periods=1).std()
            
            # Moving averages
            df[f'P3_ma_{window}'] = df['P3'].rolling(window, min_periods=1).mean()
            df[f'P3_dist_ma_{window}'] = (df['P3'] - df[f'P3_ma_{window}']) / (df[f'P3_ma_{window}'] + 1e-10)
            
            # Z-score
            df[f'P3_zscore_{window}'] = (df['P3'] - df[f'P3_ma_{window}']) / (df[f'P3_vol_{window}'] + 1e-10)
        
        # Price spreads
        if 'P1' in df.columns and 'P2' in df.columns:
            df['spread_P1_P3'] = df['P1'] - df['P3']
            df['spread_P2_P3'] = df['P2'] - df['P3']
            
            for window in [10, 20]:
                df[f'spread_P1_P3_ma_{window}'] = df['spread_P1_P3'].rolling(window, min_periods=1).mean()
        
        # Key lag features
        for lag in [1, 3, 5, 10]:
            df[f'P3_lag_{lag}'] = df['P3'].shift(lag)
        
        # Volatility regime
        df['vol_regime_fast'] = df['P3'].pct_change().rolling(20, min_periods=1).std()
        df['vol_regime_slow'] = df['P3'].pct_change().rolling(50, min_periods=1).std()
        df['vol_ratio'] = df['vol_regime_fast'] / (df['vol_regime_slow'] + 1e-10)
        
        return df
    
    def sanitize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean features."""
        df = df.copy()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.fillna(0, inplace=True)
        return df
    
    def select_features(self, df: pd.DataFrame) -> List[str]:
        """Select feature columns."""
        exclude_patterns = ['ts_ns', 'target_']
        feature_cols = [col for col in df.columns 
                       if not any(pattern in col for pattern in exclude_patterns)]
        return feature_cols


class EnsembleModel:
    """Ensemble model with multiple horizons."""
    
    def __init__(self, horizons: List[int] = [10, 20, 30, 50]):
        self.horizons = horizons
        self.models_direction = {}
        self.models_magnitude = {}
        self.feature_cols = None
        
    def train(self, X: pd.DataFrame, df: pd.DataFrame, feature_cols: List[str]) -> None:
        """Train ensemble of models for different horizons."""
        self.feature_cols = feature_cols
        
        print(f"Training ensemble models for horizons: {self.horizons}")
        
        for horizon in self.horizons:
            print(f"  Training horizon {horizon}...")
            
            # Get targets for this horizon
            y_direction = df[f'target_direction_{horizon}']
            y_magnitude = df[f'target_return_{horizon}']
            
            # Remove missing targets
            valid_idx = ~(y_direction.isna() | y_magnitude.isna())
            X_train = X.loc[valid_idx, feature_cols]
            y_dir_train = y_direction[valid_idx]
            y_mag_train = y_magnitude[valid_idx]
            
            if len(X_train) == 0:
                continue
            
            # Optimized LightGBM parameters for speed
            params_direction = {
                'objective': 'multiclass',
                'num_class': 3,
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'max_depth': 6,
                'min_data_in_leaf': 100,
                'lambda_l1': 0.5,
                'lambda_l2': 0.5
            }
            
            params_magnitude = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'max_depth': 6,
                'min_data_in_leaf': 100,
                'lambda_l1': 0.5,
                'lambda_l2': 0.5
            }
            
            # Train direction model - faster
            y_dir_class = (y_dir_train + 1).astype(int)
            train_data_dir = lgb.Dataset(X_train, label=y_dir_class)
            self.models_direction[horizon] = lgb.train(
                params_direction,
                train_data_dir,
                num_boost_round=150,
                valid_sets=[train_data_dir],
                callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=0)]
            )
            
            # Train magnitude model - faster
            train_data_mag = lgb.Dataset(X_train, label=y_mag_train)
            self.models_magnitude[horizon] = lgb.train(
                params_magnitude,
                train_data_mag,
                num_boost_round=150,
                valid_sets=[train_data_mag],
                callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=0)]
            )
        
        print("Ensemble training complete!")
    
    def predict(self, X: pd.DataFrame) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        """Generate ensemble predictions."""
        X_pred = X[self.feature_cols]
        
        predictions_direction = {}
        predictions_magnitude = {}
        
        for horizon in self.horizons:
            if horizon in self.models_direction:
                # Direction
                dir_probs = self.models_direction[horizon].predict(X_pred)
                predictions_direction[horizon] = np.argmax(dir_probs, axis=1) - 1
                
                # Magnitude
                predictions_magnitude[horizon] = self.models_magnitude[horizon].predict(X_pred)
        
        return predictions_direction, predictions_magnitude


class EnhancedExecutionEngine:
    """Enhanced execution with dynamic position sizing and better signal generation."""
    
    def __init__(self, transaction_cost: float = 0.0001, max_position: int = 10):
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        self.reset()
        
    def reset(self):
        """Reset state."""
        self.position = 0
        self.entry_price = 0.0
        self.cumulative_pnl = 0.0
        self.realized_pnl = 0.0
        self.transaction_costs = 0.0
        self.trades = []
        
    def generate_signal(self, predictions_dir: Dict[int, int], 
                       predictions_mag: Dict[int, float],
                       current_vol: float,
                       base_threshold: float = 0.0005) -> Tuple[int, int]:
        """
        Generate trading signal with dynamic position sizing.
        
        Returns:
            Tuple of (direction, position_size)
        """
        if not predictions_dir or not predictions_mag:
            return 0, 0
        
        # Ensemble predictions - weighted by horizon (shorter = more weight)
        horizons = sorted(predictions_dir.keys())
        weights = [1.0 / (i + 1) for i in range(len(horizons))]
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Weighted direction vote
        direction_vote = 0
        magnitude_avg = 0
        confidence_scores = []
        
        for horizon, weight in zip(horizons, weights):
            direction_vote += predictions_dir[horizon] * weight
            magnitude_avg += abs(predictions_mag[horizon]) * weight
            
            # Confidence based on magnitude and consistency
            if abs(predictions_mag[horizon]) > base_threshold:
                confidence_scores.append(weight)
        
        # Determine direction
        if direction_vote > 0.3:
            direction = 1
        elif direction_vote < -0.3:
            direction = -1
        else:
            direction = 0
        
        # Volatility-adaptive threshold
        adaptive_threshold = base_threshold * (1.0 + current_vol * 100)
        
        # Check if signal is strong enough
        if magnitude_avg < adaptive_threshold:
            return 0, 0
        
        # Dynamic position sizing based on confidence
        confidence = sum(confidence_scores)
        
        if confidence > 0.8 and magnitude_avg > adaptive_threshold * 2:
            position_size = self.max_position
        elif confidence > 0.6 and magnitude_avg > adaptive_threshold * 1.5:
            position_size = max(5, self.max_position // 2)
        elif confidence > 0.4 and magnitude_avg > adaptive_threshold:
            position_size = max(3, self.max_position // 3)
        else:
            position_size = 1
        
        return direction, position_size
    
    def execute_iteration(self, timestamp: int, price: float, direction: int, 
                         position_size: int, pred_mag: float) -> Dict:
        """Execute one iteration with dynamic position sizing."""
        prev_position = self.position
        trade_cost = 0.0
        realized_pnl_step = 0.0
        
        # Target position
        target_position = direction * position_size
        
        # Calculate MTM PnL
        mtm_pnl = 0.0
        if self.position != 0:
            mtm_pnl = self.position * (price - self.entry_price)
        
        # Execute position change
        if target_position != prev_position:
            # Calculate cost
            position_change = abs(target_position - prev_position)
            trade_cost = self.transaction_cost * abs(price) * position_change
            
            # Realize PnL if reducing/closing position
            if prev_position != 0:
                if (prev_position > 0 and target_position < prev_position) or \
                   (prev_position < 0 and target_position > prev_position):
                    # Closing some/all position
                    closed_size = min(abs(prev_position), abs(prev_position - target_position))
                    realized_pnl_step = np.sign(prev_position) * closed_size * (price - self.entry_price)
                    self.realized_pnl += realized_pnl_step
            
            # Update position
            self.position = target_position
            if target_position != 0:
                # Weighted average entry price if adding to position
                if np.sign(prev_position) == np.sign(target_position) and prev_position != 0:
                    # Adding to existing position
                    added_size = abs(target_position - prev_position)
                    total_size = abs(target_position)
                    self.entry_price = (abs(prev_position) * self.entry_price + added_size * price) / total_size
                else:
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
                'position_size': position_size,
                'direction': direction,
                'pred_magnitude': pred_mag,
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
    
    def get_trade_log(self) -> pd.DataFrame:
        """Get trade log."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(self.trades)


class EnhancedStrategy:
    """Enhanced strategy with ensemble models and better execution."""
    
    def __init__(self, horizons: List[int] = [10, 20, 30, 50], 
                 transaction_cost: float = 0.0001,
                 max_position: int = 10):
        self.horizons = horizons
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        self.feature_engine = EnhancedFeatureEngine(horizons=horizons)
        self.model = EnsembleModel(horizons=horizons)
        self.execution_engine = EnhancedExecutionEngine(
            transaction_cost=transaction_cost,
            max_position=max_position
        )
        
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare data."""
        df = self.feature_engine.create_targets(df)
        df = self.feature_engine.engineer_features(df)
        df = self.feature_engine.sanitize(df)
        feature_cols = self.feature_engine.select_features(df)
        return df, feature_cols
    
    def train_on_historical_data(self, historical_files: List[str]) -> None:
        """Train on historical data."""
        print(f"Training on {len(historical_files)} historical days...")
        
        all_data = []
        for i, file_path in enumerate(historical_files):
            df = pd.read_csv(file_path)
            df, feature_cols = self.prepare_data(df)
            all_data.append(df)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(historical_files)} days")
        
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Total training samples: {len(combined_df)}")
        
        # Train ensemble
        valid_idx = ~combined_df[f'target_direction_{self.horizons[0]}'].isna()
        X = combined_df[valid_idx]
        
        self.model.train(X, combined_df, feature_cols)
        
    def execute_day(self, df: pd.DataFrame, 
                   base_threshold: float = 0.0005) -> Tuple[pd.DataFrame, Dict]:
        """Execute strategy on a day."""
        df, feature_cols = self.prepare_data(df)
        self.execution_engine.reset()
        
        results = []
        min_warmup = max(self.horizons) + 100  # Warm-up period
        
        for idx in range(len(df)):
            if idx < min_warmup:
                continue
            
            current_row = df.iloc[idx:idx+1]
            
            # Generate predictions
            try:
                predictions_dir, predictions_mag = self.model.predict(current_row)
                pred_dir_dict = {h: predictions_dir[h][0] for h in self.horizons if h in predictions_dir}
                pred_mag_dict = {h: predictions_mag[h][0] for h in self.horizons if h in predictions_mag}
            except:
                pred_dir_dict = {}
                pred_mag_dict = {}
            
            # Current volatility
            current_vol = df.iloc[idx]['vol_regime_fast'] if 'vol_regime_fast' in df.columns else 0.0001
            
            # Generate signal
            direction, position_size = self.execution_engine.generate_signal(
                pred_dir_dict, pred_mag_dict, current_vol, base_threshold
            )
            
            # Execute
            timestamp = int(df.iloc[idx]['ts_ns'])
            price = float(df.iloc[idx]['P3'])
            avg_mag = np.mean(list(pred_mag_dict.values())) if pred_mag_dict else 0.0
            
            result = self.execution_engine.execute_iteration(
                timestamp, price, direction, position_size, avg_mag
            )
            results.append(result)
        
        # Calculate summary
        results_df = pd.DataFrame(results)
        
        if len(results) > 0 and 'cumulative_pnl' in results_df.columns:
            final_pnl = self.execution_engine.cumulative_pnl
            total_trades = len(self.execution_engine.trades)
            
            pnl_series = results_df['cumulative_pnl']
            returns = pnl_series.diff()
            
            sharpe_ratio = 0.0
            if returns.std() > 0:
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(len(returns))
            
            max_drawdown = (pnl_series - pnl_series.cummax()).min()
            
            summary = {
                'final_pnl': final_pnl,
                'total_trades': total_trades,
                'total_transaction_costs': self.execution_engine.transaction_costs,
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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Enhanced Intraday Strategy V2')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--train_dir', type=str, default='train')
    parser.add_argument('--transaction_cost', type=float, default=0.0001)
    parser.add_argument('--confidence_threshold', type=float, default=0.0005)
    parser.add_argument('--train_days', type=int, default=80)
    parser.add_argument('--max_position', type=int, default=10)
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ENHANCED INTRADAY STRATEGY V2 - HFT OPTIMIZED")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Horizons: [10, 20, 30, 50]")
    print(f"  Transaction Cost: {args.transaction_cost * 100:.3f}%")
    print(f"  Base Threshold: {args.confidence_threshold}")
    print(f"  Training Days: {args.train_days}")
    print(f"  Max Position Size: {args.max_position}")
    print()
    
    strategy = EnhancedStrategy(
        horizons=[10, 20, 30, 50],
        transaction_cost=args.transaction_cost,
        max_position=args.max_position
    )
    
    # Get training files
    train_dir = Path(args.train_dir)
    all_train_files = sorted(train_dir.glob('*.csv'), key=lambda x: int(x.stem))
    train_files = all_train_files[:args.train_days]
    
    if len(train_files) > 0:
        strategy.train_on_historical_data([str(f) for f in train_files])
    
    # Execute
    print(f"\nExecuting on {args.input}...")
    input_df = pd.read_csv(args.input)
    results_df, summary = strategy.execute_day(input_df, args.confidence_threshold)
    
    # Save trade log
    trade_log = strategy.execution_engine.get_trade_log()
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
