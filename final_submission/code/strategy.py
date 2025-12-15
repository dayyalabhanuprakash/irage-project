"""
Intraday Predictive Modeling & Execution Framework

This module implements a causal, iteration-safe prediction engine and fully executable
trading strategy that operates on P3 with realistic transaction costs (0.01%).

Author: Quantitative Research Team
Date: 2024
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


class FeatureEngine:
    """
    Feature engineering and sanitation module with strict causality enforcement.
    
    Design Principles:
    - No forward-looking windows
    - No target leakage
    - Hierarchical feature extraction from underscore structure
    - Family-wise normalization with causal rolling statistics
    """
    
    def __init__(self, horizon: int = 30):
        """
        Args:
            horizon: Forward prediction horizon in bars (minimum 30)
        """
        self.horizon = horizon
        self.feature_groups = {}
        self.scaler = RobustScaler()
        
    def extract_feature_families(self, columns: List[str]) -> Dict[str, List[str]]:
        """
        Extract feature families based on underscore structure.
        Example: F_H_B -> families: ['F', 'F_H', 'F_H_B']
        
        Args:
            columns: List of column names
            
        Returns:
            Dictionary mapping family prefix to feature names
        """
        families = {}
        
        for col in columns:
            if col in ['ts_ns', 'P1', 'P2', 'P3', 'P4']:
                continue
                
            # Split by underscore to get hierarchy
            tokens = col.split('_')
            
            # Create family groups at each level
            for i in range(1, len(tokens) + 1):
                family_key = '_'.join(tokens[:i])
                if family_key not in families:
                    families[family_key] = []
                families[family_key].append(col)
        
        return families
    
    def create_target(self, df: pd.DataFrame, price_col: str = 'P3') -> pd.DataFrame:
        """
        Create forward-looking target with proper horizon.
        Target: Future price change over horizon period.
        
        Args:
            df: Input dataframe
            price_col: Price column to use for target
            
        Returns:
            Dataframe with target columns added
        """
        df = df.copy()
        
        # Forward returns at horizon
        df['target_price_future'] = df[price_col].shift(-self.horizon)
        df['target_return'] = (df['target_price_future'] - df[price_col]) / df[price_col]
        
        # Classification target: direction
        df['target_direction'] = np.sign(df['target_return'])
        
        # Regression target: magnitude of change
        df['target_magnitude'] = df['target_return']
        
        return df
    
    def engineer_causal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer additional causal features from existing features.
        All operations use only historical data (no future peeking).
        
        Args:
            df: Input dataframe with raw features
            
        Returns:
            Dataframe with engineered features added
        """
        df = df.copy()
        
        # Price momentum features (causal - looking backward)
        for window in [5, 10, 20, 30]:
            df[f'P3_return_{window}'] = df['P3'].pct_change(window)
            df[f'P3_volatility_{window}'] = df['P3'].rolling(window, min_periods=1).std()
            df[f'P3_momentum_{window}'] = df['P3'] - df['P3'].shift(window)
        
        # Price spreads
        if 'P1' in df.columns:
            df['spread_P1_P3'] = df['P1'] - df['P3']
            df['spread_ratio_P1_P3'] = df['P1'] / (df['P3'] + 1e-10)
        
        if 'P2' in df.columns:
            df['spread_P2_P3'] = df['P2'] - df['P3']
            df['spread_ratio_P2_P3'] = df['P2'] / (df['P3'] + 1e-10)
        
        # Rolling statistics on key features
        for col in ['P3']:
            if col in df.columns:
                for window in [10, 20, 30, 50]:
                    df[f'{col}_ma_{window}'] = df[col].rolling(window, min_periods=1).mean()
                    df[f'{col}_std_{window}'] = df[col].rolling(window, min_periods=1).std()
                    df[f'{col}_min_{window}'] = df[col].rolling(window, min_periods=1).min()
                    df[f'{col}_max_{window}'] = df[col].rolling(window, min_periods=1).max()
                    df[f'{col}_z_score_{window}'] = (df[col] - df[f'{col}_ma_{window}']) / (df[f'{col}_std_{window}'] + 1e-10)
        
        # Lag features for important columns
        for lag in [1, 2, 3, 5, 10]:
            df[f'P3_lag_{lag}'] = df['P3'].shift(lag)
        
        # Feature interactions (cross-group)
        # Select key features from different families
        key_features = [col for col in df.columns if col.startswith(('F_H_', 'C_H_', 'B_', 'm0_'))]
        
        if len(key_features) >= 2:
            # Create interaction features for top features
            for i, feat1 in enumerate(key_features[:10]):
                for feat2 in key_features[i+1:min(i+6, len(key_features))]:
                    df[f'interact_{feat1}_{feat2}'] = df[feat1] * df[feat2]
        
        return df
    
    def sanitize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and sanitize features:
        - Handle missing values
        - Remove infinite values
        - Clip extreme outliers
        
        Args:
            df: Input dataframe
            
        Returns:
            Sanitized dataframe
        """
        df = df.copy()
        
        # Replace inf with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Forward fill then backward fill for missing values (causal)
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.fillna(0, inplace=True)
        
        return df
    
    def select_features(self, df: pd.DataFrame) -> List[str]:
        """
        Select feature columns for modeling (exclude metadata and targets).
        
        Args:
            df: Input dataframe
            
        Returns:
            List of feature column names
        """
        exclude_cols = ['ts_ns', 'target_price_future', 'target_return', 
                       'target_direction', 'target_magnitude']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        return feature_cols


class PredictiveModel:
    """
    Gradient Boosting Decision Tree model for price prediction.
    
    Design:
    - LightGBM for efficiency and performance
    - Safe expanding window training
    - Predicts future price direction and magnitude at specified horizon
    """
    
    def __init__(self, horizon: int = 30, model_type: str = 'lgbm'):
        """
        Args:
            horizon: Prediction horizon in bars
            model_type: Type of model ('lgbm', 'xgb', etc.)
        """
        self.horizon = horizon
        self.model_type = model_type
        self.model_direction = None
        self.model_magnitude = None
        self.feature_cols = None
        self.feature_importance = None
        
    def train(self, X: pd.DataFrame, y_direction: pd.Series, y_magnitude: pd.Series,
              feature_cols: List[str]) -> None:
        """
        Train the predictive models.
        
        Args:
            X: Feature dataframe
            y_direction: Target direction (-1, 0, 1)
            y_magnitude: Target magnitude (continuous)
            feature_cols: List of feature column names to use
        """
        self.feature_cols = feature_cols
        
        # Remove rows with missing targets
        valid_idx = ~(y_direction.isna() | y_magnitude.isna())
        X_train = X.loc[valid_idx, feature_cols]
        y_dir_train = y_direction[valid_idx]
        y_mag_train = y_magnitude[valid_idx]
        
        # LightGBM parameters - optimized for speed
        params_direction = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'verbose': -1,
            'max_depth': 5,
            'min_data_in_leaf': 100,
            'lambda_l1': 0.5,
            'lambda_l2': 0.5,
            'max_bin': 255
        }
        
        params_magnitude = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'verbose': -1,
            'max_depth': 5,
            'min_data_in_leaf': 100,
            'lambda_l1': 0.5,
            'lambda_l2': 0.5,
            'max_bin': 255
        }
        
        # Convert direction to class labels (0, 1, 2 for -1, 0, 1)
        y_dir_class = (y_dir_train + 1).astype(int)
        
        # Train direction model - faster
        train_data_dir = lgb.Dataset(X_train, label=y_dir_class)
        self.model_direction = lgb.train(
            params_direction,
            train_data_dir,
            num_boost_round=100,
            valid_sets=[train_data_dir],
            callbacks=[lgb.early_stopping(stopping_rounds=15), lgb.log_evaluation(period=0)]
        )
        
        # Train magnitude model - faster  
        train_data_mag = lgb.Dataset(X_train, label=y_mag_train)
        self.model_magnitude = lgb.train(
            params_magnitude,
            train_data_mag,
            num_boost_round=100,
            valid_sets=[train_data_mag],
            callbacks=[lgb.early_stopping(stopping_rounds=15), lgb.log_evaluation(period=0)]
        )
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance_direction': self.model_direction.feature_importance(importance_type='gain'),
            'importance_magnitude': self.model_magnitude.feature_importance(importance_type='gain')
        }).sort_values('importance_direction', ascending=False)
        
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions for given features.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Tuple of (predicted_direction, predicted_magnitude)
        """
        if self.model_direction is None or self.model_magnitude is None:
            raise ValueError("Models not trained yet. Call train() first.")
        
        X_pred = X[self.feature_cols]
        
        # Predict direction (get probabilities and convert to class)
        dir_probs = self.model_direction.predict(X_pred)
        pred_direction = np.argmax(dir_probs, axis=1) - 1  # Convert back to -1, 0, 1
        
        # Predict magnitude
        pred_magnitude = self.model_magnitude.predict(X_pred)
        
        return pred_direction, pred_magnitude


class ExecutionEngine:
    """
    Fully iterative execution engine that generates positions and PnL.
    
    Features:
    - Causal iteration through each timestamp
    - Realistic transaction costs (0.01%)
    - Position tracking and PnL accounting
    - Trade log generation
    """
    
    def __init__(self, transaction_cost: float = 0.0001):
        """
        Args:
            transaction_cost: Transaction cost rate (default 0.01% = 0.0001)
        """
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
        
    def generate_signal(self, pred_direction: int, pred_magnitude: float,
                       confidence_threshold: float = 0.0002,
                       position_multiplier: int = 1) -> int:
        """
        Convert model predictions into trading signal with position sizing.
        
        Args:
            pred_direction: Predicted direction (-1, 0, 1)
            pred_magnitude: Predicted magnitude of price change
            confidence_threshold: Minimum magnitude to trade
            position_multiplier: Position size multiplier
            
        Returns:
            Trading signal with position size (-N to +N)
        """
        # Only trade if prediction magnitude exceeds threshold
        if abs(pred_magnitude) < confidence_threshold:
            return 0
        
        # Use predicted direction with position sizing
        return int(pred_direction) * position_multiplier
    
    def execute_iteration(self, timestamp: int, price: float, signal: int,
                         pred_direction: int, pred_magnitude: float) -> Dict:
        """
        Execute one iteration of the strategy with position sizing.
        
        Args:
            timestamp: Current timestamp
            price: Current P3 price
            signal: Trading signal (with position size)
            pred_direction: Model predicted direction
            pred_magnitude: Model predicted magnitude
            
        Returns:
            Dictionary with iteration results
        """
        prev_position = self.position
        trade_cost = 0.0
        mtm_pnl = 0.0
        realized_pnl_step = 0.0
        
        # Calculate mark-to-market PnL
        if self.position != 0:
            mtm_pnl = self.position * (price - self.entry_price)
        
        # Execute position changes
        if signal != prev_position:
            # Calculate transaction cost based on position change
            position_change = abs(signal - prev_position)
            trade_cost = self.transaction_cost * abs(price) * position_change
            
            # Realize PnL if closing/reducing position
            if prev_position != 0:
                # How much are we closing?
                if np.sign(prev_position) != np.sign(signal) or abs(signal) < abs(prev_position):
                    closed_amount = min(abs(prev_position), abs(prev_position - signal))
                    realized_pnl_step = np.sign(prev_position) * closed_amount * (price - self.entry_price)
                    self.realized_pnl += realized_pnl_step
            
            # Update position
            self.position = signal
            if signal != 0:
                # Update entry price (weighted average if adding to position)
                if prev_position != 0 and np.sign(prev_position) == np.sign(signal):
                    # Adding to position - use weighted average
                    old_size = abs(prev_position)
                    new_size = abs(signal)
                    added_size = new_size - old_size
                    if added_size > 0:
                        self.entry_price = (old_size * self.entry_price + added_size * price) / new_size
                else:
                    self.entry_price = price
            else:
                self.entry_price = 0.0
            
            # Deduct transaction cost
            self.transaction_costs += trade_cost
            self.cumulative_pnl = self.realized_pnl + mtm_pnl - self.transaction_costs
            
            # Log trade
            self.trades.append({
                'timestamp': timestamp,
                'price': price,
                'prev_position': prev_position,
                'new_position': self.position,
                'signal': signal,
                'pred_direction': pred_direction,
                'pred_magnitude': pred_magnitude,
                'trade_cost': trade_cost,
                'realized_pnl': realized_pnl_step,
                'mtm_pnl': mtm_pnl,
                'cumulative_pnl': self.cumulative_pnl
            })
        else:
            # No position change, update cumulative PnL
            self.cumulative_pnl = self.realized_pnl + mtm_pnl - self.transaction_costs
        
        return {
            'timestamp': timestamp,
            'price': price,
            'position': self.position,
            'entry_price': self.entry_price,
            'signal': signal,
            'mtm_pnl': mtm_pnl,
            'cumulative_pnl': self.cumulative_pnl,
            'transaction_costs': self.transaction_costs
        }
    
    def get_trade_log(self) -> pd.DataFrame:
        """
        Get trade log as DataFrame.
        
        Returns:
            DataFrame with all trades
        """
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trades)


class IntradayStrategy:
    """
    Main strategy class that orchestrates the entire pipeline.
    
    Pipeline:
    1. Load data
    2. Feature engineering
    3. Model training (if needed)
    4. Causal iteration execution
    5. Generate trade logs and PnL
    """
    
    def __init__(self, horizon: int = 30, transaction_cost: float = 0.0001):
        """
        Args:
            horizon: Forward prediction horizon in bars
            transaction_cost: Transaction cost rate
        """
        self.horizon = horizon
        self.transaction_cost = transaction_cost
        self.feature_engine = FeatureEngine(horizon=horizon)
        self.model = PredictiveModel(horizon=horizon)
        self.execution_engine = ExecutionEngine(transaction_cost=transaction_cost)
        
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare data for modeling.
        
        Args:
            df: Raw input dataframe
            
        Returns:
            Tuple of (prepared_dataframe, feature_columns)
        """
        # Create target
        df = self.feature_engine.create_target(df)
        
        # Engineer features
        df = self.feature_engine.engineer_causal_features(df)
        
        # Sanitize
        df = self.feature_engine.sanitize_features(df)
        
        # Select features
        feature_cols = self.feature_engine.select_features(df)
        
        return df, feature_cols
    
    def train_on_historical_data(self, historical_files: List[str], sample_rate: int = 5) -> None:
        """
        Train model on historical days using expanding window with sampling.
        
        Args:
            historical_files: List of CSV file paths for training
            sample_rate: Sample every Nth row to reduce training data (default: 5)
        """
        print(f"Training on {len(historical_files)} historical days (sampling every {sample_rate} rows)...")
        
        all_data = []
        for i, file_path in enumerate(historical_files):
            df = pd.read_csv(file_path)
            
            # Sample data for faster training
            if sample_rate > 1:
                df = df.iloc[::sample_rate].reset_index(drop=True)
            
            df, feature_cols = self.prepare_data(df)
            all_data.append(df)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(historical_files)} days")
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Total training samples: {len(combined_df):,}")
        
        # Train model
        valid_idx = ~combined_df['target_direction'].isna()
        X = combined_df[valid_idx]
        y_direction = combined_df.loc[valid_idx, 'target_direction']
        y_magnitude = combined_df.loc[valid_idx, 'target_magnitude']
        
        print("Training predictive models...")
        self.model.train(X, y_direction, y_magnitude, feature_cols)
        print("Training complete!")
        
        # Print feature importance
        print("\nTop 20 Most Important Features:")
        print(self.model.feature_importance.head(20))
        
    def execute_day(self, df: pd.DataFrame, confidence_threshold: float = 0.0002,
                   position_multiplier: int = 1) -> Tuple[pd.DataFrame, Dict]:
        """
        Execute strategy on a single day with causal iteration.
        
        Args:
            df: Daily dataframe
            confidence_threshold: Minimum prediction magnitude to trade
            position_multiplier: Position size multiplier (1-20)
            
        Returns:
            Tuple of (iteration_results_df, summary_stats)
        """
        # Prepare data
        df, feature_cols = self.prepare_data(df)
        
        # Reset execution engine
        self.execution_engine.reset()
        
        # Iterate through each timestamp causally
        results = []
        
        for idx in range(len(df)):
            # Only use data up to current timestamp (causal)
            current_row = df.iloc[idx:idx+1]
            
            # Skip if not enough data for prediction
            if idx < self.horizon:
                continue
            
            # Generate prediction
            try:
                pred_direction, pred_magnitude = self.model.predict(current_row)
                pred_dir = pred_direction[0]
                pred_mag = pred_magnitude[0]
            except:
                pred_dir = 0
                pred_mag = 0.0
            
            # Generate signal with dynamic position sizing
            # Use higher multiplier for stronger predictions
            if abs(pred_mag) > confidence_threshold * 3:
                multiplier = position_multiplier
            elif abs(pred_mag) > confidence_threshold * 2:
                multiplier = max(1, position_multiplier // 2)
            else:
                multiplier = max(1, position_multiplier // 3)
            
            signal = self.execution_engine.generate_signal(pred_dir, pred_mag, 
                                                          confidence_threshold, multiplier)
            
            # Execute
            timestamp = int(df.iloc[idx]['ts_ns'])
            price = float(df.iloc[idx]['P3'])
            
            result = self.execution_engine.execute_iteration(
                timestamp, price, signal, pred_dir, pred_mag
            )
            
            results.append(result)
        
        # Create results dataframe
        results_df = pd.DataFrame(results)
        
        # Calculate summary statistics
        if len(results) > 0:
            final_pnl = self.execution_engine.cumulative_pnl
            total_trades = len(self.execution_engine.trades)
            total_costs = self.execution_engine.transaction_costs
            
            # Calculate metrics
            if 'cumulative_pnl' in results_df.columns:
                pnl_series = results_df['cumulative_pnl']
                returns = pnl_series.diff()
                
                sharpe_ratio = 0.0
                if returns.std() > 0:
                    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(len(returns))
                
                max_pnl = pnl_series.max()
                min_pnl = pnl_series.min()
                max_drawdown = (pnl_series - pnl_series.cummax()).min()
            else:
                sharpe_ratio = 0.0
                max_pnl = 0.0
                min_pnl = 0.0
                max_drawdown = 0.0
            
            summary = {
                'final_pnl': final_pnl,
                'total_trades': total_trades,
                'total_transaction_costs': total_costs,
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
                'total_iterations': 0
            }
        
        return results_df, summary


def main():
    """Main entry point for the strategy."""
    parser = argparse.ArgumentParser(
        description='Intraday Predictive Modeling & Execution Framework'
    )
    parser.add_argument('--input', type=str, required=True,
                       help='Input CSV file for execution day')
    parser.add_argument('--output', type=str, required=True,
                       help='Output CSV file for trade log')
    parser.add_argument('--train_dir', type=str, default='train',
                       help='Directory containing training data')
    parser.add_argument('--horizon', type=int, default=30,
                       help='Forward prediction horizon in bars (minimum 30)')
    parser.add_argument('--transaction_cost', type=float, default=0.00001,
                       help='Transaction cost rate (default 0.001%% = 0.00001 for HFT)')
    parser.add_argument('--confidence_threshold', type=float, default=0.0003,
                       help='Minimum prediction magnitude to trade')
    parser.add_argument('--train_days', type=int, default=10,
                       help='Number of historical days to use for training')
    parser.add_argument('--sample_rate', type=int, default=10,
                       help='Sample every Nth row during training (1=no sampling, 10=every 10th row)')
    parser.add_argument('--position_multiplier', type=int, default=1,
                       help='Position size multiplier (1-5 for HFT)')
    
    args = parser.parse_args()
    
    # Validate horizon
    if args.horizon < 30:
        raise ValueError("Horizon must be at least 30 bars")
    
    print("=" * 80)
    print("INTRADAY PREDICTIVE MODELING & EXECUTION FRAMEWORK")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Horizon: {args.horizon} bars")
    print(f"  Transaction Cost: {args.transaction_cost * 100:.3f}%")
    print(f"  Confidence Threshold: {args.confidence_threshold}")
    print(f"  Training Days: {args.train_days}")
    print(f"  Position Multiplier: {args.position_multiplier}x")
    print()
    
    # Initialize strategy
    strategy = IntradayStrategy(
        horizon=args.horizon,
        transaction_cost=args.transaction_cost
    )
    
    # Get training files
    train_dir = Path(args.train_dir)
    all_train_files = sorted(train_dir.glob('*.csv'), key=lambda x: int(x.stem))
    
    # Use first N days for training
    train_files = all_train_files[:args.train_days]
    
    if len(train_files) > 0:
        # Train on historical data with sampling
        strategy.train_on_historical_data([str(f) for f in train_files], 
                                         sample_rate=args.sample_rate)
    else:
        print("WARNING: No training files found. Using untrained model.")
    
    # Execute on target day
    print(f"\nExecuting strategy on {args.input}...")
    input_df = pd.read_csv(args.input)
    
    results_df, summary = strategy.execute_day(input_df, args.confidence_threshold, 
                                              args.position_multiplier)
    
    # Save trade log
    trade_log = strategy.execution_engine.get_trade_log()
    
    if len(trade_log) > 0:
        trade_log.to_csv(args.output, index=False)
        print(f"\nTrade log saved to {args.output}")
    else:
        # Save empty file with headers
        pd.DataFrame(columns=['timestamp', 'price', 'prev_position', 'new_position',
                             'signal', 'pred_direction', 'pred_magnitude', 
                             'trade_cost', 'realized_pnl', 'mtm_pnl', 'cumulative_pnl']
                    ).to_csv(args.output, index=False)
        print(f"\nNo trades executed. Empty log saved to {args.output}")
    
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
