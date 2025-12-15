#!/usr/bin/env python3
"""
Feature importance analysis tool.

Trains a model and analyzes which features are most predictive.
Helps understand what drives the model's predictions.

Usage:
    python feature_importance_analyzer.py --train_days 30 --output feature_analysis.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from strategy import IntradayStrategy
import json


def analyze_feature_families(importance_df):
    """Analyze feature importance by family groups."""
    
    # Extract family from feature names
    def get_family(feature_name):
        if '_' not in feature_name:
            return 'other'
        parts = feature_name.split('_')
        return parts[0]  # First token
    
    importance_df['family'] = importance_df['feature'].apply(get_family)
    
    # Group by family
    family_importance = importance_df.groupby('family').agg({
        'importance_direction': 'sum',
        'importance_magnitude': 'sum',
        'feature': 'count'
    }).rename(columns={'feature': 'feature_count'})
    
    family_importance = family_importance.sort_values('importance_direction', ascending=False)
    
    return family_importance


def main():
    parser = argparse.ArgumentParser(description='Analyze feature importance')
    parser.add_argument('--train_days', type=int, default=30, help='Number of days for training')
    parser.add_argument('--train_dir', type=str, default='train', help='Training data directory')
    parser.add_argument('--output', type=str, default='feature_analysis.csv', help='Output CSV file')
    parser.add_argument('--horizon', type=int, default=30, help='Prediction horizon')
    parser.add_argument('--top_n', type=int, default=50, help='Number of top features to display')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)
    print(f"Training on {args.train_days} days...")
    print(f"Horizon: {args.horizon} bars")
    print()
    
    # Initialize strategy
    strategy = IntradayStrategy(horizon=args.horizon)
    
    # Get training files
    train_dir = Path(args.train_dir)
    all_files = sorted(train_dir.glob('*.csv'), key=lambda x: int(x.stem))
    train_files = [str(f) for f in all_files[:args.train_days]]
    
    if len(train_files) == 0:
        print("ERROR: No training files found!")
        return
    
    # Train model
    strategy.train_on_historical_data(train_files)
    
    # Get feature importance
    importance_df = strategy.model.feature_importance
    
    print("\n" + "=" * 80)
    print(f"TOP {args.top_n} MOST IMPORTANT FEATURES")
    print("=" * 80)
    print("\nFor Direction Prediction:")
    print(importance_df.head(args.top_n).to_string(index=False))
    
    print("\n" + "=" * 80)
    print("FEATURE FAMILY ANALYSIS")
    print("=" * 80)
    
    family_importance = analyze_feature_families(importance_df)
    print("\nImportance by Feature Family:")
    print(family_importance.to_string())
    
    print("\n" + "=" * 80)
    print("TOP MAGNITUDE PREDICTORS")
    print("=" * 80)
    top_magnitude = importance_df.nlargest(20, 'importance_magnitude')
    print(top_magnitude[['feature', 'importance_magnitude']].to_string(index=False))
    
    # Save detailed results
    importance_df.to_csv(args.output, index=False)
    print(f"\n✓ Detailed feature importance saved to: {args.output}")
    
    # Save family analysis
    family_file = args.output.replace('.csv', '_families.csv')
    family_importance.to_csv(family_file)
    print(f"✓ Family analysis saved to: {family_file}")
    
    # Generate summary statistics
    summary = {
        'total_features': len(importance_df),
        'total_importance_direction': float(importance_df['importance_direction'].sum()),
        'total_importance_magnitude': float(importance_df['importance_magnitude'].sum()),
        'top_10_features_direction': importance_df.head(10)['feature'].tolist(),
        'top_10_features_magnitude': importance_df.nlargest(10, 'importance_magnitude')['feature'].tolist(),
        'feature_families': {
            family: {
                'count': int(row['feature_count']),
                'importance_direction': float(row['importance_direction']),
                'importance_magnitude': float(row['importance_magnitude'])
            }
            for family, row in family_importance.iterrows()
        }
    }
    
    summary_file = args.output.replace('.csv', '_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary statistics saved to: {summary_file}")
    
    # Feature engineering recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    # Check which engineered features are important
    engineered_patterns = ['return_', 'volatility_', 'momentum_', 'spread_', 'ma_', 'std_', 
                          'z_score_', 'lag_', 'interact_', 'min_', 'max_']
    
    engineered_features = importance_df[
        importance_df['feature'].str.contains('|'.join(engineered_patterns))
    ]
    
    if len(engineered_features) > 0:
        print("\n✓ Engineered features are contributing to model performance!")
        print(f"  {len(engineered_features)} engineered features in top predictors")
        print("\nTop engineered features:")
        print(engineered_features.head(10)[['feature', 'importance_direction']].to_string(index=False))
    else:
        print("\n⚠ Few engineered features in top predictors")
        print("  Consider creating more derived features")
    
    # Identify dominant families
    top_family = family_importance.index[0]
    top_family_pct = (family_importance.iloc[0]['importance_direction'] / 
                     family_importance['importance_direction'].sum() * 100)
    
    print(f"\n✓ Dominant feature family: {top_family} ({top_family_pct:.1f}% of importance)")
    
    if top_family_pct > 50:
        print("  ⚠ Model heavily relies on one family - consider diversification")
    else:
        print("  ✓ Good feature diversity across families")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
