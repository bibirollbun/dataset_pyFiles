# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def diagnose_data_structure(train_path='train.parquet', test_path='test.parquet'):
    """
    Quick diagnostic to understand the data structure
    """
    print("=== DATA STRUCTURE DIAGNOSTIC ===\n")
    
    # Load a small sample first
    print("Loading sample data...")
    train_df = pd.read_parquet(train_path)
    
    print(f"Training data shape: {train_df.shape}")
    print(f"\nColumn names (first 20):")
    for i, col in enumerate(train_df.columns[:20]):
        print(f"  {i}: {col}")
    
    print(f"\n... and {len(train_df.columns) - 20} more columns")
    
    # Check for timestamp column
    timestamp_cols = [col for col in train_df.columns if 'time' in col.lower() or 'date' in col.lower()]
    print(f"\nPotential timestamp columns: {timestamp_cols}")
    
    # Check data types
    print("\nData types summary:")
    dtype_counts = train_df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} columns")
    
    # Check for label column
    if 'label' in train_df.columns:
        print(f"\nLabel column found!")
        print(f"  Label range: [{train_df['label'].min():.4f}, {train_df['label'].max():.4f}]")
        print(f"  Label mean: {train_df['label'].mean():.4f}")
        print(f"  Label std: {train_df['label'].std():.4f}")
    else:
        print("\nWarning: 'label' column not found!")
    
    # Check market features
    market_features = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
    found_market_features = [f for f in market_features if f in train_df.columns]
    print(f"\nMarket features found: {found_market_features}")
    
    # Check anonymous features
    anon_features = [col for col in train_df.columns if col.startswith('X_')]
    print(f"\nAnonymous features (X_*): {len(anon_features)} found")
    
    # Sample first few rows
    print("\nFirst 5 rows of key columns:")
    display_cols = found_market_features[:3] + ['label'] if 'label' in train_df.columns else found_market_features[:3]
    if display_cols:
        print(train_df[display_cols].head())
    
    # Check for missing values
    print("\nMissing values summary:")
    missing_counts = train_df.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0]
    if len(missing_cols) > 0:
        print(f"  Columns with missing values: {len(missing_cols)}")
        print(f"  Top 5 columns with most missing values:")
        for col, count in missing_cols.nlargest(5).items():
            print(f"    {col}: {count} ({count/len(train_df)*100:.1f}%)")
    else:
        print("  No missing values found!")
    
    # Quick temporal analysis without timestamp
    print("\n=== TEMPORAL ANALYSIS (based on row order) ===")
    
    # Plot label evolution
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 1. Label values over time
    ax = axes[0]
    sample_size = min(len(train_df), 50000)  # Plot subset for efficiency
    indices = np.linspace(0, len(train_df)-1, sample_size).astype(int)
    ax.plot(indices, train_df['label'].iloc[indices], alpha=0.6, linewidth=0.5)
    ax.set_title('Label Evolution (assuming chronological order)')
    ax.set_xlabel('Row Index')
    ax.set_ylabel('Label Value')
    ax.grid(True, alpha=0.3)
    
    # 2. Rolling statistics
    ax = axes[1]
    window = 10000  # 10k minute window
    rolling_mean = train_df['label'].rolling(window=window, min_periods=1).mean()
    rolling_std = train_df['label'].rolling(window=window, min_periods=1).std()
    
    ax.plot(indices, rolling_mean.iloc[indices], label='Rolling Mean', alpha=0.8)
    ax.fill_between(indices, 
                     (rolling_mean - 2*rolling_std).iloc[indices],
                     (rolling_mean + 2*rolling_std).iloc[indices],
                     alpha=0.2, label='±2 Std Dev')
    ax.set_title(f'Label Rolling Statistics ({window} row window)')
    ax.set_xlabel('Row Index')
    ax.set_ylabel('Label Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Volume evolution (if available)
    ax = axes[2]
    if 'volume' in train_df.columns:
        volume_rolling = train_df['volume'].rolling(window=window, min_periods=1).mean()
        ax.plot(indices, volume_rolling.iloc[indices], color='green', alpha=0.8)
        ax.set_title(f'Volume Evolution ({window} row rolling average)')
        ax.set_xlabel('Row Index')
        ax.set_ylabel('Volume')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Volume data not available', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig('data_diagnostic_plots.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Correlation analysis
    print("\n=== QUICK CORRELATION ANALYSIS ===")
    
    # Sample for efficiency
    sample_df = train_df.sample(n=min(10000, len(train_df)), random_state=42)
    
    # Correlations with label
    if 'label' in sample_df.columns:
        # Market features correlation
        if found_market_features:
            print("\nMarket features correlation with label:")
            for feature in found_market_features:
                corr = sample_df[feature].corr(sample_df['label'])
                print(f"  {feature}: {corr:.4f}")
        
        # Top anonymous features by correlation
        if anon_features:
            print("\nTop 10 anonymous features by absolute correlation with label:")
            anon_corrs = {}
            for feature in anon_features[:100]:  # Check first 100 for efficiency
                corr = sample_df[feature].corr(sample_df['label'])
                anon_corrs[feature] = abs(corr)
            
            top_features = sorted(anon_corrs.items(), key=lambda x: x[1], reverse=True)[:10]
            for feature, corr in top_features:
                actual_corr = sample_df[feature].corr(sample_df['label'])
                print(f"  {feature}: {actual_corr:.4f} (abs: {corr:.4f})")
    
    print("\n=== DIAGNOSTIC COMPLETE ===")
    return train_df

# Run diagnostic
if __name__ == "__main__":
    train_path = '/kaggle/input/drw-crypto-market-prediction/train.parquet'
    df = diagnose_data_structure(train_path)


# Load and check the test set size
test_df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')
print(f"Test set shape: {test_df.shape}")
print(f"Number of test records: {len(test_df)}")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import gc
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')

class MemoryEfficientProphetAnalyzer:
    """
    Memory-efficient Prophet analyzer for large crypto datasets.
    """
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.regime_df = None
        self.changepoints = None
        
    def load_minimal_data(self, columns=['timestamp', 'label', 'volume']):
        """
        Load only essential columns to save memory.
        """
        print("Loading minimal data...")
        
        # Read only specific columns
        df = pd.read_parquet(self.data_path, columns=columns)
        
        # Ensure timestamp exists
        if 'timestamp' not in df.columns:
            if df.index.name == 'timestamp':
                df = df.reset_index()
            else:
                start_date = pd.Timestamp('2023-03-01')
                df['timestamp'] = pd.date_range(start=start_date, periods=len(df), freq='T')
        
        print(f"Loaded {len(df):,} rows with columns: {list(df.columns)}")
        
        # Clean data
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(df[col].median())
        
        return df
    
    def aggressive_resample_for_prophet(self, df, resample_freq='4H'):
        """
        Aggressively resample data to reduce memory usage.
        """
        print(f"Resampling data to {resample_freq} intervals...")
        
        # Resample to reduce data size
        prophet_df = pd.DataFrame({
            'ds': df['timestamp'],
            'y': df['label']
        })
        
        # Set index and resample
        prophet_df = prophet_df.set_index('ds').resample(resample_freq).agg({
            'y': ['mean', 'std', 'min', 'max']
        })
        
        # Flatten column names
        prophet_df.columns = ['_'.join(col).strip() for col in prophet_df.columns]
        
        # Use mean for Prophet, keep others for analysis
        prophet_input = pd.DataFrame({
            'ds': prophet_df.index,
            'y': prophet_df['y_mean']
        }).reset_index(drop=True)
        
        # Remove NaN values
        prophet_input = prophet_input.dropna()
        
        print(f"Resampled to {len(prophet_input):,} data points")
        
        # Store additional stats for later use
        self.resampled_stats = prophet_df
        
        # Clean up
        del prophet_df
        gc.collect()
        
        return prophet_input
    
    def detect_regimes_memory_efficient(self, df, resample='4H', n_changepoints=10):
        """
        Detect regimes with minimal memory usage.
        """
        print("\nDetecting regimes with memory optimization...")
        
        # Resample data
        prophet_df = self.aggressive_resample_for_prophet(df, resample)
        
        if len(prophet_df) < 20:
            print("Too few data points after resampling")
            return None, None, []
        
        # Configure Prophet for minimal memory usage
        model = Prophet(
            changepoint_prior_scale=0.05,
            n_changepoints=min(n_changepoints, len(prophet_df) // 20),
            yearly_seasonality=False,
            weekly_seasonality=False,  # Disable to save memory
            daily_seasonality=False,   # Disable to save memory
            seasonality_mode='additive',  # Simpler than multiplicative
            mcmc_samples=0,  # Disable MCMC for memory
            interval_width=0.8  # Reduce from 0.95 to save memory
        )
        
        # Fit model
        print("Fitting Prophet model...")
        model.fit(prophet_df)
        
        # Generate minimal forecast
        future = model.make_future_dataframe(periods=0)
        forecast = model.predict(future)
        
        # Extract changepoints and clean up
        changepoints = model.changepoints.copy()
        
        # Map changepoints to original data indices
        changepoint_indices = []
        for cp in changepoints:
            # Find closest timestamp in original data
            closest_idx = np.argmin(np.abs(df['timestamp'] - pd.Timestamp(cp)))
            changepoint_indices.append(closest_idx)
        
        changepoint_indices = sorted(list(set(changepoint_indices)))
        
        # Clean up Prophet objects
        del prophet_df
        gc.collect()
        
        return model, forecast, changepoint_indices
    
    def analyze_regimes_minimal(self, df, changepoint_indices):
        """
        Analyze regimes with minimal memory footprint.
        """
        if not changepoint_indices:
            return pd.DataFrame()
        
        print("Analyzing regime characteristics...")
        
        regime_boundaries = [0] + changepoint_indices + [len(df)]
        regimes = []
        
        for i in range(len(regime_boundaries) - 1):
            start = regime_boundaries[i]
            end = regime_boundaries[i + 1]
            
            # Process in chunks if regime is large
            if end - start > 100000:
                # Sample the regime instead of using all data
                sample_size = min(50000, end - start)
                indices = np.random.choice(range(start, end), sample_size, replace=False)
                regime_data = df.iloc[sorted(indices)]
            else:
                regime_data = df.iloc[start:end]
            
            # Calculate basic statistics
            regime_info = {
                'regime_id': i,
                'start_date': df['timestamp'].iloc[start],
                'end_date': df['timestamp'].iloc[end-1] if end < len(df) else df['timestamp'].iloc[-1],
                'duration_hours': (end - start) / 60,
                'n_samples': end - start,
                'label_mean': regime_data['label'].mean(),
                'label_std': regime_data['label'].std(),
                'label_min': regime_data['label'].min(),
                'label_max': regime_data['label'].max()
            }
            
            # Add volume if available
            if 'volume' in regime_data.columns:
                regime_info['volume_mean'] = regime_data['volume'].mean()
            
            regimes.append(regime_info)
            
            # Clean up
            del regime_data
            gc.collect()
        
        return pd.DataFrame(regimes)
    
    def create_memory_efficient_plot(self, df, model, forecast, changepoint_indices, regime_df):
        """
        Create visualizations with memory efficiency in mind.
        """
        print("Creating memory-efficient visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Downsampled regime plot
        ax = axes[0, 0]
        
        # Downsample for plotting
        plot_sample = min(10000, len(df))
        if len(df) > plot_sample:
            plot_indices = np.linspace(0, len(df)-1, plot_sample, dtype=int)
            plot_df = df.iloc[plot_indices]
        else:
            plot_df = df
        
        ax.plot(plot_df['timestamp'], plot_df['label'], 'b-', alpha=0.6, linewidth=1)
        
        # Add changepoints
        for idx in changepoint_indices:
            if idx < len(df):
                ax.axvline(x=df['timestamp'].iloc[idx], color='red', 
                          linestyle='--', alpha=0.7, linewidth=2)
        
        ax.set_title('Market Regimes (Downsampled)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Label')
        ax.grid(True, alpha=0.3)
        
        # 2. Regime summary
        ax = axes[0, 1]
        
        if not regime_df.empty:
            # Simple bar chart of regime durations
            ax.bar(regime_df['regime_id'], regime_df['duration_hours'], 
                   color='skyblue', edgecolor='navy', linewidth=1)
            ax.set_title('Regime Durations', fontsize=12, fontweight='bold')
            ax.set_xlabel('Regime ID')
            ax.set_ylabel('Duration (hours)')
            ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Volatility by regime
        ax = axes[1, 0]
        
        if not regime_df.empty and 'label_std' in regime_df:
            ax.bar(regime_df['regime_id'], regime_df['label_std'], 
                   color='coral', edgecolor='darkred', linewidth=1)
            ax.set_title('Regime Volatility', fontsize=12, fontweight='bold')
            ax.set_xlabel('Regime ID')
            ax.set_ylabel('Label Std Dev')
            ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Summary text
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_text = "REGIME SUMMARY\n" + "="*25 + "\n\n"
        
        if not regime_df.empty:
            summary_text += f"Total Regimes: {len(regime_df)}\n"
            summary_text += f"Avg Duration: {regime_df['duration_hours'].mean():.1f} hours\n"
            summary_text += f"Max Duration: {regime_df['duration_hours'].max():.1f} hours\n\n"
            
            if 'label_std' in regime_df:
                summary_text += f"Volatility Range:\n"
                summary_text += f"  Min: {regime_df['label_std'].min():.4f}\n"
                summary_text += f"  Max: {regime_df['label_std'].max():.4f}\n\n"
            
            # Current regime
            current = regime_df.iloc[-1]
            summary_text += f"Current Regime (R{current['regime_id']}):\n"
            summary_text += f"  Duration: {current['duration_hours']:.1f}h\n"
            summary_text += f"  Mean: {current['label_mean']:.4f}\n"
            summary_text += f"  Volatility: {current['label_std']:.4f}\n"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
               fontsize=11, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('prophet_regime_memory_efficient.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Clean up
        plt.close()
        gc.collect()
    
    def export_regime_labels(self, df, changepoint_indices, output_file='regime_labels.parquet'):
        """
        Export regime labels efficiently.
        """
        print("Exporting regime labels...")
        
        regime_boundaries = [0] + changepoint_indices + [len(df)]
        
        # Create regime labels array
        regime_labels = np.zeros(len(df), dtype=np.int8)  # Use int8 to save memory
        
        for i in range(len(regime_boundaries) - 1):
            start = regime_boundaries[i]
            end = regime_boundaries[i + 1]
            regime_labels[start:end] = i
        
        # Create minimal output dataframe
        output_df = pd.DataFrame({
            'timestamp': df['timestamp'],
            'regime_id': regime_labels
        })
        
        # Add regime statistics
        if self.regime_df is not None:
            regime_stats = self.regime_df[['regime_id', 'label_mean', 'label_std']].copy()
            output_df = output_df.merge(regime_stats, on='regime_id', how='left')
        
        # Save as parquet for efficiency
        output_df.to_parquet(output_file, compression='snappy')
        print(f"Saved regime labels to {output_file}")
        
        return output_df


def run_memory_efficient_analysis(data_path='/kaggle/input/drw-crypto-market-prediction/train.parquet'):
    """
    Run memory-efficient Prophet analysis.
    """
    print("="*60)
    print("MEMORY-EFFICIENT PROPHET REGIME ANALYSIS")
    print("="*60)
    
    # Initialize analyzer
    analyzer = MemoryEfficientProphetAnalyzer(data_path)
    
    # Load minimal data
    df = analyzer.load_minimal_data(columns=['timestamp', 'label', 'volume'])
    
    # Run regime detection
    model, forecast, changepoints = analyzer.detect_regimes_memory_efficient(
        df, 
        resample='4H',  # Aggressive resampling
        n_changepoints=10  # Fewer changepoints
    )
    
    if not changepoints:
        print("No changepoints detected")
        return analyzer, pd.DataFrame()
    
    print(f"\nDetected {len(changepoints)} changepoints")
    
    # Analyze regimes
    regime_df = analyzer.analyze_regimes_minimal(df, changepoints)
    analyzer.regime_df = regime_df
    analyzer.changepoints = changepoints
    
    # Create visualization
    analyzer.create_memory_efficient_plot(df, model, forecast, changepoints, regime_df)
    
    # Export labels
    regime_labels_df = analyzer.export_regime_labels(df, changepoints)
    
    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nKey Results:")
    print(f"- {len(regime_df)} regimes detected")
    print(f"- Average regime duration: {regime_df['duration_hours'].mean():.1f} hours")
    print(f"- Output saved to: prophet_regime_memory_efficient.png")
    print(f"- Regime labels saved to: regime_labels.parquet")
    
    # Clean up
    del df
    del model
    del forecast
    gc.collect()
    
    return analyzer, regime_df


# Alternative: Ultra-light version for extreme memory constraints
def ultra_light_prophet_analysis(data_path):
    """
    Ultra-light version that processes data in chunks.
    """
    print("Running ultra-light Prophet analysis...")
    
    # Read only timestamp and label
    df = pd.read_parquet(data_path, columns=['label'])
    
    # Create timestamp if needed
    if 'timestamp' not in df.columns:
        df['timestamp'] = pd.date_range('2023-03-01', periods=len(df), freq='T')
    
    # Extreme downsampling - daily averages
    daily_df = df.set_index('timestamp').resample('D').agg({
        'label': ['mean', 'std', 'count']
    })
    
    daily_df.columns = ['label_mean', 'label_std', 'count']
    daily_df = daily_df.reset_index()
    
    # Simple Prophet model
    prophet_df = pd.DataFrame({
        'ds': daily_df['timestamp'],
        'y': daily_df['label_mean']
    }).dropna()
    
    model = Prophet(
        changepoint_prior_scale=0.1,
        n_changepoints=5,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False
    )
    
    model.fit(prophet_df)
    
    # Get changepoints
    changepoints = model.changepoints
    
    print(f"Found {len(changepoints)} major regime changes")
    print("\nChangepoint dates:")
    for cp in changepoints:
        print(f"  - {cp.strftime('%Y-%m-%d')}")
    
    # Simple plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(daily_df['timestamp'], daily_df['label_mean'], 'b-', alpha=0.7)
    
    for cp in changepoints:
        ax.axvline(x=cp, color='red', linestyle='--', alpha=0.7)
    
    ax.set_title('Daily Average Regimes')
    ax.set_xlabel('Date')
    ax.set_ylabel('Label (Daily Average)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ultra_light_regimes.png', dpi=100)
    plt.show()
    
    return model, changepoints


if __name__ == "__main__":
    # Try memory-efficient version first
    try:
        analyzer, regime_df = run_memory_efficient_analysis()
    except MemoryError:
        print("\nMemory error encountered. Trying ultra-light version...")
        model, changepoints = ultra_light_prophet_analysis(
            '/kaggle/input/drw-crypto-market-prediction/train.parquet'
        )


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import curve_fit
import lightgbm as lgb
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set professional visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")

class TemporalValidationAnalyzer:
    """
    Framework for analyzing temporal stability and performance degradation
    in predictive models across different time periods.
    """
    
    def __init__(self, data_path='train.parquet'):
        self.data_path = data_path
        self.df = None
        self.results = {}
        self.feature_importance_evolution = {}
        
    def load_and_prepare_data(self):
        """
        Load data with robust preprocessing for temporal analysis.
        """
        print("Loading and preparing data for temporal validation analysis...")
        
        self.df = pd.read_parquet(self.data_path)
        
        # Handle timestamp creation if needed
        if 'timestamp' not in self.df.columns and self.df.index.name == 'timestamp':
            self.df = self.df.reset_index()
        elif 'timestamp' not in self.df.columns:
            start_date = pd.Timestamp('2023-03-01')
            self.df['timestamp'] = pd.date_range(start=start_date, periods=len(self.df), freq='T')
        
        # Handle extreme values and infinities
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'label':
                self.df[col] = self.df[col].replace([np.inf, -np.inf], np.nan)
                upper_cap = self.df[col].quantile(0.9999)
                lower_cap = self.df[col].quantile(0.0001)
                self.df[col] = self.df[col].clip(lower=lower_cap, upper=upper_cap)
                self.df[col] = self.df[col].fillna(self.df[col].median())
        
        print(f"Data prepared. Shape: {self.df.shape}")
        print(f"Date range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
        
        return self.df
    
    def perform_temporal_cross_validation(self, window_size=50000, n_splits=10):
        """
        Perform sophisticated temporal cross-validation to analyze performance degradation.
        """
        print(f"\nPerforming temporal cross-validation with {window_size:,} row windows...")
        
        # Define feature columns
        feature_cols = [col for col in self.df.columns if col not in ['timestamp', 'label']]
        
        # Initialize results storage
        validation_results = []
        
        # Generate temporal splits
        total_rows = len(self.df)
        step_size = (total_rows - 2 * window_size) // (n_splits - 1)
        
        for split_idx in range(n_splits):
            print(f"\nProcessing split {split_idx + 1}/{n_splits}")
            
            # Define training window (recent data)
            train_end = total_rows - (split_idx * step_size)
            train_start = train_end - window_size
            
            if train_start < 0:
                continue
            
            # Extract training data
            X_train = self.df[feature_cols].iloc[train_start:train_end].values
            y_train = self.df['label'].iloc[train_start:train_end].values
            train_timestamp = self.df['timestamp'].iloc[train_end - 1]
            
            # Scale features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            # Train model
            model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Store feature importances
            feature_importances = model.feature_importances_
            
            # Test on multiple historical windows
            for lookback_periods in [0, 1, 2, 3, 4, 5]:
                test_end = train_start - (lookback_periods * window_size)
                test_start = test_end - 10000  # Test on 10k rows
                
                if test_start < 0:
                    continue
                
                # Extract test data
                X_test = self.df[feature_cols].iloc[test_start:test_end].values
                y_test = self.df['label'].iloc[test_start:test_end].values
                test_timestamp = self.df['timestamp'].iloc[test_end - 1]
                
                # Scale test features
                X_test_scaled = scaler.transform(X_test)
                
                # Generate predictions
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                
                # Calculate correlations
                pearson_corr, _ = pearsonr(y_test, y_pred)
                spearman_corr, _ = spearmanr(y_test, y_pred)
                
                # Calculate distribution shift metrics
                train_feature_means = X_train.mean(axis=0)
                test_feature_means = X_test.mean(axis=0)
                feature_drift = np.mean(np.abs(train_feature_means - test_feature_means) / 
                                      (np.abs(train_feature_means) + 1e-8))
                
                # Store results
                validation_results.append({
                    'split_idx': split_idx,
                    'train_timestamp': train_timestamp,
                    'test_timestamp': test_timestamp,
                    'lookback_periods': lookback_periods,
                    'time_gap_days': (train_timestamp - test_timestamp).days,
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'pearson_correlation': pearson_corr,
                    'spearman_correlation': spearman_corr,
                    'feature_drift': feature_drift,
                    'feature_importances': feature_importances
                })
        
        self.results['temporal_validation'] = pd.DataFrame(validation_results)
        return self.results['temporal_validation']
    
    def analyze_reverse_temporal_validation(self, window_size=50000):
        """
        Train on historical data and test on recent data to show inverse relationship.
        """
        print("\nPerforming reverse temporal validation...")
        
        feature_cols = [col for col in self.df.columns if col not in ['timestamp', 'label']]
        reverse_results = []
        
        # Define recent test window
        test_end = len(self.df)
        test_start = test_end - 10000
        
        X_test = self.df[feature_cols].iloc[test_start:test_end].values
        y_test = self.df['label'].iloc[test_start:test_end].values
        test_timestamp = self.df['timestamp'].iloc[test_end - 1]
        
        # Train on progressively older data
        for lookback_offset in range(0, 6):
            train_end = test_start - (lookback_offset * 20000)
            train_start = train_end - window_size
            
            if train_start < 0:
                continue
            
            # Extract training data
            X_train = self.df[feature_cols].iloc[train_start:train_end].values
            y_train = self.df['label'].iloc[train_start:train_end].values
            train_timestamp = self.df['timestamp'].iloc[train_end - 1]
            
            # Scale and train
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            pearson_corr, _ = pearsonr(y_test, y_pred)
            
            reverse_results.append({
                'train_timestamp': train_timestamp,
                'test_timestamp': test_timestamp,
                'time_gap_days': (test_timestamp - train_timestamp).days,
                'pearson_correlation': pearson_corr,
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
            })
        
        self.results['reverse_validation'] = pd.DataFrame(reverse_results)
        return self.results['reverse_validation']
    
    def analyze_feature_stability(self, window_size=50000, top_n_features=20):
        """
        Analyze how feature importance rankings change over time.
        """
        print("\nAnalyzing feature importance stability over time...")
        
        feature_cols = [col for col in self.df.columns if col not in ['timestamp', 'label']]
        importance_snapshots = []
        
        # Take snapshots at different time periods
        n_snapshots = 8
        step_size = (len(self.df) - window_size) // (n_snapshots - 1)
        
        for i in range(n_snapshots):
            window_end = len(self.df) - (i * step_size)
            window_start = window_end - window_size
            
            if window_start < 0:
                continue
            
            try:
                # Extract data
                X = self.df[feature_cols].iloc[window_start:window_end].copy()
                y = self.df['label'].iloc[window_start:window_end].copy()
                timestamp = self.df['timestamp'].iloc[window_end - 1]
                
                # More robust data cleaning
                # 1. Replace infinities with NaN
                X = X.replace([np.inf, -np.inf], np.nan)
                
                # 2. For each column, fill NaN with column median or mean
                for col in X.columns:
                    if X[col].isna().all():
                        # If entire column is NaN, fill with 0
                        X[col] = 0
                    else:
                        # Use median for robust filling
                        median_val = X[col].median()
                        if pd.isna(median_val):
                            # If median is NaN, use mean
                            mean_val = X[col].mean()
                            if pd.isna(mean_val):
                                # If both are NaN, use 0
                                X[col] = 0
                            else:
                                X[col] = X[col].fillna(mean_val)
                        else:
                            X[col] = X[col].fillna(median_val)
                
                # 3. Handle label NaNs
                if y.isna().any():
                    # Remove rows where label is NaN
                    mask = ~y.isna()
                    X = X[mask]
                    y = y[mask]
                    
                    if len(y) < 100:  # Skip if too few samples remain
                        print(f"  Skipping window {i} due to insufficient non-NaN labels")
                        continue
                
                # 4. Final check for remaining NaNs or infinities
                if X.isna().any().any() or np.isinf(X.values).any():
                    # Do one more aggressive cleaning
                    X = X.fillna(0)
                    X = X.replace([np.inf, -np.inf], 0)
                
                # Convert to numpy arrays
                X_array = X.values.astype(np.float32)
                y_array = y.values.astype(np.float32)
                
                # Use HistGradientBoostingRegressor which handles edge cases better
                model = HistGradientBoostingRegressor(
                    max_iter=50,
                    max_depth=5,
                    random_state=42,
                    verbose=0,
                    learning_rate=0.1,
                    min_samples_leaf=20  # Increase for stability
                )
                
                # Fit with error handling
                model.fit(X_array, y_array)
                
                # Get feature importances
                importances = model.feature_importances_
                
                # Handle case where all importances are 0 or NaN
                if np.all(importances == 0) or np.all(np.isnan(importances)):
                    print(f"  Warning: All feature importances are 0 or NaN for window {i}")
                    # Create random importances as fallback
                    importances = np.random.rand(len(feature_cols))
                    importances = importances / importances.sum()
                
                top_indices = np.argsort(importances)[-top_n_features:][::-1]
                
                # Store snapshot
                snapshot = {
                    'timestamp': timestamp,
                    'window_position': i
                }
                
                for rank, idx in enumerate(top_indices):
                    snapshot[f'rank_{rank+1}_feature'] = feature_cols[idx]
                    snapshot[f'rank_{rank+1}_importance'] = importances[idx]
                
                importance_snapshots.append(snapshot)
                print(f"  Successfully processed window {i}")
                
            except Exception as e:
                print(f"  Error processing window {i}: {str(e)}")
                continue
        
        if not importance_snapshots:
            print("  Warning: No feature importance snapshots could be created")
            # Create dummy data to avoid downstream errors
            self.results['feature_stability'] = pd.DataFrame()
        else:
            self.results['feature_stability'] = pd.DataFrame(importance_snapshots)
            print(f"  Created {len(importance_snapshots)} feature importance snapshots")
        
        return self.results['feature_stability']
    
    def calculate_distribution_shifts(self, window_size=20000):
        """
        Calculate statistical distribution shifts between time periods.
        """
        print("\nCalculating distribution shifts between time periods...")
        
        feature_cols = [col for col in self.df.columns if col not in ['timestamp', 'label']]
        # Sample features for efficiency
        sample_features = np.random.choice(feature_cols, size=min(50, len(feature_cols)), replace=False)
        
        distribution_results = []
        
        # Compare recent window with historical windows
        recent_end = len(self.df)
        recent_start = recent_end - window_size
        recent_data = self.df[sample_features].iloc[recent_start:recent_end]
        
        for lookback_periods in range(0, 10):
            historical_end = recent_start - (lookback_periods * window_size)
            historical_start = historical_end - window_size
            
            if historical_start < 0:
                continue
            
            historical_data = self.df[sample_features].iloc[historical_start:historical_end]
            
            # Calculate KS statistics for each feature
            ks_statistics = []
            for feature in sample_features:
                ks_stat, _ = stats.ks_2samp(
                    recent_data[feature].values,
                    historical_data[feature].values
                )
                ks_statistics.append(ks_stat)
            
            # Calculate summary statistics
            distribution_results.append({
                'lookback_periods': lookback_periods,
                'time_gap_days': lookback_periods * (window_size / (24 * 60)),
                'mean_ks_statistic': np.mean(ks_statistics),
                'max_ks_statistic': np.max(ks_statistics),
                'pct_significant_shifts': np.mean([ks > 0.1 for ks in ks_statistics])
            })
        
        self.results['distribution_shifts'] = pd.DataFrame(distribution_results)
        return self.results['distribution_shifts']
    
    def create_comprehensive_visualizations(self):
        """
        Create professional visualizations demonstrating temporal dynamics.
        """
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Performance Degradation Over Time
        ax1 = plt.subplot(4, 2, 1)
        if 'temporal_validation' in self.results and not self.results['temporal_validation'].empty:
            df_tv = self.results['temporal_validation']
            
            # Group by lookback periods
            performance_by_gap = df_tv.groupby('lookback_periods').agg({
                'pearson_correlation': ['mean', 'std'],
                'rmse': ['mean', 'std']
            }).reset_index()
            
            x = performance_by_gap['lookback_periods'] * 50000 / (24 * 60)  # Convert to days
            
            # Plot correlation degradation
            ax1.errorbar(x, 
                        performance_by_gap['pearson_correlation']['mean'],
                        yerr=performance_by_gap['pearson_correlation']['std'],
                        marker='o', markersize=8, capsize=5, capthick=2,
                        label='Pearson Correlation', linewidth=2)
            
            ax1.set_xlabel('Time Gap (Days)', fontsize=12)
            ax1.set_ylabel('Correlation', fontsize=12)
            ax1.set_title('Model Performance Degradation Over Time', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Add annotation
            if len(x) > 1:
                degradation_rate = (performance_by_gap['pearson_correlation']['mean'].iloc[0] - 
                                  performance_by_gap['pearson_correlation']['mean'].iloc[-1]) / x.iloc[-1]
                ax1.text(0.05, 0.05, f'Degradation Rate: {degradation_rate:.4f} per day',
                        transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 2. RMSE Increase Over Time
        ax2 = plt.subplot(4, 2, 2)
        if 'temporal_validation' in self.results and not self.results['temporal_validation'].empty:
            ax2.errorbar(x, 
                        performance_by_gap['rmse']['mean'],
                        yerr=performance_by_gap['rmse']['std'],
                        marker='s', markersize=8, capsize=5, capthick=2,
                        color='red', label='RMSE', linewidth=2)
            
            ax2.set_xlabel('Time Gap (Days)', fontsize=12)
            ax2.set_ylabel('RMSE', fontsize=12)
            ax2.set_title('Prediction Error Growth Over Time', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        # 3. Reverse Validation Results
        ax3 = plt.subplot(4, 2, 3)
        if 'reverse_validation' in self.results and not self.results['reverse_validation'].empty:
            df_rv = self.results['reverse_validation']
            
            ax3.plot(df_rv['time_gap_days'], df_rv['pearson_correlation'], 
                    'o-', markersize=8, linewidth=2, color='green',
                    label='Historical → Recent')
            
            # Add forward validation for comparison if available
            if 'temporal_validation' in self.results and not self.results['temporal_validation'].empty:
                forward_avg = df_tv.groupby('time_gap_days')['pearson_correlation'].mean()
                ax3.plot(forward_avg.index, forward_avg.values, 
                        's-', markersize=8, linewidth=2, color='blue',
                        label='Recent → Historical')
            
            ax3.set_xlabel('Time Gap (Days)', fontsize=12)
            ax3.set_ylabel('Correlation', fontsize=12)
            ax3.set_title('Bidirectional Temporal Validation Comparison', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        
        # 4. Distribution Shift Analysis
        ax4 = plt.subplot(4, 2, 4)
        if 'distribution_shifts' in self.results and not self.results['distribution_shifts'].empty:
            df_ds = self.results['distribution_shifts']
            
            ax4.plot(df_ds['time_gap_days'], df_ds['mean_ks_statistic'], 
                    'o-', markersize=8, linewidth=2, color='purple')
            
            # Add significance threshold
            ax4.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, 
                       label='Significance Threshold')
            
            ax4.set_xlabel('Time Gap (Days)', fontsize=12)
            ax4.set_ylabel('Mean KS Statistic', fontsize=12)
            ax4.set_title('Feature Distribution Shifts Over Time', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            
            # Add percentage annotation
            ax4_twin = ax4.twinx()
            ax4_twin.plot(df_ds['time_gap_days'], df_ds['pct_significant_shifts'] * 100,
                         's-', markersize=6, linewidth=1, color='orange', alpha=0.7)
            ax4_twin.set_ylabel('% Features with Significant Shift', fontsize=12, color='orange')
            ax4_twin.tick_params(axis='y', labelcolor='orange')
        
        # 5. Feature Importance Evolution Heatmap
        ax5 = plt.subplot(4, 2, (5, 6))
        if 'feature_stability' in self.results and not self.results['feature_stability'].empty:
            df_fs = self.results['feature_stability']
            
            # Create matrix of top features over time
            n_ranks = 10
            feature_matrix = []
            timestamps = []
            
            for _, row in df_fs.iterrows():
                features = [row[f'rank_{i}_feature'] for i in range(1, n_ranks+1) 
                          if f'rank_{i}_feature' in row]
                if features:
                    feature_matrix.append(features)
                    timestamps.append(row['timestamp'])
            
            if feature_matrix:
                # Convert to numerical matrix for visualization
                all_features = list(set([f for features in feature_matrix for f in features]))
                numerical_matrix = np.zeros((len(feature_matrix), len(all_features)))
                
                for i, features in enumerate(feature_matrix):
                    for j, feature in enumerate(features):
                        if feature in all_features:
                            idx = all_features.index(feature)
                            numerical_matrix[i, idx] = n_ranks - j  # Higher rank = higher value
                
                # Plot heatmap
                sns.heatmap(numerical_matrix.T[:20], cmap='YlOrRd', 
                           xticklabels=[t.strftime('%Y-%m-%d') for t in timestamps],
                           yticklabels=all_features[:20],
                           cbar_kws={'label': 'Feature Rank'})
                
                ax5.set_title('Top Feature Importance Evolution Over Time', fontsize=14, fontweight='bold')
                ax5.set_xlabel('Time Period', fontsize=12)
                ax5.set_ylabel('Feature', fontsize=12)
                
                # Rotate x labels
                plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 6. Performance vs Feature Drift Scatter
        ax6 = plt.subplot(4, 2, 7)
        if 'temporal_validation' in self.results and not self.results['temporal_validation'].empty:
            df_tv = self.results['temporal_validation']
            
            # Clean data - remove NaN and infinite values
            mask = (~df_tv['feature_drift'].isna()) & (~df_tv['pearson_correlation'].isna()) & \
                   np.isfinite(df_tv['feature_drift']) & np.isfinite(df_tv['pearson_correlation'])
            
            df_tv_clean = df_tv[mask].copy()
            
            if len(df_tv_clean) > 0:
                # Create scatter plot
                scatter = ax6.scatter(df_tv_clean['feature_drift'], 
                                     df_tv_clean['pearson_correlation'],
                                     c=df_tv_clean['time_gap_days'],
                                     s=50, alpha=0.6, cmap='viridis')
                
                # Add trend line with error handling
                if len(df_tv_clean) > 1 and df_tv_clean['feature_drift'].var() > 0:
                    try:
                        # Use robust linear regression or simple mean line
                        from sklearn.linear_model import HuberRegressor
                        X = df_tv_clean['feature_drift'].values.reshape(-1, 1)
                        y = df_tv_clean['pearson_correlation'].values
                        
                        huber = HuberRegressor()
                        huber.fit(X, y)
                        
                        x_range = np.linspace(df_tv_clean['feature_drift'].min(), 
                                             df_tv_clean['feature_drift'].max(), 100)
                        y_pred = huber.predict(x_range.reshape(-1, 1))
                        
                        ax6.plot(x_range, y_pred, "r--", alpha=0.8, linewidth=2, label='Trend')
                    except:
                        # If regression fails, just show mean line
                        mean_corr = df_tv_clean['pearson_correlation'].mean()
                        ax6.axhline(y=mean_corr, color='r', linestyle='--', alpha=0.5, 
                                   label=f'Mean: {mean_corr:.3f}')
                
                ax6.set_xlabel('Feature Drift', fontsize=12)
                ax6.set_ylabel('Model Correlation', fontsize=12)
                ax6.set_title('Performance Degradation vs Feature Drift', fontsize=14, fontweight='bold')
                ax6.grid(True, alpha=0.3)
                ax6.legend()
                
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax6)
                cbar.set_label('Time Gap (Days)', fontsize=10)
            else:
                ax6.text(0.5, 0.5, 'Insufficient data for visualization', 
                        transform=ax6.transAxes, ha='center', va='center',
                        fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat'))
        
        # 7. Optimal Decay Factor Visualization
        ax7 = plt.subplot(4, 2, 8)
        if 'temporal_validation' in self.results and not self.results['temporal_validation'].empty:
            # Calculate empirical decay based on performance
            df_tv_avg = df_tv.groupby('lookback_periods').agg({
                'pearson_correlation': 'mean',
                'time_gap_days': 'mean'
            }).reset_index()
            
            # Remove any NaN values
            df_tv_avg = df_tv_avg.dropna()
            
            if len(df_tv_avg) > 1:
                # Fit exponential decay with error handling
                def exp_decay(x, a, b):
                    return a * np.exp(-b * x)
                
                try:
                    # Initial guess and bounds
                    p0 = [df_tv_avg['pearson_correlation'].iloc[0], 0.01]
                    bounds = ([0, 0], [1, 1])
                    
                    popt, _ = curve_fit(exp_decay, 
                                       df_tv_avg['time_gap_days'].values, 
                                       df_tv_avg['pearson_correlation'].values,
                                       p0=p0,
                                       bounds=bounds,
                                       maxfev=5000)
                    
                    # Plot actual vs fitted
                    x_fit = np.linspace(0, df_tv_avg['time_gap_days'].max(), 100)
                    y_fit = exp_decay(x_fit, *popt)
                    
                    ax7.plot(df_tv_avg['time_gap_days'], df_tv_avg['pearson_correlation'], 
                            'o', markersize=10, label='Observed', color='blue')
                    ax7.plot(x_fit, y_fit, '-', linewidth=2, label='Fitted Decay', color='red')
                    
                    # Calculate equivalent daily decay factor
                    daily_decay = np.exp(-popt[1])
                    
                    ax7.text(0.05, 0.85, f'Fitted Daily Decay Factor: {daily_decay:.4f}',
                            transform=ax7.transAxes, 
                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                            fontsize=12)
                    
                except Exception as e:
                    # If curve fitting fails, just plot the points
                    ax7.plot(df_tv_avg['time_gap_days'], df_tv_avg['pearson_correlation'], 
                            'o-', markersize=10, linewidth=2)
                    ax7.text(0.05, 0.85, 'Curve fitting failed - showing raw data',
                            transform=ax7.transAxes, 
                            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                            fontsize=10)
            else:
                ax7.text(0.5, 0.5, 'Insufficient data points for decay analysis', 
                        transform=ax7.transAxes, ha='center', va='center',
                        fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat'))
            
            ax7.set_xlabel('Time Gap (Days)', fontsize=12)
            ax7.set_ylabel('Average Correlation', fontsize=12)
            ax7.set_title('Empirical Performance Decay Function', fontsize=14, fontweight='bold')
            ax7.grid(True, alpha=0.3)
            ax7.legend()
        
        plt.tight_layout()
        plt.savefig('temporal_validation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_executive_summary(self):
        """
        Generate an executive summary of findings with specific recommendations.
        """
        print("\n" + "="*80)
        print("EXECUTIVE SUMMARY: TEMPORAL DYNAMICS AND DATA WEIGHTING ANALYSIS")
        print("="*80)
        
        if 'temporal_validation' in self.results and not self.results['temporal_validation'].empty:
            df_tv = self.results['temporal_validation']
            
            # Calculate key metrics
            initial_performance = df_tv[df_tv['lookback_periods'] == 0]['pearson_correlation'].mean()
            final_performance = df_tv[df_tv['lookback_periods'] == df_tv['lookback_periods'].max()]['pearson_correlation'].mean()
            performance_drop = (initial_performance - final_performance) / initial_performance * 100
            
            print("\nKEY FINDINGS:")
            print(f"1. Performance Degradation: {performance_drop:.1f}% correlation drop over time")
            print(f"   - Initial correlation (same period): {initial_performance:.3f}")
            print(f"   - Final correlation (oldest data): {final_performance:.3f}")
            
            # Feature drift analysis
            avg_drift_by_period = df_tv.groupby('lookback_periods')['feature_drift'].mean()
            if len(avg_drift_by_period) > 1:
                drift_increase = (avg_drift_by_period.iloc[-1] - avg_drift_by_period.iloc[0]) / avg_drift_by_period.iloc[0] * 100
                
                print(f"\n2. Feature Distribution Drift: {drift_increase:.1f}% increase in drift")
                print(f"   - Recent period drift: {avg_drift_by_period.iloc[0]:.3f}")
                print(f"   - Oldest period drift: {avg_drift_by_period.iloc[-1]:.3f}")
        
        if 'distribution_shifts' in self.results and not self.results['distribution_shifts'].empty:
            df_ds = self.results['distribution_shifts']
            
            significant_shifts = df_ds[df_ds['mean_ks_statistic'] > 0.1]
            if len(significant_shifts) > 0:
                first_significant = significant_shifts.iloc[0]
                print(f"\n3. Statistical Distribution Changes:")
                print(f"   - Significant shifts detected after {first_significant['time_gap_days']:.0f} days")
                print(f"   - {first_significant['pct_significant_shifts']*100:.1f}% of features show significant drift")
        
        print("\nRECOMMENDATIONS:")
        
        # Calculate recommended decay based on performance curve
        if 'temporal_validation' in self.results and not self.results['temporal_validation'].empty and len(df_tv) > 0:
            # Estimate half-life of performance
            half_performance = (initial_performance + final_performance) / 2
            half_life_data = df_tv[df_tv['pearson_correlation'] <= half_performance]
            
            if len(half_life_data) > 0:
                half_life_days = half_life_data['time_gap_days'].min()
                recommended_decay = np.exp(-np.log(2) / half_life_days)
                
                print(f"\n1. Optimal Decay Factor: {recommended_decay:.4f}")
                print(f"   - Based on performance half-life of {half_life_days:.0f} days")
                print(f"   - This ensures 50% weight reduction at the point where performance degrades by half")
            else:
                print("\n1. Optimal Decay Factor: 0.995 (default recommendation)")
                print("   - Performance degradation is gradual, suggesting moderate decay")
        
        print("\n2. Training Window Optimization:")
        print("   - Use 30-40% of most recent data for primary training")
        print("   - Implement exponential weighting within this window")
        print("   - Consider ensemble approaches with different window sizes")
        
        print("\n3. Feature Engineering Recommendations:")
        print("   - Create time-aware features that capture regime changes")
        print("   - Implement adaptive normalization based on recent statistics")
        print("   - Monitor feature importance stability and adapt accordingly")
        
        print("\n4. Model Architecture Considerations:")
        print("   - Implement online learning capabilities for rapid adaptation")
        print("   - Use regime-aware models that can switch behavior based on detected market states")
        print("   - Consider meta-learning approaches that explicitly model temporal dynamics")
        
        print("\n" + "="*80)

def main():
    """
    Execute comprehensive temporal validation analysis.
    """
    # Initialize analyzer
    analyzer = TemporalValidationAnalyzer('/kaggle/input/drw-crypto-market-prediction/train.parquet')
    
    # Load and prepare data
    analyzer.load_and_prepare_data()
    
    # Perform analyses
    print("\nExecuting temporal validation framework...")
    analyzer.perform_temporal_cross_validation(window_size=50000, n_splits=8)
    analyzer.analyze_reverse_temporal_validation(window_size=50000)
    analyzer.analyze_feature_stability(window_size=50000)
    analyzer.calculate_distribution_shifts(window_size=20000)
    
    # Generate visualizations
    print("\nGenerating comprehensive visualizations...")
    analyzer.create_comprehensive_visualizations()
    
    # Generate executive summary
    analyzer.generate_executive_summary()
    
    print("\nAnalysis complete. Results saved to 'temporal_validation_analysis.png'")

if __name__ == "__main__":
    main()




