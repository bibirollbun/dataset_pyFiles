import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, jaccard_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class PumpFunTokenPredictor:
    def __init__(self, root_path, september_path):
        self.root_path = root_path
        self.september_path = september_path
        self.model = None
        self.feature_columns = []
        self.categorical_features = []

    def load_data(self):
        """Load and combine all dataset files including September 2025 data"""
        print("Loading dataset files...")

        # Load evaluation chunks
        eval_files = [
            f"{self.root_path}/evaluation_set_30s_chunk_001.csv",
            f"{self.root_path}/evaluation_set_30s_chunk_002.csv",
            f"{self.root_path}/evaluation_set_30s_chunk_003.csv",
            f"{self.root_path}/evaluation_set_30s_chunk_004.csv",
            f"{self.root_path}/evaluation_set_30s_chunk_005.csv"
        ]

        # Load September 2025 chunks
        september_files = [
            f"{self.september_path}/september_2025_first30s_chunk_001.csv",
            f"{self.september_path}/september_2025_first30s_chunk_002.csv",
            f"{self.september_path}/september_2025_first30s_chunk_003.csv",
            f"{self.september_path}/september_2025_first30s_chunk_004.csv",
            f"{self.september_path}/september_2025_first30s_chunk_005.csv",
            f"{self.september_path}/september_2025_first30s_chunk_006.csv",
            f"{self.september_path}/september_2025_first30s_chunk_007.csv",
            f"{self.september_path}/september_2025_first30s_chunk_008.csv",
            f"{self.september_path}/september_2025_first30s_chunk_009.csv",
            f"{self.september_path}/september_2025_first30s_chunk_010.csv",
            f"{self.september_path}/september_2025_first30s_chunk_011.csv",
            f"{self.september_path}/september_2025_first30s_chunk_012.csv",
            f"{self.september_path}/september_2025_first30s_chunk_013.csv",
            f"{self.september_path}/september_2025_first30s_chunk_014.csv",
            f"{self.september_path}/september_2025_first30s_chunk_015.csv"
        ]

        # Load sample dataset for training reference
        sample_df = pd.read_csv(f"{self.root_path}/Sample_Dataset.csv")

        # Combine September 2025 chunks for additional training data
        september_dfs = []
        for file in september_files:
            try:
                df = pd.read_csv(file)
                september_dfs.append(df)
                print(f"Loaded {file.split('/')[-1]} with shape {df.shape}")
            except Exception as e:
                print(f"Error loading {file}: {e}")

        combined_september_df = pd.concat(september_dfs, ignore_index=True)
        print(f"Combined September 2025 dataset shape: {combined_september_df.shape}")

        # Combine all evaluation chunks
        eval_dfs = []
        for file in eval_files:
            try:
                df = pd.read_csv(file)
                eval_dfs.append(df)
                print(f"Loaded {file.split('/')[-1]} with shape {df.shape}")
            except Exception as e:
                print(f"Error loading {file}: {e}")

        combined_eval_df = pd.concat(eval_dfs, ignore_index=True)
        print(f"Combined evaluation dataset shape: {combined_eval_df.shape}")

        # Combine sample and September data for enhanced training
        combined_training_df = pd.concat([sample_df, combined_september_df], ignore_index=True)
        print(f"Combined training dataset shape: {combined_training_df.shape}")

        return combined_training_df, combined_eval_df

    def perform_eda(self, df, title="Dataset EDA"):
        """Comprehensive Exploratory Data Analysis"""
        print(f"\n{'='*60}")
        print(f"{title}")
        print(f"{'='*60}")
        
        # Basic statistics
        print(f"\nDataset Shape: {df.shape}")
        print(f"Unique Tokens: {df['mint_token_id'].nunique()}")
        print(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print("\nMissing Values:")
            print(missing[missing > 0])
        
        # Create visualization
        fig = plt.figure(figsize=(20, 12))
        
        # Trade mode distribution
        plt.subplot(3, 4, 1)
        df['trade_mode'].value_counts().plot(kind='bar', color=['green', 'red'])
        plt.title('Trade Mode Distribution')
        plt.xlabel('Trade Mode')
        plt.ylabel('Count')
        
        # Market cap distribution
        plt.subplot(3, 4, 2)
        plt.hist(df['market_cap_usd'].clip(upper=df['market_cap_usd'].quantile(0.95)), 
                 bins=50, edgecolor='black')
        plt.title('Market Cap Distribution (95th percentile)')
        plt.xlabel('Market Cap (USD)')
        plt.ylabel('Frequency')
        
        # Volume analysis
        plt.subplot(3, 4, 3)
        plt.hist(df['sol_volume'].clip(upper=df['sol_volume'].quantile(0.95)), 
                 bins=50, edgecolor='black', color='orange')
        plt.title('SOL Volume Distribution (95th percentile)')
        plt.xlabel('SOL Volume')
        plt.ylabel('Frequency')
        
        # Buy vs Sell ratio
        plt.subplot(3, 4, 4)
        buy_sell_data = df.groupby('mint_token_id').agg({
            'buy_count': 'sum',
            'sell_count': 'sum'
        }).reset_index()
        plt.scatter(buy_sell_data['buy_count'], buy_sell_data['sell_count'], 
                   alpha=0.5, s=10)
        plt.title('Buy Count vs Sell Count per Token')
        plt.xlabel('Buy Count')
        plt.ylabel('Sell Count')
        
        # Holder statistics
        plt.subplot(3, 4, 5)
        plt.hist(df['current_holders'].clip(upper=df['current_holders'].quantile(0.95)), 
                 bins=30, edgecolor='black', color='purple')
        plt.title('Current Holders Distribution')
        plt.xlabel('Current Holders')
        plt.ylabel('Frequency')
        
        # RSI distribution
        plt.subplot(3, 4, 6)
        plt.hist(df['relative_strength_index'], bins=50, edgecolor='black', color='teal')
        plt.title('RSI Distribution')
        plt.xlabel('RSI')
        plt.ylabel('Frequency')
        
        # Rate of change
        plt.subplot(3, 4, 7)
        plt.hist(df['rate_of_change'].clip(lower=df['rate_of_change'].quantile(0.05),
                                           upper=df['rate_of_change'].quantile(0.95)), 
                 bins=50, edgecolor='black', color='brown')
        plt.title('Rate of Change Distribution (5-95th percentile)')
        plt.xlabel('Rate of Change')
        plt.ylabel('Frequency')
        
        # Creator sold analysis
        plt.subplot(3, 4, 8)
        df['creator_sold'].value_counts().plot(kind='bar', color=['blue', 'red'])
        plt.title('Creator Sold Status')
        plt.xlabel('Creator Sold')
        plt.ylabel('Count')
        
        # Liquidity ratio
        plt.subplot(3, 4, 9)
        plt.hist(np.log10(df['liquidity_ratio'] + 1e-10), bins=50, edgecolor='black', color='cyan')
        plt.title('Liquidity Ratio (log scale)')
        plt.xlabel('Log10(Liquidity Ratio)')
        plt.ylabel('Frequency')
        
        # Top 10 holder concentration
        plt.subplot(3, 4, 10)
        plt.hist(df['top10_percent_total'].clip(upper=df['top10_percent_total'].quantile(0.95)), 
                 bins=50, edgecolor='black', color='magenta')
        plt.title('Top 10 Holder Concentration')
        plt.xlabel('Top 10 %')
        plt.ylabel('Frequency')
        
        # Bollinger position
        plt.subplot(3, 4, 11)
        plt.hist(df['bollinger_relative_position'], bins=50, edgecolor='black', color='olive')
        plt.title('Bollinger Relative Position')
        plt.xlabel('Position')
        plt.ylabel('Frequency')
        
        # Money Flow Index
        plt.subplot(3, 4, 12)
        plt.hist(df['money_flow_index'], bins=50, edgecolor='black', color='pink')
        plt.title('Money Flow Index Distribution')
        plt.xlabel('MFI')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
        
        # Correlation heatmap for key features
        print("\nGenerating correlation heatmap...")
        numeric_cols = ['market_cap_usd', 'sol_volume', 'token_volume', 'buy_count', 
                       'sell_count', 'current_holders', 'relative_strength_index', 
                       'rate_of_change', 'liquidity_ratio', 'buy_sell_ratio']
        
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        if len(available_cols) > 0:
            fig, ax = plt.subplots(figsize=(12, 10))
            correlation_matrix = df[available_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, square=True, ax=ax)
            plt.title('Feature Correlation Heatmap')
            plt.tight_layout()
            plt.show()

    def safe_timestamp_conversion(self, timestamp_series):
        """Safely convert timestamp series handling invalid hour values"""
        converted_timestamps = []

        for ts in timestamp_series:
            try:
                # Try direct conversion first
                if isinstance(ts, str):
                    # Handle the specific case of "26:28.7" format
                    if ':' in ts and len(ts.split(':')) > 1:
                        parts = ts.split(':')
                        if len(parts) >= 2:
                            hour_part = parts[0]
                            # Check if hour is invalid (>23)
                            if hour_part.isdigit() and int(hour_part) >= 24:
                                # Normalize the hour (wrap around)
                                normalized_hour = int(hour_part) % 24
                                remaining_parts = ':'.join(parts[1:])
                                normalized_ts = f"{normalized_hour:02d}:{remaining_parts}"
                                # Try to create a proper datetime
                                # If it's just time without date, add a dummy date
                                if ' ' not in normalized_ts:
                                    normalized_ts = f"2024-01-01 {normalized_ts}"
                                converted_ts = pd.to_datetime(normalized_ts)
                            else:
                                converted_ts = pd.to_datetime(ts)
                        else:
                            converted_ts = pd.to_datetime(ts)
                    else:
                        converted_ts = pd.to_datetime(ts)
                else:
                    converted_ts = pd.to_datetime(ts)
                converted_timestamps.append(converted_ts)
            except Exception as e:
                # Fallback: use current timestamp or a default
                print(f"Warning: Could not parse timestamp {ts}: {e}")
                converted_timestamps.append(pd.Timestamp('2024-01-01'))  # Default fallback

        return pd.Series(converted_timestamps, index=timestamp_series.index)

    def safe_column_operation(self, df, col_name, default_value=0):
        """Safely access a column, return default value if column doesn't exist"""
        if col_name in df.columns:
            return df[col_name]
        else:
            print(f"Warning: Column '{col_name}' not found, using default value {default_value}")
            return pd.Series([default_value] * len(df), index=df.index)

    def engineer_features(self, df):
        """Engineer sophisticated features from the raw data"""
        print("Engineering features...")

        # Create a copy to avoid modifying original
        features_df = df.copy()

        # Convert timestamp to datetime using safe conversion
        print("Converting timestamps...")
        features_df['timestamp'] = self.safe_timestamp_conversion(features_df['timestamp'])

        # Extract time-based features from valid timestamps
        features_df['hour'] = features_df['timestamp'].dt.hour
        features_df['day_of_week'] = features_df['timestamp'].dt.dayofweek
        features_df['minute'] = features_df['timestamp'].dt.minute

        # Use safe column access for all feature engineering operations
        rate_of_change = self.safe_column_operation(features_df, 'rate_of_change', 0)
        momentum = self.safe_column_operation(features_df, 'momentum', 0)
        sol_volume = self.safe_column_operation(features_df, 'sol_volume', 0)
        virtual_sol_reserves = self.safe_column_operation(features_df, 'virtual_sol_reserves', 1)
        virtual_token_reserves = self.safe_column_operation(features_df, 'virtual_token_reserves', 1)
        market_cap_usd = self.safe_column_operation(features_df, 'market_cap_usd', 1)
        top10_percent_total = self.safe_column_operation(features_df, 'top10_percent_total', 0)
        current_holders = self.safe_column_operation(features_df, 'current_holders', 1)
        total_holders = self.safe_column_operation(features_df, 'total_holders', 1)
        total_count = self.safe_column_operation(features_df, 'total_count', 1)
        buy_count = self.safe_column_operation(features_df, 'buy_count', 0)
        sell_count = self.safe_column_operation(features_df, 'sell_count', 0)
        creator_balance = self.safe_column_operation(features_df, 'creator_balance', 0)
        creator_sold = self.safe_column_operation(features_df, 'creator_sold', 0)
        relative_strength_index = self.safe_column_operation(features_df, 'relative_strength_index', 50)
        money_flow_index = self.safe_column_operation(features_df, 'money_flow_index', 50)
        bollinger_relative_position = self.safe_column_operation(features_df, 'bollinger_relative_position', 0)
        volume_oscillator = self.safe_column_operation(features_df, 'volume_oscillator', 0)
        liquidity_ratio = self.safe_column_operation(features_df, 'liquidity_ratio', 0)
        buy_sell_ratio = self.safe_column_operation(features_df, 'buy_sell_ratio', 1)
        holder_ratio = self.safe_column_operation(features_df, 'holder_ratio', 0)
        token_volume = self.safe_column_operation(features_df, 'token_volume', 0)
        consumed_gas = self.safe_column_operation(features_df, 'consumed_gas', 1)
        fee = self.safe_column_operation(features_df, 'fee', 0)

        # Price and volume momentum features
        features_df['price_momentum'] = rate_of_change * momentum
        features_df['volume_price_trend'] = sol_volume * rate_of_change

        # Liquidity and market depth features
        features_df['liquidity_depth'] = virtual_sol_reserves / (virtual_token_reserves + 1)
        features_df['market_depth_ratio'] = market_cap_usd / (sol_volume + 1)

        # Holder concentration features
        features_df['holder_concentration'] = top10_percent_total / (current_holders + 1)
        features_df['holder_growth_rate'] = current_holders / (total_holders + 1)

        # Transaction intensity features
        features_df['tx_intensity'] = total_count / (sol_volume + 1)
        features_df['buy_pressure'] = buy_count / (total_count + 1)
        features_df['sell_pressure'] = sell_count / (total_count + 1)

        # Creator behavior features
        features_df['creator_engagement'] = creator_balance / (market_cap_usd + 1)
        features_df['creator_sell_ratio'] = creator_sold.astype(int) * creator_balance

        # Technical indicator combinations
        features_df['rsi_mfi_sync'] = (relative_strength_index - 50) * (money_flow_index - 50)
        features_df['bollinger_volatility'] = bollinger_relative_position * volume_oscillator

        # Risk-adjusted features
        features_df['risk_adjusted_return'] = rate_of_change / (sol_volume + 1)
        features_df['volatility_efficiency'] = rate_of_change / (volume_oscillator.abs() + 1)

        # Network effect features
        features_df['network_effect'] = current_holders * buy_sell_ratio
        features_df['community_strength'] = holder_ratio * features_df['buy_pressure']

        # Gas and fee efficiency
        features_df['gas_efficiency'] = sol_volume / (consumed_gas + 1)
        features_df['fee_to_volume_ratio'] = fee / (sol_volume + 1)

        # Market microstructure features
        features_df['order_imbalance'] = (buy_count - sell_count) / (total_count + 1)
        features_df['volume_imbalance'] = (token_volume * features_df['buy_pressure']) / (sol_volume + 1)

        # Time decay features (within 30-second window)
        features_df['time_decay_factor'] = np.exp(-features_df['minute'] / 30)

        print(f"Original features: {len(df.columns)}, Engineered features: {len(features_df.columns)}")
        return features_df

    def create_target_variable(self, df, strategy='composite'):
        """
        Create target variable using sophisticated composite scoring
        In real scenario, this would use the actual target token list
        """
        print("Creating target variable...")

        # Create composite score based on multiple success indicators
        df = df.copy()

        # Use safe column access for target creation
        rate_of_change = self.safe_column_operation(df, 'rate_of_change', 0)
        sol_volume = self.safe_column_operation(df, 'sol_volume', 0)
        current_holders = self.safe_column_operation(df, 'current_holders', 0)
        liquidity_ratio = self.safe_column_operation(df, 'liquidity_ratio', 0)
        relative_strength_index = self.safe_column_operation(df, 'relative_strength_index', 50)
        money_flow_index = self.safe_column_operation(df, 'money_flow_index', 50)
        bollinger_relative_position = self.safe_column_operation(df, 'bollinger_relative_position', 50)
        buy_sell_ratio = self.safe_column_operation(df, 'buy_sell_ratio', 1)
        volume_oscillator = self.safe_column_operation(df, 'volume_oscillator', 0)

        # Price momentum score
        price_score = np.tanh(rate_of_change * 10)

        # Volume strength score
        volume_quantile = sol_volume.quantile(0.75) if sol_volume.quantile(0.75) > 0 else 1
        volume_score = np.tanh(sol_volume / volume_quantile)

        # Holder growth score
        holder_score = np.tanh(current_holders / 10)

        # Liquidity quality score
        liquidity_score = 1 - np.tanh(liquidity_ratio * 1e7)

        # Technical indicator score
        tech_score = (
            (relative_strength_index > 60).astype(int) * 0.2 +
            (money_flow_index > 60).astype(int) * 0.2 +
            (bollinger_relative_position > 60).astype(int) * 0.2 +
            (buy_sell_ratio > 1.5).astype(int) * 0.2 +
            (volume_oscillator > 0).astype(int) * 0.2
        )

        # Composite target score
        composite_score = (
            price_score * 0.25 +
            volume_score * 0.20 +
            holder_score * 0.15 +
            liquidity_score * 0.15 +
            tech_score * 0.25
        )

        # Create binary target (top 20% as positive class)
        threshold = composite_score.quantile(0.80) if len(composite_score) > 0 else 0.5
        df['is_target'] = (composite_score >= threshold).astype(int)

        target_ratio = df['is_target'].mean()
        print(f"Target class distribution: {target_ratio:.3f} ({df['is_target'].sum()} positive samples)")

        return df

    def select_features(self, df):
        """Select final feature set for model training"""
        # Remove identifier and target columns
        exclude_cols = ['timestamp', 'mint_token_id', 'holder', 'creator', 'is_target', 'index']

        # Get all available columns
        all_columns = [col for col in df.columns if col not in exclude_cols]

        # Identify categorical features - FIXED: Only truly categorical columns
        categorical_features = []
        for col in all_columns:
            # Only consider string/object type columns as categorical
            # AND ensure they have reasonable number of categories (not IDs)
            if df[col].dtype == 'object':
                n_unique = df[col].nunique()
                # Only categorical if it's a string column with reasonable cardinality
                if 1 < n_unique < 50:  # More restrictive threshold
                    categorical_features.append(col)
                    print(f"Categorical feature detected: {col} with {n_unique} unique values")

        # All other columns are numerical
        numerical_features = [col for col in all_columns if col not in categorical_features]

        print(f"Selected {len(numerical_features)} numerical features and {len(categorical_features)} categorical features")

        self.feature_columns = numerical_features + categorical_features
        self.categorical_features = categorical_features

        return numerical_features, categorical_features

    def prepare_model_data(self, df):
        """Prepare data for model training"""
        X = df[self.feature_columns].copy()
        y = df['is_target'].copy()

        # Handle missing values and ensure correct dtypes
        for col in X.columns:
            if col in self.categorical_features:
                # Convert categorical to string explicitly
                X[col] = X[col].astype(str)
                if X[col].isnull().any():
                    X[col].fillna('missing', inplace=True)
            else:
                # Ensure numerical columns are float
                X[col] = pd.to_numeric(X[col], errors='coerce')
                if X[col].isnull().any():
                    X[col].fillna(X[col].median(), inplace=True)

        return X, y

    def train_model(self, X_train, y_train, X_val=None, y_val=None):
        """Train CatBoost model with sophisticated parameter tuning"""
        print("Training CatBoost model...")

        # Get categorical feature indices
        cat_feature_indices = [X_train.columns.get_loc(col) for col in self.categorical_features if col in X_train.columns]

        print(f"Using {len(cat_feature_indices)} categorical features at indices: {cat_feature_indices}")

        # Create catboost pool with indices instead of names
        train_pool = Pool(
            X_train,
            y_train,
            cat_features=cat_feature_indices  # Use indices instead of column names
        )

        val_pool = None
        if X_val is not None and y_val is not None:
            val_pool = Pool(
                X_val,
                y_val,
                cat_features=cat_feature_indices  # Use indices instead of column names
            )

        # Initialize CatBoost with optimized parameters
        self.model = CatBoostClassifier(
            iterations=2000,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=3,
            border_count=128,
            loss_function='Logloss',
            eval_metric='F1',
            random_seed=42,
            early_stopping_rounds=100,
            verbose=100,
            use_best_model=True
        )

        # Train model
        if val_pool is not None:
            self.model.fit(
                train_pool,
                eval_set=val_pool,
                plot=True,
                verbose=True
            )
        else:
            self.model.fit(
                train_pool,
                plot=True,
                verbose=True
            )

        print("Model training completed!")
        return self.model

    def evaluate_model_with_jaccard(self, X_test, y_test):
        """Comprehensive model evaluation with Jaccard Index focus"""
        print("\n" + "="*50)
        print("MODEL EVALUATION - JACCARD INDEX")
        print("="*50)

        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        # Calculate Jaccard Index (Intersection-over-Union)
        jaccard = jaccard_score(y_test, y_pred)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Classification report
        cr = classification_report(y_test, y_pred, output_dict=True)

        print(f"\n{'='*50}")
        print(f"JACCARD INDEX (IoU): {jaccard:.4f}")
        print(f"{'='*50}")
        print(f"Recall: {cr['1']['recall']:.4f}")
        print(f"Precision: {cr['1']['precision']:.4f}")
        print(f"F1-Score: {cr['1']['f1-score']:.4f}")
        print(f"Accuracy: {cr['accuracy']:.4f}")
        
        # Check if recall meets minimum requirement
        if cr['1']['recall'] >= 0.75:
            print(f"✓ Recall requirement MET (≥75%)")
        else:
            print(f"✗ Recall requirement NOT MET (<75%)")

        # Plot confusion matrix with Jaccard annotation
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Not Target', 'Target'],
                   yticklabels=['Not Target', 'Target'],
                   ax=axes[0])
        axes[0].set_title(f'Confusion Matrix\nJaccard Index: {jaccard:.4f}')
        axes[0].set_ylabel('Actual')
        axes[0].set_xlabel('Predicted')
        
        # Add Jaccard calculation visualization
        TP = cm[1, 1]
        FP = cm[0, 1]
        FN = cm[1, 0]
        
        axes[0].text(0.5, -0.15, 
                    f'Jaccard = TP/(TP+FP+FN) = {TP}/({TP}+{FP}+{FN}) = {jaccard:.4f}',
                    transform=axes[0].transAxes,
                    ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # ROC-like visualization for probability distribution
        axes[1].hist(y_pred_proba[y_test == 0], bins=50, alpha=0.5, label='Negative Class', color='blue')
        axes[1].hist(y_pred_proba[y_test == 1], bins=50, alpha=0.5, label='Positive Class', color='red')
        axes[1].set_xlabel('Prediction Probability')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Prediction Probability Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

        # Plot feature importance
        feature_importance = self.model.get_feature_importance()
        feature_names = self.feature_columns

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False).head(20)

        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title('Top 20 Feature Importance')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()

        return {
            'jaccard_index': jaccard,
            'confusion_matrix': cm,
            'classification_report': cr,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }

    def create_submission(self, eval_df, threshold=0.5):
        """Create submission CSV with predictions in required format"""
        print("Creating submission file...")

        # Engineer features for evaluation set
        eval_features = self.engineer_features(eval_df)

        # Prepare features
        X_eval = eval_features[self.feature_columns].copy()

        # Handle missing values in evaluation set
        for col in X_eval.columns:
            if col in self.categorical_features:
                # Convert categorical to string explicitly
                X_eval[col] = X_eval[col].astype(str)
                if X_eval[col].isnull().any():
                    X_eval[col].fillna('missing', inplace=True)
            else:
                # Ensure numerical columns are float
                X_eval[col] = pd.to_numeric(X_eval[col], errors='coerce')
                if X_eval[col].isnull().any():
                    X_eval[col].fillna(X_eval[col].median(), inplace=True)

        # Get predictions
        predictions_proba = self.model.predict_proba(X_eval)[:, 1]
        predictions = (predictions_proba >= threshold).astype(int)

        # Create submission dataframe with required format: mint_token_id, is_target
        submission_df = pd.DataFrame({
            'mint_token_id': eval_df['mint_token_id'],
            'is_target': predictions
        })

        # Save submission file as exactly "submission.csv"
        submission_filename = "submission.csv"
        submission_df.to_csv(submission_filename, index=False)

        print(f"\nSubmission file created: {submission_filename}")
        print(f"Total rows in submission: {len(submission_df)}")
        print(f"Tokens predicted as target (is_target=1): {submission_df['is_target'].sum()}")
        print(f"Tokens predicted as non-target (is_target=0): {len(submission_df) - submission_df['is_target'].sum()}")
        print(f"Target percentage: {submission_df['is_target'].mean()*100:.2f}%")

        # Verify the format
        print(f"\nSubmission format verification:")
        print(f"Columns: {list(submission_df.columns)}")
        print(f"First few rows:")
        print(submission_df.head())

        return submission_df

    def run_pipeline(self):
        """Execute complete ML pipeline"""
        print("\n" + "="*60)
        print("PUMP.FUN TOKEN PREDICTOR - ML PIPELINE")
        print("="*60 + "\n")

        # Load data
        train_df, eval_df = self.load_data()

        # Perform EDA on training data
        self.perform_eda(train_df, "Training Data EDA")

        # Engineer features
        train_features = self.engineer_features(train_df)

        # Create target variable
        train_features = self.create_target_variable(train_features)

        # Select features
        self.select_features(train_features)

        # Prepare model data
        X, y = self.prepare_model_data(train_features)

        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Further split training data for validation
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        print(f"\nData split:")
        print(f"Training: {X_train_final.shape[0]} samples")
        print(f"Validation: {X_val.shape[0]} samples")
        print(f"Test: {X_test.shape[0]} samples")

        # Train model
        self.train_model(X_train_final, y_train_final, X_val, y_val)

        # Evaluate model
        results = self.evaluate_model_with_jaccard(X_test, y_test)

        # Create submission
        submission_df = self.create_submission(eval_df)

        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)

        return results, submission_df


# Main execution
if __name__ == "__main__":
    # Define paths to your data
    ROOT_PATH = "/kaggle/input/alpha-radar-solana-sprint"
    SEPTEMBER_PATH = "/kaggle/input/pumpfun-30s-september-2025"

    # Initialize predictor
    predictor = PumpFunTokenPredictor(ROOT_PATH, SEPTEMBER_PATH)

    # Run complete pipeline
    results, submission_df = predictor.run_pipeline()

    # Display final metrics
    print("\n" + "="*60)
    print("FINAL METRICS SUMMARY")
    print("="*60)
    print(f"Jaccard Index: {results['jaccard_index']:.4f}")
    print(f"Recall: {results['classification_report']['1']['recall']:.4f}")
    print(f"Precision: {results['classification_report']['1']['precision']:.4f}")
    print(f"F1-Score: {results['classification_report']['1']['f1-score']:.4f}")
    print("="*60)

