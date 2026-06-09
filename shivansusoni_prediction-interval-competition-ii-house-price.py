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




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load the data
# Note: Update these paths based on your Kaggle environment
train_df = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv')
test_df = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/test.csv')
sample_submission = pd.read_csv('/kaggle/input/prediction-interval-competition-ii-house-price/sample_submission.csv')

print("Data Shapes:")
print(f"Training data: {train_df.shape}")
print(f"Test data: {test_df.shape}")
print(f"Sample submission: {sample_submission.shape}")

# Check columns
print("\nColumns in training but not in test:")
print(set(train_df.columns) - set(test_df.columns))

# Basic data exploration
print("\nTraining data info:")
print(train_df.info())

print("\nTarget variable (sale_price) statistics:")
print(train_df['sale_price'].describe())

# Check for missing values
print("\nMissing values in training data:")
print(train_df.isnull().sum()[train_df.isnull().sum() > 0])

print("\nMissing values in test data:")
print(test_df.isnull().sum()[test_df.isnull().sum() > 0])

# Visualize target distribution
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(train_df['sale_price'], bins=50, edgecolor='black')
plt.title('Sale Price Distribution')
plt.xlabel('Sale Price')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(np.log1p(train_df['sale_price']), bins=50, edgecolor='black')
plt.title('Log(Sale Price) Distribution')
plt.xlabel('Log(Sale Price)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Feature engineering and preprocessing
def preprocess_data(df):
    """Preprocess the data for modeling"""
    df_processed = df.copy()
    
    # Handle missing values
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].median())
    
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df_processed[col] = df_processed[col].fillna('Unknown')
    
    # Convert date to datetime features
    if 'sale_date' in df_processed.columns:
        df_processed['sale_date'] = pd.to_datetime(df_processed['sale_date'])
        df_processed['sale_year'] = df_processed['sale_date'].dt.year
        df_processed['sale_month'] = df_processed['sale_date'].dt.month
        df_processed['sale_day'] = df_processed['sale_date'].dt.day
        df_processed = df_processed.drop('sale_date', axis=1)
    
    return df_processed

# Prepare features
X_train = train_df.drop(['sale_price', 'id'], axis=1)
y_train = train_df['sale_price']
X_test = test_df.drop(['id'], axis=1)

# Preprocess data
X_train_processed = preprocess_data(X_train)
X_test_processed = preprocess_data(X_test)

# Get numeric columns only for initial model
numeric_features = X_train_processed.select_dtypes(include=[np.number]).columns
X_train_numeric = X_train_processed[numeric_features]
X_test_numeric = X_test_processed[numeric_features]

# Ensure test has same columns as train
common_cols = list(set(X_train_numeric.columns) & set(X_test_numeric.columns))
X_train_numeric = X_train_numeric[common_cols]
X_test_numeric = X_test_numeric[common_cols]

print(f"\nNumber of features used: {len(common_cols)}")

# Split training data for validation
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_numeric, y_train, test_size=0.2, random_state=42
)

# Quantile Random Forest for Prediction Intervals
from sklearn.ensemble import RandomForestRegressor

class QuantileRandomForest:
    """Random Forest for prediction intervals using quantile predictions"""
    
    def __init__(self, n_estimators=100, random_state=42, n_jobs=-1):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.rf = None
        
    def fit(self, X, y):
        """Fit the random forest model"""
        self.rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            max_features='sqrt',
            min_samples_leaf=10
        )
        self.rf.fit(X, y)
        return self
    
    def predict_quantiles(self, X, quantiles=[0.05, 0.95]):
        """Predict quantiles using the random forest"""
        # Get predictions from all trees
        all_predictions = []
        for tree in self.rf.estimators_:
            pred = tree.predict(X)
            all_predictions.append(pred)
        
        all_predictions = np.array(all_predictions).T
        
        # Calculate quantiles
        quantile_predictions = []
        for q in quantiles:
            q_pred = np.percentile(all_predictions, q * 100, axis=1)
            quantile_predictions.append(q_pred)
        
        return np.array(quantile_predictions).T

# Train the model
print("\nTraining Quantile Random Forest...")
qrf = QuantileRandomForest(n_estimators=100, random_state=42)
qrf.fit(X_tr, y_tr)

# Make predictions on validation set
val_intervals = qrf.predict_quantiles(X_val, quantiles=[0.05, 0.95])
val_lower = val_intervals[:, 0]
val_upper = val_intervals[:, 1]

# Calculate coverage and interval width
coverage = np.mean((y_val >= val_lower) & (y_val <= val_upper))
avg_interval_width = np.mean(val_upper - val_lower)

print(f"\nValidation Results:")
print(f"Coverage: {coverage:.2%} (target: 90%)")
print(f"Average interval width: ${avg_interval_width:,.0f}")

# Winkler Score calculation
def winkler_score(y_true, lower, upper, alpha=0.1):
    """Calculate Winkler score for prediction intervals"""
    interval_width = upper - lower
    
    # Penalty for predictions outside interval
    lower_penalty = 2 * (lower - y_true) / alpha * (y_true < lower)
    upper_penalty = 2 * (y_true - upper) / alpha * (y_true > upper)
    
    return interval_width + lower_penalty + upper_penalty

# Calculate Winkler score on validation set
val_winkler_scores = winkler_score(y_val, val_lower, val_upper)
mean_winkler_score = np.mean(val_winkler_scores)
print(f"Mean Winkler Score: {mean_winkler_score:,.0f}")

# Visualize prediction intervals on a sample
plt.figure(figsize=(12, 6))
sample_idx = np.random.choice(len(y_val), 100, replace=False)
sample_idx_sorted = sample_idx[np.argsort(y_val.iloc[sample_idx])]

plt.scatter(range(len(sample_idx)), y_val.iloc[sample_idx_sorted], 
           color='black', s=20, label='Actual', zorder=3)
plt.fill_between(range(len(sample_idx)), 
                val_lower[sample_idx_sorted], 
                val_upper[sample_idx_sorted],
                alpha=0.3, color='blue', label='90% Prediction Interval')
plt.xlabel('Sample Index (sorted by actual price)')
plt.ylabel('Sale Price')
plt.title('Prediction Intervals vs Actual Prices (100 samples)')
plt.legend()
plt.tight_layout()
plt.show()

# Feature importance
feature_importance = pd.DataFrame({
    'feature': common_cols,
    'importance': qrf.rf.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
top_features = feature_importance.head(20)
plt.barh(top_features['feature'], top_features['importance'])
plt.xlabel('Importance')
plt.title('Top 20 Feature Importances')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Train final model on full training data
print("\nTraining final model on full dataset...")
qrf_final = QuantileRandomForest(n_estimators=200, random_state=42)
qrf_final.fit(X_train_numeric, y_train)

# Make predictions on test set
test_intervals = qrf_final.predict_quantiles(X_test_numeric, quantiles=[0.05, 0.95])
test_lower = test_intervals[:, 0]
test_upper = test_intervals[:, 1]

# Create submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'pi_lower': test_lower,
    'pi_upper': test_upper
})

# Ensure predictions are reasonable
submission['pi_lower'] = submission['pi_lower'].clip(lower=0)
submission['pi_upper'] = submission['pi_upper'].clip(lower=submission['pi_lower'] + 1000)

print("\nSubmission statistics:")
print(submission.describe())

# Save submission
submission.to_csv('submission.csv', index=False)
print("\nSubmission saved to 'submission.csv'")

# Advanced approach: Gradient Boosting with Quantile Loss
from sklearn.ensemble import GradientBoostingRegressor

print("\n" + "="*50)
print("Alternative Approach: Gradient Boosting Quantile Regression")
print("="*50)

# Train models for lower and upper quantiles
gb_lower = GradientBoostingRegressor(
    loss='quantile', 
    alpha=0.05,  # 5th percentile
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

gb_upper = GradientBoostingRegressor(
    loss='quantile', 
    alpha=0.95,  # 95th percentile
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

print("Training lower quantile model...")
gb_lower.fit(X_tr, y_tr)

print("Training upper quantile model...")
gb_upper.fit(X_tr, y_tr)

# Predictions on validation set
gb_val_lower = gb_lower.predict(X_val)
gb_val_upper = gb_upper.predict(X_val)

# Calculate metrics
gb_coverage = np.mean((y_val >= gb_val_lower) & (y_val <= gb_val_upper))
gb_avg_width = np.mean(gb_val_upper - gb_val_lower)
gb_winkler = np.mean(winkler_score(y_val, gb_val_lower, gb_val_upper))

print(f"\nGradient Boosting Validation Results:")
print(f"Coverage: {gb_coverage:.2%}")
print(f"Average interval width: ${gb_avg_width:,.0f}")
print(f"Mean Winkler Score: {gb_winkler:,.0f}")

# Tips for improvement
print("\n" + "="*50)
print("Tips for Improving Your Model:")
print("="*50)
print("1. Feature Engineering:")
print("   - Create interaction features (e.g., sqft * grade)")
print("   - Add neighborhood-based features")
print("   - Engineer date-based features (seasonality, days on market)")
print("   - Use target encoding for categorical variables")
print("\n2. Model Ensemble:")
print("   - Combine multiple models (RF, GB, XGBoost, LightGBM)")
print("   - Use different quantile estimation methods")
print("   - Consider conformalized quantile regression")
print("\n3. Calibration:")
print("   - Adjust quantiles based on validation coverage")
print("   - Use isotonic regression for calibration")
print("   - Consider local (feature-dependent) interval widths")
print("\n4. Advanced Techniques:")
print("   - Neural networks with quantile loss")
print("   - Bayesian approaches for uncertainty quantification")
print("   - Conformal prediction for guaranteed coverage")

