!pip install --upgrade imbalanced-learn


# Standard data manipulation and analysis
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier

# Utilities
import os
import warnings
import joblib

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# Visualization settings
sns.set_theme(style='whitegrid', palette='Set2')
plt.rcParams.update({
    'figure.dpi': 100,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11
})

# Suppress warnings
warnings.filterwarnings('ignore')


# Load dataset
data_path = "/kaggle/input/bank-marketing-prediction/train.csv"
df = pd.read_csv(data_path)

# Display basic information
print("Dataset Overview:")
print("-" * 50)
print(f"Shape: {df.shape}")
print("\nFirst few rows:")
display(df.head())

# Data info
print("\nDataset Information:")
print("-" * 50)
df.info()

# Check missing values
print("\nMissing Values:")
print("-" * 50)
missing = df.isnull().sum()
missing_pct = (df.isnull().mean() * 100)
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Missing Percentage': missing_pct
})

missing_df = missing_df[missing_df['Missing Count'] > 0]

# Check duplicates
duplicates = df.duplicated().sum()
print(f"\nDuplicate rows: {duplicates} ({duplicates/len(df)*100:.2f}%)")

# Identify numeric and categorical columns
num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(include=['object']).columns

print("\nFeature Types:")
print("-" * 50)
print(f"Numeric features ({len(num_cols)}): {', '.join(num_cols)}")
print(f"Categorical features ({len(cat_cols)}): {', '.join(cat_cols)}")


# Target Variable Analysis
plt.figure(figsize=(12, 5))

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Target distribution
target_dist = df['y'].value_counts()
target_pct = df['y'].value_counts(normalize=True) * 100

# Plot 1: Counts
sns.barplot(x=target_dist.index, y=target_dist.values, ax=ax1)
ax1.set_title('Target Variable Distribution (Counts)')
ax1.set_xlabel('Target (y)')
ax1.set_ylabel('Count')
for i, v in enumerate(target_dist.values):
    ax1.text(i, v, f'{v:,}', ha='center', va='bottom')

# Plot 2: Percentages
sns.barplot(x=target_pct.index, y=target_pct.values, ax=ax2)
ax2.set_title('Target Variable Distribution (%)')
ax2.set_xlabel('Target (y)')
ax2.set_ylabel('Percentage')
for i, v in enumerate(target_pct.values):
    ax2.text(i, v, f'{v:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Print statistics
print("\nTarget Variable Statistics:")
print("-" * 50)
print("\nCounts:")
print(target_dist)
print("\nPercentages:")
print(target_pct)

# Calculate imbalance ratio
imbalance_ratio = target_dist.max() / target_dist.min()
print(f"\nClass Imbalance Ratio: {imbalance_ratio:.2f}:1")


# Numerical Features Analysis
numeric_features = [col for col in num_cols if col not in ['SampleId']]

# Create subplots for each numeric feature
n_features = len(numeric_features)
n_cols = 2
n_rows = (n_features + 1) // 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
axes = axes.flatten()

for idx, col in enumerate(numeric_features):
    # Distribution plot
    sns.histplot(data=df, x=col, kde=True, ax=axes[idx])
    axes[idx].set_title(f'Distribution of {col}')
    
    # Add summary statistics
    stats = df[col].describe()
    stats_text = f'Mean: {stats["mean"]:.2f}\nStd: {stats["std"]:.2f}\nSkew: {df[col].skew():.2f}'
    axes[idx].text(0.95, 0.95, stats_text,
                  transform=axes[idx].transAxes,
                  verticalalignment='top',
                  horizontalalignment='right',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Remove empty subplots if any
for idx in range(n_features, len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.show()

# Print summary statistics
print("\nNumerical Features Summary:")
print("-" * 50)
display(df[numeric_features].describe())


# Categorical Features Analysis
categorical_features = [col for col in cat_cols if col != 'y']

# Plot distribution for each categorical feature
for col in categorical_features:
    plt.figure(figsize=(10, 5))
    
    # Calculate value counts and percentages
    value_counts = df[col].value_counts()
    value_percentages = df[col].value_counts(normalize=True) * 100
    
    # Create bar plot
    sns.barplot(x=value_counts.values, y=value_counts.index)
    plt.title(f'Distribution of {col}')
    plt.xlabel('Count')
    
    # Add percentage labels
    for i, (count, percentage) in enumerate(zip(value_counts, value_percentages)):
        plt.text(count, i, f'{count:,} ({percentage:.1f}%)', 
                va='center', ha='left', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Print target rate by category
    print(f"\nTarget Rate by {col}:")
    print("-" * 50)
    target_rates = df.groupby(col)['y'].mean().sort_values(ascending=False) * 100
    print(target_rates.to_frame('Target Rate (%)').round(2))


# Correlation Analysis
numeric_df = df[numeric_features].copy()

# Calculate correlation matrix
corr_matrix = numeric_df.corr()

# Create correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, 
            annot=True,
            fmt='.2f',
            square=True,
            cmap='RdBu',
            center=0,
            vmin=-1, 
            vmax=1)
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
plt.show()

# Print strongest correlations
print("\nStrongest Correlations:")
print("-" * 50)
# Get upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
# Find strongest correlations
strongest_corr = (upper.unstack()
                 .sort_values(key=abs, ascending=False)
                 .drop_duplicates())
print(strongest_corr[:10].to_frame('correlation'))


# 1. Feature Selection
# Drop low-value features based on EDA
df_processed = df.copy()
df_processed.drop(columns=['SampleId', 'default', 'loan'], inplace=True)

print("Feature Selection:")
print("-" * 50)
print(f"Original features: {df.shape[1]}")
print(f"Selected features: {df_processed.shape[1]}")
print("Dropped features: SampleId, default, loan")

# 2. Handle Unknown Values in Categorical Features
print("\nUnknown Values in Categorical Features:")
print("-" * 50)
for col in df_processed.select_dtypes(include='object').columns:
    if col != 'y':  # Skip target variable
        unknown_count = (df_processed[col] == 'unknown').sum()
        if unknown_count > 0:
            print(f"{col:15s}: {unknown_count:5d} ({unknown_count/len(df_processed)*100:.2f}%)")

# 3. Age Binning
df_processed['age'] = pd.cut(df_processed['age'], 
                           bins=[17, 29, 45, 60, 95], 
                           labels=['young', 'adult', 'middle_aged', 'senior'])

print("\nAge Categories Distribution:")
print("-" * 50)
print(df_processed['age'].value_counts().sort_index())


# 4. Feature Engineering and Transformations

# 4.1 Log transformation for numerical features
for col in ['balance', 'duration', 'campaign']:
    # Handle negative values by shifting
    min_val = df_processed[col].min()
    if min_val <= 0:
        shift = abs(min_val) + 1
        df_processed[col] = np.log1p(df_processed[col] + shift)
    else:
        df_processed[col] = np.log1p(df_processed[col])
    
    print(f"\n{col} transformation results:")
    print(f"Range: [{df_processed[col].min():.2f}, {df_processed[col].max():.2f}]")
    print(f"Skewness: {df_processed[col].skew():.2f}")

# 4.2 Cyclical encoding for month
month_mapping = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}

df_processed['month_num'] = df_processed['month'].map(month_mapping)
df_processed['month_sin'] = np.sin(2 * np.pi * df_processed['month_num'] / 12)
df_processed['month_cos'] = np.cos(2 * np.pi * df_processed['month_num'] / 12)
df_processed.drop(columns=['month', 'month_num'], inplace=True)

print("\nCyclical month features created: month_sin, month_cos")


# 5. Feature Encoding

# 5.1 One-hot encoding for nominal categorical variables
df_processed = pd.get_dummies(df_processed, 
                            columns=['job', 'marital', 'contact', 'poutcome', 'age'],
                            drop_first=True,
                            dtype=int)

# 5.2 Ordinal encoding for education
df_processed['education'] = df_processed['education'].map({
    'unknown': 0, 'primary': 1, 'secondary': 2, 'tertiary': 3
})

# 5.3 Binary encoding for housing and target
df_processed['housing'] = df_processed['housing'].map({'yes': 1, 'no': 0})
df_processed['y'] = df_processed['y'].astype(int)

# Rename target for clarity
df_processed.rename(columns={'y': 'subscribed'}, inplace=True)

print("Final Dataset Overview:")
print("-" * 50)
print(f"Shape: {df_processed.shape}")
print(f"Features: {df_processed.shape[1]}")
print(f"Samples: {df_processed.shape[0]}")
print("\nFeature List:")
for i, col in enumerate(df_processed.columns, 1):
    print(f"{i:2d}. {col}")


# Split features and target
X = df_processed.drop(columns=['subscribed'])
y = df_processed['subscribed']

print("Training Data Overview:")
print("-" * 50)
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"\nClass distribution:\n{y.value_counts(normalize=True) * 100}")

# Create and configure the model pipeline
pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=74, k_neighbors=5)),
    ('scaler', RobustScaler()),
    ('model', RandomForestClassifier(
        n_estimators=168,
        max_depth=14,
        max_features='sqrt',
        min_samples_split=9,
        min_samples_leaf=4,
        bootstrap=False,
        random_state=74
    ))
])

# Train the model
print("\nTraining model...")
pipeline.fit(X, y)

print("\nModel training completed!")

# Save model artifacts
output_dir = './output'
os.makedirs(output_dir, exist_ok=True)

# Save pipeline
model_path = os.path.join(output_dir, 'model_pipeline.joblib')
joblib.dump(pipeline, model_path)
print(f"\nModel saved to: {model_path}")

# Save feature names
features_path = os.path.join(output_dir, 'feature_names.joblib')
joblib.dump(X.columns, features_path)
print(f"Feature names saved to: {features_path}")

# Save preprocessed data
processed_path = os.path.join(output_dir, 'train_preprocessed.csv')
df_processed.to_csv(processed_path, index=False)
print(f"Preprocessed data saved to: {processed_path}")


# Save model artifacts
output_dir = '/kaggle/working/output'
os.makedirs(output_dir, exist_ok=True)

# Save pipeline
model_path = os.path.join(output_dir, 'model_pipeline.joblib')
joblib.dump(pipeline, model_path)
print(f"Model saved to: {model_path}")

# Save feature names
features_path = os.path.join(output_dir, 'feature_names.joblib')
joblib.dump(X.columns, features_path)
print(f"Feature names saved to: {features_path}")

# Save preprocessed training data
processed_path = "/kaggle/working/output/train_preprocessed.csv"
os.makedirs(os.path.dirname(processed_path), exist_ok=True)
df_processed.to_csv(processed_path, index=False)
print(f"Preprocessed data saved to: {processed_path}")

print("\nAll artifacts saved successfully!")
print("The model is now ready for deployment.")


# Prediction Class Implementation
class BankMarketingPredictor:
    def __init__(self, model_pipeline=pipeline, feature_names=X.columns):
        """
        Initialize the predictor with model pipeline and feature names.
        
        Args:
            model_pipeline: Trained model pipeline
            feature_names: List of feature names expected by the model
        """
        self.model = model_pipeline
        self.feature_names = feature_names
        
    def preprocess_data(self, data, is_single_sample=False):
        """
        Preprocess input data to match model requirements.
        """
        # Convert single sample to DataFrame if needed
        if is_single_sample:
            data = pd.DataFrame([data])
            
        df = data.copy()
        
        # Drop unnecessary columns if present
        drop_cols = ['SampleId', 'default', 'loan']
        df = df.drop(columns=[col for col in drop_cols if col in df.columns])
        
        # Age binning
        df['age'] = pd.cut(df['age'], 
                          bins=[17, 29, 45, 60, 95], 
                          labels=['young', 'adult', 'middle_aged', 'senior'])
        
        # Log transformations
        for col in ['balance', 'duration', 'campaign']:
            min_val = df[col].min()
            if min_val <= 0:
                shift = abs(min_val) + 1
                df[col] = np.log1p(df[col] + shift)
            else:
                df[col] = np.log1p(df[col])
        
        # Month encoding
        month_mapping = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        df['month_num'] = df['month'].map(month_mapping)
        df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)
        df.drop(columns=['month', 'month_num'], inplace=True)
        
        # Categorical encoding
        df = pd.get_dummies(df, 
                          columns=['job', 'marital', 'contact', 'poutcome', 'age'],
                          drop_first=True)
        
        # Ordinal encoding for education
        df['education'] = df['education'].map({
            'unknown': 0, 'primary': 1, 'secondary': 2, 'tertiary': 3
        })
        
        # Binary encoding for housing
        df['housing'] = df['housing'].map({'yes': 1, 'no': 0})
        
        # Ensure all expected features are present
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
                
        return df[self.feature_names]
    
    def predict(self, data):
        """
        Make predictions on input data.
        
        Args:
            data: DataFrame or dict containing input features
            
        Returns:
            DataFrame with SampleId and predictions
        """
        # Store original SampleId if present
        sample_ids = None
        if isinstance(data, pd.DataFrame) and 'SampleId' in data.columns:
            sample_ids = data['SampleId'].copy()
        elif isinstance(data, dict) and 'SampleId' in data:
            sample_ids = pd.Series([data['SampleId']])
        
        # Preprocess the data
        is_single_sample = isinstance(data, dict)
        preprocessed_data = self.preprocess_data(data, is_single_sample)
        
        # Make predictions
        predictions = self.model.predict(preprocessed_data)
        
        # Create results DataFrame
        if sample_ids is None:
            sample_ids = pd.Series(range(len(predictions)))
        
        results = pd.DataFrame({
            'SampleId': sample_ids,
            'y': predictions.astype(int)
        })
        
        return results


# Initialize predictor with our trained model
predictor = BankMarketingPredictor(pipeline, X.columns)

# Example 1: Single Sample Prediction
sample_data = {
    'age': 41,
    'job': 'entrepreneur',
    'marital': 'married',
    'education': 'tertiary',
    'default': 'no',
    'balance': 1500,
    'housing': 'yes',
    'loan': 'no',
    'contact': 'cellular',
    'day': 15,
    'month': 'may',
    'duration': 240,
    'campaign': 2,
    'pdays': -1,
    'previous': 0,
    'poutcome': 'unknown'
}

# Predict single sample
print("Single Sample Prediction:")
print("-" * 50)
single_prediction = predictor.predict(sample_data)
print(single_prediction)

print("\nPrediction explanation:")
print("0 = No subscription")
print("1 = Will subscribe")


# Example 2: Batch Predictions
# Load test data
test_data = pd.read_csv("/kaggle/input/bank-marketing-prediction/test.csv")

# Make predictions on test data
print("Batch Predictions:")
print("-" * 50)
batch_predictions = predictor.predict(test_data)

# Display summary of predictions
positives = batch_predictions['y'].sum()
total = len(batch_predictions)
print(f"\nPrediction Summary:")
print(f"Total predictions: {total}")
print(f"Positive predictions (y=1): {positives:,d} ({(positives/total)*100:.1f}%)")
print(f"Negative predictions (y=0): {total-positives:,d} ({((total-positives)/total)*100:.1f}%)")

# Save predictions to file
output_path = "/kaggle/working/output/notebook_predictions.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
batch_predictions.to_csv(output_path, index=False)
print(f"\nPredictions saved to: {output_path}")

# Display first few predictions
print("\nFirst 10 predictions:")
batch_predictions.head(10)

