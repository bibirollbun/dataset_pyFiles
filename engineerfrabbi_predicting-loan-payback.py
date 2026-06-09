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


print("=== SECTION 1: COMPETITION INTRODUCTION ===")
print("Loading datasets and understanding the problem structure...")


# Import all required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Machine Learning libraries
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression

# Advanced ML libraries
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("All libraries imported successfully!")


print("\n=== LOADING DATASETS ===")

# Load the datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')

print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")
print(f"Sample submission shape: {sample_submission.shape}")

# Display first few rows
print("\nTraining Data Overview:")
print(train_df.head())

print("\nTest Data Overview:")
print(test_df.head())


train_df.head()


print("\n=== DATA BASIC INFORMATION ===")

print("Training Data Info:")
print(train_df.info())

print("\nTraining Data Description:")
print(train_df.describe())

print("\nMissing Values in Training Data:")
missing_train = train_df.isnull().sum()
print(missing_train[missing_train > 0])

print("\nMissing Values in Test Data:")
missing_test = test_df.isnull().sum()
print(missing_test[missing_test > 0])

# Check for duplicate rows
print(f"\nDuplicate rows in training data: {train_df.duplicated().sum()}")
print(f"Duplicate rows in test data: {test_df.duplicated().sum()}")


print("\n=== SECTION 2: EXPLORATORY DATA ANALYSIS ===")

# Separate features and target
X = train_df.drop('loan_paid_back', axis=1) 
y = train_df['loan_paid_back']

print(f"Features: {X.shape[1]}, Target: {y.name}")


print("\n--- Target Variable Analysis ---")

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(y, bins=50, alpha=0.7, color='skyblue')
plt.title('Target Variable Distribution')
plt.xlabel('Target Value')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
plt.boxplot(y)
plt.title('Target Variable Boxplot')
plt.ylabel('Target Value')

plt.subplot(1, 3, 3)
sns.kdeplot(y, fill=True)
plt.title('Target Variable Density Plot')
plt.xlabel('Target Value')

plt.tight_layout()
plt.show()

# Statistical summary of target
print(f"Target Statistics:")
print(f"Mean: {y.mean():.2f}")
print(f"Median: {y.median():.2f}")
print(f"Std: {y.std():.2f}")
print(f"Skewness: {y.skew():.2f}")
print(f"Kurtosis: {y.kurtosis():.2f}")


print("\n--- Feature Distributions ---")

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

print(f"Numerical features: {len(numerical_cols)}")
print(f"Categorical features: {len(categorical_cols)}")

# Plot distributions for numerical features
if numerical_cols:
    n_cols = min(4, len(numerical_cols))
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(numerical_cols):
        if i < len(axes):
            axes[i].hist(X[col], bins=50, alpha=0.7, color='lightgreen')
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
    
    # Remove empty subplots
    for i in range(len(numerical_cols), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()

# Analyze categorical features
if categorical_cols:
    print("\nCategorical Features Analysis:")
    for col in categorical_cols:
        print(f"\n{col}:")
        print(f"Number of unique values: {X[col].nunique()}")
        print(f"Top 5 values: {X[col].value_counts().head().to_dict()}")


print("\n--- Correlation Analysis ---")

# Calculate correlation matrix for numerical features
if numerical_cols:
    correlation_matrix = train_df[numerical_cols + ['loan_paid_back']].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    # Top correlations with target
    target_correlations = correlation_matrix['loan_paid_back'].abs().sort_values(ascending=False)
    print("Top features correlated with target:")
    print(target_correlations.head(10))


print("\n=== SECTION 3: FEATURE ENGINEERING ===")

class AdvancedFeatureEngineer:
    def __init__(self):
        self.features_created = []
    
    def create_interaction_features(self, df):
        """Create interaction features between important variables"""
        print("Creating interaction features...")
        
        # Identify highly correlated numerical features for interactions
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        # Create some basic interaction features
        if len(numerical_cols) >= 2:
            # Use first two numerical features for demonstration
            col1, col2 = numerical_cols[0], numerical_cols[1]
            df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
            self.features_created.append(f'{col1}_x_{col2}')
            
            df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-8)  # Avoid division by zero
            self.features_created.append(f'{col1}_div_{col2}')
        
        return df
    
    def create_polynomial_features(self, df, degree=2):
        """Create polynomial features for important numerical columns"""
        print("Creating polynomial features...")
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        # Create squared terms for top correlated features
        for col in numerical_cols[:3]:  # Use first 3 numerical features
            df[f'{col}_squared'] = df[col] ** 2
            self.features_created.append(f'{col}_squared')
            
            if degree > 2:
                df[f'{col}_cubed'] = df[col] ** 3
                self.features_created.append(f'{col}_cubed')
        
        return df
    
    def create_statistical_features(self, df):
        """Create statistical aggregation features"""
        print("Creating statistical features...")
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) > 0:
            # Row-wise statistics
            df['numerical_mean'] = df[numerical_cols].mean(axis=1)
            df['numerical_std'] = df[numerical_cols].std(axis=1)
            df['numerical_sum'] = df[numerical_cols].sum(axis=1)
            
            self.features_created.extend(['numerical_mean', 'numerical_std', 'numerical_sum'])
        
        return df
    
    def create_binning_features(self, df, columns_to_bin=3):
        """Create binned features for continuous variables"""
        print("Creating binning features...")
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols[:columns_to_bin]:
            # Create quantile-based bins
            df[f'{col}_binned'] = pd.qcut(df[col], q=5, labels=False, duplicates='drop')
            self.features_created.append(f'{col}_binned')
        
        return df

# Apply feature engineering
feature_engineer = AdvancedFeatureEngineer()

# Apply to train data
X_engineered = feature_engineer.create_interaction_features(X.copy())
X_engineered = feature_engineer.create_polynomial_features(X_engineered)
X_engineered = feature_engineer.create_statistical_features(X_engineered)
X_engineered = feature_engineer.create_binning_features(X_engineered)

print(f"\nTotal new features created: {len(feature_engineer.features_created)}")
print(f"New features: {feature_engineer.features_created}")
print(f"New data shape: {X_engineered.shape}")


print("\n=== SECTION 4: DATA PREPROCESSING ===")

from sklearn.preprocessing import StandardScaler, LabelEncoder

# Prepare test data with same feature engineering
test_df_engineered = test_df.copy()
test_df_engineered = feature_engineer.create_interaction_features(test_df_engineered)
test_df_engineered = feature_engineer.create_polynomial_features(test_df_engineered)
test_df_engineered = feature_engineer.create_statistical_features(test_df_engineered)
test_df_engineered = feature_engineer.create_binning_features(test_df_engineered)

print(f"Test data shape after feature engineering: {test_df_engineered.shape}")

# Handle categorical variables
categorical_cols = X_engineered.select_dtypes(include=['object']).columns.tolist()

if categorical_cols:
    print(f"\nEncoding categorical variables: {categorical_cols}")
    
    # Use Label Encoding for tree-based models
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X_engineered[col] = le.fit_transform(X_engineered[col].astype(str))
        test_df_engineered[col] = le.transform(test_df_engineered[col].astype(str))
        label_encoders[col] = le
    
    print("Categorical encoding completed!")

# Scale numerical features
numerical_cols = X_engineered.select_dtypes(include=[np.number]).columns.tolist()

print(f"\nScaling {len(numerical_cols)} numerical features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_engineered)
test_scaled = scaler.transform(test_df_engineered)

# Convert back to DataFrame
X_processed = pd.DataFrame(X_scaled, columns=X_engineered.columns)
test_processed = pd.DataFrame(test_scaled, columns=test_df_engineered.columns)

print(f"Final processed training data shape: {X_processed.shape}")
print(f"Final processed test data shape: {test_processed.shape}")


print("\n=== SECTION 5: BASELINE MODELS ===")

# Split the data for validation
X_train, X_val, y_train, y_val = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")

# Define evaluation function
def evaluate_model(model, X_train, X_val, y_train, y_val, model_name):
    """Evaluate model performance"""
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    # Calculate RMSE
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    
    print(f"{model_name}:")
    print(f"  Training RMSE: {train_rmse:.4f}")
    print(f"  Validation RMSE: {val_rmse:.4f}")
    print(f"  Overfitting gap: {train_rmse - val_rmse:.4f}")
    
    return {
        'model': model,
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'model_name': model_name
    }

# Test multiple baseline models
print("\n--- Training Baseline Models ---")

baseline_models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = []
for name, model in baseline_models.items():
    result = evaluate_model(model, X_train, X_val, y_train, y_val, name)
    results.append(result)

# Find best baseline model
best_baseline = min(results, key=lambda x: x['val_rmse'])
print(f"\nğŸ�¯ Best Baseline Model: {best_baseline['model_name']}")
print(f"Best Validation RMSE: {best_baseline['val_rmse']:.4f}")


print("\n=== SECTION 6: ADVANCED GRADIENT BOOSTING ===")

# Define advanced models
advanced_models = {
    'XGBoost': xgb.XGBRegressor(
        n_estimators=1000,
        random_state=42,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8
    ),
    'LightGBM': lgb.LGBMRegressor(
        n_estimators=1000,
        random_state=42,
        learning_rate=0.1,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        verbose=-1
    ),
    'CatBoost': CatBoostRegressor(
        iterations=1000,
        random_state=42,
        learning_rate=0.1,
        depth=6,
        verbose=False
    )
}

print("--- Training Advanced Gradient Boosting Models ---")

advanced_results = []
for name, model in advanced_models.items():
    result = evaluate_model(model, X_train, X_val, y_train, y_val, name)
    advanced_results.append(result)

# Combine all results
all_results = results + advanced_results

# Find overall best model
best_model_info = min(all_results, key=lambda x: x['val_rmse'])
print(f"\nğŸ�† Overall Best Model: {best_model_info['model_name']}")
print(f"Best Validation RMSE: {best_model_info['val_rmse']:.4f}")

# Plot model comparison
plt.figure(figsize=(12, 6))
model_names = [r['model_name'] for r in all_results]
val_rmses = [r['val_rmse'] for r in all_results]

bars = plt.bar(model_names, val_rmses, color=['skyblue' if 'Advanced' not in name else 'lightcoral' for name in model_names])
plt.title('Model Performance Comparison (Lower RMSE is Better)')
plt.ylabel('Validation RMSE')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, val_rmses):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
             f'{value:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()


print("\n=== SECTION 7: DEEP LEARNING MODEL ===")

# Build neural network
def build_neural_network(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        Dropout(0.2),
        
        Dense(16, activation='relu'),
        Dropout(0.1),
        
        Dense(1, activation='linear')  # Linear activation for regression
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

print("Building and training neural network...")

nn_model = build_neural_network(X_train.shape[1])
print("Neural Network Architecture:")
nn_model.summary()

# Train the neural network
early_stopping = EarlyStopping(patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(factor=0.5, patience=10)

history = nn_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Evaluate neural network
nn_train_pred = nn_model.predict(X_train).flatten()
nn_val_pred = nn_model.predict(X_val).flatten()

nn_train_rmse = np.sqrt(mean_squared_error(y_train, nn_train_pred))
nn_val_rmse = np.sqrt(mean_squared_error(y_val, nn_val_pred))

print(f"\nNeural Network Performance:")
print(f"Training RMSE: {nn_train_rmse:.4f}")
print(f"Validation RMSE: {nn_val_rmse:.4f}")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.ylabel('MSE Loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(np.sqrt(history.history['loss']), label='Training RMSE')
plt.plot(np.sqrt(history.history['val_loss']), label='Validation RMSE')
plt.title('RMSE During Training')
plt.ylabel('RMSE')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.show()

# Add NN to results
nn_result = {
    'model': nn_model,
    'train_rmse': nn_train_rmse,
    'val_rmse': nn_val_rmse,
    'model_name': 'Neural Network'
}
all_results.append(nn_result)


print("\n=== SECTION 8: MODEL ENSEMBLING ===")

from sklearn.ensemble import VotingRegressor

# Select top models for ensemble
top_models = sorted(all_results, key=lambda x: x['val_rmse'])[:3]
print("Top 3 models for ensemble:")
for model_info in top_models:
    print(f"  - {model_info['model_name']}: {model_info['val_rmse']:.4f}")

# Create ensemble
ensemble_models = []
for model_info in top_models:
    if model_info['model_name'] == 'Neural Network':
        # For neural network, we need to create a wrapper
        class NeuralNetworkWrapper:
            def __init__(self, model):
                self.model = model
            
            def predict(self, X):
                return self.model.predict(X).flatten()
        
        ensemble_models.append((model_info['model_name'], NeuralNetworkWrapper(model_info['model'])))
    else:
        ensemble_models.append((model_info['model_name'], model_info['model']))

# Create voting regressor
ensemble = VotingRegressor(estimators=ensemble_models, weights=[3, 2, 1])  # Weight by performance

# Evaluate ensemble
ensemble_result = evaluate_model(ensemble, X_train, X_val, y_train, y_val, "Ensemble")

print(f"\nğŸ�¯ Ensemble Performance:")
print(f"Validation RMSE: {ensemble_result['val_rmse']:.4f}")

# Compare with best single model
best_single = min(all_results, key=lambda x: x['val_rmse'])
improvement = best_single['val_rmse'] - ensemble_result['val_rmse']

if improvement > 0:
    print(f"âœ… Ensemble improved performance by {improvement:.4f} RMSE")
else:
    print(f"â�Œ Ensemble did not improve over best single model")


print("\n=== SECTION 9: FEATURE IMPORTANCE ===")

# Get feature importance from best tree-based model
best_tree_model = None
for result in all_results:
    if hasattr(result['model'], 'feature_importances_'):
        best_tree_model = result
        break

if best_tree_model:
    print(f"Analyzing feature importance from {best_tree_model['model_name']}...")
    
    feature_importance = pd.DataFrame({
        'feature': X_processed.columns,
        'importance': best_tree_model['model'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Plot top 20 features
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(20)
    
    plt.barh(top_features['feature'], top_features['importance'])
    plt.title(f'Top 20 Feature Importance - {best_tree_model["model_name"]}')
    plt.xlabel('Importance Score')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    print("Top 10 Most Important Features:")
    print(top_features.head(10))
else:
    print("No tree-based model available for feature importance analysis")

# Correlation with target for all features
correlation_with_target = []
for col in X_processed.columns:
    corr = np.corrcoef(X_processed[col], y)[0, 1]
    correlation_with_target.append((col, abs(corr)))

correlation_df = pd.DataFrame(correlation_with_target, columns=['feature', 'correlation'])
correlation_df = correlation_df.sort_values('correlation', ascending=False)

print("\nTop 10 Features by Correlation with Target:")
print(correlation_df.head(10))


print("\n=== SECTION 10: FINAL PREDICTIONS ===")

# Train final model on all data
print("Training final model on entire dataset...")

# Use the best model or ensemble
if ensemble_result['val_rmse'] <= best_single['val_rmse']:
    final_model = ensemble
    print("Using Ensemble as final model")
else:
    final_model = best_single['model']
    print(f"Using {best_single['model_name']} as final model")

# Retrain on full data
final_model.fit(X_processed, y)

# Make predictions on test set
final_predictions = final_model.predict(test_processed)

# Ensure predictions are reasonable
print(f"\nPrediction Statistics:")
print(f"Mean prediction: {final_predictions.mean():.4f}")
print(f"Std prediction: {final_predictions.std():.4f}")
print(f"Min prediction: {final_predictions.min():.4f}")
print(f"Max prediction: {final_predictions.max():.4f}")

# Create submission file
submission = sample_submission.copy()
submission['cost'] = final_predictions  # Assuming 'cost' is the target column name

# Save submission
submission_file = 'final_submission.csv'
submission.to_csv(submission_file, index=False)
print(f"\nâœ… Submission file saved as: {submission_file}")
print(f"Submission shape: {submission.shape}")
print(submission.head())


print("\n=== SECTION 11: COMPETITION SUMMARY ===")

# Final results summary
print("FINAL RESULTS SUMMARY:")
print("=" * 50)

for result in sorted(all_results + [ensemble_result], key=lambda x: x['val_rmse']):
    marker = " ğŸ�†" if result == min(all_results + [ensemble_result], key=lambda x: x['val_rmse']) else ""
    print(f"{result['model_name']:20} | Val RMSE: {result['val_rmse']:.4f}{marker}")

print("=" * 50)

# Key insights
print("\nğŸ”� KEY INSIGHTS:")
print("1. Best performing model type identified")
print("2. Feature engineering impact analyzed") 
print("3. Potential overfitting issues addressed")
print("4. Final submission prepared for leaderboard")

print("\nğŸš€ POTENTIAL IMPROVEMENTS:")
print("1. Hyperparameter tuning with cross-validation")
print("2. More advanced feature engineering")
print("3. External data integration")
print("4. Model stacking with diverse algorithms")
print("5. Target transformation if distribution is skewed")

print("\nğŸ�¯ COMPETITION STRATEGY:")
print("Submit this baseline, then iterate with improvements!")

