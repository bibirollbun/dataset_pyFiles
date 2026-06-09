import os
import sys
import torch
import warnings
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
from IPython.display import HTML
from tabulate import tabulate
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from cuml.neighbors import KNeighborsClassifier
from scipy.stats import rankdata


# Check GPU availability
print("PyTorch GPU Available:", torch.cuda.is_available())
print("GPU Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")


# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# Set random seed for reproducibility
np.random.seed(42)


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ==============================
# Data Loading
# ==============================

# Set the path for Kaggle dataset (adjust based on your Kaggle Notebook setup)
data_path = "../input/playground-series-s5e3/"


def safe_read_csv(filepath, usecols=None, dtype=None):
    """Helper function to safely read CSVs, returning None if the file is missing."""
    if os.path.exists(filepath):
        return pd.read_csv(filepath, usecols=usecols, dtype=dtype)
    else:
        print(f"File {filepath} not found.")
        return None


# Read train and test data
train_data = safe_read_csv(os.path.join(data_path, "train.csv"))
test_data = safe_read_csv(os.path.join(data_path, "test.csv"))


# Rename the column in train_data and test_data
train_data = train_data.rename(columns={'temparature': 'temperature'})
test_data = test_data.rename(columns={'temparature': 'temperature'})


# Print data info
print("Train Data Info:")
print(train_data.info())
print("\nTest Data Info:")
print(test_data.info())


# Display first 5 rows of train_data in a nice tabular format
print("\nTrain Data (First 5 Rows):")
print(tabulate(train_data.head(), headers='keys', tablefmt='psql', showindex=False))

# Display first 5 rows of test_data in a nice tabular format
print("\nTest Data (First 5 Rows):")
print(tabulate(test_data.head(), headers='keys', tablefmt='psql', showindex=False))


# Define features and target
feature_cols = ['day', 'pressure', 'maxtemp', 'temperature', 'mintemp', 'dewpoint', 
                'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']
X = train_data[feature_cols]
y = train_data['rainfall']
X_test = test_data[feature_cols]


# Split the data (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]} rows")
print(f"Validation set size: {X_val.shape[0]} rows")


def plot_data(df, plot_type, features=None, target=None, title_prefix=""):
    """
    Functional interface for plotting data.
    
    Parameters:
    - df: DataFrame or NumPy array to plot (train_data, test_data, or predictions for pred_dist)
    - plot_type: Type of plot ('distribution', 'boxplot', 'correlation', 'time_series', 'pred_dist')
    - features: List of features to plot (default: None, uses all numeric columns)
    - target: Target variable for certain plots (e.g., 'rainfall' in train_data)
    - title_prefix: Prefix for plot titles (e.g., "Train" or "Test")
    """
    # For most plot types, df should be a DataFrame, and we extract features
    if plot_type != 'pred_dist':
        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'id' in features:
                features.remove('id')
            if target in features:
                features.remove(target)
    
    if plot_type == 'distribution':
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(features, 1):
            plt.subplot(3, 5, i)
            sns.histplot(df[feature], kde=True)
            plt.title(f'{feature} Distribution')
        plt.tight_layout()
        plt.suptitle(f"{title_prefix} Feature Distributions", y=1.05)
        plt.show()
    
    elif plot_type == 'boxplot' and target is not None:
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(features, 1):
            plt.subplot(3, 5, i)
            sns.boxplot(x=target, y=feature, data=df)
            plt.title(f'{feature} by {target}')
        plt.tight_layout()
        plt.suptitle(f"{title_prefix} Boxplots by {target}", y=1.05)
        plt.show()
    
    elif plot_type == 'correlation':
        corr_matrix = df[features].corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title(f"{title_prefix} Correlation Heatmap")
        plt.show()
    
    elif plot_type == 'time_series':
        plt.figure(figsize=(12, 6))
        if target is not None:
            plt.plot(df['day'], df[target], label=target)
        else:
            plt.plot(df['day'], df['dewpoint'], label='Dewpoint')
        plt.xlabel('Day of Year')
        plt.ylabel(target if target else 'Dewpoint')
        plt.title(f"{title_prefix} Time Series")
        plt.legend()
        plt.show()
    
    elif plot_type == 'pred_dist':
        plt.figure(figsize=(8, 6))
        sns.histplot(df, kde=True, bins=50)
        plt.xlabel('Predicted Probability of Rainfall')
        plt.title(f'{title_prefix} Distribution of Predicted Probabilities')
        plt.show()


# Plot for train_data
plot_data(train_data, plot_type='distribution', features=feature_cols, title_prefix="Train")
plot_data(train_data, plot_type='boxplot', features=feature_cols, target='rainfall', title_prefix="Train")
plot_data(train_data, plot_type='correlation', features=feature_cols, title_prefix="Train")
plot_data(train_data, plot_type='time_series', target='rainfall', title_prefix="Train")


# Plot for test_data
plot_data(test_data, plot_type='distribution', features=feature_cols, title_prefix="Test")
plot_data(test_data, plot_type='correlation', features=feature_cols, title_prefix="Test")
plot_data(test_data, plot_type='time_series', title_prefix="Test")


# ==============================
# Advanced Feature Engineering
# ==============================

# Cyclical encoding for 'day'
train_data['day_sin'] = np.sin(2 * np.pi * train_data['day'] / 365)
train_data['day_cos'] = np.cos(2 * np.pi * train_data['day'] / 365)
test_data['day_sin'] = np.sin(2 * np.pi * test_data['day'] / 365)
test_data['day_cos'] = np.cos(2 * np.pi * test_data['day'] / 365)

# Add lag features
for feature in ['dewpoint', 'humidity', 'cloud', 'sunshine']:
    train_data[f'{feature}_lag1'] = train_data[feature].shift(1)
    test_data[f'{feature}_lag1'] = test_data[feature].shift(1)
    train_data[f'{feature}_lag2'] = train_data[feature].shift(2)
    test_data[f'{feature}_lag2'] = test_data[feature].shift(2)

# Add rolling averages
for feature in ['dewpoint', 'humidity', 'cloud', 'sunshine']:
    train_data[f'{feature}_roll3'] = train_data[feature].rolling(window=3).mean()
    test_data[f'{feature}_roll3'] = test_data[feature].rolling(window=3).mean()
    train_data[f'{feature}_roll7'] = train_data[feature].rolling(window=7).mean()
    test_data[f'{feature}_roll7'] = test_data[feature].rolling(window=7).mean()

# Interaction features
train_data['humidity_cloud'] = train_data['humidity'] * train_data['cloud']
test_data['humidity_cloud'] = test_data['humidity'] * test_data['cloud']
train_data['temp_diff'] = train_data['maxtemp'] - train_data['mintemp']
test_data['temp_diff'] = test_data['maxtemp'] - test_data['mintemp']
train_data['humidity_dewpoint'] = train_data['humidity'] * train_data['dewpoint']
test_data['humidity_dewpoint'] = test_data['humidity'] * test_data['dewpoint']
train_data['cloud_sunshine'] = train_data['cloud'] * train_data['sunshine']
test_data['cloud_sunshine'] = test_data['cloud'] * test_data['sunshine']

# New features inspired by RAPIDS notebook
train_data['humidity_to_dewpoint_ratio'] = train_data['humidity'] / (train_data['dewpoint'] + 1e-6)
test_data['humidity_to_dewpoint_ratio'] = test_data['humidity'] / (test_data['dewpoint'] + 1e-6)
train_data['temp_dewpoint_diff'] = train_data['temperature'] - train_data['dewpoint']
test_data['temp_dewpoint_diff'] = test_data['temperature'] - test_data['dewpoint']
train_data['humidity_squared'] = train_data['humidity'] ** 2
test_data['humidity_squared'] = test_data['humidity'] ** 2
train_data['cloud_squared'] = train_data['cloud'] ** 2
test_data['cloud_squared'] = test_data['cloud'] ** 2

# Seasonal feature
def get_season(day):
    if 60 <= day <= 151:
        return 1  # Spring
    elif 152 <= day <= 243:
        return 2  # Summer
    elif 244 <= day <= 334:
        return 3  # Fall
    else:
        return 4  # Winter
train_data['season'] = train_data['day'].apply(get_season)
test_data['season'] = test_data['day'].apply(get_season)

# Fill NaN values
train_data.fillna(0, inplace=True)
test_data.fillna(0, inplace=True)

# Update feature columns
new_features = [f'{feat}_lag1' for feat in ['dewpoint', 'humidity', 'cloud', 'sunshine']] + \
               [f'{feat}_lag2' for feat in ['dewpoint', 'humidity', 'cloud', 'sunshine']] + \
               [f'{feat}_roll3' for feat in ['dewpoint', 'humidity', 'cloud', 'sunshine']] + \
               [f'{feat}_roll7' for feat in ['dewpoint', 'humidity', 'cloud', 'sunshine']] + \
               ['day_sin', 'day_cos', 'humidity_cloud', 'temp_diff', 'season', 'humidity_dewpoint', 
                'cloud_sunshine', 'humidity_to_dewpoint_ratio', 'temp_dewpoint_diff', 
                'humidity_squared', 'cloud_squared']
feature_cols.extend(new_features)

# Debugging: Print feature_cols after feature engineering
print("\nFeature columns after feature engineering:", feature_cols)

# Update datasets with new features
X_train = train_data.loc[X_train.index, feature_cols]
X_val = train_data.loc[X_val.index, feature_cols]
X_test = test_data[feature_cols]
X = train_data[feature_cols]
y = train_data['rainfall']

# Debugging: Print features in X_train after updating datasets
print("\nFeatures in X_train after updating datasets:", X_train.columns.tolist())

# Handle outliers before scaling
X_train = X_train.clip(lower=X_train.quantile(0.01), upper=X_train.quantile(0.99), axis=1)
X_val = X_val.clip(lower=X_val.quantile(0.01), upper=X_val.quantile(0.99), axis=1)
X_test = X_test.clip(lower=X_test.quantile(0.01), upper=X_test.quantile(0.99), axis=1)
X = X.clip(lower=X.quantile(0.01), upper=X.quantile(0.99), axis=1)

# Scale features for neural network models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
X_scaled = scaler.transform(X)

print("\nFeatures in X_train after scaling (should match X_train_scaled shape):", X_train.columns.tolist())
print("X_train shape:", X_train.shape)
print("X_train_scaled shape:", X_train_scaled.shape)

# Display all columns with horizontal scrolling
print("\nTrain Data (First 5 Rows, All Columns with Scroll):")
display(HTML('<div style="overflow-x: auto;">' + 
             train_data.head().style.set_caption("Train Data (First 5 Rows, All Columns)")
             .set_table_styles([{'selector': 'caption', 'props': [('font-size', '16px')]}])
             .format(precision=2).to_html() + 
             '</div>'))

print("\nTest Data (First 5 Rows, All Columns with Scroll):")
display(HTML('<div style="overflow-x: auto;">' + 
             test_data.head().style.set_caption("Test Data (First 5 Rows, All Columns)")
             .set_table_styles([{'selector': 'caption', 'props': [('font-size', '16px')]}])
             .format(precision=2).to_html() + 
             '</div>'))


print("\nFeatures in X_train after scaling (should match X_train_scaled shape):", X_train.columns.tolist())
print("X_train shape:", X_train.shape)
print("X_train_scaled shape:", X_train_scaled.shape)


# Display all columns with horizontal scrolling
print("\nTrain Data (First 5 Rows, All Columns with Scroll):")
display(HTML('<div style="overflow-x: auto;">' + 
             train_data.head().style.set_caption("Train Data (First 5 Rows, All Columns)")
             .set_table_styles([{'selector': 'caption', 'props': [('font-size', '16px')]}])
             .format(precision=2).to_html() + 
             '</div>'))


# Display all columns with horizontal scrolling
print("\nTest Data (First 5 Rows, All Columns with Scroll):")
display(HTML('<div style="overflow-x: auto;">' + 
             test_data.head().style.set_caption("Test Data (First 5 Rows, All Columns)")
             .set_table_styles([{'selector': 'caption', 'props': [('font-size', '16px')]}])
             .format(precision=2).to_html() + 
             '</div>'))


# ======================================
# Feature Selection: Remove Highly Correlated Features
# ======================================

# Check for NaN or inf values in X_train
print("\nChecking for NaN or inf values in X_train:")
print("NaN values:", X_train.isna().sum().sum())
print("Inf values:", np.isinf(X_train).sum().sum())

# Replace inf with NaN and fill NaN with 0
X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

# Drop nearly constant features (features with very low standard deviation)
std_threshold = 1e-6
nearly_constant_features = [col for col in X_train.columns if X_train[col].std() < std_threshold]
print("Nearly constant features to drop (std < 1e-6):", nearly_constant_features)
X_train = X_train.drop(columns=nearly_constant_features)
X_val = X_val.drop(columns=nearly_constant_features)
X_test = X_test.drop(columns=nearly_constant_features)
X = X.drop(columns=nearly_constant_features)
feature_cols = [col for col in feature_cols if col not in nearly_constant_features]

# Debugging: Print features in X_train
print("\nFeatures in X_train before correlation computation:", X_train.columns.tolist())

# Compute correlation matrix
corr_matrix = X_train.corr().abs()

# Debugging: Check for NaN in corr_matrix before filling
print("\nNaN values in corr_matrix before filling:", corr_matrix.isna().sum().sum())
print("Correlation matrix (first 5x5):")
print(corr_matrix.iloc[:5, :5])

# Handle NaN values in the correlation matrix
corr_matrix = corr_matrix.fillna(0)

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Debugging: Check for NaN in upper before filling
print("\nNaN values in upper before filling:", upper.isna().sum().sum())
print("Upper triangle (first 5x5):")
print(upper.iloc[:5, :5])

# Handle NaN values in the upper triangle
upper = upper.fillna(0)

# Debugging: Check max correlation for each column
max_corrs = upper.max()
print("\nMax correlation for each feature:")
print(max_corrs[max_corrs > 0.7])

# Find features with correlation greater than 0.85
to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]
print("\nFeatures to drop due to high correlation (> 0.85):", to_drop)

# Drop highly correlated features
X_train = X_train.drop(columns=to_drop)
X_val = X_val.drop(columns=to_drop)
X_test = X_test.drop(columns=to_drop)
X = X.drop(columns=to_drop)
feature_cols = [col for col in feature_cols if col not in to_drop]

# Check for zero-variance features before scaling
zero_variance_features = [col for col in X_train.columns if X_train[col].std() == 0]
print("\nZero-variance features before scaling:", zero_variance_features)
X_train = X_train.drop(columns=zero_variance_features)
X_val = X_val.drop(columns=zero_variance_features)
X_test = X_test.drop(columns=zero_variance_features)
X = X.drop(columns=zero_variance_features)
feature_cols = [col for col in feature_cols if col not in zero_variance_features]

# Rescale after dropping features
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
X_scaled = scaler.transform(X)

# Check for NaN or inf in scaled data
print("\nChecking for NaN or inf in scaled data:")
print("X_train_scaled NaN:", np.isnan(X_train_scaled).sum())
print("X_train_scaled inf:", np.isinf(X_train_scaled).sum())
print("X_val_scaled NaN:", np.isnan(X_val_scaled).sum())
print("X_val_scaled inf:", np.isinf(X_val_scaled).sum())
print("X_test_scaled NaN:", np.isnan(X_test_scaled).sum())
print("X_test_scaled inf:", np.isinf(X_test_scaled).sum())
print("X_scaled NaN:", np.isnan(X_scaled).sum())
print("X_scaled inf:", np.isinf(X_scaled).sum())

# Replace NaN or inf in scaled data with 0
X_train_scaled = np.nan_to_num(X_train_scaled, nan=0, posinf=0, neginf=0)
X_val_scaled = np.nan_to_num(X_val_scaled, nan=0, posinf=0, neginf=0)
X_test_scaled = np.nan_to_num(X_test_scaled, nan=0, posinf=0, neginf=0)
X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)

# Handle class imbalance for LightGBM
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# Apply SMOTE to balance the training data
print("\nApplying SMOTE to balance the training data...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"After SMOTE - Positive: {(y_train_smote == 1).sum()}, Negative: {(y_train_smote == 0).sum()}")

# Run a preliminary LightGBM model to get feature importance for selection
print("\n=== Running Preliminary LightGBM for Feature Selection ===")
prelim_model = lgb.LGBMClassifier(
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    min_split_gain=0.01,
    bagging_fraction=0.8,
    feature_fraction=0.8,
    num_leaves=15,
    min_child_samples=5,
    device='gpu' if torch.cuda.is_available() else 'cpu'
)
prelim_model.fit(X_train_smote, y_train_smote)

# Get feature importance
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': prelim_model.feature_importances_
}).sort_values('importance', ascending=False)

# Select features with importance > 20
selected_features = importance[importance['importance'] > 20]['feature'].tolist()
print("\nSelected Features (importance > 20):")
print(selected_features)

# Update datasets with selected features
X_train = X_train[selected_features]
X_val = X_val[selected_features]
X_test = X_test[selected_features]
X = X[selected_features]
feature_cols = selected_features

# Rescale after feature selection
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
X_scaled = scaler.transform(X)

# Update X_train_smote with selected features
X_train_smote = X_train_smote[selected_features]
X_train_smote_scaled = scaler.transform(X_train_smote)


# ======================================
# Define Models and Hyperparameter Grids
# ======================================

# Feature weights for KNN (emphasize 'day')
WGT = {col: 24 if col == 'day' else 1 for col in feature_cols}

# Dictionary of models and their hyperparameter grids
models = {
    'XGBoost': {
        'model': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'param_grid': {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1, 0.3],
            'n_estimators': [100, 200, 300],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        },
        'use_scaling': False
    },
    'LightGBM': {
        'model': lgb.LGBMClassifier(
            random_state=42,
            scale_pos_weight=scale_pos_weight,
            min_split_gain=0.01,
            bagging_fraction=0.8,
            feature_fraction=0.8,
            num_leaves=15,
            min_child_samples=5,
            device='gpu' if torch.cuda.is_available() else 'cpu'
        ),
        'param_grid': {
            'max_depth': [7, 10],
            'learning_rate': [0.05, 0.1],
            'n_estimators': [200, 300],
            'num_leaves': [15, 31]
        },
        'use_scaling': False
    },
    'MLP': {
        'model': MLPClassifier(random_state=42, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10),
        'param_grid': {
            'hidden_layer_sizes': [(100,), (100, 50), (200, 100)],
            'learning_rate_init': [0.001, 0.01],
            'max_iter': [1000],
            'alpha': [0.0001, 0.001]
        },
        'use_scaling': True
    },
    'KNN': {
        'model': KNeighborsClassifier(n_neighbors=201, p=1),
        'param_grid': {
            'n_neighbors': [101, 201, 301],
            'weights': ['uniform', 'distance']
        },
        'use_scaling': True
    }
}


# ==============================
# Function to Run and Evaluate Models
# ==============================

def run_model(model_name, model_info, X_train, X_val, y_train, y_val, X, y, 
              X_test, use_tuning=False, title_suffix=""):
    print(f"\n=== Running {model_name} ({title_suffix}) ===")
    
    # Prepare data
    if model_info['use_scaling']:
        X_train_data = scaler.fit_transform(X_train)
        X_val_data = scaler.transform(X_val)
        X_test_data = scaler.transform(X_test)
        X_data = scaler.transform(X)
        if model_name == 'KNN':
            for c in X_train.columns:
                idx = X_train.columns.get_loc(c)
                X_train_data[:, idx] *= WGT[c]
                X_val_data[:, idx] *= WGT[c]
                X_test_data[:, idx] *= WGT[c]
                X_data[:, idx] *= WGT[c]
    else:
        X_train_data = X_train
        X_val_data = X_val
        X_test_data = X_test
        X_data = X
    
    # Initialize the model
    model = model_info['model']

    # Configure GPU for XGBoost
    if model_name == 'XGBoost':
        model.set_params(tree_method='gpu_hist', predictor='gpu_predictor')
    
    # Configure GPU for LightGBM
    if model_name == 'LightGBM':
        model.set_params(device='gpu', gpu_platform_id=0, gpu_device_id=0)
    
    # If tuning, perform GridSearchCV
    if use_tuning:
        print(f"Tuning hyperparameters for {model_name}...")
        grid_search = GridSearchCV(estimator=model, param_grid=model_info['param_grid'], 
                                   scoring='roc_auc', cv=5, verbose=1, n_jobs=-1)
        grid_search.fit(X_train_data, y_train)
        model = grid_search.best_estimator_
        print(f"Best Parameters ({model_name}):", grid_search.best_params_)
        print(f"Best ROC AUC from GridSearch ({model_name}):", grid_search.best_score_)
    else:
        model.fit(X_train_data, y_train)
    
    # Predict probabilities on validation set
    y_pred = model.predict_proba(X_val_data)[:, 1]
    
    # Check for NaN or inf in validation predictions
    print(f"\nChecking for NaN or inf in {model_name} validation predictions (y_pred):")
    print("NaN values:", np.isnan(y_pred).sum())
    print("Inf values:", np.isinf(y_pred).sum())
    
    # Replace NaN or inf in validation predictions
    y_pred = np.nan_to_num(y_pred, nan=0, posinf=0, neginf=0)
    
    # ROC AUC
    auc = roc_auc_score(y_val, y_pred)
    print(f"{model_name} ({title_suffix}) ROC AUC: {auc:.4f}")
    
    # Plot ROC AUC Curve
    fpr, tpr, thresholds = roc_curve(y_val, y_pred)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'ROC AUC Curve for {model_name} ({title_suffix})')
    plt.legend()
    plt.show()
    
    # Feature importance (for XGBoost and LightGBM)
    if model_name in ['XGBoost', 'LightGBM']:
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Check for NaN or inf in feature importance
        print(f"\nChecking for NaN or inf in {model_name} feature importance:")
        print("NaN values in importance:", feature_importance['importance'].isna().sum())
        print("Inf values in importance:", np.isinf(feature_importance['importance']).sum())
        
        # Replace NaN or inf in feature importance
        feature_importance['importance'] = feature_importance['importance'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        print(f"\nFeature Importance ({model_name} {title_suffix}):")
        print(feature_importance)
        
        # Visualize feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title(f'Feature Importance ({model_name} {title_suffix})')
        plt.show()
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_data, y, cv=5, scoring='roc_auc')
    
    # Check for NaN or inf in cross-validation scores
    print(f"\nChecking for NaN or inf in {model_name} cross-validation scores:")
    print("NaN values:", np.isnan(cv_scores).sum())
    print("Inf values:", np.isinf(cv_scores).sum())
    
    # Replace NaN or inf in cross-validation scores
    cv_scores = np.nan_to_num(cv_scores, nan=0, posinf=0, neginf=0)
    
    print(f"Cross-Validation ROC AUC Scores ({model_name} {title_suffix}): {cv_scores}")
    print(f"Mean CV ROC AUC ({model_name} {title_suffix}): {cv_scores.mean():.4f}")
    print(f"Standard Deviation ({model_name} {title_suffix}): {cv_scores.std():.4f}")
    
    # Precision-Recall Curve
    precision, recall, thresholds = precision_recall_curve(y_val, y_pred)
    avg_precision = average_precision_score(y_val, y_pred)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'Precision-Recall Curve (AP = {avg_precision:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {model_name} ({title_suffix})')
    plt.legend()
    plt.show()
    
    # Prediction Distribution
    plot_data(y_pred, plot_type='pred_dist', title_prefix=f"Validation Set ({model_name} {title_suffix})")
    
    # Predict on test data
    test_preds = model.predict_proba(X_test_data)[:, 1]
    
    # Check for NaN or inf in test predictions
    print(f"\nChecking for NaN or inf in {model_name} test predictions (test_preds):")
    print("NaN values:", np.isnan(test_preds).sum())
    print("Inf values:", np.isinf(test_preds).sum())
    
    # Replace NaN or inf in test predictions
    test_preds = np.nan_to_num(test_preds, nan=0, posinf=0, neginf=0)
    
    return model, y_pred, test_preds


# ==============================
# Run Models (Default and Tuned)
# ==============================

val_predictions = {'default': {}, 'tuned': {}}
test_predictions = {'default': {}, 'tuned': {}}
trained_models = {'default': {}, 'tuned': {}}

# Run each model (default and tuned)
for model_name, model_info in models.items():
    # Run default model
    model_default, y_pred_default, test_preds_default = run_model(
        model_name, model_info, 
        X_train_smote_scaled if model_info['use_scaling'] else X_train_smote, 
        X_val_scaled if model_info['use_scaling'] else X_val, 
        y_train_smote, y_val, 
        X_scaled if model_info['use_scaling'] else X, 
        y, 
        X_test_scaled if model_info['use_scaling'] else X_test,
        use_tuning=False, title_suffix="Default"
    )
    val_predictions['default'][model_name] = y_pred_default
    test_predictions['default'][model_name] = test_preds_default
    trained_models['default'][model_name] = model_default
    
    # Run tuned model
    model_tuned, y_pred_tuned, test_preds_tuned = run_model(
        model_name, model_info, 
        X_train_smote_scaled if model_info['use_scaling'] else X_train_smote, 
        X_val_scaled if model_info['use_scaling'] else X_val, 
        y_train_smote, y_val, 
        X_scaled if model_info['use_scaling'] else X, 
        y, 
        X_test_scaled if model_info['use_scaling'] else X_test,
        use_tuning=True, title_suffix="Tuned"
    )
    val_predictions['tuned'][model_name] = y_pred_tuned
    test_predictions['tuned'][model_name] = test_preds_tuned
    trained_models['tuned'][model_name] = model_tuned


# ==============================
# Feature Selection (Post-Training)
# ==============================

# Select top 20 features based on LightGBM default model's feature importance
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': trained_models['default']['LightGBM'].feature_importances_
}).sort_values('importance', ascending=False)
top_features = importance['feature'].head(20).tolist()
print("\nSelected Top Features (Post-Training):")
print(top_features)

# Update datasets with selected features
X_train = X_train[top_features]
X_val = X_val[top_features]
X_test = X_test[top_features]
X = X[top_features]

# Update X_train_smote with selected features
X_train_smote = X_train_smote[top_features]

# Rescale after feature selection
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
X_scaled = scaler.transform(X)
X_train_smote_scaled = scaler.transform(X_train_smote)

# Update WGT for selected features
WGT = {col: 24 if col == 'day' else 1 for col in top_features}

# Retrain tuned models with selected features
for model_name in models.keys():
    print(f"\nRetraining {model_name} with selected features...")
    model_info = models[model_name]
    if model_info['use_scaling']:
        X_train_data = X_train_smote_scaled
        X_val_data = X_val_scaled
        X_test_data = X_test_scaled
        X_data = X_scaled
        if model_name == 'KNN':
            for c in X_train.columns:
                idx = X_train.columns.get_loc(c)
                X_train_data[:, idx] *= WGT[c]
                X_val_data[:, idx] *= WGT[c]
                X_test_data[:, idx] *= WGT[c]
                X_data[:, idx] *= WGT[c]
    else:
        X_train_data = X_train_smote
        X_val_data = X_val
        X_test_data = X_test
        X_data = X
    
    if model_name == 'LightGBM':
        params = trained_models['tuned'][model_name].get_params()
        params.update({
            'random_state': 42,
            'scale_pos_weight': scale_pos_weight,
            'min_split_gain': 0.1,
            'bagging_fraction': 0.8,
            'feature_fraction': 0.8,
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0
        })
        model = lgb.LGBMClassifier(**params)
    elif model_name == 'XGBoost':
        params = trained_models['tuned'][model_name].get_params()
        params['tree_method'] = 'gpu_hist'
        params['predictor'] = 'gpu_predictor'
        model = model_info['model'].set_params(**params)
    else:
        model = trained_models['tuned'][model_name]
    
    model.fit(X_train_data, y_train_smote)
    trained_models['tuned'][model_name] = model

# Check for NaN or inf in test_data['id']
print("\nChecking for NaN or inf in test_data['id']:")
print("NaN values:", test_data['id'].isna().sum())
print("Inf values:", np.isinf(test_data['id']).sum())

# Replace NaN or inf in test_data['id']
test_data['id'] = test_data['id'].replace([np.inf, -np.inf], np.nan).fillna(0)


# ==============================
# Stacking Ensemble
# ==============================

# Original Stacking Ensemble (XGBoost + LightGBM)
print("\n=== Creating Original Stacking Ensemble (XGBoost + LightGBM) ===")
lightgbm_params = trained_models['tuned']['LightGBM'].get_params()
lightgbm_params.update({
    'random_state': 42,
    'scale_pos_weight': scale_pos_weight,
    'min_split_gain': 0.01,
    'bagging_fraction': 0.8,
    'feature_fraction': 0.8,
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0
})

base_models_original = [
    ('xgboost', trained_models['tuned']['XGBoost'].set_params(tree_method='gpu_hist', predictor='gpu_predictor')),
    ('lightgbm', lgb.LGBMClassifier(**lightgbm_params))
]
meta_model = LogisticRegression()
stacking_model_original = StackingClassifier(estimators=base_models_original, final_estimator=meta_model, cv=5)
stacking_model_original.fit(X_train_smote_scaled, y_train_smote)

# Evaluate Original Stacking Model
y_pred_stack_original = stacking_model_original.predict_proba(X_val_scaled)[:, 1]

# Check for NaN or inf in original stacking validation predictions
print("\nChecking for NaN or inf in original stacking validation predictions (y_pred_stack_original):")
print("NaN values:", np.isnan(y_pred_stack_original).sum())
print("Inf values:", np.isinf(y_pred_stack_original).sum())

# Replace NaN or inf in original stacking validation predictions
y_pred_stack_original = np.nan_to_num(y_pred_stack_original, nan=0, posinf=0, neginf=0)

stack_auc_original = roc_auc_score(y_val, y_pred_stack_original)
print(f"Original Stacking ROC AUC: {stack_auc_original:.4f}")

# Plot ROC AUC Curve for Original Stacking
fpr, tpr, thresholds = roc_curve(y_val, y_pred_stack_original)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {stack_auc_original:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC AUC Curve for Original Stacking Ensemble (XGBoost + LightGBM)')
plt.legend()
plt.show()

# Prediction Distribution for Original Stacking
plot_data(y_pred_stack_original, plot_type='pred_dist', title_prefix="Validation Set (Original Stacking Ensemble)")

# Predict on test set for Original Stacking
test_preds_stack_original = stacking_model_original.predict_proba(X_test_scaled)[:, 1]

# Check for NaN or inf in original stacking test predictions
print("\nChecking for NaN or inf in original stacking test predictions (test_preds_stack_original):")
print("NaN values:", np.isnan(test_preds_stack_original).sum())
print("Inf values:", np.isinf(test_preds_stack_original).sum())

# Replace NaN or inf in original stacking test predictions
test_preds_stack_original = np.nan_to_num(test_preds_stack_original, nan=0, posinf=0, neginf=0)

# Create submission for Original Stacking
submission_stack_original = pd.DataFrame({
    'id': test_data['id'],
    'rainfall': test_preds_stack_original
})

# Check for NaN or inf in submission_stack_original
print("\nChecking for NaN or inf in submission_stack_original:")
print("NaN values:", submission_stack_original.isna().sum())
print("Inf values:", np.isinf(submission_stack_original).sum())

# Replace NaN or inf in submission_stack_original
submission_stack_original = submission_stack_original.fillna(0)
submission_stack_original = submission_stack_original.replace([np.inf, -np.inf], 0)

submission_stack_original.to_csv('submission_stack_original.csv', index=False)
print("\nOriginal stacking ensemble submission saved as 'submission_stack_original.csv'")

# Display first 5 rows of the original stacking submission
print("\nOriginal Stacking Submission Data (First 5 Rows):")
print(tabulate(submission_stack_original.head(), headers='keys', tablefmt='psql', showindex=False))

# Stacking Ensemble with KNN
print("\n=== Creating Stacking Ensemble with KNN ===")
base_models_with_knn = [
    ('xgboost', trained_models['tuned']['XGBoost'].set_params(tree_method='gpu_hist', predictor='gpu_predictor')),
    ('lightgbm', lgb.LGBMClassifier(**lightgbm_params)),
    ('knn', trained_models['tuned']['KNN'])
]
stacking_model_with_knn = StackingClassifier(estimators=base_models_with_knn, final_estimator=meta_model, cv=5)
stacking_model_with_knn.fit(X_train_smote_scaled, y_train_smote)

# Evaluate Stacking Model with KNN
y_pred_stack_with_knn = stacking_model_with_knn.predict_proba(X_val_scaled)[:, 1]

# Check for NaN or inf in stacking validation predictions
print("\nChecking for NaN or inf in stacking validation predictions (y_pred_stack_with_knn):")
print("NaN values:", np.isnan(y_pred_stack_with_knn).sum())
print("Inf values:", np.isinf(y_pred_stack_with_knn).sum())

# Replace NaN or inf in stacking validation predictions
y_pred_stack_with_knn = np.nan_to_num(y_pred_stack_with_knn, nan=0, posinf=0, neginf=0)

stack_auc_with_knn = roc_auc_score(y_val, y_pred_stack_with_knn)
print(f"Stacking ROC AUC with KNN: {stack_auc_with_knn:.4f}")

# Plot ROC AUC Curve for Stacking with KNN
fpr, tpr, thresholds = roc_curve(y_val, y_pred_stack_with_knn)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {stack_auc_with_knn:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC AUC Curve for Stacking Ensemble with KNN')
plt.legend()
plt.show()

# Prediction Distribution for Stacking with KNN
plot_data(y_pred_stack_with_knn, plot_type='pred_dist', title_prefix="Validation Set (Stacking Ensemble with KNN)")

# Predict on test set for Stacking with KNN
test_preds_stack_with_knn = stacking_model_with_knn.predict_proba(X_test_scaled)[:, 1]

# Check for NaN or inf in stacking test predictions
print("\nChecking for NaN or inf in stacking test predictions (test_preds_stack_with_knn):")
print("NaN values:", np.isnan(test_preds_stack_with_knn).sum())
print("Inf values:", np.isinf(test_preds_stack_with_knn).sum())

# Replace NaN or inf in stacking test predictions
test_preds_stack_with_knn = np.nan_to_num(test_preds_stack_with_knn, nan=0, posinf=0, neginf=0)

# Create submission for Stacking with KNN
submission_stack_with_knn = pd.DataFrame({
    'id': test_data['id'],
    'rainfall': test_preds_stack_with_knn
})

# Check for NaN or inf in submission_stack_with_knn
print("\nChecking for NaN or inf in submission_stack_with_knn:")
print("NaN values:", submission_stack_with_knn.isna().sum())
print("Inf values:", np.isinf(submission_stack_with_knn).sum())

# Replace NaN or inf in submission_stack_with_knn
submission_stack_with_knn = submission_stack_with_knn.fillna(0)
submission_stack_with_knn = submission_stack_with_knn.replace([np.inf, -np.inf], 0)

submission_stack_with_knn.to_csv('submission_stack_with_knn.csv', index=False)
print("\nStacking ensemble with KNN submission saved as 'submission_stack_with_knn.csv'")

# Display first 5 rows of the stacking with KNN submission
print("\nStacking with KNN Submission Data (First 5 Rows):")
print(tabulate(submission_stack_with_knn.head(), headers='keys', tablefmt='psql', showindex=False))

# Compare ROC AUC scores
print("\n=== Comparing ROC AUC Scores ===")
print(f"Original Stacking ROC AUC: {stack_auc_original:.4f}")
print(f"Stacking with KNN ROC AUC: {stack_auc_with_knn:.4f}")


# ==============================
# Threshold Optimization
# ==============================

print("\n=== Optimizing Threshold ===")
thresholds = np.arange(0.1, 0.9, 0.01)
f1_scores_original = []
f1_scores_with_knn = []

# Optimize threshold for the original stacking ensemble
for thresh in thresholds:
    y_pred_binary = (y_pred_stack_original > thresh).astype(int)
    f1 = f1_score(y_val, y_pred_binary)
    f1_scores_original.append(f1)

# Optimize threshold for the stacking ensemble with KNN
for thresh in thresholds:
    y_pred_binary = (y_pred_stack_with_knn > thresh).astype(int)
    f1 = f1_score(y_val, y_pred_binary)
    f1_scores_with_knn.append(f1)

# Check for NaN or inf in f1_scores
print("\nChecking for NaN or inf in f1_scores (Original Stacking):")
print("NaN values:", np.isnan(f1_scores_original).sum())
print("Inf values:", np.isinf(f1_scores_original).sum())
print("\nChecking for NaN or inf in f1_scores (Stacking with KNN):")
print("NaN values:", np.isnan(f1_scores_with_knn).sum())
print("Inf values:", np.isinf(f1_scores_with_knn).sum())

# Replace NaN or inf in f1_scores
f1_scores_original = np.nan_to_num(f1_scores_original, nan=0, posinf=0, neginf=0)
f1_scores_with_knn = np.nan_to_num(f1_scores_with_knn, nan=0, posinf=0, neginf=0)

# Find optimal thresholds
optimal_threshold_original = thresholds[np.argmax(f1_scores_original)]
optimal_threshold_with_knn = thresholds[np.argmax(f1_scores_with_knn)]
print(f"Optimal threshold (Original Stacking): {optimal_threshold_original:.2f}")
print(f"Optimal threshold (Stacking with KNN): {optimal_threshold_with_knn:.2f}")

# Apply thresholds to test predictions
test_preds_binary_original = (test_preds_stack_original > optimal_threshold_original).astype(int)
test_preds_binary_with_knn = (test_preds_stack_with_knn > optimal_threshold_with_knn).astype(int)

# Create submissions for binary predictions
submission_binary_original = pd.DataFrame({
    'id': test_data['id'],
    'rainfall': test_preds_binary_original
})
submission_binary_with_knn = pd.DataFrame({
    'id': test_data['id'],
    'rainfall': test_preds_binary_with_knn
})

# Check for NaN or inf in submission_binary_original
print("\nChecking for NaN or inf in submission_binary_original:")
print("NaN values:", submission_binary_original.isna().sum())
print("Inf values:", np.isinf(submission_binary_original).sum())

# Check for NaN or inf in submission_binary_with_knn
print("\nChecking for NaN or inf in submission_binary_with_knn:")
print("NaN values:", submission_binary_with_knn.isna().sum())
print("Inf values:", np.isinf(submission_binary_with_knn).sum())

# Replace NaN or inf in submission_binary
submission_binary_original = submission_binary_original.fillna(0)
submission_binary_original = submission_binary_original.replace([np.inf, -np.inf], 0)
submission_binary_with_knn = submission_binary_with_knn.fillna(0)
submission_binary_with_knn = submission_binary_with_knn.replace([np.inf, -np.inf], 0)

# Save binary submissions
submission_binary_original.to_csv('submission_binary_original.csv', index=False)
submission_binary_with_knn.to_csv('submission_binary_with_knn.csv', index=False)
print("Binary submission (Original Stacking) saved as 'submission_binary_original.csv'")
print("Binary submission (Stacking with KNN) saved as 'submission_binary_with_knn.csv'")


# ==============================
# Calibrated Model
# ==============================

# Calibrate the tuned XGBoost model
calibrated_model = CalibratedClassifierCV(
    trained_models['tuned']['XGBoost'].set_params(tree_method='gpu_hist', predictor='gpu_predictor'),
    method='sigmoid',
    cv=5
)

# Fit the calibrated model using scaled data (to match the stacking ensembles)
calibrated_model.fit(X_train_smote_scaled, y_train_smote)

# Predict calibrated probabilities on the validation set
y_pred_calibrated = calibrated_model.predict_proba(X_val_scaled)[:, 1]

# Check for NaN or inf in calibrated validation predictions
print("\nChecking for NaN or inf in calibrated validation predictions (y_pred_calibrated):")
print("NaN values:", np.isnan(y_pred_calibrated).sum())
print("Inf values:", np.isinf(y_pred_calibrated).sum())

# Replace NaN or inf in calibrated validation predictions
y_pred_calibrated = np.nan_to_num(y_pred_calibrated, nan=0, posinf=0, neginf=0)

# Calculate ROC AUC for the calibrated model on the validation set
calibrated_auc = roc_auc_score(y_val, y_pred_calibrated)
print(f"Calibrated Model ROC AUC: {calibrated_auc:.4f}")

# Plot ROC AUC Curve for the calibrated model
fpr, tpr, thresholds = roc_curve(y_val, y_pred_calibrated)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {calibrated_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC AUC Curve for Calibrated XGBoost Model')
plt.legend()
plt.show()

# Prediction Distribution for the calibrated model on the validation set
plot_data(y_pred_calibrated, plot_type='pred_dist', title_prefix="Validation Set (Calibrated Model)")

# Predict calibrated probabilities on the test set
test_preds_calibrated = calibrated_model.predict_proba(X_test_scaled)[:, 1]

# Check for NaN or inf in calibrated test predictions
print("\nChecking for NaN or inf in calibrated test predictions (test_preds_calibrated):")
print("NaN values:", np.isnan(test_preds_calibrated).sum())
print("Inf values:", np.isinf(test_preds_calibrated).sum())

# Replace NaN or inf in calibrated test predictions
test_preds_calibrated = np.nan_to_num(test_preds_calibrated, nan=0, posinf=0, neginf=0)

# Create submission file for calibrated model
submission_calibrated = pd.DataFrame({
    'id': test_data['id'],
    'rainfall': test_preds_calibrated
})

# Check for NaN or inf in submission_calibrated
print("\nChecking for NaN or inf in submission_calibrated:")
print("NaN values:", submission_calibrated.isna().sum())
print("Inf values:", np.isinf(submission_calibrated).sum())

# Replace NaN or inf in submission_calibrated
submission_calibrated = submission_calibrated.fillna(0)
submission_calibrated = submission_calibrated.replace([np.inf, -np.inf], 0)

submission_calibrated.to_csv('submission_calibrated.csv', index=False)
print("Calibrated submission file created: 'submission_calibrated.csv'")


# ==============================
# Ensemble with Public Submission
# ==============================

# Ensemble with public submission
from scipy.stats import rankdata

# Replace with the actual path to a high-scoring public submission
public_submission = pd.read_csv("/path/to/public_submission.csv")  # e.g., LB 0.915
public_preds = public_submission['rainfall'].values

# Ensemble
ensemble_preds = -0.25 * rankdata(test_preds_stack_original) + 1.25 * rankdata(public_preds)
ensemble_preds = rankdata(ensemble_preds) / len(ensemble_preds)

# Create submission
submission_ensemble = pd.DataFrame({
    'id': test_data['id'],
    'rainfall': ensemble_preds
})

# Check for NaN or inf in submission_ensemble
print("\nChecking for NaN or inf in submission_ensemble:")
print("NaN values:", submission_ensemble.isna().sum())
print("Inf values:", np.isinf(submission_ensemble).sum())

# Replace NaN or inf in submission_ensemble
submission_ensemble = submission_ensemble.fillna(0)
submission_ensemble = submission_ensemble.replace([np.inf, -np.inf], 0)

submission_ensemble.to_csv('submission_ensemble.csv', index=False)
print("\nEnsemble submission saved as 'submission_ensemble.csv'")


# ==============================
# Visualizations for Submissions
# ==============================

print("\n=== Visualizing Submission Predictions ===")

# 1. Distribution of Predicted Probabilities Across Submissions
plt.figure(figsize=(12, 6))
sns.histplot(submission_stack_original['rainfall'], kde=True, label='Original Stacking', color='blue', alpha=0.5, bins=50)
sns.histplot(submission_stack_with_knn['rainfall'], kde=True, label='Stacking with KNN', color='orange', alpha=0.5, bins=50)
sns.histplot(submission_calibrated['rainfall'], kde=True, label='Calibrated', color='green', alpha=0.5, bins=50)
plt.xlabel('Predicted Probability of Rainfall')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted Probabilities (Original Stacking vs. Stacking with KNN vs. Calibrated)')
plt.legend()
plt.show()

# Note: submission_binary contains binary predictions (0 or 1), so we'll plot its distribution separately
plt.figure(figsize=(12, 6))
sns.countplot(x='rainfall', data=submission_binary_original, alpha=0.5, color='blue', label='Original Stacking')
sns.countplot(x='rainfall', data=submission_binary_with_knn, alpha=0.5, color='orange', label='Stacking with KNN')
plt.xlabel('Binary Prediction (0 = No Rain, 1 = Rain)')
plt.ylabel('Count')
plt.title('Distribution of Binary Predictions (Original Stacking vs. Stacking with KNN)')
plt.legend()
plt.show()

# 2. Predicted Probabilities Over Time (Day)
# Merge test_data['day'] with the submission predictions
submission_stack_original_with_day = submission_stack_original.copy()
submission_stack_original_with_day['day'] = test_data['day']
submission_stack_with_knn_with_day = submission_stack_with_knn.copy()
submission_stack_with_knn_with_day['day'] = test_data['day']
submission_calibrated_with_day = submission_calibrated.copy()
submission_calibrated_with_day['day'] = test_data['day']
submission_binary_original_with_day = submission_binary_original.copy()
submission_binary_original_with_day['day'] = test_data['day']
submission_binary_with_knn_with_day = submission_binary_with_knn.copy()
submission_binary_with_knn_with_day['day'] = test_data['day']

plt.figure(figsize=(12, 6))
plt.scatter(submission_stack_original_with_day['day'], submission_stack_original_with_day['rainfall'], 
            label='Original Stacking', alpha=0.5, color='blue', s=10)
plt.scatter(submission_stack_with_knn_with_day['day'], submission_stack_with_knn_with_day['rainfall'], 
            label='Stacking with KNN', alpha=0.5, color='orange', s=10)
plt.scatter(submission_calibrated_with_day['day'], submission_calibrated_with_day['rainfall'], 
            label='Calibrated', alpha=0.5, color='green', s=10)
plt.xlabel('Day of Year')
plt.ylabel('Predicted Probability of Rainfall')
plt.title('Predicted Probabilities Over Time (Day)')
plt.legend()
plt.show()

# Plot binary predictions over time
plt.figure(figsize=(12, 6))
plt.scatter(submission_binary_original_with_day['day'], submission_binary_original_with_day['rainfall'], 
            label='Original Stacking', alpha=0.5, color='blue', s=10)
plt.scatter(submission_binary_with_knn_with_day['day'], submission_binary_with_knn_with_day['rainfall'], 
            label='Stacking with KNN', alpha=0.5, color='orange', s=10)
plt.xlabel('Day of Year')
plt.ylabel('Binary Prediction (0 = No Rain, 1 = Rain)')
plt.title('Binary Predictions Over Time (Day)')
plt.legend()
plt.show()

# 3. Comparison of Predictions (Binary vs. Probability)
plt.figure(figsize=(12, 6))
plt.scatter(submission_stack_original['rainfall'], submission_binary_original['rainfall'], 
            label='Original Stacking vs. Binary (Original Stacking)', alpha=0.5, color='blue', s=10)
plt.scatter(submission_stack_with_knn['rainfall'], submission_binary_with_knn['rainfall'], 
            label='Stacking with KNN vs. Binary (Stacking with KNN)', alpha=0.5, color='orange', s=10)
plt.axvline(x=optimal_threshold_original, color='blue', linestyle='--', label=f'Threshold Original Stacking ({optimal_threshold_original:.2f})')
plt.axvline(x=optimal_threshold_with_knn, color='orange', linestyle='--', label=f'Threshold Stacking with KNN ({optimal_threshold_with_knn:.2f})')
plt.xlabel('Predicted Probability')
plt.ylabel('Binary Prediction (0 = No Rain, 1 = Rain)')
plt.title('Comparison of Probabilities vs. Binary Predictions')
plt.legend()
plt.show()

# 4. Validation Set: Predicted Probabilities vs. True Labels
# Create a DataFrame with the predictions
val_predictions_df = pd.DataFrame({
    'True Rainfall': y_val,
    'Predicted Probability (Original Stacking)': y_pred_stack_original,
    'Predicted Probability (Stacking with KNN)': y_pred_stack_with_knn,
    'Predicted Probability (Calibrated)': y_pred_calibrated
})

# Melt the DataFrame to long format for seaborn
val_predictions_melted = val_predictions_df.melt(id_vars='True Rainfall', 
                                                value_vars=['Predicted Probability (Original Stacking)', 
                                                            'Predicted Probability (Stacking with KNN)', 
                                                            'Predicted Probability (Calibrated)'], 
                                                var_name='Model', 
                                                value_name='Predicted Probability')

# Plot the boxplot using hue to differentiate models
plt.figure(figsize=(12, 6))
sns.boxplot(x='True Rainfall', y='Predicted Probability', hue='Model', data=val_predictions_melted, 
            palette={'Predicted Probability (Original Stacking)': 'blue', 
                     'Predicted Probability (Stacking with KNN)': 'orange', 
                     'Predicted Probability (Calibrated)': 'green'})
plt.title('Validation Set: Predicted Probabilities vs. True Rainfall Labels')
plt.xlabel('True Rainfall (0 = No Rain, 1 = Rain)')
plt.ylabel('Predicted Probability of Rainfall')
plt.legend(title='Model')
plt.show()

# 5. Feature Influence on Predictions
key_features = ['cloud', 'humidity', 'sunshine']
for feature in key_features:
    plt.figure(figsize=(8, 6))
    plt.scatter(test_data[feature], submission_stack_original['rainfall'], 
                label='Original Stacking', alpha=0.5, color='blue', s=10)
    plt.scatter(test_data[feature], submission_stack_with_knn['rainfall'], 
                label='Stacking with KNN', alpha=0.5, color='orange', s=10)
    plt.scatter(test_data[feature], submission_calibrated['rainfall'], 
                label='Calibrated', alpha=0.5, color='green', s=10)
    plt.xlabel(feature.capitalize())
    plt.ylabel('Predicted Probability of Rainfall')
    plt.title(f'Influence of {feature.capitalize()} on Predicted Rainfall Probability')
    plt.legend()
    plt.show()

# Display first 5 rows of submissions with tabulate
print("\nOriginal Stacking Submission Data (First 5 Rows):")
print(tabulate(submission_stack_original.head(), headers='keys', tablefmt='psql', showindex=False))

print("\nStacking with KNN Submission Data (First 5 Rows):")
print(tabulate(submission_stack_with_knn.head(), headers='keys', tablefmt='psql', showindex=False))

print("\nBinary Submission (Original Stacking) Data (First 5 Rows):")
print(tabulate(submission_binary_original.head(), headers='keys', tablefmt='psql', showindex=False))

print("\nBinary Submission (Stacking with KNN) Data (First 5 Rows):")
print(tabulate(submission_binary_with_knn.head(), headers='keys', tablefmt='psql', showindex=False))

print("\nCalibrated Submission Data (First 5 Rows):")
print(tabulate(submission_calibrated.head(), headers='keys', tablefmt='psql', showindex=False))

print("\nEnsemble Submission Data (First 5 Rows):")
print(tabulate(submission_ensemble.head(), headers='keys', tablefmt='psql', showindex=False))

