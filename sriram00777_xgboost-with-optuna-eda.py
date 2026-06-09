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


train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")


train.columns


train.head()


train.shape


test.shape


train.describe()


train.dtypes


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# Set seaborn style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 5)

# Target variable distribution
plt.figure(figsize=(15, 5))

# Subplot 1: Histogram with KDE
plt.subplot(1, 3, 1)
sns.histplot(data=train, x='accident_risk', kde=True, bins=30, alpha=0.7)
plt.axvline(train['accident_risk'].mean(), color='red', linestyle='--', 
           label=f'Mean: {train["accident_risk"].mean():.3f}')
plt.axvline(train['accident_risk'].median(), color='orange', linestyle='--', 
           label=f'Median: {train["accident_risk"].median():.3f}')
plt.title('Accident Risk Distribution')
plt.legend()

# Subplot 2: Box plot
plt.subplot(1, 3, 2)
sns.boxplot(y=train['accident_risk'])
plt.title('Accident Risk Box Plot')

# Subplot 3: Density plot
plt.subplot(1, 3, 3)
sns.kdeplot(data=train, x='accident_risk', fill=True)
plt.title('Accident Risk Density')

plt.tight_layout()
plt.show()



numerical_cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents', 'accident_risk']

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 15)

# 1. Distribution plots for all numerical variables
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.ravel()

for i, col in enumerate(numerical_cols):
    # Histogram with KDE
    sns.histplot(data=train, x=col, kde=True, ax=axes[i], bins=30, alpha=0.7)
    axes[i].axvline(train[col].mean(), color='red', linestyle='--', 
                   label=f'Mean: {train[col].mean():.3f}')
    axes[i].axvline(train[col].median(), color='orange', linestyle='--', 
                   label=f'Median: {train[col].median():.3f}')
    axes[i].set_title(f'Distribution of {col}')
    axes[i].legend()

# Remove empty subplot
if len(numerical_cols) < len(axes):
    fig.delaxes(axes[-1])

plt.tight_layout()
plt.show()



categorical_cols = ['road_type', 'lighting', 'weather', 'time_of_day', 
                   'road_signs_present', 'public_road', 'holiday', 'school_season']
object_cols = ['road_type', 'lighting', 'weather', 'time_of_day']
boolean_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 15)

# 1. Count plots for all categorical variables
fig, axes = plt.subplots(2, 4, figsize=(20, 12))
axes = axes.ravel()

for i, col in enumerate(categorical_cols):
    ax = sns.countplot(data=train, x=col, ax=axes[i])
    axes[i].set_title(f'Distribution of {col}')
    axes[i].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                   (p.get_x() + p.get_width()/2., p.get_height()),
                   ha='center', va='bottom', fontsize=10)
    
    # Add percentage labels
    total = len(train)
    for p in ax.patches:
        percentage = f'{100 * p.get_height() / total:.1f}%'
        ax.annotate(percentage, 
                   (p.get_x() + p.get_width()/2., p.get_height()/2),
                   ha='center', va='center', fontsize=8, color='white', weight='bold')

plt.tight_layout()
plt.show()



from scipy import stats



# 2. Z-Score Method (Standard Deviation)
def detect_outliers_zscore(df, columns, threshold=3):
    outlier_dict = {}
    outlier_summary = {}
    
    for col in columns:
        z_scores = np.abs(stats.zscore(df[col]))
        outliers = df[z_scores > threshold].index
        outlier_dict[col] = outliers
        
        outlier_summary[col] = {
            'count': len(outliers),
            'percentage': len(outliers) / len(df) * 100,
            'threshold': threshold,
            'mean': df[col].mean(),
            'std': df[col].std(),
            'max_zscore': z_scores.max(),
            'outlier_values': df.loc[outliers, col].values if len(outliers) > 0 else []
        }
    
    return outlier_dict, outlier_summary

# Apply Z-Score method
zscore_outliers, zscore_summary = detect_outliers_zscore(train, numerical_cols, threshold=3)

print("\n=== Z-SCORE OUTLIER DETECTION RESULTS ===")
for col, summary in zscore_summary.items():
    print(f"\n{col.upper()}:")
    print(f"  Outliers: {summary['count']} ({summary['percentage']:.2f}%)")
    print(f"  Threshold: {summary['threshold']} standard deviations")
    print(f"  Mean: {summary['mean']:.4f}, Std: {summary['std']:.4f}")
    print(f"  Max Z-Score: {summary['max_zscore']:.4f}")
    if len(summary['outlier_values']) > 0:
        print(f"  Outlier values: {summary['outlier_values'][:5]}...")  # Show first 5



numerical_cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents', 'accident_risk']
corr = train[numerical_cols].corr()

# Create mask for upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(12, 10))
sns.heatmap(corr, 
            mask=mask,                    
            annot=True,
            cmap="RdBu_r",
            center=0,
            square=True,
            fmt='.3f',
            cbar_kws={"shrink": 0.8},
            linewidths=0.5)

plt.title("Triangular Correlation Matrix", fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.show()



train.drop(['id'], axis=1, inplace=True)


train.head()


train.isnull().sum()


test.isnull().sum()


from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

def encode_categorical_features(train_df, test_df, categorical_cols):
    """
    Label encode categorical columns
    Fit on train, transform both train and test
    """
    encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        
        # Handle boolean columns
        if train_df[col].dtype == 'bool':
            train_df[col] = train_df[col].astype(str)
            test_df[col] = test_df[col].astype(str)
        
        # Fit on train and transform
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        
        # Handle unseen categories in test
        test_values = test_df[col].astype(str)
        unseen = set(test_values.unique()) - set(le.classes_)
        if unseen:
            print(f"Warning: {col} has unseen categories: {unseen}, replacing with first class")
            for unseen_val in unseen:
                test_values = test_values.replace(unseen_val, le.classes_[0])
        
        # Transform test
        test_df[col] = le.transform(test_values)
        
        # Store encoder
        encoders[col] = le
        
        print(f"{col}: {dict(zip(le.classes_, range(len(le.classes_))))}")
    
    return train_df, test_df, encoders



def normalize_features(train_df, test_df, cols, method='zscore'):
    """
    Normalize numerical columns
    Fit on train, transform both train and test
    """
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'zscore':
        scaler = StandardScaler()
    else:
        raise ValueError("Choose 'minmax' or 'zscore'.")

    # Fit scaler on train, transform train and test
    train_df[cols] = scaler.fit_transform(train_df[cols])
    test_df[cols] = scaler.transform(test_df[cols])
    
    print(f"Applied {method} normalization to: {cols}")
    return train_df, test_df, scaler



cat_cols = [col for col in train.columns if train.dtypes[col] in ['object', 'bool']]
cat_cols


train, test, cat_encoders = encode_categorical_features(train, test, cat_cols)


cat_cols


num_cols = [col for col in train.columns if col not in cat_cols and col not in ['accident_risk', 'id']]
num_cols


train, test, scalers = normalize_features(train, test, num_cols)


import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Assume you have processed train and test data with cat_cols and num_cols
# train_processed, test_processed, cat_cols, num_cols are available

# Prepare features and target
feature_cols = cat_cols + num_cols
X_train = train[feature_cols]
y_train = train['accident_risk']
X_test = test[feature_cols]

print(f"Feature columns: {feature_cols}")
print(f"Training shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

# Optuna objective function with GPU optimization
def objective(trial):
    """Optuna objective function for XGBoost hyperparameter tuning"""
    
    # Suggest hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 10),
        
        # GPU settings
        'tree_method': 'gpu_hist',
        'gpu_id': 0,
        'predictor': 'gpu_predictor',
        
        # Fixed parameters
        'random_state': 42,
        'verbosity': 0,
        'objective': 'reg:squarederror'
    }
    
    # Create XGBoost model
    model = xgb.XGBRegressor(**params)
    
    # 5-fold cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=kfold, 
                           scoring='neg_root_mean_squared_error', n_jobs=1)
    
    return scores.mean()  # Return negative RMSE (Optuna maximizes)




# Create Optuna study
print("Creating Optuna study...")
study = optuna.create_study(
    direction='maximize',  # Maximize negative RMSE (minimize RMSE)
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20)
)

# Run optimization
print("Starting Optuna hyperparameter optimization with GPU...")
print("This may take 30-60 minutes depending on n_trials...")

study.optimize(objective, n_trials=30, timeout=3600)  # 1 hour timeout

# Get best results
best_params = study.best_params
best_score = study.best_value

print(f"\n{'='*50}")
print(f"OPTIMIZATION COMPLETED")
print(f"{'='*50}")
print(f"Best RMSE: {-best_score:.4f}")
print(f"Best parameters:")
for key, value in best_params.items():
    print(f"  {key}: {value}")




# Train final model with best parameters
print(f"\nTraining final model with best parameters...")
final_params = best_params.copy()
final_params.update({
    'tree_method': 'gpu_hist',
    'gpu_id': 0,
    'predictor': 'gpu_predictor',
    'random_state': 42,
    'verbosity': 1  # Show training progress
})

final_model = xgb.XGBRegressor(**final_params)
final_model.fit(X_train, y_train)

# Validation on training data
train_pred = final_model.predict(X_train)
train_rmse = mean_squared_error(y_train, train_pred, squared=False)
train_mae = mean_absolute_error(y_train, train_pred)

print(f"\nTraining Metrics:")
print(f"  RMSE: {train_rmse:.4f}")
print(f"  MAE: {train_mae:.4f}")

# Make predictions on test set
print(f"\nGenerating predictions on test set...")
test_predictions = final_model.predict(X_test)

print(f"Prediction statistics:")
print(f"  Min: {test_predictions.min():.4f}")
print(f"  Max: {test_predictions.max():.4f}")
print(f"  Mean: {test_predictions.mean():.4f}")
print(f"  Std: {test_predictions.std():.4f}")

# Create submission file
submission = pd.DataFrame({
    'id': test['id'],  # Note: assuming 'id' column exists in test
    'accident_risk': test_predictions
})

# Save submission
submission.to_csv('submission.csv', index=False)
print(f"\n{'='*50}")
print(f"SUBMISSION CREATED")
print(f"{'='*50}")
print(f"File saved: submission.csv")
print(f"Submission shape: {submission.shape}")
print(f"\nFirst 5 predictions:")
print(submission.head())




# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

# Save feature importance
feature_importance.to_csv('feature_importance.csv', index=False)
print(f"\nFeature importance saved to: feature_importance.csv")

# Optuna study results
print(f"\nOptuna Study Summary:")
print(f"  Number of trials: {len(study.trials)}")
print(f"  Best trial number: {study.best_trial.number}")
print(f"  Study duration: {study.trials[-1].datetime_complete - study.trials[0].datetime_start}")






