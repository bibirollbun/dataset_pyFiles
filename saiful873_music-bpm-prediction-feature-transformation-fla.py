import warnings
warnings.simplefilter('ignore')


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


import polars as pl


train_pdf = pl.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test_pdf = pl.read_csv("/kaggle/input/playground-series-s5e9/test.csv")


train_pdf.head()


train_pdf.describe()


train_pdf.columns


import seaborn as sns
import matplotlib.pyplot as plt 


# Set up subplots: 2 rows x 5 columns = 10 plots
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten() # make it 1D for easy iteration

for i, col in enumerate([col for col in train_pdf.columns if col not in ['id']]):
    sns.histplot(data=train_pdf, x=col, ax=axes[i], kde=True, bins=20, color='#00BFFF')

    # Calculate mean and median
    mean_val = train_pdf[col].mean()
    median_val = train_pdf[col].median()

    # Add vertical lines
    axes[i].axvline(mean_val, color='#FF0040', linestyle='--', linewidth=2, label="mean")
    axes[i].axvline(median_val, color='#FFBF00', linestyle='-.', linewidth=2, label="median")

    # Title and legend
    axes[i].set_title(col)
    axes[i].legend()

plt.tight_layout()
plt.show()


import lightgbm as lgb
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import (MinMaxScaler, StandardScaler, RobustScaler, 
                                 PowerTransformer, QuantileTransformer)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import time


id_col = 'id',
target_col = 'BeatsPerMinute'

feature_cols = [col for col in train_pdf.columns if col not in [id_col, target_col]]

X = train_pdf[feature_cols]
y = train_pdf[target_col]


lgbm_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 1500,
    'verbose': -1,
    'random_state': 42,
    # 'device': 'gpu'
}

# Cross-validation setup
cv = KFold(n_splits=5, shuffle=True, random_state=42)


# Create all preprocessing combinations
preprocessors = {
    # 1. Baseline - no preprocessing
    'Baseline': None,
    
    # 2-4. Scaling only
    'MinMax': MinMaxScaler(),
    'Standard': StandardScaler(), 
    'Robust': RobustScaler(),
    
    # 5-6. Transform only
    'PowerTransform': PowerTransformer(method='yeo-johnson'),
    'QuantileTransform': QuantileTransformer(output_distribution='normal'),
    
    # 7-12. Combined approaches
    'MinMax_Power': Pipeline([
        ('scaler', MinMaxScaler()),
        ('transformer', PowerTransformer(method='yeo-johnson'))
    ]),
    'Standard_Power': Pipeline([
        ('scaler', StandardScaler()),
        ('transformer', PowerTransformer(method='yeo-johnson'))
    ]),
    'Robust_Power': Pipeline([
        ('scaler', RobustScaler()),
        ('transformer', PowerTransformer(method='yeo-johnson'))
    ]),
    'MinMax_Quantile': Pipeline([
        ('scaler', MinMaxScaler()),
        ('transformer', QuantileTransformer(output_distribution='normal'))
    ]),
    'Standard_Quantile': Pipeline([
        ('scaler', StandardScaler()),
        ('transformer', QuantileTransformer(output_distribution='normal'))
    ]),
    'Robust_Quantile': Pipeline([
        ('scaler', RobustScaler()),
        ('transformer', QuantileTransformer(output_distribution='normal'))
    ])
}

print(f"Total experiments: {len(preprocessors)}")


results = []

for name, preprocessor in preprocessors.items():
    print(f"\nExperiment: {name}")
    start_time = time.time()

    try:
        if preprocessor is None:
            # Baseline - no preprocessing
            model = lgb.LGBMRegressor(**lgbm_params)
            scores = cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1)
        else:
            # Create pipeline with preprocessing + model
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', lgb.LGBMRegressor(**lgbm_params))
            ])
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1)

        rmse_scores = -scores
        
        # Store results
        result = {
            'experiment': name,
            'mean_RMSE': rmse_scores.mean(),
            'std_RMSE': rmse_scores.std(),
            'min_RMSE': rmse_scores.min(),
            'max_RMSE': rmse_scores.max(),
            'cv_scores': rmse_scores.tolist(),
            'runtime_seconds': time.time() - start_time
        }
        results.append(result)
        
        print(f"  Mean RMSE: {rmse_scores.mean():.4f} (Â±{rmse_scores.std():.4f})")
        print(f"  Runtime: {result['runtime_seconds']:.1f} seconds")
        
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        result = {
            'experiment': name,
            'mean_RMSE': np.nan,
            'std_RMSE': np.nan,
            'min_RMSE': np.nan,
            'max_RMSE': np.nan,
            'cv_Scores': [],
            'runtime_seconds': time.time() - start_time,
            'error': str(e)
        }
        results.append(result)

print("\n" + "="*60)
print("All experiments completed!")
        


# Create results DataFrame
results_df = pd.DataFrame(results)

# Sort by Mean RMSE (best first)
results_df = results_df.sort_values('mean_RMSE').reset_index(drop=True)

# Display results
print("\nğŸ“Š EXPERIMENT RESULTS (Sorted by Mean RMSE)")
print("="*80)

display_cols = ['experiment', 'mean_RMSE', 'std_RMSE', 'runtime_seconds']
print(results_df[display_cols].to_string(index=False, float_format='%.4f'))

# Find best performing method
best_method = results_df.iloc[0]
baseline_rmse = results_df[results_df['experiment'] == 'Baseline']['mean_RMSE'].iloc[0]

print(f"\nğŸ�† BEST METHOD: {best_method['experiment']}")
print(f"   RMSE: {best_method['mean_RMSE']:.4f} (Â±{best_method['std_RMSE']:.4f})")
print(f"   Improvement vs Baseline: {baseline_rmse - best_method['mean_RMSE']:.4f} RMSE units")
print(f"   Improvement %: {((baseline_rmse - best_method['mean_RMSE']) / baseline_rmse * 100):.2f}%")

# Statistical significance check (simple)
print(f"\nğŸ“ˆ TOP 3 METHODS:")
for i in range(min(3, len(results_df))):
    row = results_df.iloc[i]
    improvement = baseline_rmse - row['mean_RMSE']
    improvement_pct = (improvement / baseline_rmse) * 100
    print(f"   {i+1}. {row['experiment']}: {row['mean_RMSE']:.4f} (Â±{row['std_RMSE']:.4f}) "
          f"[+{improvement_pct:.2f}% vs baseline]")


from flaml import AutoML


import sklearn
sklearn.__version__


print("Applying PowerTransform...")
transformer = PowerTransformer(method='yeo-johnson').set_output(transform='pandas')
X_transformed = transformer.fit_transform(X)
# X_transformed = pd.DataFrame(X_transformed, columns=feature_cols)


type(X_transformed)


y = y.to_pandas()


# FLAML configuration
automl = AutoML()


automl.fit(
    X_train=X_transformed,
    y_train=y,
    task='regression',
    metric='rmse',
    estimator_list=['lgbm'],  # Only LGBM
    time_budget=5400,  # 90 minutes
    retrain_full=True,
    log_file_name="lgbm_automl.log",
    eval_method="holdout",
    split_ratio=0.1,
    seed=42,
    early_stop=True,
    n_jobs=-1
)


plt.barh(automl.feature_names_in_, automl.feature_importances_)


from flaml.automl.data import get_output_from_log
import numpy as np

time_history, best_valid_loss_history, valid_loss_history, config_history, metric_history = get_output_from_log(filename='lgbm_automl.log', time_budget=5400)
plt.title('Learning Curve')
plt.xlabel('Wall Clock Time (s)')
plt.ylabel('Validation Negative RMSE')
plt.step(time_history, 1 - np.array(best_valid_loss_history), where='post')
plt.show()


# Generate predictions
print("Generating predictions on test set...")
X_test = test_pdf[feature_cols]
X_test = transformer.transform(X_test)
test_predictions = automl.predict(X_test)

print(f"âœ… Predictions generated!")
print(f"Prediction distribution:")
print(f"  - Prediction range: [{test_predictions.min():.6f}, {test_predictions.max():.6f}]")



# Create submission dataframe
submission = pd.DataFrame()

# Add ID column (adjust based on your competition format)
if 'id' in test_pdf.columns:
    submission['id'] = test_pdf['id']
elif 'Id' in test_pdf.columns:
    submission['Id'] = test_pdf['Id']
else:
    # If no ID column, create index-based IDs
    submission['id'] = range(len(test_pdf))
    print("âš ï¸�  No ID column found, using index as ID")

# Add predictions (adjust column name based on competition requirements)
# Common formats: 'y', 'target', 'prediction', 'Survived', etc.
SUBMISSION_TARGET_COLUMN = 'BeatsPerMinute'  # Change this to match your competition

if SUBMISSION_TARGET_COLUMN == 'BeatsPerMinute':
    # For binary classification, some competitions want probabilities, others want binary
    # Check your competition requirements!
    
    # Option 1: Prediction of regression (more common)
    submission[SUBMISSION_TARGET_COLUMN] = test_predictions


print(f"Submission format:")
print(f"  Columns: {list(submission.columns)}")
print(f"  Shape: {submission.shape}")
print(f"  Sample:")
print(submission.head())

# Save submission file
submission_filename = 'submission.csv'
submission.to_csv(submission_filename, index=False)

print(f"\nâœ… SUBMISSION SAVED: {submission_filename}")




