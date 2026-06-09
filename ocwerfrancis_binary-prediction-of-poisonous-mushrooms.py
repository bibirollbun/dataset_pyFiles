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


import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (12, 8)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv("/kaggle/input/playground-series-s4e8/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s4e8/test.csv")
submission_df = pd.read_csv("/kaggle/input/playground-series-s4e8/sample_submission.csv")

train_df.shape, test_df.shape, submission_df.shape


sample = int(0.8 * len(train_df))
train_df = train_df[sample:]


train_df.shape,test_df.shape


train_df.info()


train_df.describe().T


# 1. Check for missing values
print(train_df.isnull().sum())

# 2. Examine target distribution
print(train_df['class'].value_counts())


for col in train_df.select_dtypes(include='object').columns:
    if col != 'class':
        print(f"{col}: {train_df[col].nunique()} unique values")


train_df=train_df.drop(['id'],axis=1)


round(train_df['class'].value_counts(normalize=True) *100)


train_df.hist(bins=60,figsize=(30,20))


corr_matrix = train_df.corr(numeric_only=True)

plt.figure(figsize=(12,8))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
)


numeric_df = [col for col in train_df.columns if train_df[col].dtype != 'O']
sns.pairplot(train_df[numeric_df],diag_kind='kde',corner=True)
plt.suptitle('Scatter Matrix (Seaborn Pairplot)', fontsize=22, y=1.02)
plt.show()



categorical_cols = [col for col in train_df.columns.drop('class') if train_df[col].dtype == 'O']
numerical_cols = [col for col in train_df.columns if train_df[col].dtype != 'O']
print("FEATURE TYPE SUMMARY")
print("="*80)
print(f"\n Numerical features ({len(numerical_cols)}):")
for i, col in enumerate(numerical_cols, 1):
    print(f"   {i}. {col}")

print(f"\n Categorical features ({len(categorical_cols)}):")
for i, col in enumerate(categorical_cols, 1):
    print(f"   {i}. {col}")

print(f"\n Total predictive features: {len(numerical_cols) + len(categorical_cols)}")


# Statistical summary of numerical features
print("NUMERICAL FEATURES - STATISTICAL SUMMARY")
print("="*80)

numerical_stats = train_df[numerical_cols].describe().T
numerical_stats['missing'] = train_df[numerical_cols].isnull().sum().values
numerical_stats['skewness'] = train_df[numerical_cols].skew().values
numerical_stats['kurtosis'] = train_df[numerical_cols].kurtosis().values

display(numerical_stats.style.background_gradient(cmap='coolwarm', subset=['mean', 'std', 'skewness', 'kurtosis']))


# Distribution of numerical features
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
axes = axes.flatten()

for idx, col in enumerate(numerical_cols):
    # Plot distribution with KDE
    axes[idx].hist(train_df[col], bins=50, alpha=0.6, color='steelblue', edgecolor='black', density=True, label='Histogram')
    
    # Add KDE
    train_df[col].plot(kind='kde', ax=axes[idx], color='red', linewidth=2, label='KDE')
    
    axes[idx].set_title(f'{col} Distribution', fontsize=13, fontweight='bold', pad=10)
    axes[idx].set_xlabel(col, fontsize=11)
    axes[idx].set_ylabel('Density', fontsize=11)
    axes[idx].grid(alpha=0.3, linestyle='--')
    
    # Add statistics box
    mean_val = train_df[col].mean()
    median_val = train_df[col].median()
    std_val = train_df[col].std()
    
    axes[idx].axvline(mean_val, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Mean: {mean_val:.2f}')
    axes[idx].axvline(median_val, color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'Median: {median_val:.2f}')
    
    axes[idx].legend(fontsize=9, loc='upper right')

plt.suptitle('Numerical Features Distribution Analysis', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.show()


# Categorical features summary
print("CATEGORICAL FEATURES - DETAILED ANALYSIS")
print("="*80)

for col in categorical_cols:
    print(f"Feature: {col.upper()}")
    print(f"{'='*80}")
    print(f"Unique values: {train_df[col].nunique()}")
    print(f"Most common: {train_df[col].mode()[0]}")
    print(f"\nValue Counts:")
    
    value_counts_df = pd.DataFrame({
        'Value': train_df[col].value_counts().index,
        'Count': train_df[col].value_counts().values,
        'Percentage': (train_df[col].value_counts(normalize=True) * 100).values
    })
    display(value_counts_df.head(10).style.background_gradient(cmap='Blues', subset=['Count', 'Percentage']))


# Outlier detection using IQR method
print("OUTLIER DETECTION (IQR METHOD)")
print("="*80)

outlier_summary = []

for col in numerical_cols:
    Q1 = train_df[col].quantile(0.25)
    Q3 = train_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = train_df[(train_df[col] < lower_bound) | (train_df[col] > upper_bound)][col]
    outlier_count = len(outliers)
    outlier_pct = (outlier_count / len(train_df)) * 100
    
    outlier_summary.append({
        'Feature': col,
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'Lower_Bound': lower_bound,
        'Upper_Bound': upper_bound,
        'Outlier_Count': outlier_count,
        'Outlier_Percentage': outlier_pct
    })

outlier_df = pd.DataFrame(outlier_summary).sort_values('Outlier_Percentage', ascending=False)
display(outlier_df.style.background_gradient(cmap='Reds', subset=['Outlier_Count', 'Outlier_Percentage']))

print(f"\n Outlier Summary:")
print(f"   - Total features with outliers: {(outlier_df['Outlier_Count'] > 0).sum()}")
print(f"   - Average outlier percentage: {outlier_df['Outlier_Percentage'].mean():.2f}%")


train_df = train_df.copy()
test_df = test_df.copy()


def create_composite_features(df):
    df_new = df.copy()
    
    # 1. Size-related features
    df_new['cap_area'] = np.pi * (df['cap-diameter'] / 2) ** 2
    df_new['stem_cross_section'] = np.pi * (df['stem-width'] / 2) ** 2
    df_new['stem_height_to_width'] = df['stem-height'] / (df['stem-width'] + 1e-6)  # Avoid division by zero
    df_new['size_ratio'] = df['cap-diameter'] / (df['stem-height'] + 1e-6)
    
    return df_new

train_df = create_composite_features(train_df)
test_df = create_composite_features(test_df)


from sklearn.model_selection import train_test_split


train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42)


print('train_df.shape :', train_df.shape)
print('val_df.shape :', val_df.shape)
print('test_df.shape :', test_df.shape)


train_df.columns


input_cols = list(train_df.columns[1:])
input_cols


target_col = 'class'
target_col


# Training dataset inputs and target

train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_col].copy() 

# Validation dataset inputs and target

val_inputs = val_df[input_cols].copy()
val_targets = val_df[target_col].copy() 

# Testing dataset inputs and target

test_inputs = test_df[input_cols].copy()


numerical_cols = [var for var in train_inputs.columns if train_inputs[var].dtype != 'O']
numerical_cols


category_cols = [var for var in train_inputs.columns if train_inputs[var].dtype == 'O']

category_cols


train_inputs[category_cols].isnull().sum()


train_inputs[numerical_cols].isnull().sum()


from sklearn.impute import SimpleImputer


numerical_imputer = SimpleImputer(strategy='mean')


numerical_imputer.fit(train_inputs[numerical_cols])


train_inputs[numerical_cols] = numerical_imputer.transform(train_inputs[numerical_cols])
val_inputs[numerical_cols] = numerical_imputer.transform(val_inputs[numerical_cols])
test_inputs[numerical_cols] = numerical_imputer.transform(test_inputs[numerical_cols])


categorical_imputer = SimpleImputer(strategy='most_frequent')


categorical_imputer.fit(train_inputs[category_cols])


train_inputs[category_cols] = categorical_imputer.transform(train_inputs[category_cols])
val_inputs[category_cols] = categorical_imputer.transform(val_inputs[category_cols])
test_inputs[category_cols] = categorical_imputer.transform(test_inputs[category_cols])


from sklearn.preprocessing import OneHotEncoder


encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(train_inputs[category_cols])
encoded_cols = list(encoder.get_feature_names_out(category_cols))


train_inputs[encoded_cols] = encoder.transform(train_inputs[category_cols])
val_inputs[encoded_cols] = encoder.transform(val_inputs[category_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[category_cols])


from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
scaler.fit(train_inputs[numerical_cols])


train_inputs[numerical_cols] = scaler.transform(train_inputs[numerical_cols])
val_inputs[numerical_cols] = scaler.transform(val_inputs[numerical_cols])
test_inputs[numerical_cols] = scaler.transform(test_inputs[numerical_cols])


X_train = train_inputs[numerical_cols + encoded_cols]
X_val = val_inputs[numerical_cols + encoded_cols]
X_test = test_inputs[numerical_cols + encoded_cols]


import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import matthews_corrcoef, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


from catboost import CatBoostClassifier

# CatBoost handles string labels and categorical features automatically
catboost_params = {
    'iterations': 2000,               
    'learning_rate': 0.03,            
    'depth': 14,                       
    'l2_leaf_reg': 8,                 
    'random_seed': 42,
    'auto_class_weights': 'Balanced',
    'eval_metric': 'MCC',             
    'bootstrap_type': 'Bayesian',     
    'bagging_temperature': 0.3,
    'colsample_bylevel': 0.8,
    'early_stopping_rounds': 200,
    'use_best_model': True,
    'verbose': 100,
}

print("Training CatBoost with K-Fold Cross Validation")

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
mcc_scores = []
cb_models = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, train_targets)):
    print(f"\nFold {fold + 1}")
    
    X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_fold_train, y_fold_val = train_targets.iloc[train_idx], train_targets.iloc[val_idx]
    
    # Identify categorical feature indices
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
    cat_features_indices = [i for i, col in enumerate(X_train.columns) if col in categorical_features]
    
    # Train CatBoost
    model = CatBoostClassifier(**catboost_params)
    
    model.fit(
        X_fold_train, y_fold_train,
        eval_set=(X_fold_val, y_fold_val),
        cat_features=cat_features_indices,
        verbose=100
    )
    
    # Predictions
    val_pred = model.predict(X_fold_val)
    
    # Calculate MCC
    mcc = matthews_corrcoef(y_fold_val, val_pred)
    mcc_scores.append(mcc)
    cb_models.append(model)
    
    print(f"Fold {fold + 1} MCC: {mcc:.4f}")

print(f"\nCatBoost Mean MCC: {np.mean(mcc_scores):.4f} (+/- {np.std(mcc_scores):.4f})")


test_preds = model.predict(X_test)


submission_df.head()


submission_df['class'] = test_preds

# Verify the update
print("Updated submission preview:")
print(submission_df.head())
print(f"\nSubmission shape: {submission_df.shape}")

# Save the updated submission
submission_df.to_csv('submission.csv', index=False)
print("Submission file saved as 'submission.csv'")




