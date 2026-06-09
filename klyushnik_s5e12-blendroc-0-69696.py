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


import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
from matplotlib.pyplot import figure

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import mstats
from scipy.stats.mstats import winsorize

from sklearn import preprocessing
from sklearn.preprocessing import (
    LabelEncoder,
    QuantileTransformer,
    StandardScaler,
    PowerTransformer,
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    PolynomialFeatures,
    OrdinalEncoder,
    OneHotEncoder,
    FunctionTransformer,
    KBinsDiscretizer,
)
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_classif,  
    SequentialFeatureSelector,
    SelectFromModel
)
from sklearn.model_selection import (
    StratifiedKFold,  
    KFold,
    StratifiedGroupKFold,
    RepeatedStratifiedKFold,
    RepeatedKFold,
    cross_validate,
    train_test_split,
    TimeSeriesSplit,
    cross_val_score
)
from sklearn.linear_model import (
    SGDOneClassSVM,  
    LogisticRegression,  
    RidgeClassifier,
    Ridge
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import SVC
from sklearn.ensemble import (
    HistGradientBoostingClassifier,  
    ExtraTreesClassifier,  
    GradientBoostingClassifier,  
    IsolationForest,  
    BaggingClassifier,  
    RandomForestClassifier, 
    AdaBoostClassifier  
)
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,  
    precision_score,  
    recall_score,  
    f1_score,  
    classification_report, 
    confusion_matrix,  
    roc_auc_score,  
    make_scorer
)
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

import optuna
from optuna.samplers import CmaEsSampler
from optuna.pruners import MedianPruner
import optuna.visualization as vis

from catboost import CatBoostClassifier 
import xgboost as xgb
from xgboost import XGBClassifier  
from lightgbm import LGBMClassifier  
from mlxtend.classifier import StackingClassifier, StackingCVClassifier 

from category_encoders import TargetEncoder, MEstimateEncoder
# from cuml.preprocessing import TargetEncoder  

import requests
import holidays
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from category_encoders import CatBoostEncoder, LeaveOneOutEncoder

import warnings
import re
import time
import logging
from functools import partial
from itertools import combinations
from IPython.display import Image

from functools import partial

# Visualization settings
plt.style.use('ggplot')
%matplotlib inline
matplotlib.rcParams['figure.figsize'] = (12, 8)
sns.set_context("notebook", font_scale=1.2)
sns.set_style("whitegrid")

# Pandas settings
pd.options.mode.chained_assignment = None

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Warnings configuration
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

display(train.shape, test.shape)
display(train.info(), test.info())

display(train.describe().T)
display(test.describe().T)

duplicates = train.duplicated()
print(f"Number of duplicates: {duplicates.sum()}")

duplicates = test.duplicated()
print(f"Number of duplicates: {duplicates.sum()}")

train = train.drop_duplicates()

for col in train.columns:
    pct_missing = np.mean(train[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))

display(train.head(5))


plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


print(f"\nğŸ“ˆ Target Distribution:")
target_stats = train['diagnosed_diabetes'].value_counts()
print(f"  Positive (1): {target_stats[1]:,} ({target_stats[1]/len(train)*100:.1f}%)")
print(f"  Negative (0): {target_stats[0]:,} ({target_stats[0]/len(train)*100:.1f}%)")


print("\n2ï¸�âƒ£ CATEGORICAL FEATURES ANALYSIS")
categorical_cols = train.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns: {categorical_cols}")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, col in enumerate(categorical_cols):
    if idx < len(axes):
        value_counts = train[col].value_counts()
        
        cross_tab = pd.crosstab(train[col], train['diagnosed_diabetes'], normalize='index')
        
        ax = axes[idx]
        bars = value_counts.plot(kind='bar', ax=ax, alpha=0.6, label='Count')
        ax.set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Count', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.legend(['Count'], loc='upper right')
        
        ax2 = ax.twinx()
        cross_tab[1].plot(kind='line', ax=ax2, color='red', marker='o', linewidth=2, label='Diabetes %')
        ax2.set_ylabel('Diabetes %', color='red', fontsize=10)
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(0, 1)
        ax2.legend(loc='upper left')

plt.tight_layout()
plt.show()


print("\n3ï¸�âƒ£ NUMERICAL FEATURES ANALYSIS")
numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove('id')  

biomarkers = ['age', 'bmi', 'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 
              'heart_rate', 'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol', 
              'triglycerides']
lifestyle = ['alcohol_consumption_per_week', 'physical_activity_minutes_per_week', 
             'diet_score', 'sleep_hours_per_day', 'screen_time_hours_per_day']
medical_history = ['family_history_diabetes', 'hypertension_history', 'cardiovascular_history']

print(f"  Biomarkers: {len(biomarkers)} features")
print(f"  Lifestyle: {len(lifestyle)} features")
print(f"  Medical History: {len(medical_history)} features")

fig, axes = plt.subplots(5, 4, figsize=(20, 20))
axes = axes.flatten()

all_numeric = biomarkers + lifestyle
for idx, col in enumerate(all_numeric):
    if idx < len(axes) and col in train.columns:
        ax = axes[idx]
        
        for target_value in [0, 1]:
            subset = train[train['diagnosed_diabetes'] == target_value][col]
            sns.kdeplot(subset, ax=ax, label=f'Diabetes={target_value}', fill=True, alpha=0.5)
        
        ax.set_title(f'{col}', fontsize=11, fontweight='bold')
        ax.set_xlabel('')
        ax.legend()
        
        mean_val = train[col].mean()
        median_val = train[col].median()
        ax.axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='--', alpha=0.7, label=f'Median: {median_val:.2f}')

for idx in range(len(all_numeric), len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

important_biomarkers = ['age', 'bmi', 'waist_to_hip_ratio', 'systolic_bp', 'cholesterol_total', 'triglycerides']
for idx, col in enumerate(important_biomarkers):
    if idx < len(axes):
        ax = axes[idx]
        sns.boxplot(x='diagnosed_diabetes', y=col, data=train, ax=ax)
        ax.set_title(f'{col} by Diabetes Status', fontsize=12, fontweight='bold')
        ax.set_xlabel('Diabetes (0=No, 1=Yes)')
        ax.set_ylabel(col)

plt.tight_layout()
plt.show()


print("\n4ï¸�âƒ£ CORRELATION ANALYSIS")

numeric_target_corr = {}
for col in numeric_cols:
    if col != 'diagnosed_diabetes':
        corr = train[col].corr(train['diagnosed_diabetes'])
        numeric_target_corr[col] = corr

sorted_corr = sorted(numeric_target_corr.items(), key=lambda x: abs(x[1]), reverse=True)

print("\nTop 10 features correlated with diabetes:")
for col, corr in sorted_corr[:10]:
    direction = "positive" if corr > 0 else "negative"
    print(f"  {col:35} | {corr:+.4f} ({direction})")

top_features = [col for col, _ in sorted_corr[:15]] + ['diagnosed_diabetes']
corr_matrix = train[top_features].corr()

plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix of Top Features', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()


print("\n5ï¸�âƒ£ FEATURE INTERACTIONS ANALYSIS")

plt.figure(figsize=(10, 6))
scatter = plt.scatter(train['age'], train['bmi'], 
                      c=train['diagnosed_diabetes'], 
                      alpha=0.5, cmap='coolwarm', s=10)
plt.colorbar(scatter, label='Diabetes (0=No, 1=Yes)')
plt.xlabel('Age', fontsize=12)
plt.ylabel('BMI', fontsize=12)
plt.title('BMI vs Age colored by Diabetes Status', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.show()


sample_size = min(5000, len(train))
sample_df = train.sample(sample_size, random_state=42)

important_features = ['age', 'bmi', 'waist_to_hip_ratio', 'physical_activity_minutes_per_week', 
                      'diet_score', 'diagnosed_diabetes']

sns.pairplot(sample_df[important_features], 
             hue='diagnosed_diabetes',
             palette='coolwarm',
             diag_kind='kde',
             plot_kws={'alpha': 0.6, 's': 20},
             height=2.5)
plt.suptitle('Pairplot of Important Features by Diabetes Status', y=1.02, fontsize=16, fontweight='bold')
plt.show()


print("\n6ï¸�âƒ£ OUTLIER ANALYSIS")

outlier_stats = {}
for col in biomarkers + lifestyle:
    if col in train.columns:
        Q1 = train[col].quantile(0.25)
        Q3 = train[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((train[col] < lower_bound) | (train[col] > upper_bound)).sum()
        outlier_pct = outliers / len(train) * 100
        outlier_stats[col] = outlier_pct

print("Features with >5% outliers:")
for col, pct in sorted(outlier_stats.items(), key=lambda x: x[1], reverse=True):
    if pct > 5:
        print(f"  {col:35} | {pct:.2f}% outliers")


print("\n7ï¸�âƒ£ SUBGROUP ANALYSIS")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx, cat_col in enumerate(['gender', 'ethnicity', 'education_level', 'income_level', 
                               'smoking_status', 'employment_status'][:6]):
    ax = axes[idx]
    
    group_stats = train.groupby(cat_col)['diagnosed_diabetes'].mean().sort_values(ascending=False)
    
    bars = ax.bar(range(len(group_stats)), group_stats.values, color='skyblue', alpha=0.7)
    ax.set_title(f'Diabetes Rate by {cat_col}', fontsize=12, fontweight='bold')
    ax.set_xlabel(cat_col)
    ax.set_ylabel('Diabetes Rate')
    ax.set_xticks(range(len(group_stats)))
    ax.set_xticklabels(group_stats.index, rotation=45, ha='right')
    
    for bar, value in zip(bars, group_stats.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.2%}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()


X = train.drop(columns=['id','diagnosed_diabetes'])
y = train['diagnosed_diabetes']
test = test.drop(columns=['id'])

print('âœ… Everything clear')


print("ğŸ�¯ FEATURE ENGINEERING STARTED...")
print("\n1ï¸�âƒ£ Creating Combined Features...")

def bmi_category(bmi):
    if bmi < 18.5:
        return 'underweight'
    elif bmi < 25:
        return 'normal'
    elif bmi < 30:
        return 'overweight'
    else:
        return 'obese'

for df in [X, test]:
    df['bmi_category'] = df['bmi'].apply(bmi_category)
    bmi_dummies = pd.get_dummies(df['bmi_category'], prefix='bmi_cat')
    df = pd.concat([df, bmi_dummies], axis=1)

print("   âœ… BMI categories created")


def bp_category(systolic, diastolic):
    if systolic < 120 and diastolic < 80:
        return 'normal'
    elif systolic < 130 and diastolic < 80:
        return 'elevated'
    elif systolic < 140 or diastolic < 90:
        return 'hypertension_stage1'
    else:
        return 'hypertension_stage2'

for df in [X, test]:
    df['bp_category'] = df.apply(lambda row: bp_category(row['systolic_bp'], row['diastolic_bp']), axis=1)
    bp_dummies = pd.get_dummies(df['bp_category'], prefix='bp_cat')
    df = pd.concat([df, bp_dummies], axis=1)

print("   âœ… Blood pressure categories created")


for df in [X, test]:
    
    df['cholesterol_ratio'] = df['cholesterol_total'] / df['hdl_cholesterol']
    
    df['ldl_hdl_ratio'] = df['ldl_cholesterol'] / df['hdl_cholesterol']
    
    df['non_hdl_cholesterol'] = df['cholesterol_total'] - df['hdl_cholesterol']
    
    df['tg_hdl_ratio'] = df['triglycerides'] / df['hdl_cholesterol']
    
    df['chol_risk_category'] = pd.cut(df['cholesterol_ratio'], 
                                      bins=[0, 3.5, 5, 10, 100],
                                      labels=['optimal', 'normal', 'borderline', 'high'])

print("   âœ… Cholesterol indices created")


for df in [X, test]:
    
    df['age_adjusted_bmi'] = df['bmi'] * (df['age'] / 50)  
    
    df['metabolic_score'] = (
        (df['waist_to_hip_ratio'] > 0.9).astype(int) +  
        (df['triglycerides'] > 150).astype(int) +      
        (df['hdl_cholesterol'] < 40).astype(int) +     
        (df['systolic_bp'] > 130).astype(int) +         
        (df['fasting_glucose'] > 100).astype(int) if 'fasting_glucose' in df.columns else 0  
    )
    
    df['age_adjusted_activity'] = df['physical_activity_minutes_per_week'] * (100 - df['age']) / 100

print("   âœ… Medical indices created")


print("\n2ï¸�âƒ£ Creating Feature Interactions...")

for df in [X, test]:
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    df['income_level'] = df['income_level'].map({
        'Low': 1, 'Lower-Middle': 2, 'Upper-Middle': 3, 'High': 4})
    df['age_bmi_interaction'] = df['age'] * df['bmi']
    df['age_activity_interaction'] = df['age'] * df['physical_activity_minutes_per_week']
    df['bmi_waist_interaction'] = df['bmi'] * df['waist_to_hip_ratio']
    df['bp_bmi_interaction'] = df['systolic_bp'] * df['bmi']
    df['age_cholesterol_interaction'] = df['age'] * df['cholesterol_total']
    
    df['age_gender_interaction'] = df['age'] * df['gender']
    df['bmi_income_interaction'] = df['bmi'] * df['income_level']
    
    df['age_family_history'] = df['age'] * df['family_history_diabetes']
    df['bmi_hypertension'] = df['bmi'] * df['hypertension_history']
    
    df['activity_diet_interaction'] = df['physical_activity_minutes_per_week'] * df['diet_score']
    df['sleep_screen_interaction'] = df['sleep_hours_per_day'] * df['screen_time_hours_per_day']
    df['alcohol_bmi_interaction'] = df['alcohol_consumption_per_week'] * df['bmi']

print("   âœ… Feature interactions created")


print("\n3ï¸�âƒ£ Creating Polynomial Features...")

key_features = ['age', 'bmi', 'waist_to_hip_ratio', 'systolic_bp', 'physical_activity_minutes_per_week']

for df in [X, test]:
    
    for feature in key_features:
        df[f'{feature}_squared'] = df[feature] ** 2
        df[f'{feature}_cubed'] = df[feature] ** 3
        df[f'{feature}_log'] = np.log1p(df[feature])
    
    df['inverse_bmi'] = 1 / (df['bmi'] + 1)
    df['inverse_age'] = 1 / (df['age'] + 1)
    
    df['age_sq_bmi'] = (df['age'] ** 2) * df['bmi']
    df['bmi_sq_waist'] = (df['bmi'] ** 2) * df['waist_to_hip_ratio']

print("   âœ… Polynomial features created")


print("\n4ï¸�âƒ£ Applying Target Encoding...")

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

for col in categorical_cols:
    if col in X.columns:
        X[f'{col}_original'] = X[col]
        test[f'{col}_original'] = test[col]

for col in categorical_cols:
    if col in X.columns:
        
        target_mean = y.groupby(X[col]).mean()
        
        X[f'{col}_target_enc'] = X[col].map(target_mean)
        
        test[f'{col}_target_enc'] = test[col].map(target_mean)
        
        overall_mean = y.mean()
        X[f'{col}_target_enc'].fillna(overall_mean, inplace=True)
        test[f'{col}_target_enc'].fillna(overall_mean, inplace=True)
        
        print(f"   âœ… Target encoded: {col}")


print("\n5ï¸�âƒ£ Creating Statistical Features...")

for df in [X, test]:
    df['lifestyle_score'] = (
        df['diet_score'] * 0.3 +
        (df['physical_activity_minutes_per_week'] / 100) * 0.3 +
        (df['sleep_hours_per_day'] / 8) * 0.2 +
        (1 / (df['screen_time_hours_per_day'] + 1)) * 0.1 +
        (1 / (df['alcohol_consumption_per_week'] + 1)) * 0.1
    )
    
    df['diabetes_risk_score'] = (
        (df['age'] / 100) * 0.15 +
        ((df['bmi'] - 25) / 10).clip(0, 1) * 0.15 +
        ((df['waist_to_hip_ratio'] - 0.85) / 0.15).clip(0, 1) * 0.15 +
        ((df['systolic_bp'] - 120) / 40).clip(0, 1) * 0.1 +
        (df['cholesterol_ratio'] / 5).clip(0, 1) * 0.1 +
        (df['family_history_diabetes']) * 0.15 +
        (df['hypertension_history']) * 0.1 +
        (1 - df['lifestyle_score']) * 0.1
    )
    
    df['age_group'] = pd.cut(df['age'], 
                            bins=[0, 30, 40, 50, 60, 100],
                            labels=['18-30', '31-40', '41-50', '51-60', '60+'])
    
    df['high_risk_bmi'] = ((df['bmi'] > 30) | 
                          ((df['bmi'] > 25) & (df['waist_to_hip_ratio'] > 0.85))).astype(int)

print("   âœ… Statistical features created")


print("\n6ï¸�âƒ£ Processing New Categorical Features...")

new_categorical = ['bmi_category', 'bp_category', 'chol_risk_category', 'age_group']

for col in new_categorical:
    if col in X.columns and col in test.columns:  
        if X[col].dtype.name == 'category':
            X[col] = X[col].astype('object')
        if test[col].dtype.name == 'category':
            test[col] = test[col].astype('object')
        
        freq = X[col].value_counts(normalize=True)
        X[f'{col}_freq_enc'] = X[col].map(freq)
        test[f'{col}_freq_enc'] = test[col].map(freq)
        
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        combined = pd.concat([X[col], test[col]], axis=0).fillna('Unknown')
        le.fit(combined)
        
        X[f'{col}_label_enc'] = le.transform(X[col].fillna('Unknown'))
        test[f'{col}_label_enc'] = le.transform(test[col].fillna('Unknown'))

print("   âœ… New categorical features processed")


print("\n7ï¸�âƒ£ Final Processing...")

cols_to_drop = []
for df in [X, test]:
    
    for col in categorical_cols + new_categorical:
        if col in df.columns and f'{col}_target_enc' in df.columns:
            cols_to_drop.append(col)
        if col in df.columns and f'{col}_label_enc' in df.columns:
            cols_to_drop.append(col)
    
    cols_to_drop = list(set(cols_to_drop))
    for col in cols_to_drop:
        if col in df.columns:
            df.drop(columns=[col], inplace=True, errors='ignore')
    
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)

print("   âœ… Final processing completed")


print("\n" + "="*80)
print("ğŸ�¯ FEATURE ENGINEERING SUMMARY")
print("="*80)

print(f"\nğŸ“Š Original features: {len([col for col in X.columns if 'original' in col or col in train.columns])}")
print(f"ğŸ“ˆ New features created: {len([col for col in X.columns if 'original' not in col and col not in train.columns])}")
print(f"ğŸ“‹ Total features after engineering: {X.shape[1]}")


feature_types = {
    'Medical Indices': ['cholesterol_ratio', 'ldl_hdl_ratio', 'non_hdl_cholesterol', 'tg_hdl_ratio', 
                       'metabolic_score', 'age_adjusted_bmi'],
    'Interactions': [col for col in X.columns if 'interaction' in col],
    'Polynomial': [col for col in X.columns if any(x in col for x in ['squared', 'cubed', 'log', 'inverse'])],
    'Target Encoded': [col for col in X.columns if 'target_enc' in col],
    'Statistical': ['lifestyle_score', 'diabetes_risk_score', 'high_risk_bmi'],
    'Categorical Encoded': [col for col in X.columns if any(x in col for x in ['freq_enc', 'label_enc', 'cat_'])]
}

print("\nğŸ“� Feature breakdown by type:")
for feat_type, features in feature_types.items():
    if features:
        print(f"  {feat_type:20} : {len(features)} features")

print("\nâœ… Most important new features created:")
important_new_features = [
    'cholesterol_ratio', 'ldl_hdl_ratio', 'diabetes_risk_score', 
    'lifestyle_score', 'age_bmi_interaction', 'bmi_waist_interaction',
    'age_squared', 'bmi_squared', 'metabolic_score'
]

for feature in important_new_features:
    if feature in X.columns:
        corr_with_target = X[feature].corr(y) if feature in X.columns else 0
        print(f"  {feature:30} | Correlation with target: {corr_with_target:+.4f}")

print("\n" + "="*80)
print("ğŸ�‰ FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
print("="*80)

print(f"\nğŸ“ˆ Final dataset shapes:")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Test shape: {test.shape}")

print(f"\nâœ… Data quality check:")
print(f"Missing values in X: {X.isnull().sum().sum()}")
print(f"Missing values in test: {test.isnull().sum().sum()}")


print("\nğŸ�¯ Applying variance threshold...")

X_processed = X.copy()
test_processed = test.copy()

string_cols = X_processed.select_dtypes(include=['object']).columns.tolist()
if string_cols:
    print(f"âš ï¸� Found string columns: {string_cols}")
    print("Encoding string columns before variance threshold...")
    
    for col in string_cols:
        le = LabelEncoder()
        combined = pd.concat([X_processed[col], test_processed[col]], axis=0).fillna('Unknown')
        le.fit(combined)
        X_processed[col] = le.transform(X_processed[col].fillna('Unknown'))
        test_processed[col] = le.transform(test_processed[col].fillna('Unknown'))
        print(f"   Encoded: {col}")

print(f"\nğŸ“Š Data types after encoding:")
print(f"  X dtypes: {set(X_processed.dtypes)}")
print(f"  Test dtypes: {set(test_processed.dtypes)}")

def variance_threshold(df, th):
    var_thres = VarianceThreshold(threshold=th)
    var_thres.fit(df)
    new_cols = var_thres.get_support()
    return df.iloc[:, new_cols]

try:
    X_filtered = variance_threshold(X_processed, 0.1)
    selected_features = list(X_filtered.columns)
    
    print(f"\nâœ… Features after variance threshold: {len(selected_features)}")
    print(f"   Removed {X_processed.shape[1] - len(selected_features)} low-variance features")
    
    X = X_filtered.copy()
    test = test_processed[selected_features]
    
    print(f"\nğŸ“‹ Selected features (first 10): {selected_features[:10]}...")
    
except Exception as e:
    print(f"â�Œ Error applying variance threshold: {e}")
    print("   Skipping variance threshold, using all features")
    
    selected_features = list(X_processed.columns)
    X = X_processed.copy()
    test = test_processed.copy()

print("\nğŸ“ˆ Final dataset shapes:")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}") 
print(f"test shape: {test.shape}")

print("\nğŸ“Š Data types summary:")
display(X.info())
print("\nTest set info:")
display(test.info())

print("\nğŸ�¯ Target distribution:")
print(y.value_counts(normalize=True))


def optimize_catboost_classification(X, y, n_trials=15, cv=5):
    
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 900),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'random_strength': trial.suggest_float('random_strength', 0, 2),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise']),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'task_type': 'GPU', 
            'verbose': False,
            'early_stopping_rounds': 100,
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.1, 10.0)
        }

        model = CatBoostClassifier(**params)
        
        scores = cross_val_score(model, X, y, cv=cv, 
                               scoring='roc_auc', n_jobs=1)
        
        return scores.mean()
    
    study = optuna.create_study(direction='maximize')  
    study.optimize(objective, n_trials=n_trials)
    
    return study

catboost_studies = []
for i in range(3):
    print(f"\nRunning CatBoost Classification optimization {i+1}/3")
    study = optimize_catboost_classification(X, y, n_trials=15)
    catboost_studies.append(study)
    print(f"Best trial {i+1}:")
    print(f"  ROC-AUC: {study.best_value:.5f}")
    print(f"  Params: {study.best_params}")

catboost_best_params = []

for i, study in enumerate(catboost_studies):
    params = study.best_params.copy()
    params['loss_function'] = 'Logloss'
    params['eval_metric'] = 'AUC'
    params['verbose'] = False
    params['task_type'] = 'GPU'
    catboost_best_params.append(params)
    print(f"\nBest parameters for model {i+1}:")
    for key, value in params.items():
        print(f"  {key}: {value}")

print("\n" + "="*50)
print("OPTIMIZATION SUMMARY")
print("="*50)
for i, (study, params) in enumerate(zip(catboost_studies, catboost_best_params)):
    print(f"Model {i+1}: ROC-AUC = {study.best_value:.5f}")

config_0 = {

        'iterations': 500,
        'depth': 8,
        'learning_rate': 0.01816721169731064,
        'l2_leaf_reg': 6.901118902592421,
        'border_count': 226,
        'random_strength': 1.2775341874094779,
        'bagging_temperature': 0.2039594590592117,
        'grow_policy': 'Depthwise',
        'min_data_in_leaf': 32,
        'scale_pos_weight': 2.8174668679789066,
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'verbose': False,
        'task_type': 'GPU'

}

config_1 = {

        'iterations': 700,
        'depth': 7,
        'learning_rate': 0.0536466110605888,
        'l2_leaf_reg': 9.14873264737344,
        'border_count': 184,
        'random_strength': 1.9114396745170756,
        'bagging_temperature': 0.08999986547803351,
        'grow_policy': 'SymmetricTree',
        'min_data_in_leaf': 17,
        'scale_pos_weight': 7.02365672600217,
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'verbose': False,
        'task_type': 'GPU'

}

config_2 = {
    
        'iterations': 550,
        'depth': 4,
        'learning_rate': 0.06273309004713383,
        'l2_leaf_reg': 6.373791122546431,
        'border_count': 212,
        'random_strength': 0.6760575153006142,
        'bagging_temperature': 0.0023685021637712238,
        'grow_policy': 'Depthwise',
        'min_data_in_leaf': 4,
        'scale_pos_weight': 2.8285899843487243,
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'verbose': False,
        'task_type': 'GPU'
    
}

config_3 = {
    
        'iterations': 498,
        'depth': 4,
        'learning_rate': 0.0911136296079596,
        'l2_leaf_reg': 9.992781300877866,
        'border_count': 255,
        'random_strength': 0.018385667495696006,
        'bagging_temperature': 0.6553300237177059,
        'grow_policy': 'Depthwise',
        'min_data_in_leaf': 1,
        'scale_pos_weight': 1.952645400055282,
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'verbose': False,
        'task_type': 'GPU'
    
}

catboost_best_params.append(config_0)
catboost_best_params.append(config_1)
catboost_best_params.append(config_2)
catboost_best_params.append(config_3)


def optimize_xgboost_classification(X, y, n_trials=15, cv=5):
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 900),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 1),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 2),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'max_delta_step': trial.suggest_int('max_delta_step', 0, 5),
            'eval_metric': 'auc',
            'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
            'max_leaves': trial.suggest_int('max_leaves', 32, 256),
            'max_bin': trial.suggest_int('max_bin', 128, 256),
            'tree_method': 'gpu_hist',
            'predictor': 'gpu_predictor',
            'sampling_method': trial.suggest_categorical('sampling_method', ['uniform', 'gradient_based']),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.1, 10.0)
        }
        
        model = xgb.XGBClassifier(**params)
        
        scores = cross_val_score(model, X, y, cv=cv, 
                               scoring='roc_auc', n_jobs=-1)
        
        return scores.mean()
    
    study = optuna.create_study(direction='maximize')  
    study.optimize(objective, n_trials=n_trials)
    
    return study


xgb_studies = []
for i in range(3):
    print(f"\nRunning XGBoost Classification optimization {i+1}/3")
    study = optimize_xgboost_classification(X, y, n_trials=15)
    xgb_studies.append(study)
    print(f"Best trial {i+1}:")
    print(f"  ROC-AUC: {study.best_value:.5f}")
    print(f"  Params: {study.best_params}")

xgb_best_params = []
for i, study in enumerate(xgb_studies):
    params = study.best_params.copy()
    params.update({
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'eval_metric': 'auc',
        'objective': 'binary:logistic'
    })
    xgb_best_params.append(params)
    print(f"\nXGBoost config {i+1}:")
    for key, value in params.items():
        print(f"  {key}: {value}")

print("\n" + "="*50)
print("XGBOOST OPTIMIZATION SUMMARY")
print("="*50)
for i, (study, params) in enumerate(zip(xgb_studies, xgb_best_params)):
    print(f"Model {i+1}: ROC-AUC = {study.best_value:.5f}")

config_0 = {
    
        'n_estimators': 719,
        'max_depth': 4,
        'learning_rate': 0.08728281120585796,
        'subsample': 0.7156366715152943,
        'colsample_bytree': 0.7108648017761406,
        'gamma': 0.9985164452475833,
        'min_child_weight': 3,
        'reg_lambda': 0.10609887430879072,
        'reg_alpha': 0.8290380863851698,
        'max_delta_step': 1,
        'grow_policy': 'depthwise',
        'max_leaves': 106,
        'max_bin': 164,
        'sampling_method': 'uniform',
        'scale_pos_weight': 5.60450500326871,
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'eval_metric': 'auc',
        'objective': 'binary:logistic'
    
}

config_1 = {
    
        'n_estimators': 467,
        'max_depth': 6,
        'learning_rate': 0.021008309623985727,
        'subsample': 0.8912825302791664,
        'colsample_bytree': 0.7195862505220774,
        'gamma': 0.3033947642678359,
        'min_child_weight': 10,
        'reg_lambda': 1.9972821207432154,
        'reg_alpha': 0.3724000278069234,
        'max_delta_step': 5,
        'grow_policy': 'depthwise',
        'max_leaves': 130,
        'max_bin': 252,
        'sampling_method': 'uniform',
        'scale_pos_weight': 3.3910516482718545,
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'eval_metric': 'auc',
        'objective': 'binary:logistic'    
}

config_2 = {
    
        'n_estimators': 771,
        'max_depth': 8,
        'learning_rate': 0.01184706113012854,
        'subsample': 0.70655959785533,
        'colsample_bytree': 0.8207416844980716,
        'gamma': 0.9337868245354732,
        'min_child_weight': 9,
        'reg_lambda': 1.8845924077378502,
        'reg_alpha': 0.4100461508041082,
        'max_delta_step': 4,
        'grow_policy': 'depthwise',
        'max_leaves': 159,
        'max_bin': 254,
        'sampling_method': 'gradient_based',
        'scale_pos_weight': 3.5556511418258503,
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'eval_metric': 'auc',
        'objective': 'binary:logistic'
    
}

config_3 = {
    
        'n_estimators': 341,
        'max_depth': 3,
        'learning_rate': 0.08436799580035606,
        'subsample': 0.9917906146194638,
        'colsample_bytree': 0.915392731070202,
        'gamma': 0.5717738334964471,
        'min_child_weight': 6,
        'reg_lambda': 1.3778372713157971,
        'reg_alpha': 0.3455064906102807,
        'max_delta_step': 2,
        'grow_policy': 'depthwise',
        'max_leaves': 160,
        'max_bin': 256,
        'sampling_method': 'gradient_based',
        'scale_pos_weight': 7.397107205074961,
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'eval_metric': 'auc',
        'objective': 'binary:logistic'
    
}

xgb_best_params.append(config_0)
xgb_best_params.append(config_1)
xgb_best_params.append(config_2)
xgb_best_params.append(config_3)


def optimize_lightgbm_classification(X, y, n_trials=15, cv=5):
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 900),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 128),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'min_child_weight': trial.suggest_float('min_child_weight', 0.001, 0.1),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 0, 10),
            'lambda_l1': trial.suggest_float('lambda_l1', 0, 1),
            'lambda_l2': trial.suggest_float('lambda_l2', 0, 1),
            'min_split_gain': trial.suggest_float('min_split_gain', 0, 0.2),
            'path_smooth': trial.suggest_float('path_smooth', 0, 1),
            'max_bin': trial.suggest_int('max_bin', 64, 255),
            'extra_trees': trial.suggest_categorical('extra_trees', [True, False]),
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'objective': 'binary',
            'metric': 'auc',
            'verbose': -1,
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.1, 10.0)
        }
        
        model = LGBMClassifier(**params)
        
        scores = cross_val_score(model, X, y, cv=cv, 
                               scoring='roc_auc')
        
        return scores.mean()
    
    study = optuna.create_study(direction='maximize') 
    study.optimize(objective, n_trials=n_trials)
    
    return study


lgbm_studies = []
for i in range(3):
    print(f"\nRunning LightGBM Classification optimization {i+1}/3")
    study = optimize_lightgbm_classification(X, y, n_trials=15)
    lgbm_studies.append(study)
    print(f"Best trial {i+1}:")
    print(f"  ROC-AUC: {study.best_value:.5f}")
    print(f"  Params: {study.best_params}")

lgbm_best_params = []
for i, study in enumerate(lgbm_studies):
    params = study.best_params.copy()
    params.update({
        'objective': 'binary',
        'metric': 'auc',
        'device': 'gpu',
        'verbose': -1
    })
    lgbm_best_params.append(params)
    print(f"\nLightGBM config {i+1}:")
    for key, value in params.items():
        print(f"  {key}: {value}")

print("\n" + "="*50)
print("LIGHTGBM OPTIMIZATION SUMMARY")
print("="*50)
for i, (study, params) in enumerate(zip(lgbm_studies, lgbm_best_params)):
    print(f"Model {i+1}: ROC-AUC = {study.best_value:.5f}")

config_0 = {
    
        'n_estimators': 567,
        'max_depth': 6,
        'learning_rate': 0.017610848997737235,
        'num_leaves': 20,
        'min_child_samples': 5,
        'min_child_weight': 0.07867583365026909,
        'feature_fraction': 0.5126361649285935,
        'bagging_fraction': 0.999106225684828,
        'bagging_freq': 7,
        'lambda_l1': 0.984648051201791,
        'lambda_l2': 0.9560952146250232,
        'min_split_gain': 0.1967277362694852,
        'path_smooth': 0.014026385204785045,
        'max_bin': 250,
        'extra_trees': False,
        'scale_pos_weight': 8.387051997229882,
        'objective': 'binary',
        'metric': 'auc',
        'device': 'gpu',
        'verbose': -1
    
}

config_1 = {
    
        'n_estimators': 781,
        'max_depth': 3,
        'learning_rate': 0.05707680994916189,
        'num_leaves': 87,
        'min_child_samples': 25,
        'min_child_weight': 0.024582185232500813,
        'feature_fraction': 0.6019476561921188,
        'bagging_fraction': 0.8091766935240692,
        'bagging_freq': 7,
        'lambda_l1': 0.7525057611601782,
        'lambda_l2': 0.7841594498355441,
        'min_split_gain': 0.1902553146225533,
        'path_smooth': 0.46157879112186484,
        'max_bin': 182,
        'extra_trees': False,
        'scale_pos_weight': 3.563614770787167,
        'objective': 'binary',
        'metric': 'auc',
        'device': 'gpu',
        'verbose': -1
    
}

config_2 = {
    
        'n_estimators': 451,
        'max_depth': 8,
        'learning_rate': 0.048781065327623226,
        'num_leaves': 36,
        'min_child_samples': 44,
        'min_child_weight': 0.07104539569974595,
        'feature_fraction': 0.5511355889937433,
        'bagging_fraction': 0.8921162947901531,
        'bagging_freq': 6,
        'lambda_l1': 0.5939775981038105,
        'lambda_l2': 0.3946539510111705,
        'min_split_gain': 0.10752701341644968,
        'path_smooth': 0.6142404799930528,
        'max_bin': 249,
        'extra_trees': False,
        'scale_pos_weight': 1.6942486539570902,
        'objective': 'binary',
        'metric': 'auc',
        'device': 'gpu',
        'verbose': -1
    
}

config_3 = {
    
        'n_estimators': 456,
        'max_depth': 9,
        'learning_rate': 0.005459216695843719,
        'num_leaves': 83,
        'min_child_samples': 24,
        'min_child_weight': 0.09946780130040314,
        'feature_fraction': 0.775690283359843,
        'bagging_fraction': 0.8302500189010505,
        'bagging_freq': 10,
        'lambda_l1': 0.7013647517774626,
        'lambda_l2': 0.45882699446815334,
        'min_split_gain': 0.08142167580357887,
        'path_smooth': 0.20695599450389324,
        'max_bin': 186,
        'extra_trees': False,
        'scale_pos_weight': 7.206269155440004,
        'objective': 'binary',
        'metric': 'auc',
        'device': 'gpu',
        'verbose': -1
    
}

lgbm_best_params.append(config_0)
lgbm_best_params.append(config_1)
lgbm_best_params.append(config_2)
lgbm_best_params.append(config_3)


def create_classification_ensemble(X, y, test, n_folds=5):
    FOLDS = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    all_oof = {}
    all_predictions = {}
    models = []

    for i, params in enumerate(catboost_best_params, 1):
        models.append((f'cat_{i}', CatBoostClassifier(**params)))
    
    for i, params in enumerate(xgb_best_params, 1):
        models.append((f'xgb_{i}', xgb.XGBClassifier(**params)))
    
    for i, params in enumerate(lgbm_best_params, 1):
        models.append((f'lgb_{i}', LGBMClassifier(**params)))
    
    for name, model in models:
        try:
            print(f"\nTraining {name}...")
            oof = np.zeros(len(X))
            pred = np.zeros(len(test))
            
            fold_auc_scores = []
            
            for fold, (trn_idx, val_idx) in enumerate(FOLDS.split(X, y)):
                X_train, y_train = X.iloc[trn_idx], y.iloc[trn_idx]
                X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
                
                model.fit(X_train, y_train)
                
                oof[val_idx] = model.predict_proba(X_val)[:, 1]
                pred += model.predict_proba(test)[:, 1] / FOLDS.n_splits
                
                fold_auc = roc_auc_score(y_val, oof[val_idx])
                fold_auc_scores.append(fold_auc)
                print(f'{name} - Fold {fold} AUC: {fold_auc:.4f}')
            
            all_oof[name] = oof
            all_predictions[name] = pred
            
            full_auc = roc_auc_score(y, oof)
            mean_fold_auc = np.mean(fold_auc_scores)
            std_fold_auc = np.std(fold_auc_scores)
            
            print(f'{name} - Full OOF AUC: {full_auc:.4f}')
            print(f'{name} - Mean Fold AUC: {mean_fold_auc:.4f} Â± {std_fold_auc:.4f}')
            
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            continue
    
    oof_df = pd.DataFrame(all_oof)
    predictions_df = pd.DataFrame(all_predictions)
    
    oof_df['target'] = y.values
    
    model_performance = {}
    for name in all_oof.keys():
        auc_score = roc_auc_score(y, all_oof[name])
        model_performance[name] = auc_score
    
    sorted_models = sorted(model_performance.items(), key=lambda x: x[1], reverse=True)
    
    print("\n" + "="*60)
    print("MODEL PERFORMANCE RANKING (by ROC-AUC):")
    print("="*60)
    for i, (name, auc) in enumerate(sorted_models, 1):
        print(f"{i:2d}. {name:20} AUC: {auc:.4f}")
    
    model_info = {
        'model_names': [name for name, _ in models],
        'num_models': len(all_oof),
        'features_used': list(X.columns),
        'model_performance': model_performance,
        'top_models': sorted_models[:10]
    }
    
    return oof_df, predictions_df, model_info

print("ğŸš€ Creating Classification Ensemble...")
oof_results, test_predictions, model_info = create_classification_ensemble(X, y, test)

print("\nğŸ�‰ Modeling completed successfully!")
print(f"Trained {model_info['num_models']} models")
print("OOF predictions shape:", oof_results.shape)
print("Test predictions shape:", test_predictions.shape)


def create_optimal_ensemble(oof_results, test_predictions, y):
   
    oof_predictions = oof_results.drop(['target'], axis=1, errors='ignore')
    y_true = oof_results['target'] if 'target' in oof_results else y
    
    ridge = Ridge(alpha=1.0)
    ridge.fit(oof_predictions, y_true)
    ridge_pred = ridge.predict(oof_predictions)
    ridge_auc = roc_auc_score(y_true, ridge_pred)
    
    def objective(weights):
        weighted_pred = np.dot(oof_predictions.values, weights)
        return -roc_auc_score(y_true, weighted_pred)  
    
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1)] * len(oof_predictions.columns)
    initial_weights = np.ones(len(oof_predictions.columns)) / len(oof_predictions.columns)
    
    result = minimize(objective, initial_weights, 
                     method='SLSQP', bounds=bounds, constraints=constraints)
    optimized_weights = result.x
    optimized_pred = np.dot(oof_predictions.values, optimized_weights)
    optimized_auc = roc_auc_score(y_true, optimized_pred)
    
    if ridge_auc >= optimized_auc:
        print(f"Using Ridge (AUC: {ridge_auc:.4f})")
        test_pred = ridge.predict(test_predictions)
        weights = ridge.coef_
        best_auc = ridge_auc
    else:
        print(f"Using Optimized Weights (AUC: {optimized_auc:.4f})")
        test_pred = np.dot(test_predictions.values, optimized_weights)
        weights = optimized_weights
        best_auc = optimized_auc
    
    weights = weights / weights.sum()
    
    return test_pred, weights, best_auc

test_pred, weights, best_auc = create_optimal_ensemble(oof_results, test_predictions, y)

print(f"\nBest Ensemble AUC: {best_auc:.4f}")
print("Final weights:")
for model, weight in zip(test_predictions.columns, weights):
    print(f"  {model}: {weight:.4f}")


sample = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')
sample['diagnosed_diabetes'] = test_pred
sample.to_csv('submission.csv', index=False)
sample.head(10)

