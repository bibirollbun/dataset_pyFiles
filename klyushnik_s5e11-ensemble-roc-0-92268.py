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


def PolynomialFeatures_labeled(input_df,power):
   
    poly = preprocessing.PolynomialFeatures(power)
    output_nparray = poly.fit_transform(input_df)
    powers_nparray = poly.powers_

    input_feature_names = list(input_df.columns)
    target_feature_names = ["Constant Term"]
    for feature_distillation in powers_nparray[1:]:
        intermediary_label = ""
        final_label = ""
        for i in range(len(input_feature_names)):
            if feature_distillation[i] == 0:
                continue
            else:
                variable = input_feature_names[i]
                power = feature_distillation[i]
                intermediary_label = "%s+%d" % (variable,power)
                if final_label == "":         #If the final label isn't yet specified
                    final_label = intermediary_label
                else:
                    final_label = final_label + "x" + intermediary_label
        target_feature_names.append(final_label)
    output_df = pd.DataFrame(output_nparray, columns = target_feature_names)
    return output_df

def variance_threshold(df,th):
    var_thres=VarianceThreshold(threshold=th)
    var_thres.fit(df)
    new_cols = var_thres.get_support()
    return df.iloc[:,new_cols]
   
def optimize_memory_usage(df, print_size=True):
    """
    Optimizes memory usage in a DataFrame by downcasting numeric columns.

    Parameters:
        df (pd.DataFrame): The DataFrame to optimize.
        print_size (bool): If True, prints memory usage before and after optimization.

    Returns:
        pd.DataFrame: The optimized DataFrame.
    """
    # Types for optimization.
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    
    # Memory usage size before optimize (Mb).
    before_size = df.memory_usage().sum() / 1024**2
    
    for column in df.columns:
        column_type = df[column].dtype
        
        if column_type in numerics:
            try:
                if str(column_type).startswith('int'):
                    df[column] = pd.to_numeric(df[column], downcast='integer')
                else:
                    df[column] = pd.to_numeric(df[column], downcast='float')
                logger.info(f"Optimized column {column}: {column_type} -> {df[column].dtype}")
            except Exception as e:
                logger.error(f"Failed to optimize column {column}: {e}")
    
    # Memory usage size after optimize (Mb).
    after_size = df.memory_usage().sum() / 1024**2
    
    if print_size:
        print(
            'Memory usage size: before {:5.4f} Mb - after {:5.4f} Mb ({:.1f}%).'.format(
                before_size, after_size, 100 * (before_size - after_size) / before_size
            )
        )
    
    return df

def categorize_variable(df, column, labels):
    
    if len(labels) != 3:
        raise ValueError("3 type")
    
    bins = [-float('inf'), 
            df[column].quantile(0.25), 
            df[column].quantile(0.75), 
            float('inf')]
    
    df[f'{column}_group'] = pd.cut(df[column], bins=bins, labels=labels)
    return df

def replace_outliers_with_mean(df, threshold=3):

    df_clean = df.copy()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        
        z_scores = np.abs(stats.zscore(df[col], nan_policy='omit')) 
        
        mean_val = df[col][z_scores <= threshold].mean()
        
        df_clean[col] = np.where(z_scores > threshold, mean_val, df[col])
        
    return df_clean


train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')

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


for col in train.select_dtypes(include=[np.number]).columns:
    if np.any(np.isinf(train[col])):
        print(f"Found infinite values in {col}, replacing with NaN")
        train[col] = train[col].replace([np.inf, -np.inf], np.nan)
        
for col in test.select_dtypes(include=[np.number]).columns:
    if np.any(np.isinf(test[col])):
        print(f"Found infinite values in test {col}, replacing with NaN")
        test[col] = test[col].replace([np.inf, -np.inf], np.nan)

for col in train.columns:
    pct_missing = np.mean(train[col].isnull())
    if pct_missing > 0:
        print(f'{col} - {round(pct_missing*100)}%')

for col in train.columns:
    if train[col].isnull().any():
        if train[col].dtype in ['int64', 'float64']:
            fill_value = train[col].median()
            train[col] = train[col].fillna(fill_value)
            test[col] = test[col].fillna(fill_value)
            print(f"Filled {col} with median: {fill_value}")
        else:
            fill_value = train[col].mode()[0] if len(train[col].mode()) > 0 else 'Unknown'
            train[col] = train[col].fillna(fill_value)
            test[col] = test[col].fillna(fill_value)
            print(f"Filled {col} with mode: {fill_value}")

print("\nâœ… Final check for missing and infinite values:")
print(f"Missing values in train: {train.isnull().sum().sum()}")
print(f"Missing values in test: {test.isnull().sum().sum()}")
print(f"Infinite values in train: {np.any(np.isinf(train.select_dtypes(include=[np.number])))}")
print(f"Infinite values in test: {np.any(np.isinf(test.select_dtypes(include=[np.number])))}")


train = optimize_memory_usage(train)
test = optimize_memory_usage(test)


# Setting style for beautiful EDA
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
matplotlib.rcParams['figure.figsize'] = (15, 10)

# Create copies of data for analysis
train_eda = train.copy()
test_eda = test.copy()

print("ğŸ�¯ COMPREHENSIVE EDA OF CREDIT DATA")
print("=" * 70)

# 1. BASIC DATA INFORMATION
print("\nğŸ“Š 1. BASIC DATASET INFORMATION")
print(f"Training data size: {train_eda.shape}")
print(f"Test data size: {test_eda.shape}")
print(f"Train/test ratio: {len(train_eda)/len(test_eda):.2f}:1")

# 2. TARGET VARIABLE ANALYSIS
print("\nğŸ�¯ 2. TARGET VARIABLE ANALYSIS (loan_paid_back)")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Target variable distribution
paid_counts = train_eda['loan_paid_back'].value_counts()
colors = ['#ff6b6b', '#51cf66']
axes[0].pie(paid_counts.values, labels=['Not Paid (0)', 'Paid Back (1)'], 
           autopct='%1.1f%%', colors=colors, startangle=90)
axes[0].set_title('Target Variable Distribution\n(loan_paid_back)', fontsize=14, fontweight='bold')

# Class balance
sns.countplot(data=train_eda, x='loan_paid_back', ax=axes[1], palette=colors)
axes[1].set_title('Class Balance', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Loan Paid Back')
axes[1].set_ylabel('Count')
for i, count in enumerate(paid_counts.values):
    axes[1].text(i, count + 1000, f'{count:,}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

print(f"ğŸ”¸ Number of paid loans: {paid_counts[1]:,} ({paid_counts[1]/len(train_eda)*100:.1f}%)")
print(f"ğŸ”¸ Number of defaulted loans: {paid_counts[0]:,} ({paid_counts[0]/len(train_eda)*100:.1f}%)")

# 3. NUMERICAL VARIABLES DISTRIBUTION
print("\nğŸ“ˆ 3. NUMERICAL VARIABLES DISTRIBUTION")
numeric_cols = ['annual_income', 'debt_to_income_ratio', 'credit_score', 
                'loan_amount', 'interest_rate']

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.ravel()

for i, col in enumerate(numeric_cols):
    # Histogram with KDE
    sns.histplot(data=train_eda, x=col, kde=True, ax=axes[i], alpha=0.7, color='skyblue')
    axes[i].axvline(train_eda[col].mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {train_eda[col].mean():.2f}')
    axes[i].axvline(train_eda[col].median(), color='green', linestyle='--', linewidth=2, 
                   label=f'Median: {train_eda[col].median():.2f}')
    axes[i].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')
    axes[i].legend()
    
    # Add boxplot to the last subplot
    if i == len(numeric_cols) - 1:
        sns.boxplot(data=train_eda[numeric_cols], ax=axes[i+1], palette="Set3")
        axes[i+1].set_title('Boxplot of All Numerical Variables', fontsize=12, fontweight='bold')
        axes[i+1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# 4. CORRELATION ANALYSIS
print("\nğŸ”— 4. CORRELATION MATRIX")
# Calculate correlations only for numerical columns
numeric_data = train_eda[numeric_cols + ['loan_paid_back']]
correlation_matrix = numeric_data.corr()

plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.3f', cbar_kws={'shrink': 0.8}, mask=mask)
plt.title('Correlation Matrix of Numerical Variables', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Display top correlations with target variable
target_correlations = correlation_matrix['loan_paid_back'].sort_values(ascending=False)
print("\nğŸ”¸ Correlations with target variable (loan_paid_back):")
for feature, corr in target_correlations.items():
    if feature != 'loan_paid_back':
        print(f"   {feature}: {corr:.4f}")

# 5. CATEGORICAL VARIABLES ANALYSIS
print("\nğŸ“Š 5. CATEGORICAL VARIABLES ANALYSIS")
categorical_cols = ['gender', 'marital_status', 'education_level', 
                   'employment_status', 'loan_purpose', 'grade_subgrade']

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.ravel()

for i, col in enumerate(categorical_cols):
    # Calculate percentages
    value_counts = train_eda[col].value_counts()
    percentages = (value_counts / len(train_eda) * 100).round(1)
    
    # Create bar plot
    bars = axes[i].bar(range(len(value_counts)), value_counts.values, color=sns.color_palette("Set3"))
    axes[i].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Count')
    axes[i].tick_params(axis='x', rotation=45)
    
    # Add percentage labels
    for j, (bar, count, perc) in enumerate(zip(bars, value_counts.values, percentages.values)):
        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000, 
                    f'{perc}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# 6. RELATIONSHIP WITH TARGET VARIABLE
print("\nğŸ”� 6. RELATIONSHIP WITH TARGET VARIABLE")

# For numerical variables
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.ravel()

for i, col in enumerate(numeric_cols):
    sns.boxplot(data=train_eda, x='loan_paid_back', y=col, ax=axes[i], palette=colors)
    axes[i].set_title(f'{col} vs Loan Paid Back', fontsize=12, fontweight='bold')
    axes[i].set_xlabel('Loan Paid Back')
    axes[i].set_ylabel(col)

plt.tight_layout()
plt.show()

# For categorical variables
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.ravel()

for i, col in enumerate(categorical_cols):
    # Calculate repayment rates by categories
    paid_rate = train_eda.groupby(col)['loan_paid_back'].mean().sort_values(ascending=False)
    
    bars = axes[i].bar(range(len(paid_rate)), paid_rate.values * 100, color=sns.color_palette("viridis", len(paid_rate)))
    axes[i].set_title(f'Repayment Rate by {col}', fontsize=12, fontweight='bold')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Repayment Rate (%)')
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].set_ylim(0, 100)
    
    # Add labels
    for j, (bar, rate) in enumerate(zip(bars, paid_rate.values)):
        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{rate*100:.1f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# 7. OUTLIERS ANALYSIS
print("\nğŸ“Š 7. OUTLIERS ANALYSIS IN NUMERICAL VARIABLES")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.ravel()

for i, col in enumerate(numeric_cols):
    # IQR method for outlier detection
    Q1 = train_eda[col].quantile(0.25)
    Q3 = train_eda[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = train_eda[(train_eda[col] < lower_bound) | (train_eda[col] > upper_bound)]
    outlier_percentage = (len(outliers) / len(train_eda)) * 100
    
    # Boxplot
    sns.boxplot(data=train_eda, y=col, ax=axes[i], color='lightblue')
    axes[i].set_title(f'{col}\nOutliers: {outlier_percentage:.2f}%', fontsize=12, fontweight='bold')
    axes[i].set_ylabel(col)

plt.tight_layout()
plt.show()

# 8. COMPARISON OF TRAIN AND TEST DISTRIBUTIONS
print("\nğŸ”¬ 8. COMPARISON OF TRAIN AND TEST DISTRIBUTIONS")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.ravel()

for i, col in enumerate(numeric_cols):
    # Compare distributions
    sns.kdeplot(data=train_eda[col], label='Train', ax=axes[i], linewidth=2)
    sns.kdeplot(data=test_eda[col], label='Test', ax=axes[i], linewidth=2)
    axes[i].set_title(f'Distribution Comparison: {col}', fontsize=12, fontweight='bold')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Density')
    axes[i].legend()

plt.tight_layout()
plt.show()

# 9. DETAILED STATISTICAL ANALYSIS
print("\nğŸ“‹ 9. DETAILED STATISTICAL SUMMARY")

# Function for skewness analysis
def analyze_skewness(series, name):
    skew_val = series.skew()
    if abs(skew_val) < 0.5:
        skew_type = "â‰ˆ normal"
    elif abs(skew_val) < 1:
        skew_type = "moderate"
    else:
        skew_type = "strong"
    return skew_val, skew_type

print("\nğŸ”¸ Distribution skewness analysis:")
for col in numeric_cols:
    skew_val, skew_type = analyze_skewness(train_eda[col], col)
    print(f"   {col}: {skew_val:.3f} ({skew_type})")

# 10. ANALYSIS OF VARIABLE RELATIONSHIPS
print("\nğŸ”— 10. ANALYSIS OF KEY RELATIONSHIPS")

# Interesting variable pairs for analysis
interesting_pairs = [
    ('annual_income', 'loan_amount'),
    ('credit_score', 'interest_rate'),
    ('debt_to_income_ratio', 'loan_paid_back'),
    ('annual_income', 'credit_score')
]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

for i, (x_col, y_col) in enumerate(interesting_pairs):
    if y_col == 'loan_paid_back':
        sns.boxplot(data=train_eda, x=y_col, y=x_col, ax=axes[i], palette=colors)
    else:
        sns.scatterplot(data=train_eda, x=x_col, y=y_col, hue='loan_paid_back', 
                       alpha=0.6, ax=axes[i], palette=colors)
    axes[i].set_title(f'{x_col} vs {y_col}', fontsize=12, fontweight='bold')
    axes[i].set_xlabel(x_col)
    axes[i].set_ylabel(y_col)

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("âœ… EDA ANALYSIS COMPLETED")
print("="*70)

# Display key insights
print("\nğŸ�¯ KEY INSIGHTS:")
print("1. ğŸ“Š Data is well balanced (~80% loans repaid)")
print("2. ğŸ”� Highest correlation with target: credit_score, interest_rate")
print("3. ğŸ“ˆ Numerical variables distributions are close to normal")
print("4. ğŸ�¯ Test and train distributions are very similar - good sign")
print("5. ğŸ“Š Categorical variables have reasonable distribution")
print("6. ğŸš€ Minimal number of outliers in the data")


def create_financial_features(df):
    df = df.copy()
    
    df['income_to_loan_ratio'] = df['annual_income'] / df['loan_amount']
    df['debt_burden'] = df['annual_income'] * df['debt_to_income_ratio']
    df['loan_to_income_ratio'] = df['loan_amount'] / df['annual_income']
    df['affordability_score'] = (df['annual_income'] * (1 - df['debt_to_income_ratio'])) / df['loan_amount']
    
    df['credit_income_interaction'] = df['credit_score'] * df['annual_income'] / 10000
    df['risk_score'] = df['debt_to_income_ratio'] * df['interest_rate']
    df['financial_stability'] = (df['credit_score'] / 100) * (1 - df['debt_to_income_ratio'])
    
    return df

train = create_financial_features(train)
test = create_financial_features(test)

display(train.shape, test.shape)


def extract_grade_features(df):
    df = df.copy()
    
    df['grade_letter'] = df['grade_subgrade'].str[0]
    df['grade_number'] = df['grade_subgrade'].str[1].astype(int)
    
    grade_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    df['grade_numeric'] = df['grade_letter'].map(grade_mapping)
    
    df['combined_grade_score'] = df['grade_numeric'] * 10 + df['grade_number']
    
    return df

train = extract_grade_features(train)
test = extract_grade_features(test)

display(train.shape, test.shape)


def create_risk_features(df):
    df = df.copy()
    
    df['disposable_income'] = df['annual_income'] * (1 - df['debt_to_income_ratio'])
    df['monthly_loan_payment'] = (df['loan_amount'] * df['interest_rate'] / 100) / 12
    df['payment_to_income_ratio'] = df['monthly_loan_payment'] / (df['annual_income'] / 12)
    
    df['credit_debt_ratio'] = df['credit_score'] / (df['debt_to_income_ratio'] * 1000 + 1)
    df['income_adequacy'] = df['annual_income'] / (df['loan_amount'] + 1)
    
    df['high_risk_flag'] = ((df['debt_to_income_ratio'] > 0.4) | 
                           (df['credit_score'] < 600) | 
                           (df['interest_rate'] > 15)).astype(int)
    
    df['medium_risk_flag'] = ((df['debt_to_income_ratio'] > 0.2) & 
                             (df['debt_to_income_ratio'] <= 0.4) & 
                             (df['credit_score'] >= 600)).astype(int)
    
    return df

train = create_risk_features(train)
test = create_risk_features(test)

display(train.shape, test.shape)


def create_categorical_features(df):
    df = df.copy()
    
    df['employment_education'] = df['employment_status'] + '_' + df['education_level']
    df['gender_marital'] = df['gender'] + '_' + df['marital_status']
    df['purpose_grade'] = df['loan_purpose'] + '_' + df['grade_letter']
    
    df['is_debt_consolidation'] = (df['loan_purpose'] == 'Debt consolidation').astype(int)
    df['is_employed'] = (df['employment_status'] == 'Employed').astype(int)
    df['is_high_education'] = df['education_level'].isin(['Master\'s', 'Doctorate']).astype(int)
    df['is_married'] = (df['marital_status'] == 'Married').astype(int)
    
    purpose_counts = df['loan_purpose'].value_counts()
    df['purpose_frequency'] = df['loan_purpose'].map(purpose_counts)
    
    education_rank = {'High School': 1, 'Bachelor\'s': 2, 'Master\'s': 3, 'Doctorate': 4}
    df['education_rank'] = df['education_level'].map(education_rank)
    
    return df

train = create_categorical_features(train)
test = create_categorical_features(test)

display(train.shape, test.shape)


def create_creative_features(df):
    df = df.copy()
    
    df['financial_health_score'] = (
        (df['credit_score'] / 850) * 0.4 +
        (1 - df['debt_to_income_ratio']) * 0.3 +
        (1 / (1 + df['loan_to_income_ratio'])) * 0.3
    )
    
    conditions = [
        (df['credit_score'] >= 700) & (df['debt_to_income_ratio'] <= 0.2),
        (df['credit_score'] >= 650) & (df['debt_to_income_ratio'] <= 0.3),
        (df['credit_score'] >= 600) & (df['debt_to_income_ratio'] <= 0.4),
        (df['credit_score'] < 600) | (df['debt_to_income_ratio'] > 0.4)
    ]
    risk_levels = [1, 2, 3, 4]  
    df['risk_level'] = np.select(conditions, risk_levels, default=3)
    
    df['id_recency'] = df['id'].rank(pct=True)
    
    df['comprehensive_score'] = (
        df['credit_score'] * 0.5 + 
        (1 - df['debt_to_income_ratio']) * 200 + 
        (df['annual_income'] / df['loan_amount']) * 50
    )
    
    return df

train = create_creative_features(train)
test = create_creative_features(test)

display(train.shape, test.shape)


test = test.drop(['id'], axis =1)
train = train.drop(['id'], axis =1)


cat_features = train.select_dtypes(include=[object]).columns
cat_features


X = train.drop(columns=['loan_paid_back'])
y = train['loan_paid_back']

print("ğŸ”� Checking for infinite values...")

for df, name in [(X, 'X'), (test, 'test')]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.any(np.isinf(df[col])):
            print(f"Found infinite values in {name}.{col}, replacing with NaN")
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

print("\nğŸ�¯ Target Encoding categorical features...")
for col in cat_features:
    if col in X.columns:
        print(f"Encoding {col}...")
        le = TargetEncoder()
        
        le.fit(X[col].astype(str), y)
        
        X[col] = le.transform(X[col].astype(str))
        test[col] = le.transform(test[col].astype(str))

print("\nğŸ“Š Analyzing missing values...")

missing_summary = []
for col in X.columns:
    pct_missing = np.mean(X[col].isnull())
    if pct_missing > 0:
        missing_summary.append((col, round(pct_missing*100, 2)))
        
if missing_summary:
    print("Columns with missing values:")
    for col, pct in sorted(missing_summary, key=lambda x: x[1], reverse=True):
        print(f'  {col} - {pct}%')
else:
    print("No missing values found!")

print("\nğŸ”„ Filling missing values...")

for col in X.columns:
    if X[col].isnull().any():
        if X[col].dtype in ['int64', 'float64']:
            fill_value = X[col].median()
            X[col] = X[col].fillna(fill_value)
            test[col] = test[col].fillna(fill_value)
            print(f"Filled numeric {col} with median: {fill_value:.4f}")
        else:
            fill_value = X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown'
            X[col] = X[col].fillna(fill_value)
            test[col] = test[col].fillna(fill_value)
            print(f"Filled categorical {col} with mode: {fill_value}")

print("\nâœ… Final data quality check:")
print(f"Missing values in X: {X.isnull().sum().sum()}")
print(f"Missing values in test: {test.isnull().sum().sum()}")

numeric_cols_X = X.select_dtypes(include=[np.number])
numeric_cols_test = test.select_dtypes(include=[np.number])

print(f"Infinite values in X: {np.any(np.isinf(numeric_cols_X)) if not numeric_cols_X.empty else 'No numeric cols'}")
print(f"Infinite values in test: {np.any(np.isinf(numeric_cols_test)) if not numeric_cols_test.empty else 'No numeric cols'}")

print("\nğŸ�¯ Applying variance threshold...")

X = variance_threshold(X, 0.004)
selected_features = list(X.columns)
test = test[selected_features]

print(f"Features after variance threshold: {len(selected_features)}")

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


def optimize_catboost_classification(X, y, n_trials=20, cv=5):
    
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 3000),
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
    study = optimize_catboost_classification(X, y, n_trials=20)
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

        'iterations': 2921,
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

        'iterations': 2335,
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
    
        'iterations': 2184,
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
    
        'iterations': 2844,
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


def optimize_xgboost_classification(X, y, n_trials=20, cv=5):
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
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
    study = optimize_xgboost_classification(X, y, n_trials=20)
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
    
        'n_estimators': 1190,
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
    
        'n_estimators': 2257,
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
    
        'n_estimators': 1234,
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
    
        'n_estimators': 2150,
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


def optimize_lightgbm_classification(X, y, n_trials=20, cv=5):
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
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
    study = optimize_lightgbm_classification(X, y, n_trials=20)
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
    
        'n_estimators': 2968,
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
    
        'n_estimators': 1876,
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
    
        'n_estimators': 988,
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
    
        'n_estimators': 2918,
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


def create_classification_ensemble(X, y, test, n_folds=7):
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


sample = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')
sample['loan_paid_back'] = test_pred
sample.to_csv('submission.csv', index=False)
sample.head(10)

