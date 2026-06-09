import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, QuantileTransformer,PowerTransformer
from scipy.stats import skew
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import (RandomForestRegressor,
                              GradientBoostingClassifier,
                              AdaBoostRegressor,
                              HistGradientBoostingRegressor,
                              BaggingRegressor,
                              StackingRegressor,
                              VotingRegressor)
from xgboost import XGBRegressor
!pip install catboost --q
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import roc_auc_score
import tensorflow as tf
import zipfile
import math
import os

#libraries to handle warnings
import warnings
warnings.filterwarnings('ignore')

#custom style for visualizations
sns.set(style='whitegrid', palette='pastel', font_scale=1.1)
plt.rcParams['figure.figsize'] = (8,5)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12


# loading the data files
train_df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')

# Test ID for Final Submission
Test_ID = test_df['id'].copy()
# Target
target = 'loan_paid_back'


# Categorical and Numeric Features
cat_cols = train_df.select_dtypes(include=['object', 'category'])
num_cols = train_df.select_dtypes(include=['int64', 'float64', 'number'])


def quick_overview(df, name):
    print(f"\n{name.upper()} SHAPE: {df.shape}")
    display(df.head())
    display(df.describe(include="all").T)

quick_overview(train_df, "train")
quick_overview(test_df , "test")

print(f"Duplicate rows (train): {train_df.duplicated().sum()}  |  (test): {test_df.duplicated().sum()}")
print("Number of missing values:")
train_df.isnull().sum()


def plot_kde(data, name, columns=None, figsize=(8, 4), fill=True, max_density=None):
    if isinstance(data, pd.Series):
        data = data.to_frame()
    columns = data.select_dtypes(include='number').columns.tolist()
    plt.figure(figsize=figsize)
    for col in columns:
        sns.kdeplot(data[col], label=col, linewidth=2,clip=(0, None),linestyle="-.")

    if max_density is not None:
        plt.ylim(0, max_density)
    plt.title(name)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()

print("KDE PLOT")
plot_kde(train_df[target], "Loan Paid Back Distribution")

print("HISTOGRAM")
sns.histplot(train_df[target], kde=False)
plt.title(f"Load Paid Back Distribution")
plt.xlabel("Load Paid Back")
plt.ylabel("Count")

plt.show()


for col in num_cols.columns:
    if col != target and col != 'id':
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        sns.histplot(train_df[col], kde=True)
        plt.title(f'Histogram of {col}')

        plt.subplot(1, 2, 2)
        sns.kdeplot(train_df[col], fill=True)
        plt.title(f'KDE plot of {col}')

        plt.tight_layout()
        plt.show()


for col in cat_cols.columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=train_df, x=col, order=train_df[col].value_counts().index)
    plt.title(f'Frequency Distribution of {col}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


print("Descriptive statistics for numerical columns:")
display(train_df.describe())

print("\nDescriptive statistics for categorical columns:")
display(train_df.describe(include='object'))


numerical_features = train_df.select_dtypes(include=np.number).columns.tolist()
numerical_features.remove('id')
if 'loan_paid_back' in numerical_features:
    numerical_features.remove('loan_paid_back')

# Plotting scatter plots for selected numerical feature pairs
selected_pairs = [
    ('annual_income', 'loan_amount'),
    ('credit_score', 'interest_rate'),
    ('annual_income', 'debt_to_income_ratio'),
    ('loan_amount', 'interest_rate')
]

for col1, col2 in selected_pairs:
    if col1 in train_df.columns and col2 in train_df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=train_df, x=col1, y=col2, alpha=0.6)
        plt.title(f'Scatter Plot of {col1} vs {col2}')
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.tight_layout()
        plt.show()


categorical_features = train_df.select_dtypes(include='object').columns.tolist()

for col in categorical_features:
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=train_df, x=col, y=target)
    plt.title(f'Box Plot of {target} by {col}')
    plt.xlabel(col)
    plt.ylabel(target)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


numerical_df = train_df.select_dtypes(include=np.number)
correlation_matrix = numerical_df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
plt.show()


categorical_features = train_df.select_dtypes(include='object').columns.tolist()

# Select two categorical features for interaction plots
if len(categorical_features) >= 2:
    cat_col1 = categorical_features[0]
    cat_col2 = categorical_features[1]

    plt.figure(figsize=(14, 7))
    sns.countplot(data=train_df, x=cat_col1, hue=cat_col2, palette='viridis')
    plt.title(f'Count of {target} by {cat_col1} and {cat_col2}')
    plt.xlabel(cat_col1)
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title=cat_col2)
    plt.tight_layout()
    plt.show()
else:
    print("Need at least two categorical features for interaction plot.")


numerical_features = train_df.select_dtypes(include=np.number).columns.tolist()
if 'id' in numerical_features:
    numerical_features.remove('id')

# Select two numerical features and a third numerical or the target
if len(numerical_features) >= 3:
    num_col1 = numerical_features[0]
    num_col2 = numerical_features[1]
    num_col3 = numerical_features[2] # Use the third numerical feature

    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=train_df, x=num_col1, y=num_col2, hue=num_col3, size=num_col3, palette='viridis', alpha=0.6)
    plt.title(f'Scatter Plot of {num_col1} vs {num_col2} colored and sized by {num_col3}')
    plt.xlabel(num_col1)
    plt.ylabel(num_col2)
    plt.tight_layout()
    plt.show()
elif len(numerical_features) >= 2 and target in train_df.columns:
     num_col1 = numerical_features[0]
     num_col2 = numerical_features[1]
     plt.figure(figsize=(12, 8))
     sns.scatterplot(data=train_df, x=num_col1, y=num_col2, hue=target, palette='viridis', alpha=0.6)
     plt.title(f'Scatter Plot of {num_col1} vs {num_col2} colored by {target}')
     plt.xlabel(num_col1)
     plt.ylabel(num_col2)
     plt.tight_layout()
     plt.show()

else:
    print("Need at least two numerical features for scatter plot with color/size variation.")



numerical_features = train_df.select_dtypes(include=np.number).columns.tolist()
if 'id' in numerical_features:
    numerical_features.remove('id')

# Select a subset of numerical features for pair plot to avoid overcrowding
selected_numerical_features = numerical_features[:5] # Select first 5 numerical features

# Use the target variable for hue if it's in the dataframe and not in the selected features
hue_var = target if target in train_df.columns and target not in selected_numerical_features else None

if len(selected_numerical_features) > 1:
    sns.pairplot(train_df[selected_numerical_features + ([hue_var] if hue_var else [])].dropna(), hue=hue_var, palette='viridis', diag_kind='kde')
    plt.suptitle('Pair Plot of Selected Numerical Features', y=1.02)
    plt.show()
else:
    print("Need at least two numerical features for a pair plot.")


numerical_features = train_df.select_dtypes(include=np.number).columns.tolist()
numerical_features.remove('id')
if 'loan_paid_back' in numerical_features:
    numerical_features.remove('loan_paid_back')

for col in numerical_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=train_df[col])
    plt.title(f'Box Plot of {col}')
    plt.xlabel(col)
    plt.show()


numerical_features = train_df.select_dtypes(include=np.number).columns.tolist()
numerical_features.remove('id')
if 'loan_paid_back' in numerical_features:
    numerical_features.remove('loan_paid_back')

outlier_analysis = {}

for col in numerical_features:
    Q1 = train_df[col].quantile(0.25)
    Q3 = train_df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = train_df[(train_df[col] < lower_bound) | (train_df[col] > upper_bound)]

    outlier_analysis[col] = {
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'num_outliers': len(outliers),
        'percentage_outliers': (len(outliers) / len(train_df)) * 100,
        'outlier_values_sample': outliers[col].sample(min(5, len(outliers))).tolist() if len(outliers) > 0 else []
    }

print("Outlier Analysis and Potential Handling Strategies:")
for col, analysis in outlier_analysis.items():
    print(f"\nFeature: {col}")
    print(f"  IQR Lower Bound: {analysis['lower_bound']:.2f}")
    print(f"  IQR Upper Bound: {analysis['upper_bound']:.2f}")
    print(f"  Number of Outliers (1.5*IQR): {analysis['num_outliers']}")
    print(f"  Percentage of Outliers: {analysis['percentage_outliers']:.2f}%")
    print(f"  Sample Outlier Values: {analysis['outlier_values_sample']}")

    # Decide on potential handling strategies (analysis only, no implementation)
    if analysis['percentage_outliers'] > 1: # Example threshold
        print("  Potential Handling Strategy: Consider capping or transformation due to significant number of outliers.")
    elif analysis['num_outliers'] > 0:
         print("  Potential Handling Strategy: Outliers present, evaluate their impact. Capping or transformation might be considered if they are influential.")
    else:
        print("  Potential Handling Strategy: No significant outliers detected by 1.5*IQR rule.")


