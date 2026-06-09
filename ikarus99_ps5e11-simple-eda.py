# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import polars as pl
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


train_df=pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv').drop(columns=['id'])
train_df


train_df['loan_paid_back'].value_counts()


test_df =pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv').drop(columns=['id'])
test_df


train_df.info()


def classify_columns(df: pl.DataFrame, unique_threshold: int = 3, seq_col = None):
    """
    Classify columns into categorical, numerical, datetime (using dtype methods).
    
    Args:
        df: Polars DataFrame to classify
        
    Returns:
        tuple: (categorical_cols: dtype, seq_cols: dtype, numerical_cols: dtype, datetime_cols: dtype)
    """
    categorical_cols = []
    numerical_cols = []
    datetime_cols = []
    seq_cols = []

    for col in df.columns:
        dtype = df[col].dtype
        n_unique = df[col].n_unique()

        # Prefer dtype helper methods when available
        try:
           if dtype.is_numeric():
               numerical_cols.append(col)
               continue
           if dtype.is_temporal():
               datetime_cols.append(col)
               continue
        except Exception:
            # ignore if dtype doesn't implement those helpers
           pass
        # Sequence column check
        if col == seq_col:
            seq_cols.append(col)
            continue

        if str(dtype).startswith(('Int', 'Float')):
            if n_unique < unique_threshold:
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
        else:
            categorical_cols.append(col)

        # Common categorical/boolean/string checks
        # if dtype in (pl.Utf8, pl.Categorical, pl.String) or dtype == pl.Boolean:
        #     categorical_cols.append(col)
        # else:
        #     # Fallback based on textual dtype name
        #     name = str(dtype).lower()
        #     if name.startswith('int') or name.startswith('float') or 'decimal' in name:
        #         numerical_cols.append(col)
        #     elif 'date' in name or 'time' in name or 'datetime' in name:
        #         datetime_cols.append(col)
        #     else:
        #         categorical_cols.append(col)

    # Prepare dtype mappings for clearer logging
    categorical_dtypes = {c: str(df[c].dtype) for c in categorical_cols}
    numerical_dtypes = {c: str(df[c].dtype) for c in numerical_cols}
    seq_dtypes = {c: str(df[c].dtype) for c in seq_cols}
    # datetime_dtypes = {c: str(df[c].dtype) for c in datetime_cols}

    print('ğŸ“‹ Column classification:')
    print(f'   Categorical: {len(categorical_cols)} columns - {categorical_cols}')
    print(f'      dtypes: {categorical_dtypes}, n_unique: {[df[c].n_unique() for c in categorical_cols]}')
    print(f'   Numerical: {len(numerical_cols)} columns - {numerical_cols}')
    print(f'      dtypes: {numerical_dtypes}')
    print(f'   Sequence: {seq_cols if seq_cols else "N/A"} / dtypes: {seq_dtypes if seq_cols else "N/A"}')
    #print(f'   Datetime: {len(datetime_cols)} columns - {datetime_cols}')
    #print(f'      dtypes: {datetime_dtypes}')

    return categorical_cols, numerical_cols, seq_cols


train_df.describe(include='all')


# FE columns Add : annual_income * debt_to_income_ratio | loan_amount 
from scipy.stats import kendalltau

train_df['income_debt_interest_ratio'] = train_df['annual_income'] * train_df['debt_to_income_ratio'] / train_df['interest_rate']
train_df['income_debt_interest_score'] = train_df['income_debt_interest_ratio'] * train_df['credit_score']
train_df['interest_loan'] = train_df['loan_paid_back'].corr(train_df['interest_rate'], method='kendall') # train_df['interest_rate'] ^ train_df['loan_paid_back']


num_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = train_df.select_dtypes(include=['object', 'category']).columns

print("Numerical Columns:", num_cols.tolist())
print("Categorical Columns:", cat_cols.tolist())


for col in num_cols:
    plt.figure(figsize=(12,4))
    
    plt.subplot(1,2,1)
    sns.histplot(train_df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    
    plt.subplot(1,2,2)
    sns.boxplot(x=train_df[col])
    plt.title(f'Boxplot of {col}')
    
    plt.show()


# Categorical vs Categorical (corrected)
for i in range(len(cat_cols)):
    for j in range(i+1, len(cat_cols)):
        ct = pd.crosstab(train_df[cat_cols[i]], train_df[cat_cols[j]])
        plt.figure(figsize=(8,5))
        sns.heatmap(ct, annot=False, cmap='YlGnBu')
        plt.title(f'{cat_cols[i]} vs {cat_cols[j]}')
        plt.xlabel(cat_cols[j])
        plt.ylabel(cat_cols[i])
        plt.show()



plt.figure(figsize=(8,6))
sns.scatterplot(
    data=train_df,
    x='annual_income',
    y='loan_amount',
    hue='grade_subgrade',
    alpha=0.7
)
plt.title('Annual Income vs Loan Amount by Grade')
plt.show()


plt.figure(figsize=(10,5))
sns.boxplot(
    data=train_df,
    x='education_level',
    y='loan_amount',
    hue='marital_status'
)
plt.title('Loan Amount by Education Level and Marital Status')
plt.xticks(rotation=45)
plt.show()


SAMPLES_COORD = 500
SEED = 42


# Correlation heatmap
corr = train_df[num_cols].corr()
plt.figure(figsize=(10,6))
mask = np.triu(np.ones_like(corr, dtype=np.bool_))
heatmap = sns.heatmap(corr, mask=mask, vmin=-1, vmax=1, annot=True)
plt.title('Correlation Heatmap (Numerical Variables)')
plt.show()


train_df['education_level']


from pandas.plotting import parallel_coordinates

subset_cols = ['education_level', 'annual_income', 'income_debt_interest_ratio', 'loan_amount', 'interest_rate']
plt.figure(figsize=(12,6))
parallel_coordinates(train_df[subset_cols].sample(SAMPLES_COORD, random_state=SEED), 'education_level', colormap='viridis')
plt.title('Parallel Coordinates Plot - Multivariate View')
plt.xticks(rotation=30)
plt.show()



# 'employment_status' 'loan_purpose' 'loan_paid_back' apply log10
subset_cols = ['education_level', 'income_debt_interest_score', 'annual_income', 'interest_loan', 'loan_amount']
plt.figure(figsize=(12,6))
parallel_coordinates(train_df[subset_cols].sample(SAMPLES_COORD, random_state=SEED), 'education_level', colormap='viridis')
plt.title('Parallel Coordinates Plot - Multivariate View')
plt.xticks(rotation=30)
plt.show()


X = train_df.drop(columns=['loan_paid_back'])
y = train_df['loan_paid_back']


num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

print("Numerical Columns:", num_cols.tolist())
print("Categorical Columns:", cat_cols.tolist())




