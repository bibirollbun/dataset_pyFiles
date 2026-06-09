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


# ğŸ“¦ 1. Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Optional (safe fallback if missingno is not installed)
try:
    import missingno as msno
    MISSINGNO_AVAILABLE = True
except ImportError:
    MISSINGNO_AVAILABLE = False

# Display all columns
pd.set_option('display.max_columns', None)
sns.set(style='whitegrid')
%matplotlib inline



# ğŸ“¥ 2. Load Dataset (change path as needed)
df = pd.read_csv('/kaggle/input/diabetes-risk-prediction/diabetes_risk_prediction_dataset.csv')  # <-- Modify this
print("âœ… Data Loaded!")



# ğŸ“Š 3. Basic Data Info
print(f"\nShape: {df.shape}")
print("\nğŸ§¾ Data Types:\n", df.dtypes)
print("\nğŸ“Œ First 5 Rows:\n", df.head())
print("\nğŸ“‰ Summary Stats (Numeric):\n", df.describe())
print("\nğŸ“‹ Summary Stats (Categorical):\n", df.describe(include='object'))



# ğŸ”� 4. Missing Values Overview
print("\nâ�“ Missing Values Per Column:\n", df.isnull().sum())

if MISSINGNO_AVAILABLE:
    msno.matrix(df)
    plt.show()
else:
    # Alternative bar plot for missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        missing.sort_values().plot(kind='barh', figsize=(10, 5), title='Missing Values per Column')
        plt.show()



# ğŸ“ˆ 5. Univariate Analysis
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object', 'category']).columns

# Numerical
for col in num_cols:
    if df[col].nunique() > 1:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col].dropna(), kde=True, bins=30)
        plt.title(f'Distribution of {col}')
        plt.show()

# Categorical
for col in cat_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=col, data=df, order=df[col].value_counts().index[:20])
    plt.title(f'Count Plot of {col}')
    plt.xticks(rotation=45)
    plt.show()



# ğŸ”— 6. Bivariate & Correlation Analysis
if len(num_cols) >= 2:
    corr = df[num_cols].corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.show()



# âš ï¸� 7. Outlier Detection
# Only on continuous numerical columns with enough unique values
for col in num_cols:
    if df[col].nunique() > 10:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.show()



# ğŸ�¯ 8. Target Variable Check (optional)
# Automatically detect a binary/class target
potential_targets = [col for col in df.columns if df[col].nunique() <= 10 and df[col].dtype != 'object']
if potential_targets:
    target_col = potential_targets[0]
    print(f"ğŸ�¯ Detected potential target: {target_col}")
    sns.countplot(x=target_col, data=df)
    plt.title(f'Target Distribution: {target_col}')
    plt.show()
else:
    print("âš ï¸� No obvious target variable found for classification.")



# ğŸ”§ 9. Data Quality Checks
print("ğŸ§¹ Duplicates:", df.duplicated().sum())

# Inconsistent text values
for col in cat_cols:
    if df[col].nunique() < 50:
        print(f"\nğŸ”¤ Unique values in {col}:\n", df[col].unique())



# ğŸ§  10. Feature Engineering Preview
# Example: Encode categorical with low cardinality
low_card_cat = [col for col in cat_cols if df[col].nunique() <= 10]
print("\nğŸ“Œ Low Cardinality Categorical Columns:\n", low_card_cat)

# Safe encoding (preview)
df_encoded = df.copy()
for col in low_card_cat:
    df_encoded[col] = df_encoded[col].astype('category').cat.codes

print("\nâœ… Sample after encoding:\n", df_encoded.head())



# 1ï¸�âƒ£ Load Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
sns.set(style='whitegrid')
%matplotlib inline

# Optional (fallback if missingno not available)
try:
    import missingno as msno
    MISSINGNO = True
except:
    MISSINGNO = False

from scipy.stats import skew, kurtosis, zscore
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE



# 2ï¸�âƒ£ Load Data (Replace the path with your dataset path)
df = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')  # <-- change here
print("âœ… Data Loaded Successfully!")



print(f"\nğŸ“¦ Shape of DataFrame: {df.shape}")
display(df.head())
display(df.tail())
display(df.sample(3))

print("\nğŸ§¾ Column Data Types:")
print(df.dtypes)

print("\nğŸ”¢ Unique values in categorical columns:")
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    print(f"{col}: {df[col].nunique()} unique values")



# Try to detect a classification target column
potential_targets = [col for col in df.columns if df[col].nunique() <= 10 and df[col].dtype != 'float64']
if potential_targets:
    target = potential_targets[0]
    print(f"\nğŸ�¯ Class Distribution (Potential Target: {target}):")
    print(df[target].value_counts())
    sns.countplot(x=target, data=df)
    plt.title("Target Class Distribution")
    plt.show()



missing = df.isnull().sum()
missing = missing[missing > 0]
print("\nğŸš« Missing Values:\n", missing)

# Visualize
if MISSINGNO:
    msno.matrix(df)
else:
    if not missing.empty:
        missing.plot(kind='barh', figsize=(10, 5), title='Missing Values by Column')
        plt.show()



print("\nğŸ“Š Summary Statistics:")
display(df.describe())

print("\nğŸ“ˆ Skewness & Kurtosis (Numeric Columns):")
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    print(f"{col}: Skew = {skew(df[col].dropna()):.2f}, Kurtosis = {kurtosis(df[col].dropna()):.2f}")



# Numeric Distribution
for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col].dropna(), kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.show()

# Categorical Count Plots
for col in cat_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=col, data=df, order=df[col].value_counts().index[:20])
    plt.title(f"Count Plot of {col}")
    plt.xticks(rotation=45)
    plt.show()



# Correlation Heatmap
if len(num_cols) >= 2:
    plt.figure(figsize=(10, 6))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm')
    plt.title("ğŸ“Œ Correlation Matrix")
    plt.show()

# Scatter Plot for top 2 numerical features
if len(num_cols) >= 2:
    sns.scatterplot(x=num_cols[0], y=num_cols[1], data=df)
    plt.title(f'Scatter: {num_cols[0]} vs {num_cols[1]}')
    plt.show()

# Boxplots (numerical vs top categorical)
for cat in cat_cols[:2]:
    for num in num_cols[:2]:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=cat, y=num, data=df)
        plt.title(f"{num} by {cat}")
        plt.xticks(rotation=45)
        plt.show()



# Boxplots
for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot: {col}')
    plt.show()

# Z-score Method (optional)
z_scores = np.abs(zscore(df[num_cols].dropna()))
outliers = (z_scores > 3).sum(axis=0)
print("\nğŸš¨ Z-score Outliers (per column):")
print(dict(zip(num_cols, outliers)))



# Encode categorical features
df_encoded = df.copy()
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

print("\nâœ… Sample after encoding:")
display(df_encoded.head())

# Binning a numeric column
if len(num_cols) > 0:
    col = num_cols[0]
    df_encoded[f"{col}_binned"] = pd.qcut(df_encoded[col], q=4, duplicates='drop')
    print(f"\nğŸªœ Binned {col} into quartiles:")
    display(df_encoded[[col, f"{col}_binned"]].head())



if 'target' in df.columns:
    print("ğŸ�¯ Target Variable Distribution")
    sns.histplot(df['target'])
    plt.title("Target Distribution")
    plt.show()

    # Correlation with numeric features
    if df['target'].dtype in ['int64', 'float64']:
        corr_with_target = df.corr()['target'].sort_values(ascending=False)
        print("\nğŸ”— Correlation with Target:\n", corr_with_target)



if df_encoded.shape[1] > 2:
    # PCA
    pca = PCA(n_components=2)
    pca_vals = pca.fit_transform(df_encoded.select_dtypes(include=np.number).fillna(0))

    plt.figure(figsize=(6, 4))
    plt.scatter(pca_vals[:, 0], pca_vals[:, 1], s=10, alpha=0.5)
    plt.title("ğŸ”» PCA 2D Projection")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()



# Duplicates
print("ğŸ§¹ Duplicate Rows:", df.duplicated().sum())

# Inconsistent Categorical Values
for col in cat_cols:
    print(f"\nğŸ”¤ Unique values in '{col}':")
    print(sorted(df[col].astype(str).str.lower().unique()))

# Logical Checks (e.g., negative age)
if 'age' in df.columns:
    print("\nâš ï¸� Rows with negative age:")
    display(df[df['age'] < 0])



# 1. Load Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, zscore
import warnings
warnings.filterwarnings('ignore')
sns.set(style='whitegrid')
%matplotlib inline

try:
    import missingno as msno
    MISSINGNO = True
except:
    MISSINGNO = False



# 2. Load Train & Test Data
train = pd.read_csv("/kaggle/input/playground-series-s3e10/train.csv")  # replace path
test = pd.read_csv("/kaggle/input/playground-series-s3e10/test.csv")    # replace path

print("âœ… Train shape:", train.shape)
print("âœ… Test shape :", test.shape)



print("ğŸ§¾ Columns in Train:")
display(train.columns)

print("ğŸ”� Data Types:")
print(train.dtypes)

print("\nğŸ�¯ Target Column Guess:")
target_col = [col for col in train.columns if col not in test.columns][0]
print("Detected target:", target_col)

print("\nğŸ“Š Train Sample:")
display(train.head())

print("\nğŸ“Š Test Sample:")
display(test.head())



def missing_vals(df, name):
    print(f"Missing values in {name} set:")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    display(missing)
    
    if MISSINGNO:
        msno.matrix(df)
    else:
        if not missing.empty:
            missing.plot(kind='barh', title=f'Missing Values in {name}', figsize=(10, 5))
            plt.show()

missing_vals(train, "Train")
missing_vals(test, "Test")



# Numerical stats
print("\nğŸ“ˆ Train Descriptive Stats:")
display(train.describe())

# Skew/Kurtosis
num_cols = train.select_dtypes(include=['int64', 'float64']).columns.drop(target_col)
for col in num_cols:
    print(f"{col}: Skew = {skew(train[col].dropna()):.2f}, Kurtosis = {kurtosis(train[col].dropna()):.2f}")



# Histogram of numeric features
for col in num_cols[:5]:
    plt.figure(figsize=(6,4))
    sns.histplot(train[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

# Count plots of categorical features
cat_cols = train.select_dtypes(include='object').columns
for col in cat_cols[:3]:
    plt.figure(figsize=(6,4))
    sns.countplot(data=train, x=col, order=train[col].value_counts().index[:10])
    plt.title(f'Count plot of {col}')
    plt.xticks(rotation=45)
    plt.show()



# Correlation Heatmap
corr = train[num_cols].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Boxplot: Top categorical vs numeric
for cat in cat_cols[:2]:
    for num in num_cols[:2]:
        plt.figure(figsize=(6,4))
        sns.boxplot(x=cat, y=num, data=train)
        plt.title(f'{num} by {cat}')
        plt.xticks(rotation=45)
        plt.show()



# Boxplots for outliers
for col in num_cols[:3]:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=train[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

# Z-score method
z_scores = np.abs(zscore(train[num_cols].dropna()))
outliers = (z_scores > 3).sum(axis=0)
print("ğŸ”� Z-Score Outliers:")
print(dict(zip(num_cols, outliers)))



# Compare distribution of common features
common_cols = [col for col in train.columns if col in test.columns and train[col].dtype != 'object']

for col in common_cols[:5]:
    plt.figure(figsize=(6,4))
    sns.kdeplot(train[col], label='Train', shade=True)
    sns.kdeplot(test[col], label='Test', shade=True)
    plt.title(f'Train vs Test Distribution: {col}')
    plt.legend()
    plt.show()



# Target distribution
if train[target_col].nunique() <= 10:
    sns.countplot(x=train[target_col])
    plt.title("Target Class Distribution")
    plt.show()
else:
    sns.histplot(train[target_col], kde=True)
    plt.title("Target Distribution")
    plt.show()

# Correlation with target
if train[target_col].dtype in ['int64', 'float64']:
    corr_with_target = train.corr()[target_col].sort_values(ascending=False)
    print("\nğŸ“Œ Correlation with Target:")
    print(corr_with_target)



from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Encode object columns
combined = pd.concat([train.drop(columns=[target_col]), test])
for col in cat_cols:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col].astype(str))

# PCA
pca = PCA(n_components=2)
pca_vals = pca.fit_transform(combined.select_dtypes(include=np.number).fillna(0))

plt.figure(figsize=(6, 4))
plt.scatter(pca_vals[:len(train), 0], pca_vals[:len(train), 1], c=train[target_col], cmap='viridis', s=10)
plt.title("PCA - Train Data Projection")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label='Target')
plt.show()



# Duplicates
print("ğŸ§¹ Duplicate Rows in Train:", train.duplicated().sum())
print("ğŸ§¹ Duplicate Rows in Test:", test.duplicated().sum())

# Inconsistent values
for col in cat_cols:
    print(f"\nUnique values in {col}:")
    print(train[col].astype(str).str.lower().value_counts().head())





