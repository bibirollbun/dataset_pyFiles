import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox


train_df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
sample_submission=pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')


train_df.head(3)


test.head(3)


train_df.dtypes


train_df.isnull().sum().sum()


fig, axes = plt.subplots(3, 2, figsize=(15, 15))

sns.countplot(data=train_df, x='job', hue='y', ax=axes[0, 0])
axes[0, 0].set_title('Job vs Target')
axes[0, 0].tick_params(axis='x', rotation=45)

sns.countplot(data=train_df, x='marital', hue='y', ax=axes[0, 1])
axes[0, 1].set_title('Marital Status vs Target')

sns.countplot(data=train_df, x='education', hue='y', ax=axes[1, 0])
axes[1, 0].set_title('Education vs Target')
axes[1, 0].tick_params(axis='x', rotation=45)

sns.countplot(data=train_df, x='housing', hue='y', ax=axes[1, 1])
axes[1, 1].set_title('Housing Loan vs Target')

sns.countplot(data=train_df, x='loan', hue='y', ax=axes[2, 0])
axes[2, 0].set_title('Personal Loan vs Target')

sns.countplot(data=train_df, x='contact', hue='y', ax=axes[2, 1])
axes[2, 1].set_title('Contact Method vs Target')

plt.tight_layout()
plt.show()


numerical_cols = train_df.select_dtypes(include='int64').drop(columns=['id', 'y']).columns.tolist()
for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(train_df[col], bins=30, kde=False)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()


numerical_cols = train_df.select_dtypes(include='int64').drop(columns=['id', 'y']).columns.tolist()
skewness = train_df[numerical_cols].skew()
print("Skewness of Numerical Features:")
print(skewness)


numerical_cols = train_df.select_dtypes(include='int64').drop(columns=['id', 'y']).columns.tolist()
high_skew_cols = train_df.select_dtypes(include='int64').drop(columns=['id', 'y']).columns[train_df.select_dtypes(include='int64').drop(columns=['id', 'y']).skew().abs() > 1]
train_df_transformed = train_df.copy()

for col in high_skew_cols:
    train_df_transformed[col] = train_df_transformed[col].apply(lambda x: np.log1p(x) if x > 0 else x)

for col in high_skew_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(train_df_transformed[col], bins=30, kde=False)
    plt.title(f'Log Transformed Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

print(train_df_transformed[high_skew_cols].skew())


numerical_cols = train_df.select_dtypes(include='int64').drop(columns=['id', 'y']).columns.tolist()
high_skew_cols = train_df[numerical_cols].skew().abs()[train_df[numerical_cols].skew().abs() > 1].index.tolist()
train_df_transformed_sqrt = train_df.copy()
train_df_transformed_cbrt = train_df.copy()

for col in high_skew_cols:
    train_df_transformed_sqrt[col] = train_df_transformed_sqrt[col].apply(lambda x: np.sqrt(x) if x > 0 else x)

for col in high_skew_cols:
    train_df_transformed_cbrt[col] = train_df_transformed_cbrt[col].apply(lambda x: np.cbrt(x) if x > 0 else x)

for col in high_skew_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(train_df_transformed_sqrt[col], bins=30, kde=False)
    plt.title(f'Square Root Transformed Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    sns.histplot(train_df_transformed_cbrt[col], bins=30, kde=False)
    plt.title(f'Cube Root Transformed Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

sqrt_skewness = train_df_transformed_sqrt[high_skew_cols].skew()
cbrt_skewness = train_df_transformed_cbrt[high_skew_cols].skew()

print("Square Root Transformation Skewness:")
print(sqrt_skewness)
print("\nCube Root Transformation Skewness:")
print(cbrt_skewness)


numerical_cols = train_df.select_dtypes(include='int64').drop(columns=['id', 'y']).columns.tolist()
high_skew_cols = train_df[numerical_cols].skew().abs()[train_df[numerical_cols].skew().abs() > 1].index.tolist()
train_df_transformed_boxcox = train_df.copy()
skipped_features = []

for col in high_skew_cols:
    if train_df_transformed_boxcox[col].nunique() == 1:
        skipped_features.append(col)
        print(f"Skipping {col} as it has constant values")
        continue
    
    if (train_df_transformed_boxcox[col] <= 0).any():
        skipped_features.append(col)
        print(f"Skipping {col} as it contains non-positive values")
        continue

    train_df_transformed_boxcox[col], _ = boxcox(train_df_transformed_boxcox[col])

for col in high_skew_cols:
    if col not in skipped_features:
        plt.figure(figsize=(8, 4))
        sns.histplot(train_df_transformed_boxcox[col], bins=30, kde=False)
        plt.title(f'Box-Cox Transformed Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

boxcox_skewness = train_df_transformed_boxcox[high_skew_cols].skew()
print("Skewness after Box-Cox Transformation:")
print(boxcox_skewness)


categorical_cols = train_df.select_dtypes(include='object').columns.tolist()
for i in range(0, len(categorical_cols), 3):
    batch_cols = categorical_cols[i:i + 3]
    fig, axes = plt.subplots(len(batch_cols), 1, figsize=(10, 4 * len(batch_cols)))

    if len(batch_cols) == 1:
        axes = [axes]

    for j, col in enumerate(batch_cols):
        sns.countplot(data=train_df, x=col, ax=axes[j],
                      order=train_df[col].value_counts().index)
        axes[j].set_title(f'Count of {col}')
        axes[j].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


top_categorical = ['job', 'month', 'education', 'poutcome', 'marital']

for col in top_categorical:
    plt.figure(figsize=(8, 4))
    sns.countplot(
        data=train_df,
        x=col,
        hue='y',
        order=train_df[col].value_counts().index  
    )
    plt.title(f'{col} by Target Variable (y)')
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=train_df, x='y', y=col)
    plt.title(f'Boxplot of {col} by Target Variable (y)')
    plt.xlabel('y')
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()


train_df['dataset'] = 'train'
test['dataset'] = 'test'
combined_df = pd.concat([train_df, test], axis=0)
numerical_cols = combined_df.select_dtypes(include='int64').drop(columns=['id']).columns.tolist()
categorical_cols = combined_df.select_dtypes(include='object').columns.tolist()

for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(data=combined_df, x=col, hue='dataset', kde=True, bins=30, common_norm=False)
    plt.title(f'Distribution of {col} in Train vs Test')
    plt.xlabel(col)
    plt.ylabel('Density')
    plt.tight_layout()
    plt.show()

for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=combined_df, x=col, hue='dataset', order=combined_df[col].value_counts().index)
    plt.title(f'Distribution of {col} in Train vs Test')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()




