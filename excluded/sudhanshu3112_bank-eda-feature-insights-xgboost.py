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


import pandas as pd

# File paths
train_path = "/kaggle/input/playground-series-s5e8/train.csv"
test_path = "/kaggle/input/playground-series-s5e8/test.csv"
submission_path = "/kaggle/input/playground-series-s5e8/sample_submission.csv"

# Reading CSV files
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
submission_df = pd.read_csv(submission_path)


print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("Sample submission shape:", submission_df.shape)



print("Train Data Sample:")
print(train_df.head())

print("\nTest Data Sample:")
print(test_df.head())

print("\nSample Submission:")
print(submission_df.head())


# Check data types and non-null counts
print("\nğŸ”¹ Train Data Info:")
print(train_df.info())

# Check for missing/null values
print("\nğŸ”¹ Null Values in Train Data:")
print(train_df.isnull().sum())

# Check categorical and numerical features
categorical_cols = train_df.select_dtypes(include='object').columns.tolist()
numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols.remove('y')  # Exclude target

print("\nğŸ”¹ Categorical Columns:", categorical_cols)
print("ğŸ”¹ Numerical Columns:", numerical_cols)




# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Warnings
import warnings
warnings.filterwarnings("ignore")

# Display settings
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)




# 2. Target Variable Distribution
sns.countplot(x='y', data=train_df)
plt.title("Target Variable Distribution (y)")
plt.xlabel("Target (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

# Display value counts and percentage
print("ğŸ”¹ Target Distribution (Counts):")
print(train_df['y'].value_counts())
print("\nğŸ”¹ Target Distribution (Percentage):")
print(train_df['y'].value_counts(normalize=True) * 100)


#  Categorical Feature Distributions by Target
cat_features = ['job', 'marital', 'education', 'housing', 'loan', 'contact', 'poutcome']

for col in cat_features:
    plt.figure(figsize=(10, 5))
    sns.countplot(data=train_df, x=col, hue='y', order=train_df[col].value_counts().index)
    plt.title(f"Distribution of {col} by Target (y)")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.legend(title='y', loc='upper right')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


#  Numerical Feature Distributions
num_features = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']

# Histograms and KDEs
for col in num_features:
    plt.figure(figsize=(10, 5))
    sns.histplot(train_df[col], bins=50, kde=True)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


# Box plots split by target
for col in num_features:
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=train_df, x='y', y=col)
    plt.title(f"{col} by Target (y)")
    plt.xlabel("y")
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()


# Correlation matrix (numerical features + target)
plt.figure(figsize=(10, 6))
corr = train_df[num_features + ['y']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap: Numerical Features and Target")
plt.tight_layout()
plt.show()


# Mean conversion rate by category
for col in cat_features:
    plt.figure(figsize=(10, 5))
    order = train_df.groupby(col)['y'].mean().sort_values(ascending=False).index
    sns.barplot(x=col, y='y', data=train_df, order=order)
    plt.title(f"Mean Target (y) by {col}")
    plt.ylabel("Average y (conversion rate)")
    plt.xlabel(col)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Cross-tab: marital vs loan
ct = pd.crosstab(train_df['marital'], train_df['loan'], normalize='index')
print("\nğŸ”¹ Crosstab: Marital vs Loan (Row-normalized):")
print(ct)

# Heatmap of above cross-tab
sns.heatmap(ct, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Crosstab Heatmap: Marital Status vs Loan")
plt.ylabel("Marital Status")
plt.xlabel("Has Loan")
plt.tight_layout()
plt.show()


# Outlier detection for balance, duration, and age
outlier_cols = ['balance', 'duration', 'age']

for col in outlier_cols:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=train_df[col])
    plt.title(f'Outlier Detection: {col}')
    plt.tight_layout()
    plt.show()


# Display 1st and 99th percentiles
for col in outlier_cols:
    q01 = train_df[col].quantile(0.01)
    q99 = train_df[col].quantile(0.99)
    print(f"{col}: 1st percentile = {q01}, 99th percentile = {q99}")


# Check how many 'unknown' values per categorical column
unknown_counts = {}

for col in categorical_cols:
    count = (train_df[col] == 'unknown').sum()
    if count > 0:
        unknown_counts[col] = count

# Print counts
for col, count in unknown_counts.items():
    print(f"{col}: {count} unknowns ({(count / len(train_df)) * 100:.2f}%)")

# Barplot
if unknown_counts:
    pd.Series(unknown_counts).sort_values(ascending=False).plot(kind='bar')
    plt.title("Count of 'unknown' per Categorical Feature")
    plt.ylabel("Number of 'unknown' values")
    plt.show()


train_df['age_bin'] = pd.cut(train_df['age'], bins=[17, 25, 35, 45, 55, 65, 100], labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])
sns.countplot(x='age_bin', data=train_df, hue='y')
plt.title("Age Bins vs Target")
plt.xlabel("Age Group")
plt.tight_layout()
plt.show()


# Mapping month to numerical value
month_map = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6,
             'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}

train_df['month_num'] = train_df['month'].map(month_map)

# Cyclical encoding
train_df['month_sin'] = np.sin(2 * np.pi * train_df['month_num'] / 12)
train_df['month_cos'] = np.cos(2 * np.pi * train_df['month_num'] / 12)

sns.scatterplot(x='month_sin', y='month_cos', hue='y', data=train_df)
plt.title("Cyclical Encoding of Month vs Target")
plt.tight_layout()
plt.show()


# Combine job and education as an interaction
train_df['job_edu'] = train_df['job'] + "_" + train_df['education']
top_10_combo = train_df['job_edu'].value_counts().head(10).index

plt.figure(figsize=(12, 6))
sns.barplot(x='job_edu', y='y', data=train_df[train_df['job_edu'].isin(top_10_combo)])
plt.title("Conversion Rate by Job + Education (Top 10 Combos)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


target_ratio = train_df['y'].value_counts(normalize=True)
print(f"Target Balance:\n{target_ratio}")


from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['y'])):
    print(f"Fold {fold + 1}:")
    print(f"  Train size: {len(train_idx)}, Validation size: {len(val_idx)}")


X = train_df.drop("y", axis=1)
y = train_df["y"]

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    print(f"Fold {fold+1}:")
    print("Train shape:", X_train.shape, "Val shape:", X_val.shape)



from xgboost import XGBClassifier

# Convert object columns to category
for col in X.select_dtypes(include="object").columns:
    X[col] = X[col].astype("category")

# Same for test_df
for col in test_df.select_dtypes(include="object").columns:
    test_df[col] = test_df[col].astype("category")

model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    use_label_encoder=False,
    eval_metric="logloss",
    enable_categorical=True  
)



from sklearn.metrics import accuracy_score
import numpy as np

scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
        enable_categorical=True
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    preds = model.predict(X_val)

    acc = accuracy_score(y_val, preds)
    scores.append(acc)

    print(f"Fold {fold+1} Accuracy: {acc:.4f}")

print("\nMean CV Accuracy:", np.mean(scores))



final_model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss",
    enable_categorical=True
)

final_model.fit(X, y)



# Age binning â†’ convert to category or string
train_df["age_bin"] = pd.cut(
    train_df["age"],
    bins=[0, 25, 35, 50, 100],
    labels=["Young", "Adult", "MidAge", "Senior"]
).astype(str)

test_df["age_bin"] = pd.cut(
    test_df["age"],
    bins=[0, 25, 35, 50, 100],
    labels=["Young", "Adult", "MidAge", "Senior"]
).astype(str)

# Month number (convert to int, not category)
month_map = {
    'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
    'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12
}

train_df["month_num"] = train_df["month"].map(month_map).astype(int)
test_df["month_num"] = test_df["month"].map(month_map).astype(int)

# Cyclical encoding of month
train_df["month_sin"] = np.sin(2 * np.pi * train_df["month_num"] / 12)
train_df["month_cos"] = np.cos(2 * np.pi * train_df["month_num"] / 12)

test_df["month_sin"] = np.sin(2 * np.pi * test_df["month_num"] / 12)
test_df["month_cos"] = np.cos(2 * np.pi * test_df["month_num"] / 12)

# Job + Education combined (keep as string)
train_df["job_edu"] = train_df["job"].astype(str) + "_" + train_df["education"].astype(str)
test_df["job_edu"] = test_df["job"].astype(str) + "_" + test_df["education"].astype(str)

# Convert object columns to category for *both* train and test after feature engineering
for col in train_df.select_dtypes(include="object").columns:
    train_df[col] = train_df[col].astype("category")

for col in test_df.select_dtypes(include="object").columns:
    test_df[col] = test_df[col].astype("category")



# Align columns - crucial for consistent feature sets
X_train_cols = X.columns
test_aligned = test_df[X_train_cols]


# Predict probabilities instead of class labels
test_probs = final_model.predict_proba(test_aligned)[:, 1]

# Save in submission format
submission_df["y"] = test_probs
submission_df.to_csv("submission.csv", index=False)

print("âœ… Submission file with probabilities saved as submission.csv")
print(submission_df.head())





