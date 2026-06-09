import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"/kaggle/input/cat-in-the-dat-ii/train.csv")

df.drop('id', axis=1, inplace=True)
df.head()
df.info()
df.describe()


# Select important features
selected_features = [
    'bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4',
    'ord_0', 'ord_1', 'ord_2',
    'nom_0', 'nom_1', 'nom_2', 'nom_3',
    'day', 'month', 'target'
]

df = df[selected_features]

# Convert binary features to numeric
bin_map = {'F': 0, 'T': 1, 'N': 0, 'Y': 1}
for col in ['bin_3', 'bin_4']:
    df[col] = df[col].replace(bin_map)

for col in ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

for col in ['ord_1', 'ord_2']:
    df[col] = df[col].astype('category').cat.codes

print("Dataset Overview")
print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

print("First 5 rows:")
print(df.head(), "\n")

print("Target distribution:")
print(df['target'].value_counts(normalize=True).round(3), "\n")

print("Data types:")
print(df.dtypes, "\n")

print("Summary statistics:")
print(df.describe(include=[np.number]).T, "\n")


# Unique value counts for categorical columns
categorical_cols = df.select_dtypes(include='object').columns.tolist()
print("Unique value counts for categorical columns:")
for col in categorical_cols:
    print(f"{col}: {df[col].nunique()} unique values")
    print(df[col].value_counts().head(), "\n")


# Target distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=df)
plt.title("Target Class Distribution")
plt.show()


# Boxplot: ord_0 vs target
plt.figure(figsize=(6, 4))
sns.boxplot(x='target', y='ord_0', data=df)
plt.title("ord_0 vs Target")
plt.show()


# Barplot: mean target by nom_1
plt.figure(figsize=(6, 4))
df.groupby('nom_1')['target'].mean().sort_values().plot(kind='bar')
plt.title("Mean Target by nom_1")
plt.ylabel("Target Rate")
plt.xticks(rotation=45)
plt.show()


# Scatterplot: ord_2 vs day
plt.figure(figsize=(6, 4))
sns.scatterplot(x='ord_2', y='day', hue='target', data=df, alpha=0.6)
plt.title("ord_2 vs day")
plt.show()


# Pairplot: bin_0, bin_1, month
sns.pairplot(df[['bin_0', 'bin_1', 'month', 'target']], hue='target')
plt.suptitle("bin_0, bin_1, month", y=1.02)
plt.show()


# Correlation heatmap (numeric only)
numeric_cols = [
    'bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4',
    'ord_0', 'ord_1', 'ord_2', 'day', 'month', 'target'
]
plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()


import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder

train=pd.read_csv(r"/kaggle/input/cat-in-the-dat-ii/train.csv")
test = pd.read_csv(r"/kaggle/input/cat-in-the-dat-ii/test.csv")
sample_submission = pd.read_csv(r"/kaggle/input/cat-in-the-dat-ii/sample_submission.csv")

# Separate features and target
X = train.drop(['id', 'target'], axis=1)
y = train['target']
X_test = test.drop(['id'], axis=1)

# missing values
for col in X.columns:
    if X[col].isnull().sum() > 0:
        if X[col].dtype == 'object':
            X[col].fillna('missing', inplace=True)
            X_test[col].fillna('missing', inplace=True)
        else:
            median = X[col].median()
            X[col].fillna(median, inplace=True)
            X_test[col].fillna(median, inplace=True)

# Encoding categorical features
cat_cols = X.select_dtypes(include='object').columns.tolist()
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[cat_cols] = encoder.fit_transform(X[cat_cols])
X_test[cat_cols] = encoder.transform(X_test[cat_cols])

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# LightGBM model
model = lgb.LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.03,
    objective='binary',
    random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate on validation set
val_preds = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, val_preds)
print("ROC AUC on validation set:", round(auc, 5))

# Predict on test set
test_preds = model.predict_proba(X_test)[:, 1]

# submission file
submission = pd.DataFrame({
    'id': test['id'],
    'target': test_preds
})
submission.to_csv("submission.csv", index=False)
print (submission.head())

