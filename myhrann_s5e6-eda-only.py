import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Optional for interactive visualization
# %matplotlib inline

# Configurations
pd.set_option('display.max_columns', None)
sns.set(style="whitegrid")

import warnings
warnings.simplefilter('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


print("Train shape:", train.shape)
print("Test shape:", test.shape)

# Quick look at the train set
display(train.head())

# Check types and missing values
train.info()
print("\nMissing values in train:")
print(train.isnull().sum())


# Visualize class distribution â€” this is a multiclass classification task
plt.figure(figsize=(12, 5))
sns.countplot(y='Fertilizer Name', data=train, order=train['Fertilizer Name'].value_counts().index)
plt.title("Distribution of Fertilizer Labels")
plt.xlabel("Count")
plt.ylabel("Fertilizer Name")
plt.tight_layout()
plt.show()

# Show class proportions
train['Fertilizer Name'].value_counts(normalize=True) * 100



# Soil and Crop Type are likely key categorical variables affecting fertilizer needs
for col in ['Soil Type', 'Crop Type']:
    plt.figure(figsize=(10, 4))
    sns.countplot(data=train, x=col, order=train[col].value_counts().index)
    plt.title(f"{col} Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Class relationship to target
    pd.crosstab(train[col], train['Fertilizer Name'], normalize='index').plot(
        kind='bar', stacked=True, figsize=(12, 5), colormap='tab20'
    )
    plt.title(f"{col} vs Fertilizer Distribution")
    plt.ylabel("Proportion")
    plt.tight_layout()
    plt.show()



# Identify numerical features automatically
numeric_cols = train.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Compare distributions between train and test
for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.kdeplot(train[col], label='Train', fill=True)
    sns.kdeplot(test[col], label='Test', fill=True)
    plt.title(f"Distribution of {col} (Train vs Test)")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Correlation matrix between numerical features
plt.figure(figsize=(10, 8))
sns.heatmap(train[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix (Numerical Features)")
plt.tight_layout()
plt.show()



# Fertilizer names often include N-P-K ratios (e.g., "14-35-14" = N=14, P=35, K=14)
def extract_npk(fert_name):
    numbers = list(map(int, re.findall(r'\d+', str(fert_name))))
    return numbers if len(numbers) >= 3 else [np.nan, np.nan, np.nan]

# Apply to dataset
train[['N', 'P', 'K']] = train['Fertilizer Name'].apply(extract_npk).apply(pd.Series)
train[['N', 'P', 'K']] = train[['N', 'P', 'K']].astype(float)

# Boxplots for NPK components
for comp in ['N', 'P', 'K']:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=comp, data=train)
    plt.title(f"{comp} Distribution from Fertilizers")
    plt.tight_layout()
    plt.show()



grouped_npk = train.groupby(['Soil Type', 'Crop Type'])[['N', 'P', 'K']].mean().reset_index()
display(grouped_npk.sort_values(by=['N', 'P', 'K'], ascending=False).head(10))


# Ensure test has similar distribution (to avoid overfitting to train)
for col in ['Soil Type', 'Crop Type']:
    train_dist = train[col].value_counts(normalize=True)
    test_dist = test[col].value_counts(normalize=True)

    compare_df = pd.DataFrame({'Train': train_dist, 'Test': test_dist}).fillna(0)
    compare_df.plot(kind='bar', figsize=(10, 4), title=f"{col} Distribution (Train vs Test)")
    plt.ylabel("Proportion")
    plt.tight_layout()
    plt.show()



import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Encode target labels
le_target = LabelEncoder()
train['FertLabel'] = le_target.fit_transform(train['Fertilizer Name'])

# Encode categorical features
for col in ['Soil Type', 'Crop Type']:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])  # ensure consistent mapping

# Feature selection
features = ['Soil Type', 'Crop Type'] + numeric_cols
X_train, X_val, y_train, y_val = train_test_split(train[features], train['FertLabel'], test_size=0.2, random_state=42)

# Train basic model
model = lgb.LGBMClassifier()
model.fit(X_train, y_train)

# Plot feature importances
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
importances.plot(kind='bar', figsize=(10, 4), title="Feature Importances (LightGBM)")
plt.tight_layout()
plt.show()



import shap

# TreeExplainer
explainer = shap.TreeExplainer(model)

# SHAP Value
shap_values = explainer.shap_values(X_val)

# SHAP Summary Plot
print("\nSHAP Summary Plot (Class 0):")
shap.summary_plot(shap_values[0], X_val, feature_names=features)

