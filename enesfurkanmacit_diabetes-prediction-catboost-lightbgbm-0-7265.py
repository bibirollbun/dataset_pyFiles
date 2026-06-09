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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

# Save test IDs for submission
test_ids = test_df['id']


print("--- Train Set Info ---")
train_df.info()

print("\n--- Test Set Info ---")
test_df.info()

print("\n--- Missing Values in Train Set ---")
print(train_df.isnull().sum())

print("\n--- Missing Values in Test Set ---")
print(test_df.isnull().sum())


train_df.head()


categorical_cols = train_df.select_dtypes(include=['object']).columns

print("--- Categorical Variable Comparison ---\n")

for col in categorical_cols:
    print(f"Column: {col}")

    train_vals = set(train_df[col].unique())
    test_vals = set(test_df[col].unique())

    only_in_train = train_vals - test_vals
    only_in_test = test_vals - train_vals

    print(f"Train value count: {len(train_vals)}")
    print(f"Test value count : {len(test_vals)}")

    if only_in_train:
        print(f"-> Present in Train but NOT in Test: {only_in_train}")

    if only_in_test:
        print(f"-> WARNING: Present in Test but NOT in Train: {only_in_test}")
        print("   (This might cause errors in the model!)")
    else:
        print("-> OK: All values in Test are known in Train.")

    print("-" * 50 + "\n")


fig, axes = plt.subplots(5, 1, figsize=(8, 25))

sns.countplot(x='diagnosed_diabetes', data=train_df, ax=axes[0], palette='viridis')
axes[0].set_title('Distribution of Diabetes Diagnosis')
axes[0].set_xlabel('Diabetes (0: No, 1: Yes)')
axes[0].set_ylabel('Count')

sns.histplot(data=train_df, x='bmi', hue='diagnosed_diabetes', kde=True, element="step", ax=axes[1], palette='husl')
axes[1].set_title('Distribution by Body Mass Index (BMI)')

sns.histplot(data=train_df, x='age', hue='diagnosed_diabetes', kde=True, element="step", ax=axes[2], palette='husl')
axes[2].set_title('Diabetes Distribution by Age')

sns.countplot(x='gender', hue='diagnosed_diabetes', data=train_df, ax=axes[3], palette='Set2')
axes[3].set_title('Diabetes Counts by Gender')

sns.countplot(x='ethnicity', hue='diagnosed_diabetes', data=train_df, ax=axes[4], palette='Set3')
axes[4].set_title('Diabetes Counts by Ethnicity')
axes[4].tick_params(axis='x', rotation=45) # Rotate labels to prevent overlap

plt.tight_layout()
plt.show()


numeric_df = train_df.select_dtypes(include=['float64', 'int64'])
corr_matrix = numeric_df.corr()
plt.figure(figsize=(20, 15))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Variables')
plt.tight_layout()
plt.show()


categorical_cols = train_df.select_dtypes(include=['object']).columns

global_rate = train_df['diagnosed_diabetes'].mean()
print(f"Global Diabetes Rate: {global_rate:.2%}")

sns.set(style="whitegrid")
n_cols = 2
n_rows = (len(categorical_cols) + 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    
    sns.barplot(x=col, y='diagnosed_diabetes', data=train_df, ax=axes[i], palette='coolwarm', errorbar=None)

    axes[i].axhline(global_rate, color='red', linestyle='--', linewidth=2, label=f'Global Avg. ({global_rate:.1%})')

    axes[i].set_title(f'{col} - Impact on Diabetes Risk')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Diabetes Probability')
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].legend()
    axes[i].set_ylim(0, 1.0) 

    for p in axes[i].patches:
        height = p.get_height()
        if height > 0:
            diff = height - global_rate 
            diff_text = f"{height:.1%}\n({diff:+.1%})" 

            axes[i].annotate(diff_text, (p.get_x() + p.get_width() / 2., height),
                             ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 5),
                             textcoords='offset points')

for i in range(len(categorical_cols), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()


cols_to_drop = [
    "id",
    "alcohol_consumption_per_week",
    "sleep_hours_per_day",
    "gender",
    "ethnicity",
    "education_level",
    "income_level",
    "smoking_status",
    "employment_status"
]

train_df = train_df.drop(cols_to_drop, axis=1, errors='ignore')
test_df = test_df.drop(cols_to_drop, axis=1, errors='ignore')

print(f"Dropped columns: {cols_to_drop}")
print(f"New Train shape: {train_df.shape}")


from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

X = train_df.drop(columns=['diagnosed_diabetes'])
y = train_df['diagnosed_diabetes']
X_test = test_df 

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train Set Size: {X_train.shape}")
print(f"Validation Set Size: {X_val.shape}")

# CATBOOST MODEL TRAINING
print("\nTraining model...")

model = CatBoostClassifier(
    iterations=2000,            
    learning_rate=0.03,         
    depth=6,                    
    eval_metric='AUC',          
    random_seed=42,
    verbose=100,                
    early_stopping_rounds=100   
)

model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    use_best_model=True,
    plot=False
)

# PERFORMANCE EVALUATION (ROC CURVE)

y_val_probs = model.predict_proba(X_val)[:, 1]

fpr, tpr, thresholds = roc_curve(y_val, y_val_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'CatBoost (AUC = {roc_auc:.5f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Random guess line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Diabetes Prediction')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

print(f"Validation Set AUC Score: {roc_auc:.5f}")


# PREDICTION

print("\nPredicting on Test set...")

test_preds = model.predict(X_test)

print("\nFeature Importance Ranking:")
feature_importance = model.get_feature_importance(prettified=True)
print(feature_importance)


from lightgbm import LGBMClassifier
from lightgbm.callback import early_stopping

X = train_df.drop(columns=['diagnosed_diabetes'])
y = train_df['diagnosed_diabetes']
X_test = test_df 

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train Set Size: {X_train.shape}")
print(f"Validation Set Size: {X_val.shape}")

# LIGHTGBM MODEL TRAINING
print("\nTraining model...")

model = LGBMClassifier(
    n_estimators=2000,          
    learning_rate=0.03,        
    num_leaves=31,              
    n_jobs=-1,
    verbose=-1                  
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='auc',          # Use ROC AUC metric
    callbacks=[early_stopping(stopping_rounds=100, verbose=False)] # Correct usage for early stopping
)

# PERFORMANCE EVALUATION (ROC CURVE)
print("\nDrawing ROC Curve...")

y_val_probs = model.predict_proba(X_val)[:, 1]

fpr, tpr, thresholds = roc_curve(y_val, y_val_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'LightGBM (AUC = {roc_auc:.5f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') 
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Diabetes Prediction')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

print(f"Validation Set AUC Score: {roc_auc:.5f}")

# PREDICTION
print("\nPredicting on Test set...")

test_preds = model.predict(X_test)

print("\nFeature Importance Ranking:")
feature_importance = pd.DataFrame({
    'Feature Id': X_train.columns,
    'Importances': model.feature_importances_
}).sort_values(by='Importances', ascending=False)
print(feature_importance)




