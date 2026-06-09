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

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


print(train.shape)
print(train.columns)
train.head()
train.info()
train.describe()
train['y'].value_counts(normalize=True)  # target imbalance?



train = train.drop('id', axis=1)
test = test.drop('id', axis=1)


plt.figure(figsize=(6, 4))
sns.set(style="whitegrid")
sns.countplot(x='y', data=train, color='#66b2a5')  # Fixed color from Set2
plt.title("Target Variable Distribution (y)", fontsize=14)
plt.xlabel("Subscribed (1 = Yes, 0 = No)")
plt.ylabel("Count")
plt.show()



numerical_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']



import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
numerical_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
color = '#66c2a5'

# Create grid
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
axes = axes.flatten()

# Plot each feature
for i, col in enumerate(numerical_columns):
    sns.histplot(train[col], kde=True, color=color, bins=30, ax=axes[i])
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')

# Hide any unused subplots
for j in range(len(numerical_columns), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()



categorical_columns = [
    'job', 'marital', 'education',
    'default', 'housing', 'loan',
    'contact', 'month', 'poutcome'
]



import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
color = '#66c2a5'

# Create a grid of plots
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()

for i, col in enumerate(categorical_columns):
    # Compute percentage of class 1 for each category
    plot_data = train.groupby(col)['y'].value_counts(normalize=True).unstack().fillna(0)[1]
    plot_data.sort_values(ascending=False).plot(
        kind='bar', ax=axes[i], color=color, edgecolor='black'
    )
    axes[i].set_title(f'Proportion of y=1 by {col}')
    axes[i].set_ylabel('Proportion of Subscribed')
    axes[i].set_xlabel(col)
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()



import seaborn as sns
import matplotlib.pyplot as plt

# Subset numerical features including target
numerical_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous', 'y']
corr_matrix = train[numerical_columns].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title("Correlation Heatmap", fontsize=16)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()



from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split

# Define categorical features
categorical_columns = [
    'job', 'marital', 'education',
    'default', 'housing', 'loan',
    'contact', 'month', 'poutcome'
]

# Prepare data
X = train.drop('y', axis=1)
y = train['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42, stratify=y)

# Indices of categorical columns
cat_features_index = [X.columns.get_loc(col) for col in categorical_columns]



from sklearn.metrics import roc_auc_score

model_cb = CatBoostClassifier(
    iterations=5000,
    learning_rate=0.1,
    # depth=6,
    # eval_metric='F1',
    # auto_class_weights='Balanced',
    cat_features=cat_features_index,
    verbose=100,
    random_state=42,
    task_type='GPU'
)

model_cb.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=200, verbose=100)

y_pred = model_cb.predict_proba(X_test)[:, 1]
roc_auc_score(y_test, y_pred)


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Compute ROC curve and ROC area
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='#66c2a5', lw=2, label=f'CatBoost (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')  # diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - CatBoost', fontsize=14)
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()



test_pred = model_cb.predict_proba(test.astype('str'))[:, 1]


sub = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")
sub['y'] = test_pred
sub.to_csv("submission.csv", index=False)
sub.head()

