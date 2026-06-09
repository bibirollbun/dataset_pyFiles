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
from scipy.stats import ttest_ind, chi2_contingency, pointbiserialr
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    LabelEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")


df.describe()


df.duplicated().sum()


df.isnull().sum()


df.nunique()


target_col = 'y'
numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
nominal_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']


all_cols = numeric_cols + nominal_cols

n = len(all_cols)
ncols = 2
nrows = (n + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows))
axes = axes.flatten()
blue_shade = '#004DAB'

for ax, col in zip(axes, all_cols):
    if col in numeric_cols:
        ax.hist(df[col], bins=30, color=blue_shade, edgecolor='black')
        ax.set_ylabel('Frequency')
    else:
        counts = df[col].value_counts().sort_index()
        ax.bar(counts.index.astype(str), counts.values, color=blue_shade)
        ax.set_ylabel('Count')
    ax.set_title(f'{col} Distribution')
    ax.set_xlabel(col)
    ax.tick_params(axis='x', rotation=0)

for ax in axes[n:]:
    fig.delaxes(ax)

plt.tight_layout()
plt.show()


cols = numeric_cols + nominal_cols
ncols = 2
nrows = (len(cols) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows))
axes = axes.flatten()
blue_shade = '#004DAB'
orange_shade = '#FFA500'

for ax, col in zip(axes, cols):
    if col in numeric_cols:
        counts = pd.crosstab(df[col].astype(int), df['y'])
        counts.plot(kind='bar', stacked=True, color=[blue_shade, orange_shade], ax=ax, legend=False)
    else:
        counts = pd.crosstab(df[col], df['y'])
        counts.plot(kind='bar', stacked=True, color=[blue_shade, orange_shade], ax=ax, legend=False)
    
    ax.set_ylim(0, counts.values.sum(axis=1).max() * 1.1)
    ax.tick_params(axis='x', bottom=False, labelbottom=False)

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, ['Introvert (0)', 'Extrovert (1)'], loc='upper right', title='Personality')
plt.tight_layout()
plt.show()


def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))

def cramers_v_matrix(df, cols):
    matrix = pd.DataFrame(index=cols, columns=cols)
    for col1 in cols:
        for col2 in cols:
            if col1 == col2:
                matrix.loc[col1, col2] = 1.0
            else:
                confusion = pd.crosstab(df[col1], df[col2])
                matrix.loc[col1, col2] = cramers_v(confusion)
    return matrix.astype(float)

cramers_matrix = cramers_v_matrix(df, nominal_cols + [target_col])

plt.figure(figsize=(10, 8))
sns.heatmap(cramers_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Cramér's V Correlation Heatmap")
plt.tight_layout()
plt.show()


X = df.drop(columns=['id', target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)



preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), nominal_cols)
    ],
    remainder='passthrough'
)


model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LGBMClassifier(random_state=42))
])

model_pipeline.fit(X_train, y_train)


y_pred = model_pipeline.predict(X_test)
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=1)
recall = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)
roc_auc = roc_auc_score(y_test, y_pred_proba)



print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, square=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()


df_test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')

X_test = df_test.drop(columns=['id'])

predictions = model_pipeline.predict(X_test)

submission_df = pd.DataFrame({'id': df_test['id'], 'y': predictions})

submission_df.to_csv('submission.csv', index=False)

