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


import warnings
warnings.filterwarnings('ignore')

train_data = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv", index_col = "id")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv", index_col = "id")


train_data.head()


X_train = train_data.drop(["Personality"], axis = 1)
y_train = train_data["Personality"]


test_data["Stage_fear"].unique()


from sklearn.preprocessing import LabelEncoder

cat_cols = ["Stage_fear", "Drained_after_socializing"]
for col in cat_cols:
    le = LabelEncoder()
    non_null_mask = X_train[col].notnull()
    X_train.loc[non_null_mask, col] = le.fit_transform(X_train.loc[non_null_mask, col])
    non_null_mask = test_data[col].notnull()
    test_data.loc[non_null_mask, col] = le.transform(test_data.loc[non_null_mask, col])

y_le = LabelEncoder()
y_train = y_le.fit_transform(y_train)


X_train.isnull().sum()


import seaborn as sns
import matplotlib.pyplot as plt
import math

# Features to plot
features = X_train.columns
n_features = len(features)

# Grid layout
n_cols = 3  # plots per row
n_rows = math.ceil(n_features / n_cols)

# Create figure and subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
axes = axes.flatten()

for i, col in enumerate(features):
    ax = axes[i]

    # Histogram with hue
    sns.histplot(data=X_train, x=col, hue=y_train, kde=True, bins=30,
                 element='step', stat='density', common_norm=False, ax=ax)
    
    # Mean & median lines
    mean_val = X_train[col].mean()
    median_val = X_train[col].median()
    
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=1, label='Mean')
    ax.axvline(median_val, color='blue', linestyle='-', linewidth=1, label='Median')
    
    ax.set_title(f'Distribution of {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Density')
    ax.legend()

# Remove extra axes if any
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()



sns.heatmap(X_train.corr(), annot = True)


Ranges = X_train.max() - X_train.min()
Ranges


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer

imputer = Pipeline(steps=[("Scaler",StandardScaler()),("Imputer",KNNImputer())])
X_train_imputed = imputer.fit_transform(X_train)
test_imputed = imputer.transform(test_data)

X_train_imputed = pd.DataFrame(X_train_imputed)
X_train_imputed.columns = X_train.columns
test_imputed = pd.DataFrame(test_imputed)
test_imputed.columns = test_data.columns


X_train_imputed.isnull().sum()


from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# param_grid = {
#     'n_estimators': [100, 200, 300, 400],
#     'max_depth': [1,5,10,20,50,100],
#     'learning_rate': [0.01, 0.05, 0.1],
# }

# grid_search = GridSearchCV(
#     estimator=xgb,
#     param_grid=param_grid,
#     scoring='accuracy',       
#     cv=5,                    
#     n_jobs=-1,                
#     verbose=1
# )

# grid_search.fit(X_train_imputed, y_train)

# print("Best Accuracy:", grid_search.best_score_)
# print("Best Parameters:", grid_search.best_params_)


model = XGBClassifier(learning_rate = 0.1, max_depth = 1, n_estimators = 200, n_jobs=-1)
model.fit(X_train_imputed, y_train)
y_test_preds = model.predict(test_imputed)


final_pred = pd.DataFrame(y_le.inverse_transform(y_test_preds))
submission = pd.concat([pd.DataFrame(test_data.index), final_pred], axis=1)
submission.to_csv('submission.csv', index=False)

