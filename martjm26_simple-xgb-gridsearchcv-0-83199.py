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


# load data
df_train = pd.read_csv("/kaggle/input/playground-series-s4e6/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s4e6/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s4e6/sample_submission.csv")
submission.head()


# check for weird types, num entries, shape, etc...
df_train.info()


# look at ranges of values, binary features, weird min or max to signify outliers, etc...
df_train.describe()


# label encode target for later use
from sklearn.preprocessing import LabelEncoder
print(df_train["Target"].head())
le = LabelEncoder()
df_train["Target"] = le.fit_transform(df_train["Target"])
df_train["Target"]


# heatmap of correlation matrix to see relationships to Target (Mutual Info prob better for this tbh)
import matplotlib.pyplot as plt
import seaborn as sns

# corr matrix relating to Target feature
corr_train = df_train.corr()
target_corr = corr_train[["Target"]]

# plot matrix on heatmap
plt.figure(figsize=(10,10))
sns.heatmap(target_corr, annot=True, cmap='cool', fmt='.2f')
plt.title("Correlation with Target")
plt.show()


# get rid of columns with super low positive or negative correlation to Target
temp = target_corr[(target_corr['Target'] < -0.1) | (target_corr['Target'] > 0.1)]
cols = temp.T.columns

# only use filtered columns now on train and test df
new_cols = df_train.columns.intersection(cols)
new_train = df_train[new_cols]
new_cols = new_cols.drop("Target")
new_test = df_test[new_cols]
new_train


# encode again :P
le = LabelEncoder()
new_train["Target"] = le.fit_transform(new_train["Target"])
new_train["Target"]


# build model, could do better with more intensive tuning I assume
# XGB because I like it for classification
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

y = new_train["Target"]
X = new_train.drop("Target", axis = "columns")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 42)

# parameter grid for grid search
param_grid = {
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 0.9, 0.98],
    'colsample_bytree': [0.6, 0.7, 0.8],
    'n_estimators': [100, 120, 150],
    'learning_rate': [0.01, 0.1, 0.2]
}

# init model
xgb = XGBClassifier(random_state=42)

# set up grid search
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    verbose=2
)

# fit grid search
grid_search.fit(X_train, y_train)


# predictions on train/validation
from sklearn.metrics import accuracy_score, classification_report

# use best model for predictions
best_xgb = grid_search.best_estimator_
y_preds = best_xgb.predict(X_val)

# metrics
print("Accuracy: ", accuracy_score(y_val, y_preds))
print("Classification Report: ", classification_report(y_val, y_preds))


# predict on test
test_predictions = best_xgb.predict(new_test)

# reverse the label encoding
test_predictions_original = le.inverse_transform(test_predictions)

# create submission
ids = df_test['id']
output_df = pd.DataFrame({'id': ids, 'Predicted Target': test_predictions_original})
output_df['Predicted Target'] = output_df['Predicted Target'].replace(0,'Dropout')
output_df['Predicted Target'] = output_df['Predicted Target'].replace(1,'Enrolled')
output_df['Predicted Target'] = output_df['Predicted Target'].replace(2,'Graduate')
output_df.to_csv("predictions.csv", index=False)

print("Predictions saved to 'predictions.csv'.")
output_df.head()




