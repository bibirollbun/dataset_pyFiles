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

train_df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


print(train_df.info())
print(train_df.describe())
print(train_df.isnull().sum())


import matplotlib.pyplot as plt
import seaborn as sns

corr_matrix = train_df.corr()
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm")
plt.show()

sns.countplot(x="rainfall", data=train_df)
plt.show()


from sklearn.impute import SimpleImputer

X = train_df.drop(columns=["id", "rainfall"])
y = train_df["rainfall"]

imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
test_imputed = imputer.transform(test_df.drop(columns=["id"]))


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
test_scaled = scaler.transform(test_imputed)


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.19, random_state=42,shuffle = False)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

val_preds = model.predict_proba(X_val)[:, 1]
print("Validation AUC-ROC:", roc_auc_score(y_val, val_preds))


from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],  
    'max_depth': [None, 10, 20],      
    'min_samples_split': [2, 5, 10],  
    'min_samples_leaf': [1, 2, 4]     
}


grid_search = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=42),
                           param_grid, scoring='roc_auc', cv=10, n_jobs=-1)
grid_search.fit(X_train, y_train)


print("Best Parameters:", grid_search.best_params_)
print("Best AUC-ROC:", grid_search.best_score_)


final_model = grid_search.best_estimator_
final_model.fit(X_scaled, y)


test_preds = final_model.predict_proba(test_scaled)[:, 1]


submission = pd.DataFrame({
    "id": test_df["id"],
    "rainfall": test_preds
})
submission.to_csv("submission.csv", index=False)

print("Submission file created successfully!")




