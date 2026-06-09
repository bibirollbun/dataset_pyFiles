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


# !git clone https://github.com/PriorLabs/tabpfn-extensions
!pip install tabpfn
# !pip install -e tabpfn-extensions


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
orig = pd.read_csv("/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv")
display(train.info(), train.head(), train.describe().T)


orig.columns = orig.columns.str.strip()
orig['rainfall'] = orig['rainfall'].str.lower().map({'yes': 1, 'no': 0})
train = train.drop(columns=['id'])
train = pd.concat([orig, train], axis=0, ignore_index=True)
train = train.fillna(train.mean())
test = test.fillna(test.mean())
train.info()


X = train.drop(columns=["rainfall"])
y = train["rainfall"]
display(X.info(), y.info())


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from tabpfn import TabPFNClassifier
# from sklearn.inspection import permutation_importance

# split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

# Fit TabPFN
# model = TabPFNClassifier(device = "cuda", random_state = 42)
# model.fit(X_train, y_train)

# Compute feature importance
# result = permutation_importance(model, X_val, y_val, scoring="roc_auc")

# Print importance scores
# for i, score in enumerate(result.importances_mean):
#    print(f"Feature Importance of {X_train.columns[i]}: {score}")


# help the skew/imbalance
smote = SMOTE(sampling_strategy=1.0, random_state=42)
X_train_resampled, y_train_resampled = X_train, y_train#smote.fit_resample(X_train, y_train)

# scale data, gave minor boost to MI scores
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_resampled), columns=X_train_resampled.columns)


from sklearn.feature_selection import SelectKBest, mutual_info_classif

# find most relevant features
selector = SelectKBest(mutual_info_classif, k=11)
X_train_selected = selector.fit_transform(X_train_resampled, y_train_resampled)
selected_features = X_train_resampled.columns[selector.get_support()]
print("Selected Features:", selected_features.tolist())

X_train_selected = X_train_resampled[selected_features]
X_val_selected = X_val[selected_features]


display(X_train_selected.info(), y_train_resampled.info())


from sklearn.metrics import roc_auc_score
from tabpfn import TabPFNClassifier

# train
model = TabPFNClassifier(device = "cuda", random_state = 12)
model.fit(X_train_selected, y_train_resampled)

# predict
val_probs = model.predict_proba(X_val_selected)
print("ROC AUC: ", roc_auc_score(y_val, val_probs[:, 1]))
# ROC AUC with no feature selection, no scaling, no smote:  0.871420668580912
# ROC AUC with feature selection, scaling, and smote: 0.8644134769895417
# ROC AUC with autotabpfn, feature selection, scaling, and smote: 0.8608835383683253
# ROC AUC with autotabpfn and no smote: 0.8702879270830589

# Using Orig Dataset + Given Dataset:
# using smote and 8 features: 0.8537081339712919
# smote and all features: 0.852811004784689
# no smote and all features: 0.8685007974481659
# no smote and 8 features: 0.8673644338118023
# no smote, all features, stratify, random = 12: 0.9004186602870813
# no smote, 8 features, stratify: 0.8973086124401913


# predict on test
test = test.drop(columns="id")
test_probs = model.predict_proba(test[selected_features])[:, 1]


# create submission
submission = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")
submission['rainfall'] = test_probs
submission.to_csv('submission_final.csv', index=False)
submission.head()




