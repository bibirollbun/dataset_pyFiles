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


#import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import warnings

warnings.filterwarnings("ignore")


df_train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


df_train.shape,df_test.shape


df_train.info()


df_train.isnull().sum()


df_test.isnull().sum()


df_train.head(3)



# List of categorical columns to encode
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

# Create a copy of your dataframe
df_encoded = df_train.copy()

# Initialize LabelEncoder
le = LabelEncoder()

# Apply label encoding to each categorical column
for col in categorical_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col])
    df_test[col] = le.fit_transform(df_test[col])

# Now df_encoded has label encoded categorical features



df_test.head(3)


df_encoded.info()




# Features to scale
features_to_scale = ['age', 'balance', 'duration']

# Create a copy of your dataframe
df_scaled = df_encoded.copy()

# Initialize scaler
scaler = StandardScaler()

# Fit and transform the selected features
df_scaled[features_to_scale] = scaler.fit_transform(df_scaled[features_to_scale])
df_test[features_to_scale] = scaler.fit_transform(df_test[features_to_scale])


df_scaled.head(3)


X = df_scaled.drop('y',axis =1)  # Drop target column
y = df_scaled['y']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



!pip install xgboost



from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC

# Example classifiers
classifiers = {
    'XGBoost': XGBClassifier(
        learning_rate=0.03679726897488401,
        max_depth=10,
        min_child_weight=3,
        gamma=0.9279624595163816,
        subsample=0.705936847613209,
        colsample_bytree=0.7936048487576377,
        n_estimators=906
    ),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000),
    #'SVM': SVC(probability=True)  # enable probability estimates for AUC
}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for name, clf in classifiers.items():
    auc_scores = []
    all_preds = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = clf.fit(X_train, y_train)

        # Predict probabilities for AUC only if supported
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            proba_test = model.predict_proba(df_test)[:, 1]
        else:
            # Use decision function for models like SVM without predict_proba
            y_pred_proba = model.decision_function(X_val)
            proba_test = model.decision_function(df_test)

        auc = roc_auc_score(y_val, y_pred_proba)
        auc_scores.append(auc)
        all_preds.append(proba_test)

    accuracy = cross_val_score(clf, X, y, cv=5, scoring='accuracy').mean()

    print(f"\n=== {name} ===")
    print(f"AUC-ROC Scores per Fold: {auc_scores}")
    print(f"Mean AUC-ROC: {np.mean(auc_scores):.4f}")
    print(f"Mean Accuracy: {accuracy:.4f}")


sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")
preds = np.mean(all_preds, axis=0)
submission = pd.DataFrame({'id': sample_submission.id, 'y': preds})
print(submission.head())
submission.to_csv('submission_xgb.csv', index=False)

