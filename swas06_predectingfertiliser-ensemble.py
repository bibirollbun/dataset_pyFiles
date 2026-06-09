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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


df_train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")


df_train.columns = df_train.columns.str.replace(' ', '_').str.lower()
df_test.columns = df_test.columns.str.replace(' ', '_').str.lower()


df_train.head(3)


import matplotlib.pyplot as plt

# Define the columns you want to plot
cat_cols = ['soil_type', 'crop_type']

# Initialize LabelEncoder for categorical columns
label_encoders = {col: LabelEncoder() for col in cat_cols}

# Apply LabelEncoder to each categorical column
for col in cat_cols:
    df_train[col] = label_encoders[col].fit_transform(df_train[col])
    df_test[col] = label_encoders[col].transform(df_test[col])

# Encode the target separately
target_le = LabelEncoder()
df_train['fertilizer_name'] = target_le.fit_transform(df_train['fertilizer_name'])



df_train.head(3)


X = df_train.drop(columns=['fertilizer_name'])
y = df_train['fertilizer_name']


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

for col in df_train.columns:
    # Reshape the column to 2D
    df_train[[col]] = scaler.fit_transform(df_train[[col]])



num_classes = len(target_le.classes_)
num_classes


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np

# Define MAP@3
def map3(actual, predicted_proba, k=3):
    top_k_preds = np.argsort(predicted_proba, axis=1)[:, ::-1][:, :k]
    score = 0.0
    for i in range(len(actual)):
        if actual[i] in top_k_preds[i]:
            rank = np.where(top_k_preds[i] == actual[i])[0][0]
            score += 1 / (rank + 1)
    return score / len(actual)

FOLDS = 15
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

n_classes = y.nunique()
oof = np.zeros((len(df_train), n_classes))
pred_prob = np.zeros((len(df_test), n_classes))
ensemble_scores = []

test = df_test[X.columns]

# Define models
logreg_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        C=1.0,
        n_jobs=-1,
        random_state=42
    )
)

xgb_model = XGBClassifier(
    objective='multi:softprob',
    num_class=n_classes,
    eval_metric='mlogloss',
    use_label_encoder=False,
    n_jobs=-1,
    random_state=42
)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    n_jobs=-1,
    random_state=42
)

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
    x_train, x_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # Train all models
    logreg_model.fit(x_train, y_train)
    xgb_model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=0)
    rf_model.fit(x_train, y_train)

    # Predict probabilities
    proba_lr_valid = logreg_model.predict_proba(x_valid)
    proba_xgb_valid = xgb_model.predict_proba(x_valid)
    proba_rf_valid = rf_model.predict_proba(x_valid)

    proba_lr_test = logreg_model.predict_proba(test)
    proba_xgb_test = xgb_model.predict_proba(test)
    proba_rf_test = rf_model.predict_proba(test)

    # Ensemble: average (equal weights)
    proba_ensemble_valid = (proba_lr_valid + proba_xgb_valid + proba_rf_valid) / 3
    proba_ensemble_test = (proba_lr_test + proba_xgb_test + proba_rf_test) / 3

    oof[valid_idx] = proba_ensemble_valid
    pred_prob += proba_ensemble_test

    score = map3(y_valid.values, proba_ensemble_valid)
    ensemble_scores.append(score)
    print(f"Fold {fold:02d}: MAP@3 = {score:.4f}")

print("\nMean MAP@3 score (3-model ensemble):", np.mean(ensemble_scores))


for col in df_test.columns:
    df_test[[col]] = scaler.fit_transform(df_test[[col]])


submission_data = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


top_3_preds = np.argsort(pred_prob, axis=1)[:, -3:][:, ::-1]
top_3_labels = target_le.inverse_transform(top_3_preds.ravel()).reshape(top_3_preds.shape)

submission = pd.DataFrame({
    'id': submission_data.id,
    'Fertilizer Name': [' '.join(row) for row in top_3_labels]
})
submission.to_csv('submission.csv', index=False)
submission.head()

