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


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder



train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')


train


train.dtypes


train = train.drop(columns=['id'])


# Encode target
fertilizer_le = LabelEncoder()
train['Fertilizer_Label'] = fertilizer_le.fit_transform(train['Fertilizer Name'])

# Encode Soil and Crop
soil_le = LabelEncoder()
crop_le = LabelEncoder()

train['Soil_Type_Label'] = soil_le.fit_transform(train['Soil Type'])
train['Crop_Type_Label'] = crop_le.fit_transform(train['Crop Type'])

test['Soil_Type_Label'] = soil_le.transform(test['Soil Type'])
test['Crop_Type_Label'] = crop_le.transform(test['Crop Type'])


feature_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Phosphorous', 'Potassium', 'Soil_Type_Label', 'Crop_Type_Label']
X = train[feature_cols]
y = train['Fertilizer_Label']
X_test = test[feature_cols]


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

val_scores = []
test_preds = np.zeros((X_test.shape[0], len(fertilizer_le.classes_)))

params = {
    "n_estimators": 202,
    "max_depth": 11,
    "learning_rate": 0.06487704576481379,
    "subsample": 0.8325030927531691,
    "colsample_bytree": 0.7282667156881286,
    "gamma": 0.07738746864437063,
    "min_child_weight": 18,
    "reg_alpha": 0.010459554158729477,
    "reg_lambda": 1.4993935282115212,
    "objective": "multi:softprob",
    "num_class": len(fertilizer_le.classes_),
    "eval_metric": "mlogloss",
    "random_state": 42,
    "tree_method": "hist",
    "device": "cuda",
    "use_label_encoder": False
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Training fold {fold + 1}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = XGBClassifier(**params, early_stopping_rounds=50)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    val_pred = model.predict_proba(X_val)
    score = log_loss(y_val, val_pred)
    print(f"Fold {fold + 1} log_loss: {score:.5f}")
    val_scores.append(score)

    test_preds += model.predict_proba(X_test) / skf.n_splits

print(f"Average CV log_loss: {np.mean(val_scores):.5f}")


top_3_preds = np.argsort(test_preds, axis=1)[:, -3:][:, ::-1]
top_3_labels = np.vectorize(lambda x: fertilizer_le.inverse_transform([x])[0])(top_3_preds)
final_preds = [' '.join(row) for row in top_3_labels]


submission = submission[['id']].copy()
submission['Fertilizer Name'] = final_preds
submission.to_csv('submission.csv', index=False)


