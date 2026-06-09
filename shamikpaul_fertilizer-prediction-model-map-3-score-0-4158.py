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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from sklearn.metrics import label_ranking_average_precision_score

# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

# Encode target
fertilizer_le = LabelEncoder()
train["Fertilizer Encoded"] = fertilizer_le.fit_transform(train["Fertilizer Name"])

# Encode categorical features
soil_le = LabelEncoder()
crop_le = LabelEncoder()
train["Soil Encoded"] = soil_le.fit_transform(train["Soil Type"])
train["Crop Encoded"] = crop_le.fit_transform(train["Crop Type"])
test["Soil Encoded"] = soil_le.transform(test["Soil Type"])
test["Crop Encoded"] = crop_le.transform(test["Crop Type"])

# Feature set
features = ["Temparature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous", "Soil Encoded", "Crop Encoded"]
X = train[features]
y = train["Fertilizer Encoded"]
X_test = test[features]

# Prepare for cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(X), len(np.unique(y))))
test_preds = np.zeros((len(X_test), len(np.unique(y))))

# Train model and collect predictions
for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"\nTraining Fold {fold+1}")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        eval_metric="MultiClass",
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50,
        task_type="CPU"
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_valid, y_valid),
        cat_features=[6, 7]
    )

    oof_preds[valid_idx] = model.predict_proba(X_valid)
    test_preds += model.predict_proba(X_test) / kf.n_splits

# Evaluate MAP@3
true_oof = pd.get_dummies(y)
map3_score = label_ranking_average_precision_score(true_oof.values, oof_preds)
print(f"\n Final MAP@3 Score: {map3_score:.4f}")

# Get Top-3 predictions for test set
top_3 = np.argsort(test_preds, axis=1)[:, -3:][:, ::-1]
top_3_labels = [
    " ".join(fertilizer_le.inverse_transform(row)) for row in top_3
]

# Create submission
submission = pd.DataFrame({
    "id": test["id"],
    "Fertilizer Name": top_3_labels
})
#submission.to_csv("submission.csv", index=False)
#print("\n Submission saved as 'submission.csv'")

