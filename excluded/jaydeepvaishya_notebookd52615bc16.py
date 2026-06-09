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


# 1. Basic Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# 2. Load Data
train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
submission_format = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")

# 3. Fix Column Typo
train.rename(columns={'Temparature': 'Temperature'}, inplace=True)
test.rename(columns={'Temparature': 'Temperature'}, inplace=True)

# 4. Encode Categorical Columns
cat_cols = [col for col in train.columns if train[col].dtype == 'object' and col != 'Fertilizer Name']

for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

# 5. Encode Target Label
target_le = LabelEncoder()
train["Fertilizer Name"] = target_le.fit_transform(train["Fertilizer Name"])

# 6. Prepare Features and Labels
X = train.drop(columns=["id", "Fertilizer Name"])
y = train["Fertilizer Name"]
X_test = test.drop(columns=["id"])

# 7. Split for Simple Validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 8. Train a Simple XGBoost Model
model = XGBClassifier(
   params = {
    'objective': 'multi:softprob',
    'num_class': y.nunique(),
    'max_depth': 10,                   # ⬇️ Reduce to prevent overfitting
    'learning_rate': 0.03,             # ⬇️ Slower learning helps generalize better
    'subsample': 0.85,
    'colsample_bytree': 0.6,           # ⬇️ Add randomness
    'colsample_bynode': 0.6,           # ⬆️ Try new param to decorrelate trees
    'max_bin': 256,
    'tree_method': 'hist',
    'random_state': 42,
    'eval_metric': 'mlogloss',
    'device': "cuda",
    'enable_categorical': True,
    'n_estimators': 12000,             # ⬆️ Train longer, early stopping will halt early
    'early_stopping_rounds': 100,
}
)

model.fit(X_train, y_train)

# 9. Predict Top 3 Classes for Test Set
pred_probs = model.predict_proba(X_test)
top_3_preds = np.argsort(pred_probs, axis=1)[:, -3:][:, ::-1]  # Top 3 predictions per row
top_3_labels = target_le.inverse_transform(top_3_preds.ravel()).reshape(top_3_preds.shape)

# 10. Prepare Submission
submission = pd.DataFrame({
    'id': submission_format['id'],
    'Fertilizer Name': [' '.join(row) for row in top_3_labels]
})

submission.to_csv('submission.csv', index=False)
print("✅ Submission saved as 'submission.csv'")





