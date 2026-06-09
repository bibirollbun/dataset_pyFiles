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


!pip install iterative-stratification


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from tqdm import tqdm
import warnings
import os

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Load data
train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")

# Rename temperature column
def rename_temperature_column(df):
    df = df.rename(columns={'Temparature': 'Temperature'})
    print("Column name corrected from 'Temparature' to 'Temperature'")
    return df

train = rename_temperature_column(train)
test = rename_temperature_column(test)

# Feature engineering
def add_features(df):
    df['Nitrogen_Potassium'] = df['Nitrogen'] * df['Potassium']
    df['Nitrogen_Phosphorous'] = df['Nitrogen'] * df['Phosphorous']
    df['Temperature_Humidity'] = df['Temperature'] * df['Humidity']
    df['Moisture_Ratio'] = df['Moisture'] / (df['Nitrogen'] + df['Potassium'] + df['Phosphorous'] + 1e-5)
    return df

train = add_features(train)
test = add_features(test)

# Identify categorical and numerical columns
cat_cols = ['Soil Type', 'Crop Type']
num_cols = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous',
            'Nitrogen_Potassium', 'Nitrogen_Phosphorous', 'Temperature_Humidity', 'Moisture_Ratio']

# Encode categorical features
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')

# Scale numerical features
scaler = StandardScaler()
train[num_cols] = scaler.fit_transform(train[num_cols])
test[num_cols] = scaler.transform(test[num_cols])

# Encode target
fer_le = LabelEncoder()
train['Fertilizer Name'] = fer_le.fit_transform(train['Fertilizer Name'])

# Prepare data
X = train.drop(columns=['id', 'Fertilizer Name'])
y = train['Fertilizer Name']
X_test = test.drop(columns=['id'])

# MAP@3 metric
def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        p = p[:k]
        score = 0.0
        hits = 0
        seen = set()
        for i, pred in enumerate(p):
            if pred in a and pred not in seen:
                hits += 1
                score += hits / (i + 1.0)
                seen.add(pred)
        return score / min(len(a), k)
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

# Cross-validation
FOLDS = 3  # Reduced to fit <10 min
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
oof = np.zeros((len(X), len(np.unique(y))))
pred_prob = np.zeros((len(X_test), len(np.unique(y))))

xgb_model = XGBClassifier(
    max_depth=12,
    colsample_bytree=0.467,
    subsample=0.86,
    n_estimators=1000,  # Reduced for speed
    learning_rate=0.03,
    gamma=0.26,
    max_delta_step=4,
    reg_alpha=2.7,
    reg_lambda=1.4,
    early_stopping_rounds=30,  # Tighter for speed
    objective='multi:softprob',
    random_state=13,
    enable_categorical=True,
    tree_method='hist',
    device='cuda'
)

for i, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    print('#' * 15, f"Fold {i+1}", '#' * 15)
    x_train, x_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    xgb_model.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],
        verbose=0
    )
    
    oof[valid_idx] = xgb_model.predict_proba(x_valid)
    pred_prob += xgb_model.predict_proba(X_test) / FOLDS

    top_3_preds = np.argsort(oof[valid_idx], axis=1)[:, -3:][:, ::-1]
    actual = [[label] for label in y_valid]
    map3_score = mapk(actual, top_3_preds)
    print(f"✅ Fold {i+1}: MAP@3 Score: {map3_score:.5f}")

# Validate shapes
print(f"OOF shape: {oof.shape}, Test pred shape: {pred_prob.shape}")

# Submission
top_3_preds = np.argsort(pred_prob, axis=1)[:, -3:][:, ::-1]
top_3_labels = fer_le.inverse_transform(top_3_preds.ravel()).reshape(top_3_preds.shape)
submission = pd.DataFrame({
    'id': submission['id'],
    'Fertilizer Name': [' '.join(row) for row in top_3_labels]
})
submission.to_csv('submission_xgb.csv', index=False)
print("✅ Submission file saved as 'submission_xgb.csv'")

