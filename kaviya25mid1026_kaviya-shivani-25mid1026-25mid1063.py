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


# mess_quality_kappa_fixed.py
# Predict mess food quality ratings (ordinal classification with RMSE metric)

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
    StackingRegressor
)
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error

# ============================
# Paths
# ============================
DATA_DIR = Path('/kaggle/input/new-vit-chennai-mess-food-predictor')  # Change to your dataset folder
TRAIN_CSV = DATA_DIR / 'train.csv'
TEST_CSV  = DATA_DIR / 'test.csv'
SAMPLE_CSV = DATA_DIR / 'sample_submission.csv'

# ============================
# Load data
# ============================
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
sample = pd.read_csv(SAMPLE_CSV)

# Detect target column
for col in ['quality', 'Quality', 'mess_food_quality', 'target']:
    if col in train.columns:
        TARGET = col
        break
else:
    TARGET = train.columns[-1]

print(f"✅ Target column detected as: {TARGET}")

y = train[TARGET]
X = train.drop(columns=[TARGET])

# Drop ID if present
if 'id' in X.columns:
    X = X.drop(columns=['id'])
if 'id' in test.columns:
    test_ids = test['id']
    test = test.drop(columns=['id'])
else:
    test_ids = np.arange(len(test))

# ============================
# Preprocessing
# ============================
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object', 'category']).columns

num_pipe = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
cat_pipe = make_pipeline(SimpleImputer(strategy='most_frequent'),
                         OneHotEncoder(handle_unknown='ignore', sparse=False))

preprocessor = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols)
])

# ============================
# Models (no internal preprocessing)
# ============================
rf = RandomForestRegressor(
    n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=42
)
hgb = HistGradientBoostingRegressor(
    max_iter=300, early_stopping=True, random_state=42
)
ridge = RidgeCV(alphas=[0.1, 1.0, 10.0])

# Stack base models
stack = StackingRegressor(
    estimators=[('rf', rf), ('hgb', hgb)],
    final_estimator=ridge,
    n_jobs=-1
)

# Full pipeline
model = make_pipeline(preprocessor, stack)

# ============================
# Train/Validation
# ============================
Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(Xtr, ytr)
ytr_pred = model.predict(Xtr)
yval_pred = model.predict(Xval)

def rmse(a, b):
    return mean_squared_error(a, b, squared=False)

print(f"Train RMSE: {rmse(ytr, ytr_pred):.4f}")
print(f"Val RMSE: {rmse(yval, yval_pred):.4f}")

# ============================
# Fit on full data
# ============================
model.fit(X, y)

# ============================
# Predict and round to integer classes
# ============================
min_rating = int(y.min())
max_rating = int(y.max())

preds = model.predict(test)
preds_rounded = np.clip(np.rint(preds), min_rating, max_rating).astype(int)

# ============================
# Save submission
# ============================
submission = pd.DataFrame({
    'id': test_ids,
    'quality': preds_rounded
})
submission.to_csv('/kaggle/working/submission.csv', index=False)

print("\n✅ submission.csv saved successfully for Kaggle upload.")
print(submission.head())


