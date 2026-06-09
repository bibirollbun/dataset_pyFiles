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


# =========================================================
# mess_quality_model.py  â€” Kaggle-ready version
# Builds a regression model to predict mess food quality
# Uses CV + Stacking + Safe preprocessing (no overfitting)
# =========================================================

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, make_scorer
import joblib

# =========================================================
# PATHS â€” Change dataset folder name as needed
# =========================================================
DATA_DIR = Path('/kaggle/input/the-vit-chennai-mess-food-predictor')  # <-- your dataset folder name
TRAIN_CSV = DATA_DIR / 'train.csv'
TEST_CSV  = DATA_DIR / 'test.csv'
SAMPLE_CSV = DATA_DIR / 'sample_submission.csv'

OUT_SUBMISSION = Path('/kaggle/working/submission.csv')
OUT_MODEL = Path('/kaggle/working/stacking_model.joblib')

# =========================================================
# UTILITIES
# =========================================================
def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

cv_scorer = make_scorer(lambda y_true, y_pred: mean_squared_error(y_true, y_pred, squared=False),
                        greater_is_better=False)

# =========================================================
# LOAD DATA
# =========================================================
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
sample_submission = pd.read_csv(SAMPLE_CSV)

# Detect target column
for possible in ['mess_food_quality', 'food_quality', 'quality', 'target', 'y']:
    if possible in train.columns:
        TARGET = possible
        break
else:
    TARGET = train.columns[-1]

print("âœ… Using target:", TARGET)

y = train[TARGET].copy()
X = train.drop(columns=[TARGET]).copy()

# Drop ID if present
test_ids = None
if 'id' in X.columns:
    X = X.drop(columns=['id'])
if 'id' in test.columns:
    test_ids = test['id']
    test = test.drop(columns=['id'])

# Align columns
common_cols = [c for c in X.columns if c in test.columns]
if len(common_cols) < X.shape[1]:
    print(f"âš ï¸� Using intersection of train/test features ({len(common_cols)} columns).")
X = X[common_cols].copy()
test = test[common_cols].copy()

# =========================================================
# FEATURE TYPES
# =========================================================
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
print(f"Numeric features: {len(numeric_cols)}, Categorical features: {len(cat_cols)}")

# =========================================================
# PREPROCESSING PIPELINES
# =========================================================
num_transformer = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler()
)

if len(cat_cols) > 0:
    cat_transformer = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder(handle_unknown='ignore', sparse=False)
    )
    preprocessor = ColumnTransformer([
        ('num', num_transformer, numeric_cols),
        ('cat', cat_transformer, cat_cols)
    ])
else:
    preprocessor = ColumnTransformer([
        ('num', num_transformer, numeric_cols)
    ])

# =========================================================
# MODELS
# =========================================================
enet = make_pipeline(preprocessor, 
                     ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=3, random_state=42, max_iter=3000))

rf = make_pipeline(preprocessor, 
                   RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5,
                                         random_state=42, n_jobs=-1))

hgb = make_pipeline(preprocessor, 
                    HistGradientBoostingRegressor(max_iter=200, early_stopping=True, random_state=42))

# =========================================================
# CROSS-VALIDATION (3-fold for speed)
# =========================================================
cv = KFold(n_splits=3, shuffle=True, random_state=42)

def cross_rmse(model):
    scores = cross_val_score(model, X, y, cv=cv, scoring=cv_scorer, n_jobs=-1)
    rmses = np.abs(scores)
    return rmses.mean(), rmses.std()

print("ğŸ”� Running 3-fold CV (this will take a few minutes)...")
for name, model in [('ElasticNet', enet), ('RandomForest', rf), ('HGB', hgb)]:
    mean_rmse, std_rmse = cross_rmse(model)
    print(f"{name:<15}  RMSE: {mean_rmse:.4f} Â± {std_rmse:.4f}")

# =========================================================
# STACKING ENSEMBLE
# =========================================================
estimators = [
    ('enet', enet),
    ('rf', rf),
    ('hgb', hgb)
]

stack = StackingRegressor(
    estimators=estimators,
    final_estimator=RidgeCV(alphas=[0.1, 1.0, 10.0]),
    n_jobs=-1
)

mean_rmse, std_rmse = cross_rmse(stack)
print(f"Stacking Ensemble RMSE: {mean_rmse:.4f} Â± {std_rmse:.4f}")

# =========================================================
# FINAL TRAINING & PREDICTION
# =========================================================
print("ğŸš€ Fitting final stacked model on full training data...")
stack.fit(X, y)
preds = stack.predict(test)

# Clamp predictions (if needed)
preds = np.clip(preds, 0, 10)

# =========================================================
# EXPORT SUBMISSION
# =========================================================
submission = sample_submission.copy()
pred_col = [c for c in submission.columns if c not in ['id']]
if len(pred_col) == 0:
    submission['prediction'] = preds
else:
    submission[pred_col[0]] = preds

submission.to_csv(OUT_SUBMISSION, index=False)
joblib.dump(stack, OUT_MODEL)
print(f"âœ… Submission saved to: {OUT_SUBMISSION}")
print(f"âœ… Model saved to: {OUT_MODEL}")
print(submission.head())

# =========================================================
# OVERFITTING CHECK
# =========================================================
Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.2, random_state=42)
stack.fit(Xtr, ytr)
print("\nğŸ”� Overfitting check:")
print("Train RMSE:", rmse(ytr, stack.predict(Xtr)))
print("Val   RMSE:", rmse(yval, stack.predict(Xval)))
print("If train RMSE << val RMSE => possible overfitting")

# =========================================================
# Done
# =========================================================



