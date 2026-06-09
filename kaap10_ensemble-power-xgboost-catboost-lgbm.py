# Install CatBoost
!pip install catboost -q

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier

warnings.filterwarnings('ignore')
print("Libraries Ready!")


# 1. Load Data
train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')

# 2. Feature Engineering Function
def feature_engineering(df):
    # BMI Category
    df['BMI_Cat'] = pd.cut(df['bmi'], bins=[0, 18.5, 24.9, 29.9, 100], labels=[0, 1, 2, 3]).astype(int)
    # Blood Pressure Risk
    df['Hypertension_Risk'] = ((df['systolic_bp'] >= 130) | (df['diastolic_bp'] >= 80)).astype(int)
    # Cholesterol Ratio
    df['Cholesterol_Ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1e-5)
    # Age Groups
    df['Age_Group'] = pd.cut(df['age'], bins=[0, 30, 50, 100], labels=[0, 1, 2]).astype(int)
    return df

train = feature_engineering(train)
test = feature_engineering(test)

# 3. Label Encoding
object_cols = train.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in object_cols:
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

X = train.drop(['id', 'diagnosed_diabetes'], axis=1)
y = train['diagnosed_diabetes']
X_test = test.drop(['id'], axis=1)

print("Data Prepared!")


# 1. XGBoost 
xgb_model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

# 2. LightGBM (Fast & Efficient)
lgbm_model = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

# 3. CatBoost (Best for Categorical Data)
cat_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.03,
    depth=6,
    random_seed=42,
    verbose=0, 
    allow_writing_files=False
)

print("Models Loaded!")


ensemble = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('lgbm', lgbm_model),
        ('cat', cat_model)
    ],
    voting='soft' # Average probability
)

print("Training Ensemble Model")
ensemble.fit(X, y)
print("Training Done!")


preds = ensemble.predict_proba(X_test)[:, 1]
submission['diagnosed_diabetes'] = preds
submission.to_csv('submission.csv', index=False)
print("Submission File Ready: submission.csv")
submission.head()

