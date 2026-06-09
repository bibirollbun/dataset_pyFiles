import numpy as np # linear algebra
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
train.head()


test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
test.head()


sample_sub = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")
sample_sub.head()


import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
import numpy as np

TARGET = 'diagnosed_diabetes'
CATS = ['gender','ethnicity','education_level','income_level',
        'smoking_status','employment_status']
NUMS = ['age','alcohol_consumption_per_week','physical_activity_minutes_per_week',
        'diet_score','sleep_hours_per_day','screen_time_hours_per_day','bmi',
        'waist_to_hip_ratio','systolic_bp','diastolic_bp','heart_rate',
        'cholesterol_total','hdl_cholesterol','ldl_cholesterol','triglycerides',
        'family_history_diabetes','hypertension_history','cardiovascular_history']

X = train[CATS + NUMS]
y = train[TARGET]
X_test = test[CATS + NUMS]

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y )


# Preprocessing
preprocess = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), NUMS),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATS)
    ]
)


X_tr_p   = preprocess.fit_transform(X_tr)
X_val_p  = preprocess.transform(X_val)
X_test_p = preprocess.transform(X_test)


# MLP model (replacing XGBoost)
mlp_clf = MLPClassifier(
    hidden_layer_sizes=(100, 50),  # Two hidden layers
    activation='relu',
    solver='adam',
    alpha=0.001,  # L2 regularization
    batch_size='auto',
    learning_rate='constant',
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42
)
mlp_clf.fit(X_tr_p, y_tr)
mlp_val_pred = mlp_clf.predict_proba(X_val_p)[:, 1]
mlp_test_pred = mlp_clf.predict_proba(X_test_p)[:, 1]

# Gradient Boosting Classifier (sklearn)
gbc_clf = GradientBoostingClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    random_state=42
)
gbc_clf.fit(X_tr_p, y_tr)
gbc_val_pred = gbc_clf.predict_proba(X_val_p)[:, 1]
gbc_test_pred = gbc_clf.predict_proba(X_test_p)[:, 1]



# Prepare stacking input features (validation predictions)
stack_X = np.column_stack((mlp_val_pred, gbc_val_pred))

# Train logistic regression meta-model
meta_clf = LogisticRegression(random_state=42)
meta_clf.fit(stack_X, y_val)

# Predict on validation set using meta-model
stack_val_pred = meta_clf.predict_proba(stack_X)[:, 1]

# Predict on test set using meta-model
stack_test_X = np.column_stack((mlp_test_pred, gbc_test_pred))
stack_test_pred = meta_clf.predict_proba(stack_test_X)[:, 1]

print("Stacking Ensemble AUC (MLP  + GBC):", roc_auc_score(y_val, stack_val_pred))


# Create submission
sub = pd.DataFrame({
    'id': test['id'],
    'diagnosed_diabetes': stack_test_pred
})

sub.to_csv('submission_stacking_mlp_gbc.csv', index=False)




