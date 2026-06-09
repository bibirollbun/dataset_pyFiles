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
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import shap
import optuna
import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

# 1. Drop ID and extract features + target
X_raw = train.drop(columns=['id', 'Personality'])
X_test_raw = test.drop(columns=['id'])

# 2. Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(train['Personality'])

# 3. One-hot encode categorical features
X = pd.get_dummies(X_raw)
X_test = pd.get_dummies(X_test_raw)

# 4. Align test set to train columns (avoid column mismatch)
X_test = X_test.reindex(columns=X.columns, fill_value=0)

# 5. Impute missing values
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)
X_test = imputer.transform(X_test)

# 6. Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# 7. Train Logistic Regression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X, y_encoded)
baseline_preds = logreg.predict(X)

# 8. Accuracy
baseline_acc = accuracy_score(y_encoded, baseline_preds)
print(f"ğŸ”¹ Baseline Logistic Regression Accuracy: {baseline_acc:.4f}")


sns.countplot(x='Personality', data=train)
plt.title('Target Distribution')
plt.show()

num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
            'Friends_circle_size', 'Post_frequency']
train[num_cols].hist(figsize=(12, 8), bins=20)
plt.suptitle("Distributions of Numeric Features", fontsize=16)
plt.show()

for col in num_cols:
    sns.boxplot(x='Personality', y=col, data=train)
    plt.title(f'{col} by Personality')
    plt.show()


X = train.drop(columns=['id', 'Personality'])
y = train['Personality']
X_test = test.drop(columns=['id'])

cat_cols = ['Stage_fear', 'Drained_after_socializing']
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    le_dict[col] = le

target_le = LabelEncoder()
y_encoded = target_le.fit_transform(y)

# Feature engineering
X['Alone_Score'] = X['Time_spent_Alone'] * X['Drained_after_socializing']
X_test['Alone_Score'] = X_test['Time_spent_Alone'] * X_test['Drained_after_socializing']
X['Post_frequency_log'] = np.log1p(X['Post_frequency'])
X_test['Post_frequency_log'] = np.log1p(X_test['Post_frequency'])

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
X_test = imputer.transform(X_test)

X_adv = np.vstack([X, X_test])
y_adv = np.hstack([np.zeros(X.shape[0]), np.ones(X_test.shape[0])])
X_train_adv, X_val_adv, y_train_adv, y_val_adv = train_test_split(X_adv, y_adv, test_size=0.2, random_state=42)
model_adv = lgb.LGBMClassifier(n_estimators=100)
model_adv.fit(X_train_adv, y_train_adv)
y_pred_adv = model_adv.predict_proba(X_val_adv)[:, 1]
auc_adv = roc_auc_score(y_val_adv, y_pred_adv)
print(f"Adversarial Validation AUC: {auc_adv:.4f}")


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds_lgb = np.zeros((X_test.shape[0], 2))
test_preds_xgb = np.zeros((X_test.shape[0], 2))
test_preds_cb = np.zeros((X_test.shape[0], 2))
val_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_encoded)):
    print(f"Fold {fold + 1}")

    # Corrected: direct indexing for NumPy arrays
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

    # LightGBM
    model_lgb = lgb.LGBMClassifier(n_estimators=1000)
    model_lgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    val_preds = model_lgb.predict(X_val)
    val_scores.append(accuracy_score(y_val, val_preds))
    test_preds_lgb += model_lgb.predict_proba(X_test) / kf.n_splits

    # XGBoost
    model_xgb = xgb.XGBClassifier(
        n_estimators=1000,
        use_label_encoder=False,
        eval_metric='logloss',
        objective='binary:logistic'
    )
    model_xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=0
    )
    test_preds_xgb += model_xgb.predict_proba(X_test) / kf.n_splits

    # CatBoost
    model_cb = cb.CatBoostClassifier(iterations=1000, verbose=0)
    model_cb.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50
    )
    test_preds_cb += model_cb.predict_proba(X_test) / kf.n_splits

print(f"Average CV Accuracy: {np.mean(val_scores):.4f}")


explainer = shap.Explainer(model_lgb)
shap_values = explainer(X[:500])
shap.summary_plot(shap_values, features=pd.DataFrame(X[:500]), feature_names=train.drop(columns=['id', 'Personality']).columns)


X = train.drop(columns=['id', 'Personality'])
y = train['Personality']
X_test = test.drop(columns=['id'])

cat_cols = ['Stage_fear', 'Drained_after_socializing']
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))


import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

# Split a holdout validation set
X_opt_train, X_opt_val, y_opt_train, y_opt_val = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

def objective(trial):
    param = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'lambda_l1': trial.suggest_float('lambda_l1', 0, 5),
        'lambda_l2': trial.suggest_float('lambda_l2', 0, 5)
    }

    dtrain = lgb.Dataset(X_opt_train, label=y_opt_train)
    dval = lgb.Dataset(X_opt_val, label=y_opt_val)

    model = lgb.train(
        param,
        dtrain,
        valid_sets=[dval],
        num_boost_round=1000,
        early_stopping_rounds=50,
        verbose_eval=False
    )

    preds = model.predict(X_opt_val)
    return log_loss(y_opt_val, preds)

# Run the tuning
# Uncomment to execute
# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=50)

# Best params
# print(" Best params:", study.best_params)

