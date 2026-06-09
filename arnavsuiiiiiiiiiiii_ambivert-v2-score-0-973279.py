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


df=pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
dt=pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
samp=pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")
org=pd.read_csv("/kaggle/input/personality-prediction-data-introvert-extrovert/personality_dataset.csv")


df.info()


dt.info()


df=df.drop(columns=['id'])
ids=dt['id']
dt=dt.drop(columns=['id'])


df.head(2)


dt.head(2)


org


import pandas as pd
import numpy as np

for df_ in [df, dt]:
    for col in df_.columns:
        if df_[col].dtype == 'object':  # Object column
            mode_val = df_[col].mode(dropna=True)[0]
            df_[col] = df_[col].fillna(mode_val)
        elif pd.api.types.is_numeric_dtype(df_[col]):
            mean_val = df_[col].mean()
            df_[col] = df_[col].fillna(mean_val).astype(int)



df.head()


dt.head()


import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations

sns.set(style='whitegrid')

# Numeric features
numeric_cols = [
    'Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
    'Friends_circle_size', 'Post_frequency'
]

# Boxplots: each numeric feature vs target hue
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x=col, hue='Personality')
    plt.title(f'{col} by Personality')
    plt.tight_layout()
    plt.show()

# Count plots for binary/categorical features with hue
for col in ['Stage_fear', 'Drained_after_socializing']:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=col, hue='Personality')
    plt.title(f'{col} Count by Personality')
    plt.tight_layout()
    plt.show()

# Scatter plots for every pair of numeric columns with target hue
for col1, col2 in combinations(numeric_cols, 2):
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x=col1, y=col2, hue='Personality', alpha=0.7)
    plt.title(f'{col1} vs {col2} by Personality')
    plt.tight_layout()
    plt.show()



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for dataset in [df, dt, org]:
    for col in dataset.columns:
        if dataset[col].dtype == 'O':
            if col == 'Personality' and (dataset is df or dataset is org):
                continue
            dataset[col] = le.fit_transform(dataset[col])



org.head()


df['Personality'] = df['Personality'].map({'Extrovert': 0, 'Introvert': 1})
org['Personality'] = org['Personality'].map({'Extrovert': 0, 'Introvert': 1})



combined = pd.concat([df, org], ignore_index=True)
num_duplicates = combined.duplicated().sum()
print("Number of duplicated rows:", num_duplicates)



from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# Features and target
X = org.drop(columns=['Personality'])
y = org['Personality']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)




model_dict = {
    "xgb": {
        'n_estimators': 535,
        'max_depth': 8,
        'learning_rate': 0.002097973384127859,
        'subsample': 0.6693808484311031,
        'colsample_bytree': 0.8259298847897588,
        'gamma': 2.880347324664747,
        'reg_alpha': 1.9846760919117905,
        'reg_lambda': 4.133694325572352,
        'min_child_weight': 2,
        'eval_metric': 'logloss',
        'use_label_encoder': False
    },
    "lgbm": {
        'n_estimators': 440,
        'max_depth': 7,
        'learning_rate': 0.0469397995554875,
        'num_leaves': 76,
        'min_child_samples': 28,
        'subsample': 0.7457511900952924,
        'colsample_bytree': 0.8990804759765815,
        'reg_alpha': 2.6671824619941886,
        'reg_lambda': 4.3579572576570005,
        'random_state': 42
    },
    "cat": {
        'iterations': 327,
        'depth': 5,
        'learning_rate': 0.05834689737939565,
        'l2_leaf_reg': 5.876071954206991,
        'border_count': 137,
        'random_strength': 0.011118773104161666,
        'bagging_temperature': 0.8625264026842994,
        'verbose': 0,
        'loss_function': 'Logloss'
    }
}



from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier



N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# Storage
oof_preds = {model: np.zeros((len(X),)) for model in model_dict}
test_preds = {model: np.zeros((len(X_test),)) for model in model_dict}



for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    # XGB
    xgb = XGBClassifier(**model_dict['xgb'])
    xgb.fit(X_tr, y_tr)
    oof_preds['xgb'][val_idx] = xgb.predict(X_val)
    test_preds['xgb'] += xgb.predict(X_test) / N_SPLITS

    # LGBM
    lgbm = LGBMClassifier(**model_dict['lgbm'])
    lgbm.fit(X_tr, y_tr)
    oof_preds['lgbm'][val_idx] = lgbm.predict(X_val)
    test_preds['lgbm'] += lgbm.predict(X_test) / N_SPLITS

    # CatBoost
    cat = CatBoostClassifier(**model_dict['cat'])
    cat.fit(X_tr, y_tr)
    oof_preds['cat'][val_idx] = cat.predict(X_val)
    test_preds['cat'] += cat.predict(X_test) / N_SPLITS



X_meta = pd.DataFrame({model: oof_preds[model] for model in model_dict})
X_meta_test = pd.DataFrame({model: test_preds[model] for model in model_dict})



import optuna

def objective(trial):
    C = trial.suggest_loguniform("C", 1e-3, 10.0)
    clf = LogisticRegression(C=C, max_iter=1000)
    clf.fit(X_meta, y)
    pred = clf.predict(X_meta)
    return f1_score(y, pred, average='macro')

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

# Final meta-learner
final_lr = LogisticRegression(C=study.best_params['C'], max_iter=1000)
final_lr.fit(X_meta, y)
final_preds = final_lr.predict(X_meta_test)



print("Final Meta Model F1:", f1_score(y_test, final_lr.predict(X_meta.loc[X_test.index]), average='macro'))
print("Classification Report:\n", classification_report(y_test, final_lr.predict(X_meta.loc[X_test.index])))



# Reinitialize test_preds for dt
test_preds_dt = {model: np.zeros((len(dt),)) for model in model_dict}



for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]

    # XGB
    xgb = XGBClassifier(**model_dict['xgb'])
    xgb.fit(X_tr, y_tr)
    test_preds_dt['xgb'] += xgb.predict(dt) / N_SPLITS

    # LGBM
    lgbm = LGBMClassifier(**model_dict['lgbm'])
    lgbm.fit(X_tr, y_tr)
    test_preds_dt['lgbm'] += lgbm.predict(dt) / N_SPLITS

    # CatBoost
    cat = CatBoostClassifier(**model_dict['cat'])
    cat.fit(X_tr, y_tr)
    test_preds_dt['cat'] += cat.predict(dt) / N_SPLITS



X_meta_dt = pd.DataFrame({model: test_preds_dt[model] for model in model_dict})



# Final stacked model prediction
stacked_preds = final_lr.predict(X_meta_dt)

# Map back to label names
label_map = {0: 'Extrovert', 1: 'Introvert'}
final_labels = [label_map[p] for p in stacked_preds]

# Create submission DataFrame
sub = pd.DataFrame({
    'id': ids,
    'Personality': final_labels
})

# Optional: Save
# sub.to_csv("submission.csv", index=False)



sub


# Count plot for df
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Personality')
plt.title('Personality Distribution in df')
plt.xlabel('Personality')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 4))
sns.countplot(data=sub, x='Personality')
plt.title('Predicted Personality Distribution in sub')
plt.xlabel('Personality')
plt.ylabel('Count')
plt.tight_layout()
plt.show()



sub.to_csv("submission.csv",index=False)

