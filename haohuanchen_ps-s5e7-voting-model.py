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


train_df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


for col in ['Stage_fear', 'Drained_after_socializing']:
    train_df[col] = train_df[col].map({'No': 0, 'Yes': 1})
    test_df[col] = test_df[col].map({'No': 0, 'Yes': 1})


num_features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
bin_features = ['Stage_fear', 'Drained_after_socializing']
features = num_features + bin_features

X = train_df[features]
X_test = test_df[features]


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(train_df['Personality'])  # Extrovert=0, Introvert=1


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=-1)),
        ('scaler', MinMaxScaler())
    ]), num_features),
    ('bin', SimpleImputer(strategy='most_frequent'), bin_features)
])

X = preprocessor.fit_transform(X)
X_test = preprocessor.transform(X_test)


# import optuna
# from lightgbm import LGBMClassifier
# from sklearn.model_selection import StratifiedKFold, cross_val_score

# def objective(trial):
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 200, 2000),
#         'max_depth': trial.suggest_int('max_depth', 2, 16),
#         'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.03, log=True),
#         'num_leaves': trial.suggest_int('num_leaves', 8, 40),
#         'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
#         'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#         'random_state': 42,
#         'force_row_wise': True,
#         'verbosity': -1
#     }
    
#     model = LGBMClassifier(**params)
#     cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     score = cross_val_score(model, X, y, cv=cv, scoring='accuracy').mean()
#     return score

# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=50)

# print("Best trial:")
# print(f"  Accuracy: {study.best_trial.value:.4f}")
# print("  Params:")
# for k, v in study.best_trial.params.items():
#     print(f"    {k}: {v}")


from lightgbm import LGBMClassifier

lgb_model = LGBMClassifier(
    random_state=42, 
    n_estimators=1000, 
    learning_rate=0.01,
    max_depth=8,
    num_leaves=20,
    min_child_samples=21,
    subsample=0.9,
    colsample_bytree=0.8,
    n_jobs=-1,
    verbose=-1
)


from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=20,
    min_samples_leaf=13,
    max_features='sqrt',
    random_state=42
)


from catboost import CatBoostClassifier

cat_model = CatBoostClassifier(
    eval_metric='Accuracy',
    n_estimators=1000,
    learning_rate=0.01,
    depth=8,
    bagging_temperature=0.8,
    od_type='Iter',
    od_wait=50,
    random_seed=42,
    verbose=False,
    max_ctr_complexity=7
)


from sklearn.ensemble import VotingClassifier

vot_model = VotingClassifier(
    estimators=[
        ('lgb', lgb_model),
        ('rf', rf_model),
        ('cat', cat_model)
    ],
    voting='soft',
    weights=[2, 1, 2],
    n_jobs=-1
)


from sklearn.model_selection import StratifiedKFold, cross_val_score

models = [
    lgb_model, 
    rf_model, 
    cat_model, 
    vot_model
]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for model in models:
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

    print(f"{model}:")
    print(f"5-Fold Accuracy: {scores}")
    print(f"Mean Accuracy: {scores.mean():.4f}")


model = vot_model
model.fit(X, y)


y_pred = model.predict(X_test)


y_pred_labels = le.inverse_transform(y_pred)

submission = pd.DataFrame({"id": test_df["id"], "Personality": y_pred_labels})
submission.to_csv("submission.csv", index=False)

print("success!")


# importances = pd.Series(model.feature_importances_, index=X.columns)
# importances.sort_values().plot(kind='barh')

