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


pip install xgboost lightgbm catboost seaborn optuna shap imbalanced-learn


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import shap
import optuna


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


print(train.info())
sns.heatmap(train.isnull(), cbar=False)
plt.title('Missing Data Heatmap')
plt.show()

sns.countplot(x='Personality', data=train)
plt.title('Target Distribution')
plt.show()


binary_map = {'Yes': 1, 'No': 0}
train['Stage_fear'] = train['Stage_fear'].map(binary_map)
train['Drained_after_socializing'] = train['Drained_after_socializing'].map(binary_map)
test['Stage_fear'] = test['Stage_fear'].map(binary_map)
test['Drained_after_socializing'] = test['Drained_after_socializing'].map(binary_map)

num_features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
cat_features = ['Stage_fear', 'Drained_after_socializing']

num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')
train[num_features] = num_imputer.fit_transform(train[num_features])
train[cat_features] = cat_imputer.fit_transform(train[cat_features])
test[num_features] = num_imputer.transform(test[num_features])
test[cat_features] = cat_imputer.transform(test[cat_features])

le = LabelEncoder()
train['Personality'] = le.fit_transform(train['Personality'])


pf = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
poly_feats = pf.fit_transform(train[num_features])
poly_df = pd.DataFrame(poly_feats, columns=pf.get_feature_names_out(num_features))
train = pd.concat([train.reset_index(drop=True), poly_df], axis=1)
test_poly = pf.transform(test[num_features])
test_poly_df = pd.DataFrame(test_poly, columns=pf.get_feature_names_out(num_features))
test = pd.concat([test.reset_index(drop=True), test_poly_df], axis=1)


X = train.drop(columns=['id', 'Personality'])
y = train['Personality']
X_test_final = test.drop(columns=['id'])

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
X_train, X_val, y_train, y_val = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)


tuned_models = {}
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Example: Tuning XGBoost with Optuna
def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }
    model = XGBClassifier(**params)
    scores = []
    for train_idx, val_idx in kfold.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx].values, X_train.iloc[val_idx].values
        y_train_fold, y_val_fold = y_train.iloc[train_idx].values, y_train.iloc[val_idx].values
        model.fit(X_train_fold, y_train_fold)
        preds = model.predict(X_val_fold)
        scores.append(accuracy_score(y_val_fold, preds))
    return np.mean(scores)

study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=30)
tuned_models['xgb'] = XGBClassifier(**study_xgb.best_params, random_state=42, use_label_encoder=False, eval_metric='logloss')
tuned_models['xgb'].fit(X_train.values, y_train.values)
print("Best XGBoost params:", study_xgb.best_params)


### Final Stacking


# stack = StackingClassifier(estimators=[(k, v) for k, v in tuned_models.items()], final_estimator=LogisticRegression(max_iter=1000, class_weight='balanced'))
# stack.fit(X_train.values, y_train.values)
# y_pred = stack.predict(X_val.values)
# print(f"\nStacked Model Accuracy: {accuracy_score(y_val, y_pred):.4f}")
# print(classification_report(y_val, y_pred))


explainer = shap.Explainer(tuned_models['xgb'], X_train.values)
shap_values = explainer(X_train.values)
shap.summary_plot(shap_values, X_train.values, plot_type="bar")
shap.summary_plot(shap_values, X_train.values)


test_preds = tuned_models['xgb'].predict(X_test_final.values)
test['Predicted_Personality'] = le.inverse_transform(test_preds)
print(test[['id', 'Predicted_Personality']].head())


submission = test[['id', 'Predicted_Personality']]
submission = submission.rename(columns={'id': 'id', 'Predicted_Personality': 'Personality'})
submission.to_csv('submission.csv', index=False)

