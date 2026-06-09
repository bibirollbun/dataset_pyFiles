pip install optuna-integration[xgboost]


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, top_k_accuracy_score

import lightgbm as lgb
import xgboost as xgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from scipy.stats import chi2_contingency

import optuna
from optuna.integration import XGBoostPruningCallback

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)


train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


train.head()


test.head()


print(train.isna().any().sum())
print(test.isna().any().sum())


correlation_matrix = train.select_dtypes(include=['int64', 'float64']).corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
            center=0, fmt='.2f', square=True, linewidths=0.5)

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.title('Correlation Matrix of Numeric Features', pad=20)
plt.tight_layout()
plt.show()


def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    min_dim = min(confusion_matrix.shape) - 1
    return np.sqrt(chi2 / (n * min_dim))

categorical_cols = train.select_dtypes(include=['object', 'category']).columns

n_cat = len(categorical_cols)
correlation_matrix = np.zeros((n_cat, n_cat))

for i in range(n_cat):
    for j in range(n_cat):
        if i == j:
            correlation_matrix[i, j] = 1.0
        else:
            confusion_matrix = pd.crosstab(train[categorical_cols[i]], train[categorical_cols[j]])
            correlation_matrix[i, j] = cramers_v(confusion_matrix)

correlation_df = pd.DataFrame(correlation_matrix, 
                            index=categorical_cols, 
                            columns=categorical_cols)

plt.figure(figsize=(12, 8))

sns.heatmap(correlation_df, annot=True, cmap='YlOrRd', 
            vmin=0, vmax=1, fmt='.2f', square=True,
            linewidths=0.5)

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.title("Cramer's V Correlation Matrix of Categorical Features", pad=20)
plt.tight_layout()
plt.show()


train['N_to_P'] = train['Nitrogen'] / (train['Phosphorous'] + 1)
train['N_to_K'] = train['Nitrogen'] / (train['Potassium'] + 1)
train['P_to_K'] = train['Phosphorous'] / (train['Potassium'] + 1)
train['NPK_total'] = train['Nitrogen'] + train['Phosphorous'] + train['Potassium']
train['NPK_std'] = train[['Nitrogen', 'Phosphorous', 'Potassium']].std(axis=1)
train['Soil_Crop'] = train['Soil Type'].astype(str) + '_' + train['Crop Type'].astype(str)
train.head()


test['N_to_P'] = test['Nitrogen'] / (test['Phosphorous'] + 1)
test['N_to_K'] = test['Nitrogen'] / (test['Potassium'] + 1)
test['P_to_K'] = test['Phosphorous'] / (test['Potassium'] + 1)
test['NPK_total'] = test['Nitrogen'] + test['Phosphorous'] + test['Potassium']
test['NPK_std'] = test[['Nitrogen', 'Phosphorous', 'Potassium']].std(axis=1)
test['Soil_Crop'] = test['Soil Type'].astype(str) + '_' + test['Crop Type'].astype(str)
test.head()


train = pd.get_dummies(train, columns=['Soil Type', 'Crop Type','Soil_Crop'], drop_first=True)
test = pd.get_dummies(test, columns=['Soil Type', 'Crop Type','Soil_Crop'], drop_first=True)

soil_type_cols = [col for col in train.columns if col.startswith('Soil Type_')]
train[soil_type_cols] = train[soil_type_cols].astype(int)
test[soil_type_cols] = test[soil_type_cols].astype(int)

crop_type_cols = [col for col in train.columns if col.startswith('Crop Type_')]
train[crop_type_cols] = train[crop_type_cols].astype(int)
test[crop_type_cols] = test[crop_type_cols].astype(int)

soil_crop_cols = [col for col in train.columns if col.startswith('Soil_Crop_')]
train[soil_crop_cols] = train[soil_crop_cols].astype(int)
test[soil_crop_cols] = test[soil_crop_cols].astype(int)


train


test


le = LabelEncoder()

train['Fertilizer Name'] = le.fit_transform(train['Fertilizer Name'])
train.head()


X = train.drop(['id','Fertilizer Name'],axis=1)
y = train['Fertilizer Name']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


def mapk(y_true, y_pred, k=3):
    total_score = 0.0
    for true_label, predicted in zip(y_true, y_pred):
        predicted = list(predicted)
        score = 0.0
        correct = 0
        for i in range(min(k, len(predicted))):
            if predicted[i] == true_label:
                correct += 1
                score += correct / (i + 1)
                break
        total_score += score
    return total_score / len(y_true)


def mapk_scorer(estimator, X, y, k=3):
    y_pred_proba = estimator.predict_proba(X)
    top_k = np.argsort(y_pred_proba, axis=1)[:, ::-1][:, :k]
    return mapk(y, top_k, k)


lgb_clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.1, max_depth=5, random_state=42)

xgb_clf = xgb.XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=5, use_label_encoder=False, eval_metric='mlogloss', random_state=42)

cat_clf = CatBoostClassifier(iterations=300, learning_rate=0.1, depth=5, random_state=42, verbose=False)

for name, model in [('LGBM', lgb_clf), ('XGBoost', xgb_clf), ('CatBoost', cat_clf)]:
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring=mapk_scorer)
    print(f"{name} MAP@3 Mean CV Score: {scores.mean():.4f}")


def objective(trial):

    params = {'iterations': trial.suggest_int('iterations', 50, 100),
    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
    'depth': trial.suggest_int('depth', 3, 7),
    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
    'random_state': 42, 'verbose': False}

    model = CatBoostClassifier(**params)

    score = cross_val_score(model, X_train, y_train, cv=3, scoring=mapk_scorer).mean()
    
    return -score

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)

print("Best MAP@3 Score:", -study.best_value)
print("Best CatBoost Parameters:", study.best_params)


optuna.visualization.plot_optimization_history(study).show()


optuna.visualization.plot_param_importances(study).show()


best_params = study.best_params
best_params['random_state'] = 42
best_params['verbose'] = False

final_model = CatBoostClassifier(**best_params)
final_model.fit(X_train, y_train)

y_pred_proba = final_model.predict_proba(X_test)
top_3 = np.argsort(y_pred_proba, axis=1)[:, ::-1][:, :3]
final_score = mapk(y_test, top_3, k=3)

print(f"Final MAP@3 on Holdout Set: {final_score:.4f}")


final_model.fit(X, y)
test_pred = final_model.predict_proba(test)

top_pred = np.argsort(test_pred, axis=1)[:, ::-1][:, :3]
top_labels = le.inverse_transform(top_pred.ravel()).reshape(-1, 3)

submission = pd.DataFrame({'id': test.id,
    'Top_1': top_labels[:, 0]})

submission


submission.to_csv('submission.csv', index=False)
print("Filed saved as 'submission.csv'")


!pip install shap -q
import shap


explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_train)


shap.summary_plot(shap_values, X_train)

