import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

import optuna

import warnings
warnings.filterwarnings('ignore')


data = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv').set_index('id')
data.head()


len(data)


data.duplicated().sum()


data.isna().sum()


print('Column vs Number of unique values')
for col in data.columns:
    print(f'**** Column: {col} ****')
    print(f'Number of unique values: {data[col].nunique()}')
    print('-'*50)


for col in data.columns[:-1]:
    plt.figure(figsize=(15, 8))
    plt.subplot(1, 2, 1)
    sns.boxplot(data=data, y=col, x='rainfall')
    plt.subplot(1, 2, 2)
    sns.histplot(data=data, x=col, hue='rainfall', kde=True)
    plt.show()


plt.figure(figsize=(10, 8))
sns.countplot(data=data, y='rainfall')
plt.show()


# for col in data.columns[:-1]:
#     Q1 = data[col].quantile(0.25)
#     Q3 = data[col].quantile(0.75)
#     IQR = Q3 - Q1
    
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR

#     data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]


plt.figure(figsize=(12, 10))
sns.heatmap(data=data.corr(), annot=True, linewidths=0.2);


data['season'] = data['day'] % 365

def get_season(day):
    month = (day % 365) // 30 + 1
    if month in [12, 1, 2]:
        return 0 #'Winter'
    elif month in [3, 4, 5]:
        return 1 #'Spring'
    elif month in [6, 7, 8]:
        return 2 #'Summer'
    else:
        return 3 #'Autumn'

data['season'] = data['day'].apply(get_season)
# data = pd.get_dummies(data, columns=['season'], dtype=int)
data.head()


data['day_of_year'] = data['day'] % 365
data['sin_day'] = np.sin(2 * np.pi * data['day_of_year'] / 365)
data['cos_day'] = np.cos(2 * np.pi * data['day_of_year'] / 365)

data.head()


data['temp_range'] = data['maxtemp'] - data['mintemp']
data['temp_dew_diff'] = data['temparature'] - data['dewpoint']

data['humid_temp'] = data['humidity'] * data['temparature']
data['cloud_sun_ratio'] = data['cloud'] / (data['sunshine'] + 1)

data['wind_speed_category'] = pd.cut(data['windspeed'], bins=[0, 10, 20, 30, 50, 100], labels=[1, 2, 3, 4, 5])
data['wind_speed_category'] = data['wind_speed_category'].astype('int')

# data['rainfall_lag1'] = data['rainfall'].shift(1).fillna(0)
# data['rainfall_lag3'] = data['rainfall'].shift(3).fillna(0)

data.head()


drop_cols = ['day', 'day_of_year', 'maxtemp',]
data.drop(columns=drop_cols, inplace=True)


# data = pd.get_dummies(data, columns=['wind_speed_category'], drop_first=True, dtype=int)


data.head()


plt.figure(figsize=(12, 12))
sns.heatmap(data=data.corr(), annot=True, linewidths=0.2);


X_train, X_test, y_train, y_test = train_test_split(data.drop('rainfall', axis=1), data['rainfall'], 
                                                    stratify=data['rainfall'], random_state=42)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc_cols  = ['pressure', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed', 'humid_temp',	'cloud_sun_ratio']
X_train[sc_cols] = sc.fit_transform(X_train[sc_cols])
X_test[sc_cols] = sc.transform(X_test[sc_cols])


X_train.head()


models = {
    'Logistic_Reg' : LogisticRegression(),
    'SVC' : LinearSVC(),
    'DT' : DecisionTreeClassifier(),
    'RF' : RandomForestClassifier(),
    'XGB': XGBClassifier(),
    'Cat' : CatBoostClassifier(verbose=0),
    'LGB': LGBMClassifier(verbose=0),
}


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = {}

for name, model in models.items():
    scores = cross_val_score(model, data.drop('rainfall', axis=1), data['rainfall'], cv=cv, scoring="roc_auc")
    cv_scores[name] = scores
    print(f"{name}: Mean ROC-AUC = {np.mean(scores):.4f}, Std Dev = {np.std(scores):.4f}")

print("\nDetailed Scores:")
for name, scores in cv_scores.items():
    print(f"{name}: {scores}")


xgb = XGBClassifier()
xgb.fit(X_train, y_train)

importances = xgb.feature_importances_
columns = X_train.columns

sorted_idx = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.barh(y=np.array(columns)[sorted_idx], width=importances[sorted_idx], color="skyblue")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("XGBoost Feature Importance")
plt.show()


cat = CatBoostClassifier(verbose=0)
cat.fit(X_train, y_train)

importances = cat.feature_importances_
columns = X_train.columns

sorted_idx = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.barh(y=np.array(columns)[sorted_idx], width=importances[sorted_idx], color="skyblue")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("CatBoost Feature Importance")
plt.show()


lgb = LGBMClassifier()
lgb.fit(X_train, y_train)

importances = lgb.feature_importances_
columns = X_train.columns

sorted_idx = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.barh(y=np.array(columns)[sorted_idx], width=importances[sorted_idx], color="skyblue")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("LGB Feature Importance")
plt.show()


import shap

explainer = shap.TreeExplainer(xgb, feature_perturbation="tree_path_dependent")

shap_values = explainer.shap_values(X_train, check_additivity=False)  

shap.summary_plot(shap_values, X_train)


model = LogisticRegression()
model.fit(X_train, y_train)
roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])


model.fit(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]))


def process_test(data:pd.DataFrame):
    data['season'] = data['day'].apply(get_season)
    
    data['day_of_year'] = data['day'] % 365
    data['sin_day'] = np.sin(2 * np.pi * data['day_of_year'] / 365)
    data['cos_day'] = np.cos(2 * np.pi * data['day_of_year'] / 365)

    data['temp_range'] = data['maxtemp'] - data['mintemp']
    data['temp_dew_diff'] = data['temparature'] - data['dewpoint']

    data['humid_temp'] = data['humidity'] * data['temparature']
    data['cloud_sun_ratio'] = data['cloud'] / (data['sunshine'] + 1)
    # print(data.head())

    data['wind_speed_category'] = pd.cut(data['windspeed'], bins=[0, 10, 20, 30, 50, 100], labels=[1, 2, 3, 4, 5])
    data['wind_speed_category'] = data['wind_speed_category'].astype('int')
    
    # data['rainfall_lag1'] = data['rainfall'].shift(1).fillna(0)
    # data['rainfall_lag3'] = data['rainfall'].shift(3).fillna(0)

    drop_cols = ['day', 'day_of_year', 'maxtemp',]
    data.drop(columns=drop_cols, inplace=True)
    data[sc_cols] = sc.transform(data[sc_cols])
    
    data.fillna(0, inplace=True, axis=0)
    return data


test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv').set_index('id')


test = process_test(test)
test.head()


submission = model.predict_proba(test)[:, 1]
subfile = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
subfile['rainfall'] = submission

subfile.to_csv('LogReg_FE2.csv', index=False)

subfile.head()


# def objective(trial):
#     n_estimators = trial.suggest_int("n_estimators", 50, 500)
#     max_depth = trial.suggest_int("max_depth", 2, 20)
#     min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
#     min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
#     max_features = trial.suggest_float("max_features", 0.1, 1.0)
    
#     clf = RandomForestClassifier(
#         n_estimators=n_estimators,
#         max_depth=max_depth,
#         min_samples_split=min_samples_split,
#         min_samples_leaf=min_samples_leaf,
#         max_features=max_features,
#         random_state=42,
#         n_jobs=-1
#     )
    
#     score = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1).mean()
#     return score

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=500, show_progress_bar=True)

# print("Best Hyperparameters:", study.best_params)

# best_params = study.best_params
# best_clf = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
# best_clf.fit(X_train, y_train)

# y_pred = best_clf.predict_proba(X_test)[:, 1]
# accuracy = roc_auc_score(y_test, y_pred)
# print("Test Accuracy:", accuracy)


params = {'n_estimators': 499, 'max_depth': 20, 
          'min_samples_split': 5, 'min_samples_leaf': 6, 
          'max_features': 0.4970916271134286}

model1 = RandomForestClassifier(**params)

model1.fit(X_train, y_train)
roc_auc_score(y_test, model1.predict_proba(X_test)[:, 1])


# def objective(trial):
#     params = {
#         "n_estimators": trial.suggest_int("n_estimators", 50, 800),
#         "max_depth": trial.suggest_int("max_depth", 2, 20),
#         "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
#         "subsample": trial.suggest_float("subsample", 0.5, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
#         "gamma": trial.suggest_float("gamma", 0, 5),
#         "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
#         "lambda": trial.suggest_float("lambda", 1e-8, 10.0),
#         "alpha": trial.suggest_float("alpha", 1e-8, 10.0),
#         # "tree_method":"gpu_hist",
#         # "devices":'0'
#     }
    
#     clf = XGBClassifier(**params, use_label_encoder=False, eval_metric="logloss")
    
#     score = cross_val_score(clf, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1).mean()
#     return score

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=1000, show_progress_bar=True)

# print("Best Hyperparameters:", study.best_params)

# best_params = study.best_params
# best_clf = XGBClassifier(**best_params, use_label_encoder=False, eval_metric="logloss")
# best_clf.fit(X_train, y_train)

# y_pred = best_clf.predict_proba(X_test)[:, 1]
# accuracy = roc_auc_score(y_test, y_pred)
# print("Test Accuracy:", accuracy)


params =    {'n_estimators': 93, 'max_depth': 20, 'learning_rate': 0.027123029711386998, 
             'subsample': 0.981593750063191, 'colsample_bytree': 0.7751554847481027, 
             'gamma': 0.2530833529943241, 'min_child_weight': 3, 'lambda': 4.721066915419485, 
             'alpha': 2.7905132243613826}

            
model2 = XGBClassifier(**params)
model2.fit(X_train, y_train)

roc_auc_score(y_test, model2.predict_proba(X_test)[:, 1])


# def objective(trial):
#     params = {
#         "n_estimators": trial.suggest_int("n_estimators", 50, 500),
#         "max_depth": trial.suggest_int("max_depth", 2, 20),
#         "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
#         "num_leaves": trial.suggest_int("num_leaves", 2, 256),
#         "subsample": trial.suggest_float("subsample", 0.5, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
#         "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0),
#         "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0),
#         "min_child_samples": trial.suggest_int("min_child_samples", 1, 100),
#         "verbose":-1
#     }
    
#     clf = LGBMClassifier(**params)
    
#     score = cross_val_score(clf, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1, verbose=0).mean()
#     return score

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=1000, show_progress_bar=True)

# print("Best Hyperparameters:", study.best_params)

# best_params = study.best_params
# best_clf = LGBMClassifier(**best_params)
# best_clf.fit(X_train, y_train)

# y_pred = best_clf.predict_proba(X_test)[:, 1]
# accuracy = roc_auc_score(y_test, y_pred)
# print("Test Accuracy:", accuracy)


params = {'n_estimators': 331, 'max_depth': 12, 'learning_rate': 0.08487661176738037, 'num_leaves': 166,
           'subsample': 0.5960443979816741, 'colsample_bytree': 0.6743393404103202, 'reg_alpha': 7.169819621428156, 
           'reg_lambda': 4.9190517223292245, 'min_child_samples': 65, 'verbose':-100}

model3 = LGBMClassifier(**params)
model3.fit(X_train, y_train)

roc_auc_score(y_test, model3.predict_proba(X_test)[:, 1])


# def objective(trial):
#     params = {
#         "iterations": trial.suggest_int("iterations", 50, 500),
#         "depth": trial.suggest_int("depth", 2, 10),
#         "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
#         "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0),
#         "border_count": trial.suggest_int("border_count", 32, 255),
#         "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
#         "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0),
#         # "od_wait": trial.suggest_int("od_wait", 10, 50),
#         # "task_type": "GPU",
#     }
    
#     clf = CatBoostClassifier(**params, verbose=0)
    
#     score = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1).mean()
#     return score

# # Run the optimization
# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=1000, show_progress_bar=True)

# # Best hyperparameters
# print("Best Hyperparameters:", study.best_params)

# best_params = study.best_params
# best_clf = CatBoostClassifier(**best_params, verbose=0)
# best_clf.fit(X_train, y_train)

# y_pred = best_clf.predict_proba(X_test)[:, 1]
# accuracy = roc_auc_score(y_test, y_pred)
# print("Test Accuracy:", accuracy)


params = {'iterations': 360, 'depth': 2, 'learning_rate': 0.02210003782641798,
          'l2_leaf_reg': 7.7966885537155, 'border_count': 140, 'bagging_temperature': 0.6001430186097687, 
          'random_strength': 8.871379893492033, 'verbose':0}


model4 = CatBoostClassifier(**params)
model4.fit(X_train, y_train)

roc_auc_score(y_test, model4.predict_proba(X_test)[:, 1])


from sklearn.ensemble import StackingClassifier

final_model = LogisticRegression()

stack = StackingClassifier(
    estimators=[
        ('rf', model1),
        ('lgb', model3),
        ('cat', model4),
        ('xgb', model2),
    ],
    final_estimator=final_model,
    cv=5
)

stack.fit(X_train, y_train)
roc_auc_score(y_test, stack.predict_proba(X_test)[:, 1])


stack


stack.fit(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]))

submission = stack.predict_proba(test)[:, 1]
subfile['rainfall'] = submission

subfile.to_csv('Optunas_Stacked_FE.csv', index=False)
subfile.head()




