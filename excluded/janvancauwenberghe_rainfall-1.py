import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import time
import random


import warnings
warnings.filterwarnings("ignore")


df_train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
df_train = df_train.rename(columns={'temparature': 'temperature'})
df_train = df_train.drop(['id', 'day'], axis=1)
df_train.head()


plt.figure(figsize=(2.5,2));
sns.countplot(data=df_train, x='rainfall', hue='rainfall', width=0.4);
plt.grid(alpha=0.3);
plt.xticks(fontsize=8);
plt.yticks(fontsize=8);
plt.legend(fontsize=8);


sns.pairplot(df_train, hue='rainfall');


plt.figure(figsize=(5,3));
sns.boxplot(x='rainfall', y='sunshine', hue ='rainfall', data=df_train, width=0.6);
plt.grid(alpha=0.3);


plt.figure(figsize=(5,3));
sns.boxplot(x='rainfall', y='cloud', hue ='rainfall', data=df_train, width=0.6);
plt.grid(alpha=0.3);


plt.figure(figsize=(5,3));
sns.boxplot(x='rainfall', y='humidity', hue ='rainfall', data=df_train, width=0.6);
plt.grid(alpha=0.3);


g = sns.pairplot(df_train[['sunshine', 'cloud', 'humidity', 'rainfall']], hue='rainfall', plot_kws={'s': 20, 'alpha': 0.9});

for ax in g.axes.flat:
    if ax is not None:
        ax.grid(alpha=0.3)
        ax.tick_params(axis='both', labelsize=8)


plt.figure(figsize=(6.5,6.5));
sns.heatmap(df_train.corr(),annot=True, cmap='coolwarm', annot_kws={"size": 8}, cbar=False);
plt.xticks(fontsize=8);
plt.yticks(fontsize=8);


df_original = pd.read_csv('/kaggle/input/rainfall-datasets/rainfall.csv')
df_original.head()


df_original = df_original.rename(columns={'temparature': 'temperature'})
df_original = df_original.drop('day', axis=1)
df_original.describe().T


df_original.columns


df_original.columns = [['pressure', 'maxtemp', 'temperature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'rainfall', 'sunshine', 
                        'winddirection', 'windspeed']]
df_original.columns = df_original.columns.get_level_values(0)


df_original[df_original[['winddirection', 'windspeed']].isna().any(axis=1)]


df_original = df_original.drop(index=160)
df_original = df_original[['pressure', 'maxtemp', 'temperature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection', 
                        'windspeed', 'rainfall']]


df_original['rainfall'] = df_original['rainfall'].map({'yes': 1, 'no': 0})


df_original.head()


plt.figure(figsize=(2.5,2));
sns.countplot(data=df_original, x='rainfall', hue='rainfall', width=0.4);
plt.grid(alpha=0.3);
plt.xticks(fontsize=8);
plt.yticks(fontsize=8);
plt.legend(fontsize=8);


g = sns.pairplot(df_original[['sunshine', 'cloud', 'humidity', 'rainfall']], hue='rainfall', plot_kws={'s': 20, 'alpha': 0.9});

for ax in g.axes.flat:
    if ax is not None:
        ax.grid(alpha=0.3)
        ax.tick_params(axis='both', labelsize=8)


df_combined = pd.concat([df_original, df_train], ignore_index=True)
df_combined.describe().T


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


y = df_combined['rainfall']
X = df_combined.drop('rainfall', axis=1)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)


scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_val = scaler.transform(X_val)


import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression


def run_model(model, X_train, y_train, X_test, y_test):
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    
    print(f'Accuracy score: {round(accuracy_score(y_test,y_pred),3)}')
    print(f'Area under ROC curve: {round(roc_auc_score(y_test, y_prob),3)}')


def objective(trial):
    params = {'n_estimators': trial.suggest_int('n_estimators', 30, 300),
        'max_features': trial.suggest_int('max_features', 3, 10),
        'bootstrap': True, 'random_state': 42}
    
    model = RandomForestClassifier(**params)
    model.fit(scaled_X_train, y_train) 

    y_pred = model.predict(scaled_X_val)
    y_prob = model.predict_proba(scaled_X_val)[:,1] 

    return roc_auc_score(y_val, y_prob)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=500, show_progress_bar=True)

print("Best hyperparameters:", study.best_params)

best_model = RandomForestClassifier(**study.best_params, random_state=42)
best_model.fit(scaled_X_train, y_train)

y_final_pred = best_model.predict(scaled_X_val)
y_final_prob = best_model.predict_proba(scaled_X_val)[:,1]

final_accuracy = accuracy_score(y_val, y_final_pred)
print(f"Final Accuracy: {final_accuracy:.4f}")

final_auc = roc_auc_score(y_val, y_final_prob)
print(f"Final AUC: {final_auc:.4f}")


def objective(trial):
    params = {'n_estimators': trial.suggest_int('n_estimators', 20, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1),
        'algorithm': 'SAMME', 'random_state': 42}
    
    model = AdaBoostClassifier(**params)
    model.fit(scaled_X_train, y_train)
    
    y_pred = model.predict(scaled_X_val)
    y_prob = model.predict_proba(scaled_X_val)[:,1]
    
    return roc_auc_score(y_val, y_prob)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=500, show_progress_bar=True)

print("Best hyperparameters:", study.best_params)

best_model = AdaBoostClassifier(**study.best_params, random_state=42, algorithm='SAMME')
best_model.fit(scaled_X_train, y_train)

y_final_pred = best_model.predict(scaled_X_val)
y_final_prob = best_model.predict_proba(scaled_X_val)[:,1]

final_accuracy = accuracy_score(y_val, y_final_pred)
print(f"Final Accuracy: {final_accuracy:.4f}")

final_auc = roc_auc_score(y_val, y_final_prob)
print(f"Final AUC: {final_auc:.4f}")


def objective(trial):
    params = {'n_estimators': trial.suggest_int('n_estimators', 20, 300), 'learning_rate': trial.suggest_float('learning_rate', 0.01, 1), 
              'max_depth': trial.suggest_int('max_depth', 2, 10), 'random_state': 42}
    
    model = GradientBoostingClassifier(**params)
    model.fit(scaled_X_train, y_train)

    y_pred = model.predict(scaled_X_val)
    y_prob = model.predict_proba(scaled_X_val)[:,1]
    
    return roc_auc_score(y_val, y_prob)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=500, show_progress_bar=True)

print("Best hyperparameters:", study.best_params)

best_model = GradientBoostingClassifier(**study.best_params, random_state=42)
best_model.fit(scaled_X_train, y_train)

y_final_pred = best_model.predict(scaled_X_val)
y_final_prob = best_model.predict_proba(scaled_X_val)[:,1]

final_accuracy = accuracy_score(y_val, y_final_pred)
print(f"Final Accuracy: {final_accuracy:.4f}")

final_auc = roc_auc_score(y_val, y_final_prob)
print(f"Final AUC: {final_auc:.4f}")


def objective(trial):
    params = {'hidden_layer_sizes': (trial.suggest_int('layer1', 5, 100), trial.suggest_int('layer2', 5, 100)),
        'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic']),
        'solver': trial.suggest_categorical('solver', ['adam', 'sgd']),
        'alpha': trial.suggest_float('alpha', 1e-5, 1e-1),
        'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'adaptive']), 'random_state': 42}
    
    model = MLPClassifier(**params, max_iter=1000)
    model.fit(scaled_X_train, y_train)

    y_pred = model.predict(scaled_X_val)
    y_prob = model.predict_proba(scaled_X_val)[:,1]
    
    return roc_auc_score(y_val, y_prob)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=500, show_progress_bar=True)

print("Best hyperparameters:", study.best_params)


def objective(trial):
    
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet', None])
    
    if penalty == None:
        solver = 'saga'
    elif penalty == 'l1':
        solver = 'saga'
    elif penalty == 'elasticnet':
        solver = 'saga'
    else:
        solver = trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'saga'])

    l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0) if penalty == 'elasticnet' else None
    C = trial.suggest_float('C', 1e-4, 1000)

    model = LogisticRegression(penalty=penalty, solver=solver, C=C, l1_ratio=l1_ratio, random_state=42, max_iter=1000)

    model.fit(scaled_X_train, y_train)

    y_pred = model.predict(scaled_X_val)
    y_prob = model.predict_proba(scaled_X_val)[:,1]

    return roc_auc_score(y_val, y_prob)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=500, show_progress_bar=True)

print("Best hyperparameters:", study.best_params)


from sklearn.ensemble import VotingClassifier


model1 = LogisticRegression(C=0.37244, penalty='l1', solver='saga', max_iter=1000, random_state=42)
model2 = RandomForestClassifier(random_state=42, n_estimators=115, bootstrap=True, max_features=3)
model3 = AdaBoostClassifier(algorithm='SAMME', n_estimators=187, learning_rate=0.99703, random_state=42)
model4 = GradientBoostingClassifier(n_estimators=92, learning_rate=0.49857, max_depth=2, random_state=42)
model5 = MLPClassifier(max_iter=754, random_state=42, hidden_layer_sizes=(36, 23), learning_rate='adaptive', alpha=0.08908, solver='adam', 
                       activation='tanh')

voting_clf = VotingClassifier(estimators=[('lr', model1), ('rf', model2), ('ada', model3), ('gb', model4), ('mlp', model5)], voting='soft')

run_model(voting_clf, scaled_X_train, y_train, scaled_X_val, y_val)


df_test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
df_test.head()


df_test.columns


df_test = df_test.rename(columns={'temparature': 'temperature'})
df_test = df_test.drop(['id', 'day'], axis=1)
df_test.head()


df_test.describe().T


df_test[df_test['winddirection'].isna()]


df_test.loc[517, 'winddirection'] = 200


df_test.loc[517]


scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(df_test)


prob = voting_clf.predict_proba(scaled_X_test)


df_sample = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
df_sample.head()


df_sample['rainfall'] = prob[:,1]


df_sample.head()


df_sample.describe().T


df_sample.to_csv('submission.csv', index=False)




