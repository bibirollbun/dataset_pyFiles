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
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from lightgbm import LGBMRegressor

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from category_encoders import TargetEncoder

import optuna
import optuna.logging

from scipy.stats import chi2_contingency

import warnings



!pip install optuna



df = pd.read_csv("/kaggle/input/heart-disease-prediction-dataquest/heart_train.csv")
df.head()


target_col = 'HeartDisease'


nominal_cols = [
    col for col in df.columns
    if (df[col].dtype == 'object' or df[col].nunique() <= 6) and col != target_col
]

blue_shade = '#6497b1'

import matplotlib.pyplot as plt

for col in nominal_cols + [target_col]:
    plt.figure(figsize=(8, 6))
    df[col].value_counts().plot(
        kind='bar',
        color=blue_shade,
        edgecolor=None
    )
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.title(f'{col} Distribution')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()



def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))

def cramers_v_matrix(df, cols):
    matrix = pd.DataFrame(index=cols, columns=cols)
    for col1 in cols:
        for col2 in cols:
            if col1 == col2:
                matrix.loc[col1, col2] = 1.0
            else:
                confusion = pd.crosstab(df[col1], df[col2])
                matrix.loc[col1, col2] = cramers_v(confusion)
    return matrix.astype(float)

cramers_matrix = cramers_v_matrix(df, nominal_cols + [target_col])

plt.figure(figsize=(10, 8))
sns.heatmap(cramers_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Cramér's V Correlation Heatmap")
plt.tight_layout()
plt.show()


warnings.filterwarnings('ignore', category=FutureWarning)

excluded = nominal_cols + [target_col]

numeric_cols = [col for col in df.columns if col not in excluded]

for feature in numeric_cols:
    plt.figure(figsize=(8, 5))
    
    sns.histplot(df[feature], kde=True, bins=30, color=blue_shade)
    plt.title(f"Histogram of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()



df['Cholesterol_Zero'] = df['Cholesterol'] == 0

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = [8, 4]

for var in numeric_cols:
    if var != 'Cholesterol':  
        plt.figure()
        sns.kdeplot(data=df, x=var, hue='Cholesterol_Zero', common_norm=False, fill=True, palette='Set2')
        plt.title(f'Distribution of {var} by Cholesterol = 0 vs Non-zero')
        plt.xlabel(var)
        plt.ylabel('Density')
        plt.legend(title='Cholesterol = 0')

for var in nominal_cols:
    plt.figure()
    prop_df = (
        df.groupby(['Cholesterol_Zero', var]).size() /
        df.groupby(['Cholesterol_Zero']).size()
    ).reset_index(name='Proportion')

    sns.barplot(data=prop_df, x=var, y='Proportion', hue='Cholesterol_Zero', palette='Set1')
    plt.title(f'{var} Distribution by Cholesterol = 0 vs Non-zero')
    plt.xticks(rotation=0)
    plt.legend(title='Cholesterol = 0')

plt.show()


for feature in numeric_cols:
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x=feature, hue="HeartDisease", kde=True, bins=30, palette="Set1")
    plt.title(f"Histogram of {feature} by HeartDisease")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()



df = pd.read_csv("/kaggle/input/heart-disease-prediction-dataquest/heart_train.csv")
df_test = pd.read_csv("/kaggle/input/heart-disease-prediction-dataquest/heart_test.csv")


df['Cholesterol'] = df['Cholesterol'].replace(0, np.nan)  
df_test['Cholesterol'] = df['Cholesterol'].replace(0, np.nan)  

df['Oldpeak'] = df['Oldpeak'].apply(lambda x: np.nan if x <= 0 else x)
df_test['Cholesterol'] = df['Cholesterol'].replace(0, np.nan)  

df_test['Cholesterol'] = df_test['Cholesterol'].replace(0, np.nan)
df_test['Oldpeak'] = df_test['Oldpeak'].apply(lambda x: np.nan if x <= 0 else x)

imputer = IterativeImputer(random_state=42)
df['Cholesterol'] = imputer.fit_transform(df[['Cholesterol']])
df_test['Cholesterol'] = imputer.transform(df_test[['Cholesterol']])

df['Oldpeak'] = imputer.fit_transform(df[['Oldpeak']])
df_test['Oldpeak'] = imputer.transform(df_test[['Oldpeak']])


def preprocess(df):
    df = df.copy()
    df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})
    df['ExerciseAngina'] = df['ExerciseAngina'].map({'Y': 1, 'N': 0})
    df['ST_Slope'] = df['ST_Slope'].map({'Up': 0, 'Flat': 1, 'Down': 2})
 
    df = pd.get_dummies(df, columns=['ChestPainType', 'RestingECG'], drop_first=True)
    return df

df = preprocess(df)
df_test = preprocess(df_test)


X = df.drop(columns='HeartDisease')
y = df['HeartDisease']


df_test = df_test.reindex(columns=X.columns, fill_value=0)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(df_test)


X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)


optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective_xgb(trial):
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1),
        'gamma': trial.suggest_loguniform('gamma', 1e-5, 1e1)
    }

    xgb_model = xgb.XGBClassifier(**param)

    score = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='accuracy').mean()
    
    return score

study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=100)

print("Best parameters for XGBoost: ", study_xgb.best_params)


best_params = study_xgb.best_params
xgb_model = xgb.XGBClassifier(**best_params)

xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_val)
y_prob_xgb = xgb_model.predict_proba(X_val)[:, 1]  

accuracy = accuracy_score(y_val, y_pred_xgb)
precision = precision_score(y_val, y_pred_xgb)
recall = recall_score(y_val, y_pred_xgb)
f1 = f1_score(y_val, y_pred_xgb)
roc_auc = roc_auc_score(y_val, y_prob_xgb)

print(f"XGBoost Model Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

cm = confusion_matrix(y_val, y_pred_xgb)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective_knn(trial):
    param = {
        'n_neighbors': trial.suggest_int('n_neighbors', 3, 20),
        'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
        'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
        'leaf_size': trial.suggest_int('leaf_size', 20, 100),
        'p': trial.suggest_int('p', 1, 2)  # L1 distance (p=1) or L2 distance (p=2)
    }

    knn_model = KNeighborsClassifier(**param)

    score = cross_val_score(knn_model, X_train, y_train, cv=5, scoring='accuracy').mean()
    
    return score

study_knn = optuna.create_study(direction='maximize')
study_knn.optimize(objective_knn, n_trials=100)

print("Best parameters for KNN: ", study_knn.best_params)


best_params_knn = study_knn.best_params
knn_model = KNeighborsClassifier(**best_params_knn)

knn_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_val)
y_prob_knn = knn_model.predict_proba(X_val)[:, 1]

accuracy = accuracy_score(y_val, y_pred_knn)
precision = precision_score(y_val, y_pred_knn)
recall = recall_score(y_val, y_pred_knn)
f1 = f1_score(y_val, y_pred_knn)
roc_auc = roc_auc_score(y_val, y_prob_knn)

print(f"KNN Model Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

cm_knn = confusion_matrix(y_val, y_pred_knn)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (KNN)')
plt.show()


optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective_rf(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
    }

    rf_model = RandomForestClassifier(**param, random_state=42)

    score = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy').mean()
    
    return score

study_rf = optuna.create_study(direction='maximize')
study_rf.optimize(objective_rf, n_trials=100)

print("Best parameters for Random Forest: ", study_rf.best_params)


best_params_rf = study_rf.best_params
rf_model = RandomForestClassifier(**best_params_rf, random_state=42)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_val)
y_prob_rf = rf_model.predict_proba(X_val)[:, 1] 

accuracy = accuracy_score(y_val, y_pred_rf)
precision = precision_score(y_val, y_pred_rf)
recall = recall_score(y_val, y_pred_rf)
f1 = f1_score(y_val, y_pred_rf)
roc_auc = roc_auc_score(y_val, y_prob_rf)

print(f"Random Forest Model Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

cm_rf = confusion_matrix(y_val, y_pred_rf)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Random Forest)')
plt.show()


y_pred_rf_test = rf_model.predict(X_test_scaled)

submission = pd.DataFrame({
    'id': range(len(y_pred_rf_test)), 
    'target': y_pred_rf_test 
})

submission.to_csv('/kaggle/working/random_forest_submission.csv', index=False)

print("Kaggle submission file created: random_forest_submission.csv")



optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective_logreg(trial):
    param = {
        'C': trial.suggest_loguniform('C', 1e-5, 1e5),
        'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
        'max_iter': trial.suggest_int('max_iter', 100, 1000),
        'penalty': trial.suggest_categorical('penalty', ['l1', 'l2'])
    }

    logreg_model = LogisticRegression(**param, random_state=42)

    score = cross_val_score(logreg_model, X_train, y_train, cv=5, scoring='accuracy').mean()
    
    return score

study_logreg = optuna.create_study(direction='maximize')
study_logreg.optimize(objective_logreg, n_trials=100)

print("Best parameters for Logistic Regression: ", study_logreg.best_params)


best_params_logreg = study_logreg.best_params
logreg_model = LogisticRegression(**best_params_logreg, random_state=42)

logreg_model.fit(X_train, y_train)

y_pred_logreg = logreg_model.predict(X_val)
y_prob_logreg = logreg_model.predict_proba(X_val)[:, 1] 

accuracy = accuracy_score(y_val, y_pred_logreg)
precision = precision_score(y_val, y_pred_logreg)
recall = recall_score(y_val, y_pred_logreg)
f1 = f1_score(y_val, y_pred_logreg)
roc_auc = roc_auc_score(y_val, y_prob_logreg)

print(f"Logistic Regression Model Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

cm_logreg = confusion_matrix(y_val, y_pred_logreg)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_logreg, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Logistic Regression)')
plt.show()


xgb_model = xgb.XGBClassifier(**study_xgb.best_params, random_state=42)
logreg_model = LogisticRegression(**study_logreg.best_params, random_state=42)
rf_model = RandomForestClassifier(**study_rf.best_params, random_state=42)
knn_model = KNeighborsClassifier(**study_knn.best_params)


def report(y_true, y_pred, y_prob, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    print(f"{model_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")


base_learners = [
    ('rf', rf_model),
    ('xgb', xgb_model),
    ('logreg', logreg_model),
    ('knn', knn_model)
]

meta_model = LogisticRegression()

stacking_model = StackingClassifier(estimators=base_learners, final_estimator=meta_model, cv=5)

stacking_model.fit(X_train, y_train)

y_pred_stack = stacking_model.predict(X_val)
y_prob_stack = stacking_model.predict_proba(X_val)[:, 1]

accuracy_stack = accuracy_score(y_val, y_pred_stack)
precision_stack = precision_score(y_val, y_pred_stack)
recall_stack = recall_score(y_val, y_pred_stack)
f1_stack = f1_score(y_val, y_pred_stack)
roc_auc_stack = roc_auc_score(y_val, y_prob_stack)

print(f"Stacking Model Accuracy: {accuracy_stack:.4f}")
print(f"Precision: {precision_stack:.4f}")
print(f"Recall: {recall_stack:.4f}")
print(f"F1-Score: {f1_stack:.4f}")
print(f"ROC-AUC: {roc_auc_stack:.4f}")

cm_stack = confusion_matrix(y_val, y_pred_stack)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_stack, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Stacking)')
plt.show()


greedy_ensemble = VotingClassifier(estimators=[
    ('rf', rf_model),
    ('xgb', xgb_model),
    ('logreg', logreg_model),
    ('knn', knn_model)
], voting='soft')

greedy_ensemble.fit(X_train, y_train)

y_pred_greedy = greedy_ensemble.predict(X_val)
y_prob_greedy = greedy_ensemble.predict_proba(X_val)[:, 1]

accuracy_greedy = accuracy_score(y_val, y_pred_greedy)
precision_greedy = precision_score(y_val, y_pred_greedy)
recall_greedy = recall_score(y_val, y_pred_greedy)
f1_greedy = f1_score(y_val, y_pred_greedy)
roc_auc_greedy = roc_auc_score(y_val, y_prob_greedy)

print(f"Greedy Ensemble Accuracy: {accuracy_greedy:.4f}")
print(f"Precision: {precision_greedy:.4f}")
print(f"Recall: {recall_greedy:.4f}")
print(f"F1-Score: {f1_greedy:.4f}")
print(f"ROC-AUC: {roc_auc_greedy:.4f}")

cm_greedy = confusion_matrix(y_val, y_pred_greedy)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_greedy, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Greedy Ensemble)')
plt.show()


soft_voting_ensemble = VotingClassifier(estimators=[
    ('rf', rf_model),
    ('xgb', xgb_model),
    ('logreg', logreg_model),
    ('knn', knn_model)
], voting='soft')

soft_voting_ensemble.fit(X_train, y_train)

y_pred_soft = soft_voting_ensemble.predict(X_val)
y_prob_soft = soft_voting_ensemble.predict_proba(X_val)[:, 1]

accuracy_soft = accuracy_score(y_val, y_pred_soft)
precision_soft = precision_score(y_val, y_pred_soft)
recall_soft = recall_score(y_val, y_pred_soft)
f1_soft = f1_score(y_val, y_pred_soft)
roc_auc_soft = roc_auc_score(y_val, y_prob_soft)

print(f"Soft Voting Ensemble Accuracy: {accuracy_soft:.4f}")
print(f"Precision: {precision_soft:.4f}")
print(f"Recall: {recall_soft:.4f}")
print(f"F1-Score: {f1_soft:.4f}")
print(f"ROC-AUC: {roc_auc_soft:.4f}")

cm_soft = confusion_matrix(y_val, y_pred_soft)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_soft, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Soft Voting)')
plt.show()


stacking_model.fit(X_train, y_train)

y_pred_stack_test = stacking_model.predict(X_test_scaled) 

submission = pd.DataFrame({
    'id': range(len(y_pred_stack_test)),  
    'target': y_pred_stack_test
})

submission.to_csv('/kaggle/working/stacking_submission.csv', index=False)

print("Kaggle submission file created: stacking_submission.csv")

