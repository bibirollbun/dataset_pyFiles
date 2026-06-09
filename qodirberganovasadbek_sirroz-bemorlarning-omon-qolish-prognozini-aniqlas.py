import numpy as np
import pandas as pd 

import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('/kaggle/input/multiclassificationtask/train.csv')
test = pd.read_csv('/kaggle/input/multiclassificationtask/test.csv')
sample_submission = pd.read_csv('/kaggle/input/multiclassificationtask/sample_submission.csv')

print("Train dataset:")
print(train.head())

print("\nTest dataset:")
print(test.head())

print("\nSample Submission:")
print(sample_submission.head())


print(test.isnull().sum())
print(train.isnull().sum())


from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

numeric_cols_mean = ['Prothrombin', 'Platelets']
numeric_cols_knn = ['Cholesterol', 'Copper', 'Alk_Phos', 'SGOT']
categorical_cols = ['Ascites', 'Hepatomegaly', 'Spiders', 'Sex', 'Edema']
all_features = numeric_cols_mean + numeric_cols_knn + categorical_cols + ['N_Days', 'Age', 'Bilirubin', 'Albumin', 'Stage']

numeric_mean_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

numeric_knn_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num_mean', numeric_mean_transformer, numeric_cols_mean),
        ('num_knn', numeric_knn_transformer, numeric_cols_knn),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

train_preprocessed = preprocessor.fit_transform(train[all_features])
test_preprocessed = preprocessor.transform(test[all_features])

train_preprocessed = pd.DataFrame(train_preprocessed, columns=numeric_cols_mean + numeric_cols_knn + categorical_cols)
test_preprocessed = pd.DataFrame(test_preprocessed, columns=numeric_cols_mean + numeric_cols_knn + categorical_cols)

for col in ['N_Days', 'Age', 'Bilirubin', 'Albumin', 'Stage']:
    train_preprocessed[col] = train[col]
    test_preprocessed[col] = test[col]

for col in categorical_cols:
    le = LabelEncoder()
    train_preprocessed[col] = le.fit_transform(train_preprocessed[col])
    test_preprocessed[col] = le.transform(test_preprocessed[col])

train_preprocessed['Status'] = train['Status']


print(train_preprocessed['Status'].value_counts())
train_preprocessed = train_preprocessed[train_preprocessed['Status'] != 'Y']
status_mapping = {'D': 0, 'C': 1, 'CL': 2}
train_preprocessed['Status'] = train_preprocessed['Status'].map(status_mapping)


for col in ['Bilirubin', 'Albumin', 'Age', 'Stage', 'Alk_Phos', 'Platelets', 'Prothrombin', 'Copper', 'Cholesterol', 'SGOT']:
    train_preprocessed[col] = pd.to_numeric(train_preprocessed[col], errors='coerce').astype(float)
    test_preprocessed[col] = pd.to_numeric(test_preprocessed[col], errors='coerce').astype(float)

train_preprocessed['Bilirubin_Albumin_Ratio'] = train_preprocessed['Bilirubin'] / train_preprocessed['Albumin']
train_preprocessed['Age_Stage'] = train_preprocessed['Age'] * train_preprocessed['Stage']
train_preprocessed['Liver_Function_Score'] = train_preprocessed['Bilirubin'] + train_preprocessed['Alk_Phos'] - train_preprocessed['Albumin']
train_preprocessed['Blood_Clotting_Index'] = train_preprocessed['Platelets'] / train_preprocessed['Prothrombin']
train_preprocessed['Log_Bilirubin'] = np.log1p(train_preprocessed['Bilirubin'].clip(lower=0))
train_preprocessed['Log_Copper'] = np.log1p(train_preprocessed['Copper'].clip(lower=0))

test_preprocessed['Bilirubin_Albumin_Ratio'] = test_preprocessed['Bilirubin'] / test_preprocessed['Albumin']
test_preprocessed['Age_Stage'] = test_preprocessed['Age'] * test_preprocessed['Stage']
test_preprocessed['Liver_Function_Score'] = test_preprocessed['Bilirubin'] + test_preprocessed['Alk_Phos'] - test_preprocessed['Albumin']
test_preprocessed['Blood_Clotting_Index'] = test_preprocessed['Platelets'] / test_preprocessed['Prothrombin']
test_preprocessed['Log_Bilirubin'] = np.log1p(test_preprocessed['Bilirubin'].clip(lower=0))
test_preprocessed['Log_Copper'] = np.log1p(test_preprocessed['Copper'].clip(lower=0))


import matplotlib.pyplot as plt
import seaborn as sns

raqamli_ustunlar = ['N_Days', 'Age', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper',
                    'Alk_Phos', 'SGOT', 'Platelets', 'Prothrombin', 'Stage',
                    'Bilirubin_Albumin_Ratio', 'Age_Stage', 'Liver_Function_Score',
                    'Blood_Clotting_Index', 'Log_Bilirubin', 'Log_Copper', 'Status']
kategorik_ustunlar = ['Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema']

plt.figure(figsize=(15, 10))
for i, col in enumerate(raqamli_ustunlar, 1):
    plt.subplot(5, 4, i)
    sns.histplot(data=train_preprocessed, x=col, kde=True)
    plt.title(col)
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 8))
for i, col in enumerate(kategorik_ustunlar, 1):
    plt.subplot(2, 3, i)
    sns.countplot(data=train_preprocessed, x=col)
    plt.title(col)
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("Stage bilan yangi korrelyatsiya:")
print(train_preprocessed.corr()['Status'].sort_values(ascending=False))


import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import log_loss, precision_score, recall_score
import optuna
import numpy as np

features = ['N_Days', 'Age', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper',
            'Alk_Phos', 'SGOT', 'Platelets', 'Prothrombin', 'Stage',
            'Bilirubin_Albumin_Ratio', 'Age_Stage', 'Liver_Function_Score',
            'Blood_Clotting_Index', 'Log_Bilirubin', 'Log_Copper',
            'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema']
X = train_preprocessed[features]
y = train_preprocessed['Status']


def objective(trial):
    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),  
        'max_depth': trial.suggest_int('max_depth', 3, 6),  
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 3.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 3.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 15.0),
        'eval_metric': 'mlogloss',
        'random_state': 42,
        'tree_method': 'gpu_hist',  
        'n_jobs': -1  
    }
    
    xgb_model = xgb.XGBClassifier(**params)
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    losses = []
    
    for train_idx, val_idx in kfold.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        xgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=100,  
            verbose=False
        )
        val_probs = xgb_model.predict_proba(X_val)
        losses.append(log_loss(y_val, val_probs))
    
    return np.mean(losses)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

best_params = study.best_params
best_params.update({
    'objective': 'multi:softprob',
    'num_class': 3,
    'eval_metric': 'mlogloss',
    'random_state': 42,
    'tree_method': 'gpu_hist',  
    'n_jobs': -1  
})

xgb_model = xgb.XGBClassifier(**best_params)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
xgb_losses = []
precisions = []
recalls = []

for train_idx, val_idx in kfold.split(X, y):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=100,  
        verbose=False
    )
    val_probs = xgb_model.predict_proba(X_val)
    val_preds = np.argmax(val_probs, axis=1)
    xgb_losses.append(log_loss(y_val, val_probs))
    precisions.append(precision_score(y_val, val_preds, average='weighted', zero_division=0))
    recalls.append(recall_score(y_val, val_preds, average='weighted', zero_division=0))

print(f"Optimizatsiya qilingan XGBoost Kross-validatsiya Log Loss: {np.mean(xgb_losses)}")
print(f"Kross-validatsiya Precision (weighted): {np.mean(precisions)}")
print(f"Kross-validatsiya Recall (weighted): {np.mean(recalls)}")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_probs = xgb_model.predict_proba(X_val)
xgb_preds = np.argmax(xgb_probs, axis=1)
xgb_loss = log_loss(y_val, xgb_probs)
precision = precision_score(y_val, xgb_preds, average='weighted', zero_division=0)
recall = recall_score(y_val, xgb_preds, average='weighted', zero_division=0)
print(f"Optimizatsiya qilingan XGBoost Log Loss: {xgb_loss}")
print(f"Yakuniy Precision (weighted): {precision}")
print(f"Yakuniy Recall (weighted): {recall}")

xgb_model.fit(X, y)


test_probs = xgb_model.predict_proba(test_preprocessed[features])
test_probs = np.clip(test_probs, 1e-15, 1 - 1e-15)

submission = pd.DataFrame({
    'id': test['id'],
    'Status_C': test_probs[:, 1],
    'Status_CL': test_probs[:, 2],
    'Status_D': test_probs[:, 0]
})
submission.to_csv('submission_xgboost_optimized.csv', index=False)
print("Submission fayli yaratildi: submission_xgboost_optimized.csv")




