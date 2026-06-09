import numpy as np 
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df=pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
df_test=pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
df_sample_submission=pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


df.info()


df.shape


df['Personality'].value_counts()


df.head(5)


print(df['Stage_fear'].unique())
print(df['Drained_after_socializing'].unique())


mapping = {'Yes': 1, 'No': 0}
df['Stage_fear'] = df['Stage_fear'].map(mapping)

mapping1 = {'Yes': 1, 'No': 0}
df['Drained_after_socializing'] = df['Drained_after_socializing'].map(mapping1)

mapping2 = {'Extrovert': 1, 'Introvert': 0}
df['Personality'] = df['Personality'].map(mapping2)


df.head(5)


df.info()


df.isnull().sum()


from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

num_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
            'Going_outside', 'Drained_after_socializing',
            'Friends_circle_size', 'Post_frequency']

imputer = KNNImputer(n_neighbors=5)
df[num_cols] = imputer.fit_transform(df[num_cols])

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])


df.isnull().sum()


import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import numpy as np

num_cols = [
    'Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
    'Going_outside', 'Drained_after_socializing',
    'Friends_circle_size', 'Post_frequency'
]

X = df[num_cols]
y = df['Personality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 5.0),  # for imbalance
        'use_label_encoder': False,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': 42
    }

    model = XGBClassifier(**params)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1_macro')
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("\nðŸŽ¯ Best Hyperparameters Found:")
print(study.best_params)

best_params = study.best_params
best_model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss', random_state=42)
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

print("\nâœ… Model Performance on Test Set:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average='macro'))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))



from sklearn.metrics import roc_auc_score
print("ROC-AUC:", roc_auc_score(y_test, best_model.predict_proba(X_test)[:,1]))

