import pandas as pd
import numpy as np
import optuna  # The Star of the Show
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings('ignore')

# Load Data
train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')

print("Ready to Optimize!")


# 1. Label Encoding for Categorical Columns
object_cols = train.select_dtypes(include=['object']).columns
le = LabelEncoder()

for col in object_cols:
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

# 2. Advanced Feature Engineering Function
def feature_engineering(df):
    # BMI Category
    df['BMI_Cat'] = pd.cut(df['bmi'], bins=[0, 18.5, 24.9, 29.9, 100], labels=[0, 1, 2, 3]).astype(int)
    
    # Blood Pressure Risk (Systolic/Diastolic)
    df['Hypertension_Risk'] = ((df['systolic_bp'] >= 130) | (df['diastolic_bp'] >= 80)).astype(int)
    
    # Cholesterol Ratio
    df['Cholesterol_Ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1e-5)
    
    # Age Groups
    df['Age_Group'] = pd.cut(df['age'], bins=[0, 30, 50, 100], labels=[0, 1, 2]).astype(int)
    
    # Central Obesity (Waist/Hip)
    df['Central_Obesity'] = (df['waist_to_hip_ratio'] > 0.9).astype(int)
    
    return df

train = feature_engineering(train)
test = feature_engineering(test)

print("Features Created Successfully!")


X = train.drop(['id', 'diagnosed_diabetes'], axis=1)
y = train['diagnosed_diabetes']
X_test = test.drop(['id'], axis=1)

def objective(trial):
    # Define the search space (Computer in range mein dhundega)
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'n_jobs': -1,
        'eval_metric': 'auc',
        'random_state': 42
    }
    
    model = XGBClassifier(**params)
    
    # Stratified K-Fold Cross Validation (Reliable Scoring)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    
    return scores.mean()

print("Optuna setup ready...")


# Create study and optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30) # 30 alag models try karega

print("Best Parameters Found: ", study.best_params)
print("Best CV Score: ", study.best_value)


# Use the best parameters found by Optuna
best_params = study.best_params
best_params['n_jobs'] = -1
best_params['random_state'] = 42

final_model = XGBClassifier(**best_params)
final_model.fit(X, y)

print("Final Model Trained!")


preds = final_model.predict_proba(X_test)[:, 1]
submission['diagnosed_diabetes'] = preds
submission.to_csv('submission.csv', index=False)
print("Submission File Created: submission.csv")
submission.head()

