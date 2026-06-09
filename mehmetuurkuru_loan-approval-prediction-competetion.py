import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier,GradientBoostingClassifier
import optuna
from optuna import create_study
import time
import warnings
warnings.filterwarnings("ignore")


df=pd.read_csv('/kaggle/input/playground-series-s4e10/train.csv')
dftest=pd.read_csv('/kaggle/input/playground-series-s4e10/test.csv')
dfnew=pd.read_csv('/kaggle/input/loan-approval-prediction/credit_risk_dataset.csv')


df = df.drop(columns=['id'])


df= pd.concat([df, dfnew], ignore_index=True)


def remove_invalid_rows(df):
    df = df[df['person_emp_length'] < df['person_age']]
    return df


def new_features(df):
    df['income_to_loan_ratio'] = df['person_income'] / df['loan_amnt']
    df['age_to_emp_length_ratio'] = df['person_emp_length'] / df['person_age']
    df['annual_interest_payment'] = df['loan_amnt'] * (df['loan_int_rate'] / 100)
    df['annual_payment_to_income_ratio'] = df['annual_interest_payment'] / df['person_income']
    df['grade_interest_interaction'] = df['loan_grade'].apply(lambda x: ord(x) - ord('A') + 1) * df['loan_int_rate']
    return df


imputer = KNNImputer(n_neighbors=5)
numeric_cols = df.select_dtypes(include=['float64', 'int']).columns
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
print(df['loan_int_rate'].isnull().sum())


def choosing_range(df):
    columns_95 = ['person_age', 'person_income', 'person_emp_length', 
                  'loan_percent_income', 'cb_person_cred_hist_length', 
                  'loan_amnt', 'loan_int_rate','income_to_loan_ratio','age_to_emp_length_ratio','annual_interest_payment','grade_interest_interaction']
    
    best_quantiles = {
        'person_age': 0.96,
        'person_income': 0.99,
        'person_emp_length': 0.97,
        'loan_percent_income': 0.99,
        'cb_person_cred_hist_length': 0.98,
        'loan_amnt': 0.98,
        'loan_int_rate': 0.98,
        'income_to_loan_ratio':0.96,
        'age_to_emp_length_ratio':0.96,
        'annual_interest_payment':0.96,
        'grade_interest_interaction':0.98
        }
    
    for col in columns_95:
        max_value = df[col].quantile(best_quantiles[col])
        df.loc[df[col] > max_value, col] = max_value

    return df


def convert_default_to_bool(df):
    df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map({'Y': True, 'N': False})
    return df


def preprocess_data(df):
    grade_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    df['loan_grade'] = df['loan_grade'].map(grade_mapping)

    df = pd.get_dummies(df, columns=['person_home_ownership', 'loan_intent'], drop_first=False)
    return df


def all_process(df):
    df= remove_invalid_rows(df)
    df= new_features(df)
    df= choosing_range(df)
    df= convert_default_to_bool(df)
    df= preprocess_data(df)
    df['person_age'] = df['person_age'].apply(lambda x: x - 20)
    df['person_income'] = df['person_income'].apply(lambda x: x - 4000)
    return(df)


df= all_process(df)


df.columns


df.describe()


X = df.drop('loan_status', axis=1)  
y = df['loan_status'] 


# Split your data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Base models - these are just templates; the actual fitted models will be stored in best_estimators
base_models = {
    'catboost': CatBoostClassifier(silent=True, random_state=42),
    'lightgbm': LGBMClassifier(verbosity=-1, random_state=42),
    'gradient_boosting': GradientBoostingClassifier(random_state=42)
}

# Optuna objective function
def objective(trial, model_name):
    if model_name == 'catboost':
        params = {
            'depth': trial.suggest_int('depth', 4, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.04, 0.09),
            'iterations': trial.suggest_int('iterations', 40, 100),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 2),
            'subsample': trial.suggest_float('subsample', 0.85, 0.95),
        }
        model = CatBoostClassifier(**params, silent=True, random_state=42)

    elif model_name == 'lightgbm':
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 25, 45),
            'max_depth': trial.suggest_int('max_depth', 4, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.04, 0.09),
            'n_estimators': trial.suggest_int('n_estimators', 70, 150),
            'subsample': trial.suggest_float('subsample', 0.85, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.75, 0.85),
        }
        model = LGBMClassifier(**params, verbosity=-1, random_state=42)

    elif model_name == 'gradient_boosting':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 70, 120),
            'learning_rate': trial.suggest_float('learning_rate', 0.04, 0.09),
            'max_depth': trial.suggest_int('max_depth', 3, 5),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 3),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 2),
            'subsample': trial.suggest_float('subsample', 0.85, 0.95)
        }
        model = GradientBoostingClassifier(**params, random_state=42)

    # Fit the model within the objective function for evaluation
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    return roc_auc

# Optuna studies for each model
best_estimators = {}
n_optuna_trials = 7

for model_name in base_models.keys():
    print(f"Optimizing {model_name} model with {n_optuna_trials} trials...")
    start_time = time.time()

    study = create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, model_name), n_trials=n_optuna_trials)

    best_params = study.best_params
    print(f"Best parameters for {model_name}: {best_params}")

    # Create a new instance of the model with the best parameters
    # and then fit it on the full training data
    model_instance = base_models[model_name].__class__(**best_params, random_state=42)
    model_instance.fit(X_train, y_train)
    best_estimators[model_name] = model_instance
    
    elapsed_time = time.time() - start_time
    print(f"Optimization for {model_name} completed in {elapsed_time:.2f} seconds.")

# Stacking model
stacking_model = StackingClassifier(
    estimators=[(name, best_estimators[name]) for name in best_estimators],
    final_estimator=LogisticRegression(random_state=42),
    cv='prefit'
)

# Train stacking model (only meta-model will be trained as base estimators are pre-fitted)
print("\nTraining the stacking model...")
start_time = time.time()
stacking_model.fit(X_train, y_train) # This will train the final_estimator
elapsed_time = time.time() - start_time
print(f"Stacking model trained in {elapsed_time:.2f} seconds.")

# Predict and evaluate
y_pred_proba = stacking_model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\nStacking Model ROC AUC Score: {roc_auc:.4f}")


dftest= all_process(dftest)


dftest= dftest.reindex(columns=df.columns, fill_value=0)


X_test_final = dftest.drop(columns=['loan_status'])  # 'id' ve 'loan_status' hariç tüm sütunlar
y_test_pred_proba = stacking_model.predict_proba(X_test_final)[:, 1]  # Tahmin olasılıkları

start_id = 58645
new_ids = range(start_id, start_id + len(y_test_pred_proba))

# Create a DataFrame to store the predictions
predictions_df = pd.DataFrame({
    'id': new_ids,
    'loan_status': y_test_pred_proba  # Olasılıkları kaydediyoruz
})

# Kaydetmek için CSV dosyası
predictions_df.to_csv('C:\\Users\\PC\\Desktop\\Çalışmalarım\\LoanCompetetion\\9619.csv', index=False)

