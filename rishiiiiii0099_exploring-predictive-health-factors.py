# prompt: mount drive

from google.colab import drive
drive.mount('/content/drive')



# Lets Unzip the data
!unzip "/content/drive/MyDrive/Predictive Health Factor /exploring-predictive-health-factors.zip" -d "/content/drive/MyDrive/Predictive Health Factor "


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Models fromm Sklearn
from sklearn.ensemble import RandomForestClassifier

# Import Metrics and reports

from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,recall_score,f1_score


train_df = pd.read_csv('/content/drive/MyDrive/Predictive Health Factor /train.csv')
test_df = pd.read_csv('/content/drive/MyDrive/Predictive Health Factor /test.csv')


# lets make a copy
train_df_copy = train_df.copy()
test_df_copy = test_df.copy()


# lets drop id
train_df_copy.drop("ID",axis=1,inplace =True)
train_df_copy.head()


test_df_copy.drop("ID",axis=1,inplace=True)
test_df_copy.head()


train_df_copy.info()


train_df_copy.describe()


test_df.drop("ID",axis=1,inplace=True)


train_df_copy.reset_index(drop=True,inplace=True)
test_df_copy.reset_index(drop=True,inplace=True)


train_df_copy.dropna(inplace=True)
train_df_copy.isnull().sum()


test_df_copy.dropna(inplace=True)
test_df_copy.isnull().sum()


train_df_copy.info(),test_df_copy.info()


# Value counts
for label,content in train_df_copy.items():
  if pd.api.types.is_object_dtype(content):
    print(label)
    print(train_df_copy[label].value_counts())
    print('--'*40)


for label,content in train_df_copy.items():
  if pd.api.types.is_numeric_dtype(content):
    print(label)
    print(train_df_copy[label].value_counts())
    print("--"*40)


test_df


import datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Define categorical and numerical columns
cat_cols = ['Age','Hormonal_Imbalance', 'Hyperandrogenism', 'Hirsutism',
            'Conception_Difficulty', 'Insulin_Resistance', 'Exercise_Frequency',
            'Exercise_Type', 'Exercise_Duration', 'Sleep_Hours', 'Exercise_Benefit']
num_cols = ['Weight_kg']

# Define Transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])

# Load and preprocess data
X = train_df_copy.drop(columns=['PCOS'])  # Drop target column
y = train_df_copy['PCOS']  # Target variable
X_test = test_df

# Convert target column to binary (if not already)
y = y.map({'Yes': 1, 'No': 0})  # Modify if different labels exist

# Split Data for Training
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Model Pipeline
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        criterion='gini',
        random_state=42,
        n_jobs=-1
    ))
])

# Train the Model
rf_pipeline.fit(X_train, y_train)

# Predictions
y_pred = rf_pipeline.predict(X_valid)
accuracy = accuracy_score(y_valid, y_pred)
print(f'Validation Accuracy: {accuracy:.4f}')

# Train on Full Data & Predict on Test
rf_pipeline.fit(X, y)
X_test_transformed = rf_pipeline.named_steps['preprocessor'].transform(X_test)
test_preds = rf_pipeline.named_steps['classifier'].predict(X_test_transformed)

# Save Trained Model
timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"classifier_model_{timestamp_str}.pkl"
joblib.dump(rf_pipeline, model_filename)
print(f"Trained model saved as {model_filename}")

# Prepare Submission File
submission = pd.DataFrame({'ID': X_test.index, 'PCOS': test_preds})
submission_filename = f"submission_{timestamp_str}.csv"
submission.to_csv(submission_filename, index=False)
print(f"Submission saved as {submission_filename}")



# prompt: install optuna

!pip install optuna



# prompt: import warning

import warnings
warnings.filterwarnings('ignore')



import datetime
import numpy as np
import pandas as pd
import joblib
import optuna
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Define categorical and numerical columns
cat_cols = ['Age','Hormonal_Imbalance', 'Hyperandrogenism', 'Hirsutism',
            'Conception_Difficulty', 'Insulin_Resistance', 'Exercise_Frequency',
            'Exercise_Type', 'Exercise_Duration', 'Sleep_Hours', 'Exercise_Benefit']
num_cols = ['Weight_kg']

# Define Transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])

# Load and preprocess data
X = train_df_copy.drop(columns=['PCOS'])  # Drop target column
y = train_df_copy['PCOS']  # Target variable
X_test = test_df

# Convert target column to binary (if not already)
y = y.map({'Yes': 1, 'No': 0})  # Modify if different labels exist

# Split Data for Training
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Optuna Hyperparameter Optimization
def objective(trial):
    """Objective function for Optuna hyperparameter tuning"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
        'max_depth': trial.suggest_int('max_depth', 5, 50, step=5),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 50, step=2),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20, step=1),
        # Change 'auto' to 'sqrt'
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
        'random_state': 42,
        'n_jobs': -1
    }

    # Define Model Pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(**params))
    ])

    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_valid)

    return accuracy_score(y_valid, y_pred)  # Optimize for accuracy

# Run Optuna Study
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=200)

# Best Parameters Found
best_params = study.best_params
print("Best Parameters Found:", best_params)

# Train on Full Dataset & Predict on Test
final_model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(**best_params))
])

final_model_pipeline.fit(X, y)
X_test_transformed = final_model_pipeline.named_steps['preprocessor'].transform(X_test)
test_preds = final_model_pipeline.named_steps['classifier'].predict(X_test_transformed)

# Save Trained Model
timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"classifier_model_{timestamp_str}.pkl"
joblib.dump(final_model_pipeline, model_filename)
print(f"Trained model saved as {model_filename}")

# Prepare Submission File
submission = pd.DataFrame({'ID': X_test.index, 'PCOS': test_preds})
submission_filename = f"submission_{timestamp_str}.csv"
submission.to_csv(submission_filename, index=False)
print(f"Submission saved as {submission_filename}")



import datetime
import numpy as np
import pandas as pd
import joblib
import optuna
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Define categorical and numerical columns
cat_cols = ['Age','Hormonal_Imbalance', 'Hyperandrogenism', 'Hirsutism',
            'Conception_Difficulty', 'Insulin_Resistance', 'Exercise_Frequency',
            'Exercise_Type', 'Exercise_Duration', 'Sleep_Hours', 'Exercise_Benefit']
num_cols = ['Weight_kg']

# Define Transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])

# Load and preprocess data
X = train_df_copy.drop(columns=['PCOS'])  # Drop target column
y = train_df_copy['PCOS']  # Target variable
X_test = test_df

# Convert target column to binary (if not already)
y = y.map({'Yes': 1, 'No': 0})  # Modify if different labels exist

# Split Data for Training
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Optuna Hyperparameter Optimization
def objective(trial):
    """Objective function for Optuna hyperparameter tuning"""
    params = {
        'C': trial.suggest_loguniform('C', 1e-4, 1e4),
        'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'saga']),
        'max_iter': trial.suggest_int('max_iter', 100, 1000, step=50),
        'random_state': 42
    }

    # Define Model Pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(**params))
    ])

    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_valid)

    return accuracy_score(y_valid, y_pred)  # Optimize for accuracy

# Run Optuna Study
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=200)

# Best Parameters Found
best_params = study.best_params
print("Best Parameters Found:", best_params)

# Train on Full Dataset & Predict on Test
final_model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(**best_params))
])

final_model_pipeline.fit(X, y)
X_test_transformed = final_model_pipeline.named_steps['preprocessor'].transform(X_test)
test_preds = final_model_pipeline.named_steps['classifier'].predict(X_test_transformed)

# Save Trained Model
timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"logistic_model_{timestamp_str}.pkl"
joblib.dump(final_model_pipeline, model_filename)
print(f"Trained model saved as {model_filename}")

# Prepare Submission File
submission = pd.DataFrame({'ID': X_test.index, 'PCOS': test_preds})
submission_filename = f"submission_{timestamp_str}.csv"
submission.to_csv(submission_filename, index=False)
print(f"Submission saved as {submission_filename}")



df_id=pd.read_csv("/content/drive/MyDrive/Predictive Health Factor /test.csv")


!pip install catboost


import datetime
import numpy as np
import pandas as pd
import joblib
import optuna
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define categorical and numerical columns
cat_cols = ['Age','Hormonal_Imbalance', 'Hyperandrogenism', 'Hirsutism',
            'Conception_Difficulty', 'Insulin_Resistance', 'Exercise_Frequency',
            'Exercise_Type', 'Exercise_Duration', 'Sleep_Hours', 'Exercise_Benefit']
num_cols = ['Weight_kg']

# Define Transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])

# Load and preprocess data
X = train_df_copy.drop(columns=['PCOS'])  # Drop target column
y = train_df_copy['PCOS']  # Target variable
X_test = test_df

# Convert target column to binary (if not already)
y = y.map({'Yes': 1, 'No': 0})  # Modify if different labels exist

# Split Data for Training
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Optuna Hyperparameter Optimization
def objective(trial):
    """Objective function for Optuna hyperparameter tuning"""
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000, step=100),
        'depth': trial.suggest_int('depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-4, 10),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'loss_function': 'Logloss',
        'eval_metric': 'Accuracy',
        'random_seed': 42,
        'verbose': 0
    }

    # Define Model Pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', CatBoostClassifier(**params))
    ])

    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_valid)

    return accuracy_score(y_valid, y_pred)  # Optimize for accuracy

# Run Optuna Study
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=200)

# Best Parameters Found
best_params = study.best_params
print("Best Parameters Found:", best_params)

# Train on Full Dataset & Predict on Test
final_model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', CatBoostClassifier(**best_params))
])

final_model_pipeline.fit(X, y)
X_test_transformed = final_model_pipeline.named_steps['preprocessor'].transform(X_test)
test_preds = final_model_pipeline.named_steps['classifier'].predict(X_test_transformed)

# Save Trained Model
timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"catboost_model_{timestamp_str}.pkl"
joblib.dump(final_model_pipeline, model_filename)
print(f"Trained model saved as {model_filename}")

# Prepare Submission File
submission = pd.DataFrame({'ID': df_id['ID'], 'PCOS': test_preds})
submission_filename = f"submission_{timestamp_str}.csv"
submission.to_csv(submission_filename, index=False)
print(f"Submission saved as {submission_filename}")



!pip install --upgrade scikit-learn xgboost


!pip install --upgrade scikit-learn


import datetime
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ✅ Load Dataset
# Assuming train_df_copy is already loaded
X = train_df_copy.drop(columns=['PCOS'])  # Drop target column
y = train_df_copy['PCOS']  # Target variable
X_test = test_df.copy()  # Ensure X_test is not modified

# ✅ Convert Target Column to Binary
y = y.map({'Yes': 1, 'No': 0})  # Adjust based on actual labels

# ✅ Define Categorical and Numerical Columns
cat_cols = [
    'Age', 'Hormonal_Imbalance', 'Hyperandrogenism', 'Hirsutism',
    'Conception_Difficulty', 'Insulin_Resistance', 'Exercise_Frequency',
    'Exercise_Type', 'Exercise_Duration', 'Sleep_Hours', 'Exercise_Benefit'
]
num_cols = ['Weight_kg']

# ✅ Define Transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# ✅ Preprocessing Pipeline
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])

# ✅ Train-Test Split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Define XGBoost Classifier (Baseline Model)
baseline_xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    objective='binary:logistic'
)

# ✅ Create Pipeline
baseline_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', baseline_xgb)
])


# ✅ Train on Full Dataset & Predict on Test Data
baseline_pipeline.fit(X, y)
X_test_transformed = baseline_pipeline.named_steps['preprocessor'].transform(X_test)
test_preds = baseline_pipeline.named_steps['classifier'].predict(X_test_transformed)

# ✅ Save Trained Model
timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"xgboost_baseline_model_{timestamp_str}.pkl"
joblib.dump(baseline_pipeline, model_filename)
print(f"Trained model saved as {model_filename}")

# ✅ Prepare Submission File
submission = pd.DataFrame({'ID': df_id['ID'], 'PCOS': test_preds})  # Change 'ID' if needed
submission_filename = f"submission_{timestamp_str}.csv"
submission.to_csv(submission_filename, index=False)
print(f"Submission saved as {submission_filename}")





