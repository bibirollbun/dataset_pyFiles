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


import matplotlib.pyplot as plt
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import xgboost as xgb
import optuna
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
from sklearn.model_selection import StratifiedKFold
# config
#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
#print(device)
class CFG:
    train_csv = '/kaggle/input/playground-series-s5e12/train.csv'
    test_csv = '/kaggle/input/playground-series-s5e12/test.csv'
    sample_submission_csv = '/kaggle/input/playground-series-s5e12/sample_submission.csv'
    N_FOLDS = 5
    RANDOM_SEED = 42


train = pd.read_csv(CFG.train_csv)
test = pd.read_csv(CFG.test_csv)
sample_submission = pd.read_csv(CFG.sample_submission_csv)


print("------------train data------------")
print(train.shape)
print(train.head())
print("------------test data------------")
print(test.shape)
print(test.head())


# basic statistics
print(train.describe())


# check if there is any nan data in it
train.isnull().sum()


numeric_cols = train.columns[train.dtypes.apply(lambda x: np.issubdtype(x, np.number))].tolist()
numeric_cols = [c for c in numeric_cols if c not in ['id', 'diagnosed_diabetes']]

fig, axs = plt.subplots(4, 5, figsize=(20, 15)) # Adjust grid size as needed
axs = axs.flatten() # Flattens the 2D grid to 1D

for i, col in enumerate(numeric_cols):
    if i < len(axs):
        axs[i].hist(train[col], bins=30, color='skyblue', edgecolor='black')
        axs[i].set_title(col)
    else:
        axs[i].axis('off') # Hide unused subplots

plt.tight_layout()
plt.show()


# we use "Label Encodeing" because its simple to set up
# gender,ethnicity,education_level,income_level,smoking_status,employment_status

encoder = LabelEncoder()
gender_encoded = encoder.fit_transform(train["gender"])
ethnicity_encoded = encoder.fit_transform(train["ethnicity"])
education_level_encoded = encoder.fit_transform(train["education_level"])
income_level_encoded = encoder.fit_transform(train["income_level"])
smoking_status_encoded = encoder.fit_transform(train["smoking_status"])
employment_status_encoded = encoder.fit_transform(train["employment_status"])


# Create subplots
fig, axs = plt.subplots(1, 6, figsize=(10, 4))  # 1 row, 4 columns
axs[0].hist(gender_encoded, color='orange', edgecolor='black')
axs[0].set_title('gender')
axs[1].hist(ethnicity_encoded, color='orange', edgecolor='black')
axs[1].set_title('ethnicity')
axs[2].hist(education_level_encoded, color='orange', edgecolor='black')
axs[2].set_title('education_level')
axs[3].hist(income_level_encoded,color='orange', edgecolor='black')
axs[3].set_title('income_level')
axs[4].hist(smoking_status_encoded, color='orange', edgecolor='black')
axs[4].set_title('smoking_status')
axs[5].hist(employment_status_encoded, color='orange', edgecolor='black')
axs[5].set_title('employment_status')
plt.tight_layout()
plt.show()


class DiabetesPreprocessor:
    def __init__(self):
        self.medians = {}
        self.encoders = {}
        self.numeric_cols = []
        self.categorical_cols = []
        
    def fit(self, df):
        """
        Learn the parameters (medians, categories) from the TRAINING data.
        """
        # Identify columns
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # 1. Learn Medians for numeric columns
        for col in self.numeric_cols:
            self.medians[col] = df[col].median()
            
        # 2. Fit Encoders for categorical columns
        # handle_unknown='use_encoded_value' prevents crashes if Test data has new categories
        for col in self.categorical_cols:
            enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            # We must reshape to (-1, 1) for sklearn encoders
            enc.fit(df[[col]].astype(str)) 
            self.encoders[col] = enc
            
        return self

    def transform(self, df):
        """
        Apply the learned parameters to the data (Train or Test).
        """
        df = df.copy()
        
        # 1. Drop irrelevant columns (ID is usually dropped, Target handled separately)
        # Note: We don't drop target here to keep X and y aligned until the end
        if 'id' in df.columns:
            df = df.drop(columns=['id'])
            
        # 2. Impute Missing Values using LEARNED medians
        for col in self.numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(self.medians.get(col, 0))
        
        # 3. Apply Encoding
        for col in self.categorical_cols:
            if col in df.columns:
                # Fill NaN in categoricals with 'Missing' before encoding to be safe
                df[col] = df[col].astype(str).fillna('Missing')
                df[col] = self.encoders[col].transform(df[[col]])
        
        # 4. Feature Engineering (Domain Specific)
        # Replacing your "Loan" features with Diabetes features
        
        # BMI is often more useful than just weight/height
        if 'weight' in df.columns and 'height' in df.columns:
             # Assuming height might be in cm, convert to meters if needed. 
             # If height is already M, remove the /100
            df['bmi_calc'] = df['weight'] / ((df['height']/100) ** 2)
        
        # Interaction: Age and Glucose levels are often correlated with risk
        if 'age' in df.columns and 'glucose_levels' in df.columns:
            df['age_glucose_interaction'] = df['age'] * df['glucose_levels']
            
        return df


# Initialize the preprocessor
preprocessor = DiabetesPreprocessor()

# Separate Target from Train for fitting (optional, but cleaner)
# It is best to calculate stats on the features, not including the target
X_train_raw = train.drop(columns=['diagnosed_diabetes'])
y_train = train['diagnosed_diabetes']
test_ids = test['id']

# FIT on Training Data Only (Learn the rules)
preprocessor.fit(X_train_raw)

# TRANSFORM both Train and Test (Apply the rules)
X_train_processed = preprocessor.transform(X_train_raw)
X_test_processed = preprocessor.transform(test)


X_train_processed.head()


X_tuning, _, y_tuning, _ = train_test_split(
    X_train_processed, y_train, 
    train_size=0.4, # Tune on 40% of data
    stratify=y_train, 
    random_state=42
)

# Define the number of folds
def objective_cv(trial, X, y, n_folds=CFG.N_FOLDS, random_seed=CFG.RANDOM_SEED):
    """
    Optuna objective function that uses Stratified K-Fold Cross-Validation.
    """
    # 1. Define Hyperparameters using Optuna trial suggestions
    param = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "use_label_encoder": False,
        "tree_method": 'hist', # Faster training method
        "booster": 'gbtree',
        "random_state": random_seed,
        
        # Hyperparameters to tune
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True), # Use log scale for LR
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
    }

    # 2. Setup Stratified K-Fold
    # Ensure stable splits regardless of data size
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    auc_scores = []
    
    # 3. Training Loop (Cross-Validation)
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Instantiate and train model
        model = xgb.XGBClassifier(**param)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Predict probabilities and calculate AUC for this fold
        y_pred_prob = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred_prob)
        auc_scores.append(auc)
        
    # 4. Return the mean AUC across all folds
    mean_auc = np.mean(auc_scores)
    #Print the result for the current trial
    print(f"Trial {trial.number:3d} finished with mean CV AUC: {mean_auc:.6f}")
    
    return mean_auc


train_x = X_train_processed 
train_y = y_train 
# Build and run Optuna study
print("Starting Optuna study with Stratified K-Fold...")
study = optuna.create_study(direction="maximize")  # Maximize the mean AUC
# Pass the full training data (X and y) to the objective function
study.optimize(lambda trial: objective_cv(trial, X_tuning, y_tuning), n_trials=20)
print("Study complete.")

# Result
print("\nðŸŽ‰ Best parameters found by CV Optuna:")
print(study.best_params)
print(f"Best Mean CV AUC: {study.best_value:.4f}")


# Final Model Training (Retrain on ALL Training Data)
# Once the best hyperparameters are found, we train the final model 
# on the ENTIRE training set (train_x and train_y) for maximum data utilization.
best_params = study.best_params

final_model = xgb.XGBClassifier(
    **best_params,
    objective="binary:logistic",
    eval_metric="auc",
    use_label_encoder=False,
    tree_method='hist',
    random_state=CFG.RANDOM_SEED
)

# Train on the full, processed training set
final_model.fit(train_x, train_y) 

# 3. Generate predictions for the submission file
# Use your processed test data (X_test_processed)
y_pred_prob_final = final_model.predict_proba(X_test_processed)[:, 1]

# 4. Submission file prepare
# IMPORTANT: Ensure the column name is 'diagnosed_diabetes'
submission = pd.DataFrame({
    'id': test['id'],  # Use the original IDs from the test set
    'diagnosed_diabetes': y_pred_prob_final
})
submission.to_csv('cv_optuna_submission.csv', index=False, header=True)
print("\nSubmission file created: cv_optuna_submission.csv")

print(f"Submission file created: {submission.shape}")
print("First 5 rows of submission:")
print(submission.head())




