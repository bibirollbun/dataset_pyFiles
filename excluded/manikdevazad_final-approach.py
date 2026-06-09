import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

# === STEP 1: Load Data ===
df = pd.read_csv("/kaggle/input/predicting-euphoria-in-the-streets/train.csv")   # change filename as needed

# === STEP 2: Basic Info ===
print("Shape of dataset:", df.shape)
print("\n--- Data Types and Null Counts ---")
print(df.info())
print("\n--- Missing Values ---")
print(df.isnull().sum())
print("\n--- Summary Statistics ---")
print(df.describe(include='all').T)


import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

ids = df['id']
X = df.drop(['id','Y'], axis=1)
y = df['Y']

print("Replacing infinite values with NaN...")
X.replace([np.inf, -np.inf], np.nan, inplace=True)

print("Imputing NaN values with the median...")
X.fillna(X.median(), inplace=True)

print("\nDataFrame after imputation:")
print(X)
print("\n" + "="*30 + "\n")

print("Applying RobustScaler...")
scaler = RobustScaler()
X_scaled_array = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled_array, index=X.index, columns=X.columns)

print(X_scaled.head())


import pandas as pd
import numpy as np

train_data = pd.read_csv('/kaggle/input/predicting-euphoria-in-the-streets/train.csv')
df_train = pd.DataFrame(train_data)

test_data = pd.read_csv('/kaggle/input/predicting-euphoria-in-the-streets/test.csv')
df_test = pd.DataFrame(test_data)
# --- End of setup ---

def clean_dataset(df):
    """Cleans a dataframe by handling infinite and missing values."""
    # Make a copy to avoid changing the original dataframe
    df_clean = df.copy()
    
    # Step 1: Replace infinite values with NaN
    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Step 2: Fill missing values (NaN) with the median of each column
    # Using the median is robust to outliers
    for col in df_clean.columns:
        if df_clean[col].isnull().any():
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
            
    return df_clean

# Clean both datasets
df_train_clean = clean_dataset(df_train)
df_test_clean = clean_dataset(df_test)
ids = df_train_clean['id']
df_train_clean = df_train_clean.drop(['id'],axis=1)
print("✅ Datasets have been cleaned.")
print(f"Original train NaNs: {df_train.isnull().sum().sum()}")
print(f"Cleaned train NaNs: {df_train_clean.isnull().sum().sum()}")


X_clean = X_scaled
y_clean = y

# Set plotting style
sns.set_style("whitegrid")
print("Plotting feature distributions...")
X_clean.hist(bins=30, figsize=(20, 15))
plt.suptitle("Histograms of Feature Distributions", size=20)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()


print("\nPlotting feature box plots to detect outliers...")
plt.figure(figsize=(20, 10))
sns.boxplot(data=X_clean)
plt.title("Box Plots of Features", size=20)
plt.xticks(rotation=45)
plt.show()


print("\nCalculating and plotting feature correlation heatmap...")
plt.figure(figsize=(18, 15))
correlation_matrix = X_clean.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title("Feature Correlation Matrix", size=20)
plt.show()


correlations = df_train_clean.corr()['Y'].abs().sort_values(ascending=False)

print("\n--- Correlation of each feature with the target Y ---")
print(correlations.drop('Y'))



CORRELATION_THRESHOLD = 0.1

important_features = correlations[correlations > CORRELATION_THRESHOLD].drop('Y').index.tolist()

print(f"\n--- Found {len(important_features)} features with absolute correlation > {CORRELATION_THRESHOLD} ---")
print(important_features)



from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np

print(f"Starting with {len(important_features)} important features.")

X = df_train_clean[important_features]
y = df_train_clean['Y']

model = LogisticRegression(random_state=42)

print("\nCalculating baseline score using 5-fold cross-validation...")
cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')

mean_score = np.mean(cv_scores)
std_dev = np.std(cv_scores)

print("\n--- Baseline Model Performance ---")
print(f"Mean AUC Score: {mean_score:.4f}")
print(f"Standard Deviation of Scores: {std_dev:.4f}")



import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.model_selection import cross_val_score

models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "LightGBM": lgb.LGBMClassifier(random_state=42)
}

print("--- Comparing Model Performance (using default settings) ---")
for name, model in models.items():
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc', n_jobs=-1)
    mean_score = np.mean(cv_scores)
    print(f"{name} Mean AUC Score: {mean_score:.4f}")



import optuna
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np

def objective_lgbm(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbose': -1,
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = lgb.LGBMClassifier(**params)
    score = cross_val_score(model, X, y, cv=5, scoring='roc_auc').mean()
    return score

optuna.logging.set_verbosity(optuna.logging.WARNING)

print("--- Running Bayesian Optimization for LightGBM (Silently) ---")
study_lgbm = optuna.create_study(direction='maximize')
study_lgbm.optimize(objective_lgbm, n_trials=50)

best_lgbm_params = study_lgbm.best_params
print(f"\nOptimization Finished!")
print(f"\nBest LightGBM Score: {study_lgbm.best_value:.4f}")
print("Best LightGBM Parameters:")
print(best_lgbm_params)



import pandas as pd
import lightgbm as lgb

test_data = pd.read_csv('/kaggle/input/predicting-euphoria-in-the-streets/test.csv')
df_test = pd.DataFrame(test_data)
df_test_clean = clean_dataset(df_test)

print("--- Creating final model using the best parameters ---")
best_params = {
    'n_estimators': 316,
    'learning_rate': 0.010436635600982121,
    'num_leaves': 34,
    'max_depth': 11,
    'reg_alpha': 0.75,
    'reg_lambda': 0.88
}
final_lgbm_model = lgb.LGBMClassifier(**best_params, random_state=42)

print("Training final model on all available training data...")
final_lgbm_model.fit(X, y)

print("Making predictions on the test set...")
X_test_final = df_test_clean[important_features]
test_ids = df_test_clean['id']
final_predictions = final_lgbm_model.predict_proba(X_test_final)[:, 1]

print("Creating new submission file: 'tuned_submission.csv'...")
optuna_tuned_submission_df = pd.DataFrame({
    'id': test_ids,
    'Y': final_predictions
})
optuna_tuned_submission_df.to_csv('optuna_tuned_submission.csv', index=False)

print("\n✅ 'tuned_submission.csv' has been created successfully!")
print("This submission is based on your tuned LightGBM model.")
print("\nHere's a preview:")
print(optuna_tuned_submission_df.head())



import pandas as pd
import numpy as np

# Model Imports
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

# Preprocessing Imports
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline



models = {
    "LightGBM": lgb.LGBMClassifier(random_state=42),
    "XGBoost": xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    "Logistic Regression": LogisticRegression(random_state=42)
}

predictions = {}

print("\n--- Training Models and Getting Predictions ---")
for name, model in models.items():
    print(f"Processing model: {name}")
    
    full_pipeline = Pipeline(steps=[
            ('classifier', model)
    ])
    
    full_pipeline.fit(X, y)
    predictions[name] = full_pipeline.predict_proba(X_test_final)[:, 1]

print("\n--- Blending Predictions and Creating Submission File ---")
ensemble_prediction = np.mean([preds for preds in predictions.values()], axis=0)

submission_df = pd.DataFrame({
    'id': test_ids,
    'Y': ensemble_prediction
})
submission_df.to_csv('ensemble_submission.csv', index=False)

print("\n✅ 'ensemble_submission.csv' has been created successfully!")
print("Here's a preview:")
print(submission_df.head())



!pip install flaml

import pandas as pd
from flaml import AutoML

X_train = df_train_clean.drop(columns='Y', axis=1)
X_test = df_test_clean
y_train = df_train_clean['Y']

automl = AutoML()
settings = {
    "time_budget": 30,
    "metric": 'roc_auc',
    "task": 'classification',
    "log_file_name": "flaml.log",
}

print("--- Running FLAML AutoML ---")
automl.fit(X_train=X_train, y_train=y_train, **settings)

print('\nBest model found:', automl.model.estimator)
print('Best hyperparameter config:', automl.best_config)

print("\n--- Making predictions on the test set ---")
predictions = automl.predict_proba(X_test)[:, 1]

submission_df = pd.DataFrame({
    'id': X_test['id'],
    'prediction': predictions
})
submission_df.to_csv('flaml_submission.csv', index=False)

print("\n✅ 'flaml_submission.csv' created successfully!")


