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


# ------------------ Libraries & Environment ------------------ #
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from category_encoders import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from tqdm import tqdm
import warnings

warnings.simplefilter("ignore")
print("Environment Ready ✅")


# ------------------ Data Loading ------------------ #
train_path = "/kaggle/input/playground-series-s5e9/train.csv"
test_path  = "/kaggle/input/playground-series-s5e9/test.csv"

train_df = pd.read_csv(train_path, index_col="id")
test_df  = pd.read_csv(test_path, index_col="id")

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")


train_df.head()


test_df.head()


train_df.columns.tolist()
test_df.columns.tolist()




# ------------------ Feature Engineering & Imputation ------------------ #
def wrangle(df):
    # Columns to check for 1.07e-06
    columns_to_impute = [
        'RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
        'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
        'TrackDurationMs', 'Energy'
    ]
    
    # Impute 1.07e-06 with mean for numerical columns
    for col in columns_to_impute:
        if col in df.columns:
            # Replace 1.07e-06 with NaN, then impute with mean
            df[col] = df[col].replace(1.07e-06, np.nan)
            df[col] = df[col].fillna(df[col].mean())
    
    # Log transformation for skewed numerical features
    df['TrackDurationMs_log'] = np.log1p(df['TrackDurationMs'])
    
    # Interaction features
    df['Energy_MoodScore'] = df['Energy'] * df['MoodScore']
    df['Rhythm_Acoustic'] = df['RhythmScore'] * df['AcousticQuality']
    df['Vocal_Instrumental'] = df['VocalContent'] * df['InstrumentalScore']
    
    # Binary features
    df['HighEnergy'] = (df['Energy'] > df['Energy'].median()).astype(int)
    df['HighLiveLikelihood'] = (df['LivePerformanceLikelihood'] > df['LivePerformanceLikelihood'].median()).astype(int)
    
    # Normalize duration to minutes
    df['TrackDurationMin'] = df['TrackDurationMs'] / (1000 * 60)
    
    # New feature: Ratio of VocalContent to InstrumentalScore
    df['Vocal_to_Instrumental'] = df['VocalContent'] / (df['InstrumentalScore'] + 1e-6)  # Avoid division by zero
    
    # Drop original columns to avoid redundancy
    df = df.drop(columns=['TrackDurationMs'], errors='ignore')
    
    return df

train_df = wrangle(train_df)
test_df  = wrangle(test_df)



# ------------------ Exploratory Data Analysis ------------------ #
# Target Distribution
print("Target Distribution:")
sns.histplot(train_df['BeatsPerMinute'], bins=50, kde=True)
plt.title("Distribution of Price")
plt.show()

# Correlation Matrix
print("Correlation Matrix:")
corr_matrix = train_df.corr(numeric_only=True)
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix of Numerical Features")
plt.show()




# ------------------ Train-Test Split ------------------ #
X = train_df.drop(columns=['BeatsPerMinute'])
y = train_df['BeatsPerMinute']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# ------------------ Robust Training & Evaluation ------------------ #
def train_evaluate_submit(models, X_train, X_val, y_train, y_val, test_df):
    results = {}
    
    # Identify categorical and numeric columns
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Preprocessor
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(use_cat_names=True, handle_unknown='ignore'), cat_cols),
        ('num', StandardScaler(), num_cols)
    ])
    
    for model in tqdm(models, desc="Training Models"):
        # Full pipeline: preprocessing + model
        pipeline = make_pipeline(
            preprocessor,
            model
        )
        
        # Fit
        pipeline.fit(X_train, y_train)
        
        # Validation predictions
        val_pred = pipeline.predict(X_val)
        
        # Metrics
        rmse_score = mean_squared_error(y_val, val_pred, squared=False)
        print(f"\nModel: {model.__class__.__name__}")
        print(f"RMSE Score: {rmse_score:.4f}")
        
        # Submission predictions
        test_pred = pipeline.predict(test_df)
        sub_df = pd.DataFrame({"BeatsPerMinute": test_pred}, index=test_df.index)
        sub_df.to_csv(f"{model.__class__.__name__}_submission.csv", index=True)
        print(f"Submission file saved: {model.__class__.__name__}_submission.csv")
        
        results[model.__class__.__name__] = rmse_score
    
    return pd.DataFrame.from_dict(results, orient='index', columns=["RMSE"]).sort_values("RMSE")



# Models to train
models = [
    XGBRegressor(random_state=42),
    CatBoostRegressor(random_state=42, verbose=0),
    LGBMRegressor(random_state=42, verbose=-1)
]




# Run training & submission
score_df = train_evaluate_submit(models, X_train, X_val, y_train, y_val, test_df)
print("\nModel Performance Summary:")
print(score_df)


# ------------------ Libraries for Model Tuning ------------------ #
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from category_encoders import OneHotEncoder
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import optuna
from tqdm import tqdm
import pandas as pd
import numpy as np
import warnings

warnings.simplefilter("ignore")

# ------------------ Robust Training, Tuning & Evaluation ------------------ #
def train_evaluate_submit(models, X_train, X_val, y_train, y_val, test_df, n_trials=20):
    results = {}
    
    # Identify categorical and numeric columns
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Preprocessor
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(use_cat_names=True, handle_unknown='ignore'), cat_cols),
        ('num', StandardScaler(), num_cols)
    ])
    
    def objective(trial, model_type):
        # Define hyperparameter search space based on model type
        if model_type == 'XGBRegressor':
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
            }
            model = XGBRegressor(**params, random_state=42)
        elif model_type == 'CatBoostRegressor':
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'iterations': trial.suggest_int('iterations', 100, 1000),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10)
            }
            model = CatBoostRegressor(**params, random_state=42, verbose=0)
        elif model_type == 'LGBMRegressor':
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0)
            }
            model = LGBMRegressor(**params, random_state=42, verbose=-1)
        
        # Create pipeline
        pipeline = make_pipeline(
            preprocessor,
            model
        )
        
        # Fit model
        pipeline.fit(X_train, y_train)
        
        # Validation predictions
        val_pred = pipeline.predict(X_val)
        
        # Calculate RMSE
        rmse = mean_squared_error(y_val, val_pred, squared=False)
        return rmse
    
    for model in tqdm(models, desc="Tuning Models"):
        model_name = model.__class__.__name__
        print(f"\nTuning {model_name}...")
        
        # Create Optuna study
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, model_name), n_trials=n_trials)
        
        # Get best parameters
        best_params = study.best_params
        print(f"Best parameters for {model_name}: {best_params}")
        
        # Train model with best parameters
        if model_name == 'XGBRegressor':
            best_model = XGBRegressor(**best_params, random_state=42)
        elif model_name == 'CatBoostRegressor':
            best_model = CatBoostRegressor(**best_params, random_state=42, verbose=0)
        elif model_name == 'LGBMRegressor':
            best_model = LGBMRegressor(**best_params, random_state=42, verbose=-1)
        
        # Create and fit pipeline with best model
        pipeline = make_pipeline(
            preprocessor,
            best_model
        )
        pipeline.fit(X_train, y_train)
        
        # Validation predictions
        val_pred = pipeline.predict(X_val)
        
        # Metrics
        rmse_score = mean_squared_error(y_val, val_pred, squared=False)
        print(f"Model: {model_name}")
        print(f"Best RMSE Score: {rmse_score:.4f}")
        
        # Submission predictions
        test_pred = pipeline.predict(test_df)
        sub_df = pd.DataFrame({"BeatsPerMinute": test_pred}, index=test_df.index)
        sub_df.to_csv(f"{model_name}_tuned_submission.csv", index=True)
        print(f"Submission file saved: {model_name}_tuned_submission.csv")
        
        results[model_name] = rmse_score
    
    return pd.DataFrame.from_dict(results, orient='index', columns=["RMSE"]).sort_values("RMSE")




# Models to tune
models = [
    XGBRegressor(random_state=42),
    CatBoostRegressor(random_state=42, verbose=0),
    LGBMRegressor(random_state=42, verbose=-1)
]

# Run training, tuning & submission
score_df = train_evaluate_submit(models, X_train, X_val, y_train, y_val, test_df, n_trials=20)
print("\nModel Performance Summary:")
print(score_df)




