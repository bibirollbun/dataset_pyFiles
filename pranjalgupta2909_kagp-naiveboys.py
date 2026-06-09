


!pip install optuna



from google.colab import drive
drive.mount('/content/drive')


import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna

from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error


train_df = pd.read_csv("train (5).csv")
test_df = pd.read_csv("test (4).csv")


train_ids = train_df['id']
test_ids = test_df['id']


print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")


def feature_engineer(df):
    df_copy = df.copy()

    df_copy['Policy Start Date'] = pd.to_datetime(df_copy['Policy Start Date'], errors='coerce')

    df_copy['Policy_Start_Year'] = df_copy['Policy Start Date'].dt.year
    df_copy['Policy_Start_Month'] = df_copy['Policy Start Date'].dt.month
    df_copy['Policy_Start_DayOfWeek'] = df_copy['Policy Start Date'].dt.dayofweek
    df_copy['Policy_Age_Days'] = (pd.to_datetime('today') - df_copy['Policy Start Date']).dt.days

    df_copy['Income_per_Child'] = df_copy['Annual Income'] / (df_copy['Number of Children'] + 1)
    df_copy['Age_x_CreditScore'] = df_copy['Age'] * df_copy['Credit Score']

    if "Age" in df_copy.columns:
        df_copy["age_bin"] = pd.cut(df_copy["Age"], bins=[0,18,25,35,50,65,100], labels=False)

    # vehicle age bin
    if "Vehicle Age" in df_copy.columns:
        df_copy["vehicle_age_bin"] = pd.cut(df_copy["Vehicle Age"].fillna(-1), bins=[-1,0,2,5,10,100], labels=False)

    # claims per duration
    if "Previous Claims" in df_copy.columns and "Insurance Duration" in df_copy.columns:
        df_copy["claims_per_year"] = df_copy["Previous Claims"] / df_copy["Insurance Duration"].replace(0, np.nan)

    if "Customer Feedback" in df_copy.columns:
        df_copy["feedback_len"] = df_copy["Customer Feedback"].fillna("").str.len()

    # Add missing value indicator columns
    for col in df_copy.columns:
        if df_copy[col].isnull().any():
            df_copy[col + '_missing'] = df_copy[col].isnull().astype(int)


    df_copy = df_copy.drop(columns=[ 'id','Policy Start Date'])
    return df_copy

train_processed = feature_engineer(train_df)
test_processed = feature_engineer(test_df)





X = train_processed.drop(columns=['Premium Amount'])
y = train_processed['Premium Amount']
y_log = np.log1p(y)


numeric_cols = X.select_dtypes(include=np.number).columns  # select only numeric columns

for col in numeric_cols:
    skewness = X[col].skew()
    print(f"{col} skewness: {skewness:.2f}")



from numpy import log1p

# List of columns to transform
skewed_cols = ['Previous Claims', 'Income_per_Child']

for col in skewed_cols:
    X[col] = np.log1p(X[col])  # log1p handles zero values safely



#import numpy as np

# Reflect and log-transform negative skew
#def transform_negative_skew(col):
#    return np.log1p(col.max() - col)

# Columns to transform
#skewed_cols_positive = ['Previous Claims', 'Income_per_Child']  # positive skew
#skewed_cols_negative = ['Annual Income']  # negative skew

# Apply log1p for positive skew
#for col in skewed_cols_positive:
 #   X[col] = np.log1p(X[col])

# Apply reflection + log1p for negative skew
#for col in skewed_cols_negative:
#    X[col] = transform_negative_skew(X[col])



categorical_features = X.select_dtypes(include=['object', 'category']).columns
numerical_features = X.select_dtypes(include=np.number).columns


from sklearn.impute import KNNImputer, SimpleImputer

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Using SimpleImputer
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)


def objective(trial):
    params = {
        'objective': 'regression_l1',
        'metric': 'rmse',
        'n_estimators': 500,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'verbose': -1,
        'n_jobs': -1,
        'seed': 30,
    }
    model = lgb.LGBMRegressor(**params)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', model)])

    kf = KFold(n_splits=5, shuffle=True, random_state=30)
    scores = cross_val_score(pipeline, X, y_log, cv=kf, scoring='neg_root_mean_squared_error')
    return -scores.mean()


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10, show_progress_bar=True)

best_params = study.best_params
print("Best RMSE:", study.best_value)
print("Best Params:", best_params)





if 'Insurance Duration_missing' not in test_processed.columns:
    test_processed['Insurance Duration_missing'] = test_processed['Insurance Duration'].isnull().astype(int)


final_lgbm = lgb.LGBMRegressor(objective='regression_l1', metric='rmse', n_estimators=2000,
                               **best_params, verbose=-1, n_jobs=-1, seed=42)

final_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('regressor', final_lgbm)])

final_pipeline.fit(X, y_log)

test_predictions_log = final_pipeline.predict(test_processed)
test_predictions = np.expm1(test_predictions_log)
test_predictions[test_predictions < 0] = 0


submission_df = pd.DataFrame({'id': test_ids, 'Premium Amount': test_predictions})
submission_df.to_csv('submission.csv', index=False)

print("Submission saved as submission.csv")
print(submission_df.head())





#import warnings

#warnings.filterwarnings('ignore')




















!pip install optuna  lightgbm
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna

from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

train_df = pd.read_csv('/kaggle/input/carnival-risk-analytics-challenge/train.csv')
test_df = pd.read_csv('/kaggle/input/carnival-risk-analytics-challenge/test.csv')

train_ids = train_df['id']
test_ids = test_df['id']

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

def feature_engineer(df):
    df_copy = df.copy()

    df_copy['Policy Start Date'] = pd.to_datetime(df_copy['Policy Start Date'], errors='coerce')

    df_copy['Policy_Start_Year'] = df_copy['Policy Start Date'].dt.year
    df_copy['Policy_Start_Month'] = df_copy['Policy Start Date'].dt.month
    df_copy['Policy_Start_DayOfWeek'] = df_copy['Policy Start Date'].dt.dayofweek
    df_copy['Policy_Age_Days'] = (pd.to_datetime('today') - df_copy['Policy Start Date']).dt.days

    df_copy['Income_per_Child'] = df_copy['Annual Income'] / (df_copy['Number of Children'] + 1)
    df_copy['Age_x_CreditScore'] = df_copy['Age'] * df_copy['Credit Score']

    df_copy = df_copy.drop(columns=[ 'id'])
    return df_copy

train_processed = feature_engineer(train_df)
test_processed = feature_engineer(test_df)

X = train_processed.drop(columns=['Premium Amount'])
y = train_processed['Premium Amount']
y_log = np.log1p(y)

categorical_features = X.select_dtypes(include=['object', 'category']).columns
numerical_features = X.select_dtypes(include=np.number).columns

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)


def objective(trial):
    params = {
        'objective': 'regression_l1',
        'metric': 'rmse',
        'n_estimators': 500,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
    }
    model = lgb.LGBMRegressor(**params)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', model)])

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y_log, cv=kf, scoring='neg_root_mean_squared_error')
    return -scores.mean()

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20, show_progress_bar=True)

best_params = study.best_params
print("Best RMSE:", study.best_value)
print("Best Params:", best_params)

final_lgbm = lgb.LGBMRegressor(objective='regression_l1', metric='rmse', n_estimators=2000,
                               **best_params, verbose=-1, n_jobs=-1, seed=42)

final_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('regressor', final_lgbm)])

final_pipeline.fit(X, y_log)

test_predictions_log = final_pipeline.predict(test_processed)
test_predictions = np.expm1(test_predictions_log)
test_predictions[test_predictions < 0] = 0

submission_df = pd.DataFrame({'id': test_ids, 'Premium Amount': test_predictions})
submission_df.to_csv('submission.csv', index=False)

print("Submission saved as submission.csv")
print(submission_df.head())




