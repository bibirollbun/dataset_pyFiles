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


!pip install lightgbm --extra-index-url https://pypi.org/simple


import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
import optuna
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


TRAIN_PATH = r'/kaggle/input/playground-series-s5e10/train.csv'
TEST_PATH = r'/kaggle/input/playground-series-s5e10/test.csv'
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file at {path} does not exist.")
        
    df = pd.read_csv(path)
    print(f"Data loaded successfully from {path}")
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df


train_df = load_data(TRAIN_PATH)
print('-'*100)
test_df = load_data(TEST_PATH)


train_df.info()


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    missing_values = df.isnull().sum()
    print("Missing values in each column:")
    print(missing_values[missing_values > 0])
    return missing_values


missing_values_train = check_missing_values(train_df)
missing_values_train.value_counts()


def check_columns(df1: pd.DataFrame, df2: pd.DataFrame):
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    
    only_in_df1 = cols1 - cols2
    only_in_df2 = cols2 - cols1
    
    if only_in_df1:
        print(f"Columns only in first DataFrame: {list(only_in_df1)}")
    else:
        print("No columns are unique to the first DataFrame.")
    
    if only_in_df2:
        print(f"Columns only in second DataFrame: {list(only_in_df2)}")
    else:
        print("No columns are unique to the second DataFrame.")


check_columns(train_df, test_df)


def one_hot_encode_for_category_columns(df: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = df.select_dtypes(include=['object', 'bool']).columns
    print(f"Categorical columns to be encoded: {list(categorical_cols)}")
    
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_data = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))
    
    df = df.drop(columns=categorical_cols).reset_index(drop=True)
    df = pd.concat([df, encoded_df], axis=1)
    print(f"Data shape after encoding: {df.shape}")
    print(f"New columns added: {list(encoded_df.columns)}")
    
    return df, encoder, categorical_cols

def one_hot_encode_for_category_columns_test(df: pd.DataFrame, encoder: OneHotEncoder, categorical_cols) -> pd.DataFrame:
    encoded_data = encoder.transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))
    
    df = df.drop(columns=categorical_cols).reset_index(drop=True)
    df = pd.concat([df, encoded_df], axis=1)
    print(f"Data shape after encoding: {df.shape}")
    print(f"New columns added: {list(encoded_df.columns)}")
    
    return df


train_df, encoder, category_columns = one_hot_encode_for_category_columns(train_df)
print('-'*100)
test_df = one_hot_encode_for_category_columns_test(test_df, encoder, category_columns)


check_columns(train_df, test_df)


import copy

def outlier_detection_with_isolation_forest(
    train_df: pd.DataFrame, 
    features: list, 
    contamination: str|float = 'auto'
):
    df = copy.deepcopy(train_df)

    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=RANDOM_STATE
    )
    df_features = df[features]
    
    iso_forest.fit(df_features)
    
    df['outlier'] = iso_forest.predict(df_features)         
    df['anomaly_score'] = iso_forest.decision_function(df_features)  
    
    outliers = df[df['outlier'] == -1]
    inliers = df[df['outlier'] == 1]
    
    print(f"Number of outliers detected: {len(outliers)}")
    print(f"Number of inliers detected: {len(inliers)}")
    
    plt.figure(figsize=(8, 4))
    plt.hist(df['anomaly_score'], bins=50, color='steelblue', edgecolor='black')
    plt.title(f"Isolation Forest anomaly scores (contamination={contamination})")
    plt.xlabel("Anomaly score (higher = more normal)")
    plt.ylabel("Frequency")
    plt.axvline(
        x=outliers['anomaly_score'].max(), 
        color='red', linestyle='--', label='Outlier threshold'
    )
    plt.legend()
    plt.show()
    
    threshold = outliers['anomaly_score'].max()
    print(f"Outlier threshold ≈ {threshold:.4f}")
    
    return outliers, inliers, df



outliers, inliers, new_df = outlier_detection_with_isolation_forest(
    train_df, 
    features=[col for col in train_df.columns if col not in ['id', 'severity', 'outlier', 'anomaly_score']],
)

del outliers, inliers, new_df


train_df = train_df.drop(columns=['id'])
y_train = train_df['accident_risk']
train_df = train_df.drop(columns=['accident_risk'])


scaler = StandardScaler()
X_scaled = scaler.fit_transform(train_df)
y_train_array = y_train.to_numpy()


params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 128,               
    "max_depth": 12,                 
    "learning_rate": 0.05,            
    "n_estimators": 1000,             
    "feature_fraction": 0.85,         
    "bagging_fraction": 0.8,          
    "bagging_freq": 5,                
    "lambda_l1": 1.0,                 
    "lambda_l2": 2.0,                 
    "min_child_samples": 30,          
    "device": "gpu",                  
    "verbosity": -1,
    "random_state": RANDOM_STATE
}


kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
rmse_list = []

for train_index, val_index in kf.split(X_scaled):
    X_train_fold, X_val_fold = X_scaled[train_index], X_scaled[val_index]
    y_train_fold, y_val_fold = y_train_array[train_index], y_train_array[val_index]
    
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        eval_metric="rmse",
    )
    
    y_pred = model.predict(X_val_fold)
    rmse = mean_squared_error(y_val_fold, y_pred, squared=False)
    print(f'RMSE: {rmse}')
    rmse_list.append(rmse)
    
print(f"Mean RMSE: {np.mean(rmse_list):.4f} ± {np.std(rmse_list):.4f}")


def eval_last_model(X_train, y_train, params):
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_scaled, y_train_array)
    preds = model.predict(X_scaled)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_train_array, preds, alpha=0.5, label="Samples")
    plt.plot([y_train_array.min(), y_train_array.max()],
             [y_train_array.min(), y_train_array.max()],
             'r--', label="Perfect Prediction")
    plt.xlabel('Actual Tm')
    plt.ylabel('Predicted Tm')
    plt.title('Actual vs Predicted Tm')
    plt.legend()
    plt.show()

    print(f"Train RMSE: {mean_squared_error(y_train_array, preds, squared=False):.4f}")
    print(f'Train MSE Score: {mean_squared_error(y_train_array, preds):.4f}')
    print(f'Train MAE Score: {mean_absolute_error(y_train_array, preds):.4f}')
    print(f'Train R2 Score: {r2_score(y_train_array, preds):.4f}')
    print(f'n samples: {len(y_train_array)}')

    return model


test_ids = test_df['id']
test_df = test_df.drop(columns=['id'])


def test_and_save(model, X_test, x_test_ids, scaler, to_save_path="submission.csv"):    
    
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    
    y_preds = model.predict(X_test_scaled)
    
    to_save = pd.DataFrame({
        "id": x_test_ids,
        "accident_risk": y_preds
    })
    
    to_save.to_csv(to_save_path, index=False)
    print(f"✅ Predictions saved to {to_save_path}")


model = eval_last_model(X_scaled, y_train_array, params)


test_and_save(model, test_df, test_ids, scaler, to_save_path="submission.csv")




