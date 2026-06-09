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


import sklearn
sklearn.__version__


import os

os.cpu_count()


from typing import Union
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


def feature_engineering_target(
    df: pd.DataFrame,
) -> Union[tuple[pd.DataFrame, pd.Series], pd.DataFrame]:
    # Your existing features
    df["BMI"] = df["Weight"] / (df["Height"] ** 2)
    df["Weight_Time_Interaction"] = df["Weight"] * df["Duration"]
    df["Height_Time_Interaction"] = df["Height"] * df["Duration"]
    df["Gender"] = pd.Categorical(df["Sex"]).codes
    df["AgeBin"] = pd.cut(
        df["Age"], bins=[0, 18, 30, 45, 60, np.inf], labels=[0, 1, 2, 3, 4]
    )

    df["Gender_Age_Interaction"] = df["Gender"] * df["Age"]
    df["Body_Temp_Heart_Rate_Ratio"] = df["Body_Temp"] / df["Heart_Rate"]

    # Additional powerful features
    df["BMI_Category"] = pd.cut(
        df["BMI"],
        bins=[0, 18.5, 25, 30, np.inf],
        labels=["Underweight", "Normal", "Overweight", "Obese"],
    )

    df["Heart_Rate_Category"] = pd.cut(
        df["Heart_Rate"],
        bins=[0, 60, 100, 140, np.inf],
        labels=["Low", "Normal", "Elevated", "High"],
    )

    # Intensity features
    df["Weight_Heart_Rate_Interaction"] = df["Weight"] * df["Heart_Rate"]
    df["BMI_Duration_Interaction"] = df["BMI"] * df["Duration"]
    df["Age_Heart_Rate_Interaction"] = df["Age"] * df["Heart_Rate"]

    # Temperature deviation (normal body temp ~98.6°F)
    df["Temp_Deviation"] = abs(df["Body_Temp"] - 98.6)

    # Physical intensity proxy
    df["Heart_Rate_Per_BMI"] = df["Heart_Rate"] / df["BMI"]

    # Polynomial features for important variables
    df["Weight_Squared"] = df["Weight"] ** 2
    df["Duration_Squared"] = df["Duration"] ** 2
    df["Heart_Rate_Squared"] = df["Heart_Rate"] ** 2

    X = df.drop(columns=["id"])
    if "Calories" in df.columns:
        return X.drop(columns=["Calories"]), df["Calories"]
    return X


def preprocessor_for(X: pd.DataFrame) -> ColumnTransformer:
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["number"]).columns.tolist()

    # Define transformers for numerical and categorical features
    numerical_transformer = Pipeline(steps=[("scaler", MinMaxScaler())])
    categorical_transformer = Pipeline(
        steps=[
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            )
        ]
    )
    # Combine transformers into a preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor


def train_xgb_gpu_tuned(X_train, y_train, X_val, y_val):
    """Train XGBoost with hyperparameter tuning on GPU"""
    
    print("Starting GPU XGBoost hyperparameter tuning...")
    import xgboost as xgb
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Parameter grid for tuning
    param_combinations = [
        # Balanced low learning rate, moderate depth
        {
            'eta': 0.03, 'max_depth': 6, 'subsample': 0.9, 'colsample_bytree': 0.9,
            'alpha': 0.1, 'lambda': 1.0
        },
        # Deeper tree, more regularization
        {
            'eta': 0.05, 'max_depth': 10, 'subsample': 0.8, 'colsample_bytree': 0.8,
            'alpha': 0.3, 'lambda': 2.0
        },
        # High colsample, aggressive learning rate, deep trees
        {
            'eta': 0.1, 'max_depth': 12, 'subsample': 0.85, 'colsample_bytree': 1.0,
            'alpha': 0.15, 'lambda': 1.5
        },
        # Conservative setup for RMSLE focus (penalize large errors)
        {
            'eta': 0.02, 'max_depth': 8, 'subsample': 0.95, 'colsample_bytree': 0.95,
            'alpha': 0.05, 'lambda': 1.0
        },
        # Moderate everything
        {
            'eta': 0.06, 'max_depth': 6, 'subsample': 0.85, 'colsample_bytree': 0.85,
            'alpha': 0.1, 'lambda': 1.1
        },
    ]
    
    best_rmsle = float('inf')
    best_model = None
    best_params = None
    
    for i, param_set in enumerate(param_combinations):
        print(f"\nTesting parameter set {i+1}/{len(param_combinations)}")
        
        # Base parameters
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmsle',
            'tree_method': 'hist',
            'gpu_id': 0,
            'seed': 42,
            'verbosity': 1,  # Less verbose for tuning
        }
        params.update(param_set)
        
        evallist = [(dtrain, 'train'), (dval, 'eval')]
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=evallist,
            early_stopping_rounds=50,
            verbose_eval=0,  # Silent during tuning
        )
        
        y_pred = model.predict(dval)
        rmsle = np.sqrt(mean_squared_error(np.log1p(y_val), np.log1p(y_pred)))
        
        print(f"RMSLE: {rmsle:.4f}")
        
        if rmsle < best_rmsle:
            best_rmsle = rmsle
            best_model = model
            best_params = params
    
    print(f"\nBest RMSLE: {best_rmsle:.4f}")
    print(f"Best parameters: {best_params}")
    
    return best_model


def create_submission_file(model, X_test, id):
    predictions = model.predict(X_test)
    submission_df = pd.DataFrame({"id": id, "Calories": predictions})
    submission_df.to_csv("/kaggle/working/submission_new.csv", index=False)
    print("Submission file created successfully.")



# Feature engineering
X, y = feature_engineering_target(train_df)
X_test = feature_engineering_target(test_df)

# Preprocessing
preprocessor = preprocessor_for(X)
X_transformed = pd.DataFrame(preprocessor.fit_transform(X), columns=preprocessor.get_feature_names_out())
X_test_transformed = pd.DataFrame(preprocessor.transform(X_test), columns=preprocessor.get_feature_names_out())
X_train, X_val, y_train, y_val = train_test_split(
    X_transformed, y, test_size=0.2, random_state=42
)
print("Data preprocessing completed.")
# Model training
model = train_xgb_gpu_tuned(X_train, y_train, X_val, y_val)


X_test_transformed.isnull().sum()


import xgboost as xgb
dtest = xgb.DMatrix(X_test_transformed, label=None)
create_submission_file(model, dtest, test_df["id"])

