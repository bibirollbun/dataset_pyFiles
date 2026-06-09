import os
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
# Suppress warnings and TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter('ignore')


# Constants
SEED = 42
FOLDS = 5
TARGET = 'Calories'

# Load data
df_train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


# Encode Gender / Sex feature
def encode_sex(df):
    if 'Gender' in df.columns:
        df['Sex'] = df['Gender'].map({'female': 1, 'male': 2})
        df.drop(columns=['Gender'], inplace=True)
    elif 'Sex' in df.columns:
        df['Sex'] = df['Sex'].map({'female': 1, 'male': 2})
    return df


# Feature Engineering
def feature_engineering(df):
    df = df.copy()
    df = encode_sex(df)
    # Drop ID columns if present
    for col in ['id', 'User_ID']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    # Combined AgeSex feature
    df['AgeSex'] = LabelEncoder().fit_transform(df['Age'].astype(str) + df['Sex'].astype(str)) + 1
    # BMI = Weight / Height^2
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)
    # Heart Rate per Age
    df['Heart_Rate_per_Age'] = df['Heart_Rate'] / df['Age']
    # Interaction of Body Temp and Heart Rate
    df['Temp_HR_Interaction'] = df['Body_Temp'] * df['Heart_Rate']
    # Heart Rate difference from mean
    df['HR_above_mean'] = df['Heart_Rate'] - df['Heart_Rate'].mean()
    # Log transform of Weight
    df['Log_Weight'] = np.log1p(df['Weight'])
    # Age squared
    df['Age_squared'] = df['Age'] ** 2
    # Weight x Height
    df['Weight_Height'] = df['Weight'] * df['Height']
    # Duration x Heart Rate
    df['Duration_HR'] = df['Duration'] * df['Heart_Rate']
    return df

# Apply feature engineering
df_train = feature_engineering(df_train)
df_test = feature_engineering(df_test)


# Prepare for CV
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
pred_test = np.zeros(len(df_test))
fold_rmse = []

for fold, (train_idx, val_idx) in enumerate(kf.split(df_train), 1):
    print(f"\n Fold {fold}/{FOLDS}")
    
    # Split train and validation data
    train_data = df_train.iloc[train_idx].copy()
    val_data = df_train.iloc[val_idx].copy()

    # Separate target and features
    y_train = np.log1p(train_data.pop(TARGET))
    y_val = np.log1p(val_data.pop(TARGET))

    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(train_data, label=y_train)
    dval = xgb.DMatrix(val_data, label=y_val)
    dtest = xgb.DMatrix(df_test)

    # XGBoost parameters
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'seed': SEED,
        'max_depth': 7,
        'learning_rate': 0.00095,
        'reg_alpha': 1,
        'reg_lambda': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'tree_method': 'hist'  # Fast CPU histogram method
    }

    # Train model with early stopping
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        evals=[(dval, 'validation')],
        early_stopping_rounds=100,
        num_boost_round=5000,
        verbose_eval=500
    )

    # Predict on validation and test
    val_pred = model.predict(dval)
    rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
    print(f"Fold {fold} RMSE: {rmse:.5f}")
    fold_rmse.append(rmse)

    pred_test += model.predict(dtest)

# Average test predictions
pred_test /= FOLDS
print("\n Fold RMSEs:", fold_rmse)
print(" Average RMSE:", np.mean(fold_rmse))



# Prepare submission
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")
submission['Calories'] = np.expm1(pred_test)
submission.to_csv("submission.csv", index=False)







