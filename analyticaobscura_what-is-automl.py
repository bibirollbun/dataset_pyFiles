%%capture
!pip install flaml 

import pandas as pd
import numpy as np
from flaml import AutoML
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv', index_col='id')
sub = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv', index_col='id')


def feature_engineering(df):
    df = df.copy()
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['Body_Temp_Duration'] = df['Body_Temp'] * df['Duration']
    df['Weight_Heart_Rate'] = df['Weight'] * df['Heart_Rate']
    df = pd.get_dummies(df, columns=['Sex'], drop_first=True)
    return df


train_fe = feature_engineering(train)
test_fe = feature_engineering(test)


X_train = train_fe.drop(columns='Calories')
y_train = np.log1p(train_fe['Calories'])


aml = AutoML()
aml.fit(
    X_train,
    y_train,
    task='regression',
    metric='rmse',  
    time_budget=3600, # 1 Hour
    eval_method='cv',
    n_splits=5,
    estimator_list=['xgboost', 'lgbm', 'catboost'], 
    ensemble=True,
    verbose=1
)


print("The best model:", aml.best_estimator)
print("Best configuration:", aml.best_config)
print("Best validation loss:", aml.best_loss)


import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

print("Train data columns:", train.columns.tolist())


def feature_engineering(df):
    df = df.copy()
    # BMI
    if 'Weight' in df.columns and 'Height' in df.columns:
        df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    else:
        print("Error: Body_Temp or Duration column is missing!")
        df['BMI'] = 0
    # Body_Temp_Duration
    if 'Body_Temp' in df.columns and 'Duration' in df.columns:
        df['Body_Temp_Duration'] = df['Body_Temp'] * df['Duration']
    else:
        print("Error: Body_Temp or Duration column is missing!")
        df['Body_Temp_Duration'] = 0
    # Weight_Heart_Rate
    if 'Weight' in df.columns and 'Heart_Rate' in df.columns:
        df['Weight_Heart_Rate'] = df['Weight'] * df['Heart_Rate']
    else:
        print("Error: Weight or Heart_Rate column is missing!")
        df['Weight_Heart_Rate'] = 0
    # Sex iÃ§in one-hot encoding
    if 'Sex' in df.columns:
        df = pd.get_dummies(df, columns=['Sex'], drop_first=True, dummy_na=False)
    else:
        print("Error: Sex column is missing!")
        df['Sex_male'] = 0
    return df


train_fe = feature_engineering(train)
print("Columns after feature engineering:", train_fe.columns.tolist())


if 'Calories' in train_fe.columns:
    X_train = train_fe.drop(columns='Calories')
    y_train = np.log1p(train_fe['Calories'])
else:
    raise ValueError("Error: Calories column missing in train_fe!")


numerical_cols = [col for col in ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 
                                  'BMI', 'Body_Temp_Duration', 'Weight_Heart_Rate'] if col in X_train.columns]
categorical_cols = [col for col in ['Sex_male'] if col in X_train.columns]
print("Numeric columns:", numerical_cols)
print("Categorical columns:", categorical_cols)


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ])

lgbm_params = {
    'n_estimators': 1125,
    'num_leaves': 110,
    'min_child_samples': 9,
    'learning_rate': 0.0179455702408711,
    'colsample_bytree': 0.5979737441060009,
    'reg_alpha': 0.001975258376030875,
    'reg_lambda': 0.005106256873241264,
    'max_bin': 2**10,  # log_max_bin=10
    'random_state': 42,
    'verbose': -1
}


pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LGBMRegressor(**lgbm_params))
])


kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(X_train.shape[0])

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"Fold {fold+1}/5")
    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
    pipeline.fit(X_train_fold, y_train_fold)
    oof_preds[val_idx] = pipeline.predict(X_val_fold)

# Validasyon RMSLE
y_pred_orig = np.expm1(oof_preds)
y_pred_orig = np.clip(y_pred_orig, a_min=0, a_max=400)
rmsle = np.sqrt(mean_squared_log_error(train['Calories'], y_pred_orig))
print(f"Validation RMSLE: {rmsle:.6f}")


df1 = pd.read_csv("/kaggle/input/my-best-sub/submission_1.csv")
df2 = pd.read_csv("/kaggle/input/my-best-sub/submission_2.csv")
df3 = pd.read_csv("/kaggle/input/my-best-sub/submission_3.csv")
df4 = pd.read_csv("/kaggle/input/my-best-sub/submission_4.csv")
df5 = pd.read_csv("/kaggle/input/my-best-sub/submission_5.csv")

ground_truth = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")  

all_preds = np.stack([df['Calories'] for df in [df1, df2, df3, df4, df5]], axis=1)
ground_truth['Calories'] = np.median(all_preds, axis=1)
ground_truth.to_csv('submission.csv', index=False)

