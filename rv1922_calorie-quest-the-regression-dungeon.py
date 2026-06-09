import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_log_error
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt  
from sklearn.model_selection import cross_val_score, KFold
from catboost import CatBoostRegressor
import optuna


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


train.head()


train.info()


cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']


le =  LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.fit_transform(test['Sex'])


def feature_engineering(df):
    # Intensity index
    df['Intensity_Index'] = df['Heart_Rate'] / df['Duration']
    
    # Log transformations 
    df['Age'] = np.log1p(df['Age'])
    df['Body_Temp'] = np.log1p(df['Body_Temp'])

    # BMR
    df['BMR'] = (
        10 * df['Weight'] + 
        6.25 * df['Height'] - 
        5 * df['Age'] + 
        np.where(df['Sex'] == 'male', 5, -161)
    )

    # Core interactions
    df['HR_Temp_Interaction'] = df['Heart_Rate'] * df['Body_Temp']
    df['HR_Duration_Interaction'] = df['Heart_Rate'] * df['Duration']
    df['Metabolic_Load'] = df['Heart_Rate'] * df['Body_Temp'] * df['Duration']  
    df['Age_Duration'] = df['Age'] * df['Duration']
    df['Age_Body_Temp'] = df['Age'] * df['Body_Temp']
    df['Duration_Body_Temp'] = df['Duration'] * df['Body_Temp']
    df['Age_Duration_Temp'] = df['Age'] * df['Duration'] * df['Body_Temp']

    # Height & Weight interactions 
    df['Height_Weight'] = df['Height'] * df['Weight']
    df['Height_Duration'] = df['Height'] * df['Duration']
    df['Weight_Duration'] = df['Weight'] * df['Duration']
    df['Weight_HeartRate'] = df['Weight'] * df['Heart_Rate']
    df['Weight_BodyTemp'] = df['Weight'] * df['Body_Temp']
    df['Height_Temp_Interaction'] = df['Height'] * df['Body_Temp']
    df['Weight_Duration_Temp'] = df['Weight'] * df['Duration'] * df['Body_Temp']
    df['Height_Duration_Temp'] = df['Height'] * df['Duration'] * df['Body_Temp']
    df['Weight_HR_Duration'] = df['Weight'] * df['Heart_Rate'] * df['Duration']
    df['Height_HR_Duration'] = df['Height'] * df['Heart_Rate'] * df['Duration']

    # Advanced exertion interactions
    df['Weight_Intensity_Index'] = df['Weight'] * df['Intensity_Index']
    df['Height_Intensity_Index'] = df['Height'] * df['Intensity_Index']
    df['Weight_HR_Temp_Interaction'] = df['Weight'] * df['HR_Temp_Interaction']
    df['Height_HR_Temp_Interaction'] = df['Height'] * df['HR_Temp_Interaction']

    # Log transform Calories only for training set
    if 'Calories' in df.columns:
        df['Calories'] = np.log1p(df['Calories'])

    return df


# Apply to both datasets
train = feature_engineering(train)
test = feature_engineering(test)


train.head()


X = train.drop(columns='Calories')
y = np.log1p(train["Calories"])  


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 1000, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth': trial.suggest_int('depth', 4, 12),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 100, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'loss_function': 'RMSE',
        'eval_metric': 'MSLE',
        'random_seed': 42,
        'task_type': 'GPU',
        'early_stopping_rounds': 50,
        'verbose': 0
    }

    model = CatBoostRegressor(**params)

    # Train model
    model.fit(
        X_train, y_train,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
        verbose=0
    )

    # Predict on validation set
    preds = model.predict(X_valid)
    
    # RMSLE: sqrt(MSLE)
    msle = mean_squared_log_error(y_valid, preds)
    rmsle = np.sqrt(msle)
    return rmsle


#study = optuna.create_study(direction='minimize')
#study.optimize(objective, n_trials=50, timeout=1800)


#print("Best trial RMSLE:", study.best_value)
#print("Best hyperparameters:")
#print(study.best_trial.params)


model = CatBoostRegressor(
    iterations=2020,
    learning_rate=0.012283827150067244,
    depth=10,
    l2_leaf_reg=0.0015895094271880666,
    bagging_temperature=0.04826822249588831,
    border_count=208,
    loss_function='RMSE',
    eval_metric='MSLE',
    task_type='GPU',
    random_seed=42,
    early_stopping_rounds=50,
    verbose=100
)

model.fit(X,y)


test_preds_log = model.predict(test)
test_preds = np.expm1(test_preds_log)  

# Prepare submission
submission['Calories'] = test_preds
submission.to_csv("submission.csv", index=False)

print(submission.head())

