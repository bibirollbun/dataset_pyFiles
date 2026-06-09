import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


# show train data sample
train_df.head(5) 


def preprocess(df):
    # labeling
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])

    # add features
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['Intensity'] = df['Heart_Rate'] / df['Duration']
    df['Temp_Deviation'] = df['Body_Temp'] - 37.0
    df['Sex_Duration'] = df['Sex'] * df['Duration']
    df['Sex_HeartRate'] = df['Sex'] * df['Heart_Rate']
    return df


# preprocessing
train_df = preprocess(train_df)
test_df = preprocess(test_df)


# prepare X data and Y data
X_features = train_df.drop(['id','Calories'],axis=1)
y_target = np.log1p(train_df['Calories'])

# split train and test set
X_train, X_test, y_train, y_test= train_test_split(X_features, y_target, test_size=0.2, random_state=0)


# kaggle's scikit-learn can't use root_mean_squared_log_error.
# so I implement function rmsle
def root_mean_squared_log_error(y, pred):
    log_y = np.log1p(y)
    log_pred = np.log1p(pred)
    squared_error = (log_y-log_pred)**2
    rmsle = np.sqrt(np.mean(squared_error))
    return rmsle


# get best parameters using optuna

# def objective(trial):
#     params = {
#         'iterations': trial.suggest_int('iterations', 100, 1000),
#         'depth': trial.suggest_int('depth', 4, 10),
#         'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
#         'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
#         'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
#         'random_strength': trial.suggest_float('random_strength', 1e-2, 10.0),
#         'border_count': trial.suggest_int('border_count', 32, 255),
#         'verbose': 0,
#         'random_state': 42
#     }
#
#
#     model = CatBoostRegressor(**params)
#     model.fit(X_train, y_train)
#     preds_log = model.predict(X_test)
#     preds = np.expm1(preds_log)
#
#     # RMSLE
#     rmsle = root_mean_squared_log_error(y_test,preds)
#     return rmsle
#
# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=50, show_progress_bar=True)ㄸ
#
# print("Best Result:", study.best_value)
# print("Best Parameters:", study.best_params_)


model = CatBoostRegressor(iterations=566,depth=10,learning_rate=0.12151953473526761,
                          l2_leaf_reg=9.930340432421506, bagging_temperature=0.2758867848756835,
                          random_strength=2.9984329992495127, border_count=147, verbose=False)


# train
model.fit(X_train, y_train)

pred_log = model.predict(X_test)
pred = np.expm1(pred_log)
y_test_exp = np.expm1(y_test)
rmse = np.sqrt(mean_squared_error(y_test_exp, pred))
rmsle = root_mean_squared_log_error(y_test_exp,pred)
print(f'RMSE : {rmse:.4f}')
print(f'RMSLE : {rmsle:.4f}')


# predict data
X_result = test_df.drop(['id'], axis=1, inplace=False)
test_pred_log = model.predict(X_result)
test_pred = np.expm1(test_pred_log)

# make submission file
test_pd = pd.DataFrame({'id':test_df['id'],'Calories':test_pred})
test_pd.to_csv("submission.csv",index=False)
test_pd.head(10)

