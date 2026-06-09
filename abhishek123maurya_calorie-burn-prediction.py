import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
df.head()


df.drop('id', inplace=True, axis=1)
df.shape


df.info()


df.describe()


gender_dist = df['Sex'].value_counts()

plt.pie(gender_dist, labels=gender_dist.index,
        autopct='%1.1f%%', startangle=140)
plt.title('Distribution across Gender')
plt.show()


df.groupby('Sex').mean()


plt.subplots(figsize=(10, 5))
for i, col in enumerate(df.columns[1:]):  
    plt.subplot(3, 3, i+1)
    sb.distplot(df[col])
plt.tight_layout()
plt.show()


df['Sex'] = df['Sex'].map({"male": 0, "female": 1})
df.head()


plt.figure(figsize=(15, 10))
sb.heatmap(df.corr()>0.8, annot=True, cbar=False)
plt.show()


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

target = df['Calories']
features = df.drop('Calories', axis=1)

x_train, x_val,\
y_train, y_val = train_test_split(features, target, test_size=0.2)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)


import optuna
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_log_error

def objective(trial):
    param = {
        'objective': 'reg:squarederror',
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth':    trial.suggest_int('max_depth', 3, 12),
        'learning_rate':trial.suggest_loguniform('learning_rate', 1e-3, 0.3),
        'subsample':    trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'reg_alpha':    trial.suggest_loguniform('reg_alpha', 1e-8, 10.0),
        'reg_lambda':   trial.suggest_loguniform('reg_lambda', 1e-8, 10.0),
        'gamma':        trial.suggest_uniform('gamma', 0.0, 5.0)
    }
    model = XGBRegressor(**param)
    model.fit(
        x_train, y_train,
        eval_set=[(x_val, y_val)],
        early_stopping_rounds=10,
        verbose=False
    )
    preds = model.predict(x_val)
    rmse = mean_squared_log_error(y_val, preds, squared=False)
    return rmse


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, show_progress_bar=True)


study.best_params



from sklearn.metrics import mean_squared_log_error


model = XGBRegressor(tree_method= 'gpu_hist',
                     predictor='gpu_predictor',
                     **study.best_params)
model.fit(x_train, y_train)
y_train_pred = model.predict(x_train)
y_val_pred = model.predict(x_val)
train_rmsle = mean_squared_log_error(y_train, y_train_pred, squared=False)
val_rmsle = mean_squared_log_error(y_val, y_val_pred, squared=False)
print(f"Training RMSLE: {train_rmsle:.4f}, Validation RMSLE: {val_rmsle:.4f}")


test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
test.head()


test.drop('id', inplace=True, axis=1)
test['Sex'] = test['Sex'].map({"male": 0, "female": 1})
test.head()


ss = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
preds = model.predict(scaler.transform(test))
ss['Calories'] = preds
ss.head()


ss.to_csv('Submission.csv', index=False)

