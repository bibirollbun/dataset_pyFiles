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


train_data = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
sample_submission_data = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')


train_df = train_data.copy()
test_df = test_data.copy()
sample_df = sample_submission_data.copy()


import pandas as pd 
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import optuna

import warnings
warnings.filterwarnings("ignore")


train_df.head()


train_df.info()


train_df.isna().sum()


train_df.describe().T


num_cols = train_df.drop(columns=['id'], axis=1).select_dtypes(include=np.number).columns.tolist()
fig, axes = plt.subplots(nrows=1, ncols=len(num_cols), figsize=(10*len(num_cols), 10))
for i, col in enumerate(num_cols):
    sns.boxplot(y=train_df[col], ax=axes[i])
    axes[i].set_title(col)
    #plt.tight_layout()
plt.show()


corr = train_df.corr(numeric_only=True)
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, vmax=1.0, vmin=-1.0)
plt.title("Feature Correlation Matrix")
plt.show()


bpm_corr = corr["BeatsPerMinute"].sort_values(ascending=False)
print("Correlation with BeatsPerMinute:\n", bpm_corr)


target_col = 'BeatsPerMinute'


# Will not be used.
def iqr_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return lower_bound, upper_bound


x = train_df.drop(columns=[target_col, 'id'])
y = train_df[target_col]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


def objective(trial):
    param = {
        "verbosity": 0,
        "objective": "reg:squarederror",
        "booster": "gbtree",
        "tree_method": "hist",  # fast, use "gpu_hist" if GPU available
        "lambda": trial.suggest_loguniform("lambda", 1e-3, 10.0),   # L2 reg
        "alpha": trial.suggest_loguniform("alpha", 1e-3, 10.0),     # L1 reg
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
        "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_uniform("gamma", 0, 5),
    }

    model = xgb.XGBRegressor(**param, random_state=42)
    model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=False)

    preds = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return rmse


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)   # try more trials for better results

print("Best trial:")
print(study.best_trial.params)


best_params = study.best_trial.params
best_model = xgb.XGBRegressor(**best_params)
best_model.fit(x_train, y_train)
y_pred = best_model.predict(x_test)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Final RMSE: {rmse:.2f}")


sample_df


test_df


sample_df['BeatsPerMinute'] = best_model.predict(test_df.drop(columns=['id'], axis=1))
sample_df


sample_df.to_csv('submission.csv', index=False)




