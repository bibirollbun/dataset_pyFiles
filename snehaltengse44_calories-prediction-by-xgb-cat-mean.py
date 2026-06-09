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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from sklearn.compose import ColumnTransformer
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import  cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import math
import seaborn as sns


train = pd.read_csv('/kaggle/input/calories-prediction/train.csv')
test = pd.read_csv('/kaggle/input/calories-prediction/test.csv')


train.describe()


"""def outliers(df):
    columns =['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate',
       'Body_Temp']
    for col in columns:
        q3 = df[col].quantile(0.75)
        q1 = df[col].quantile(0.25)
        IQR = q3-q1
        upper = q3+1.5*IQR
        lower = q3-1.5*IQR
        df[col] = np.where(df[col]>upper,upper,
                              np.where(df[col]<lower,lower,df[col]))


outliers(train)
outliers(test)
    """


sns.scatterplot(data= train, x='Heart_Rate', y='Calories', hue='Sex')
plt.xlabel('Heart_rate')
plt.ylabel('Calories')


sns.scatterplot(data= train, x='Weight', y='Calories', hue='Sex')
plt.xlabel('Weight')
plt.ylabel('Calories')


sns.scatterplot(data= train, x='Duration', y='Calories', hue='Sex')
plt.xlabel('Weight')
plt.ylabel('Calories')


plt.hist(train['Calories'],bins=10)


sns.scatterplot(data= train, x='Height', y='Calories', hue='Sex')
plt.xlabel('Height')
plt.ylabel('Calories')


#feature engineering 
# Add new features
def add_new_features(df):
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['Intensity'] = df["Heart_Rate"] / (df["Duration"] + 1e-5)
    df['Body_Temp_Duration'] = df['Body_Temp'] * df['Duration']
    df['Weight_Heart_Rate'] = df['Weight'] * df['Heart_Rate']
    
    return df

train = add_new_features(train)
test = add_new_features(test)





train.describe()


x_train, x_test, y_train,y_test= train_test_split(train.drop(columns =['Calories', 'id'], axis = 1),train['Calories'], test_size =0.3, random_state=42)


print(x_train.columns)
x_train.shape


test_drop_id = test.drop(columns ='id')


ohe = OneHotEncoder(sparse_output=False)
trf1 = ColumnTransformer(transformers=[
    ('ohe',ohe, ['Sex'])
], remainder = 'passthrough')


ss = StandardScaler()
trf2 = ColumnTransformer(transformers=[
    ('ss',ss,slice(0,9))
], remainder = 'passthrough')


pipe=Pipeline([
    ('trf1',trf1),
    ('trf2',trf2)
])


x_train_transform = pipe.fit_transform(x_train)
x_test_transform = pipe.transform(x_test)
test_transform = pipe.transform(test_drop_id)


"""test_transform =  pd.DataFrame(test_transform, columns=['Sex_1', 'Sex_2', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp',
       'BMI', 'Intensity', 'Calories_Burned']
)

x_test_transform = pd.DataFrame(x_test_transform, columns=[ 'Sex_1', 'Sex_2','Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp',
       'BMI', 'Intensity', 'Calories_Burned']
)

x_train_transform = pd.DataFrame(x_train_transform, columns=[ 'Sex_1', 'Sex_2','Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp',
       'BMI', 'Intensity', 'Calories_Burned']
)"""


test_transform =  pd.DataFrame(test_transform, columns=['Sex_1', 'Sex_2', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
)

x_test_transform = pd.DataFrame(x_test_transform, columns=[ 'Sex_1', 'Sex_2','Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
)

x_train_transform = pd.DataFrame(x_train_transform, columns=[ 'Sex_1', 'Sex_2','Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
)


test_transform =  pd.DataFrame(test_transform)

x_test_transform = pd.DataFrame(x_test_transform)

x_train_transform = pd.DataFrame(x_train_transform)


x_train_transform


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_log_error
import optuna
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_log_error
import optuna
import numpy as np

"""def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))"""

def rmsle(y_true, y_pred):
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

def objective(trial):
    # Hyperparameters to optimize
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),            # Number of boosting rounds
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),        # Step size shrinkage
        'max_depth': trial.suggest_int('max_depth', 3, 10),                       # Maximum tree depth
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),                  # Row sampling ratio
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),    # Feature sampling ratio
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),                  # L1 regularization
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),                # L2 regularization
    }

    xgb = XGBRegressor(
        **params,
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='rmse'
    )

    xgb.fit(x_train_transform, y_train)
    y_test_pred = xgb.predict(x_test_transform)

    score = rmsle(y_test,y_test_pred)
    return score


study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())  # We aim to maximize accuracy
study.optimize(objective, n_trials=100)  # Run 50 trials to find the best hyperparameters


.save_model("xgb.json")


from sklearn.metrics import accuracy_score

# Train a RandomForestClassifier using the best hyperparameters from Optuna
best_xgb = XGBRegressor(**study.best_trial.params, random_state=42)

# Fit the model to the training data
best_xgb.fit(x_train_transform, y_train)
y_test_pred_xgb = best_xgb.predict(x_test_transform)

score = rmsle(y_test,y_test_pred_xgb)
print(score)


y_pred_xgb = best_xgb.predict(test_transform)


customer_ids = test['id']  # Change 'customer_id' to the correct column name if different

# Step 2: Create a DataFrame with predictions
xgb = pd.DataFrame({
    'id': customer_ids,
    'Calories': y_pred_xgb 
})

# Step 3: Save to CSV
xgb.to_csv('submission_xgb_100.csv', index=False)


from sklearn.model_selection import KFold
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_log_error
import optuna

def rmsle(y_true, y_pred):
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 250, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'depth': trial.suggest_int('depth', 3, 6),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 3, 5),
        'loss_function': 'RMSE',
        'random_seed': 42,
        'eval_metric': 'RMSE',
        'early_stopping_rounds': 200,
        'verbose': 100,
        'task_type': 'CPU'
    }
    cat = CatBoostRegressor(**params)
    
    cat.fit(x_train_transform, y_train)
    y_test_pred = cat.predict(x_test_transform)

    score = rmsle(y_test,y_test_pred)
    return score



study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=80)

print("Trial:", study.best_trial.number)
print("RMSLE:", study.best_trial.value)


from sklearn.metrics import accuracy_score
best_cat = CatBoostRegressor(**study.best_trial.params, random_state=42)

best_cat.fit(x_train_transform, y_train)
y_test_pred = best_cat.predict(x_test_transform)

score = rmsle(y_test,y_test_pred)
print(score)


y_test_cat = best_cat.predict(test_transform)


customer_ids = test['id']  # Change 'customer_id' to the correct column name if different

# Step 2: Create a DataFrame with predictions
output_df = pd.DataFrame({
    'id': customer_ids,
    'Calories': y_test_cat
})



num_negatives = (output_df['Calories'] < 0).sum()
print(f"Number of negative values replaced: {num_negatives}")



output_df['Calories'] = output_df['Calories'].apply(lambda x: max(x, 0))



# Step 3: Save to CSV
output_df.to_csv('submission_cat.csv', index=False)


from lightgbm import LGBMRegressor

def rmsle(y_true, y_pred):
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 75, 150),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 15),
        'random_state': 42,  # Changed from random_seed to random_state for LGBMRegressor
        'verbose': -1,  # Changed from 100 to -1 to suppress output during optimization
        'colsample_bytree': 0.5979737441060009,
        'reg_alpha': 0.001975258376030875,
        'reg_lambda': 0.005106256873241264,
        'max_bin': 2**10,
    }
    
    # Remove early_stopping_rounds from params since we're not using validation here
    lgb = LGBMRegressor(**params)
    
    # Fit without early stopping
    lgb.fit(x_train_transform, y_train)
    
    # Predict and calculate score
    y_test_pred = lgb.predict(x_test_transform)
    score = rmsle(y_test, y_test_pred)
    
    return score
    
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

print("Trial:", study.best_trial.number)
print("RMSLE:", study.best_trial.value)


best_lgb = LGBMRegressor(**study.best_trial.params, random_state=42)

# Fit the model to the training data
best_lgb.fit(x_train_transform, y_train)
y_test_pred_lgb = best_lgb.predict(x_test_transform)

score = rmsle(y_test,y_test_pred_xgb)
print(score)


lgb_pred= best_lgb.predict(test_transform)
lgb_pred = pd.DataFrame({
    'id': customer_ids,
    'Calories': lgb_pred
})



lgb_pred


final_preds = np.stack([df['Calories'] for df in [xgb,output_df,lgb_pred]], axis=1)


final_preds = pd.DataFrame(final_preds)


final_preds


final_preds
mean_calories = np.mean(final_preds, axis=1)


customer_ids = test['id']  # Change 'customer_id' to the correct column name if different

# Step 2: Create a DataFrame with predictions
final_sub = pd.DataFrame({
    'id': customer_ids,
    'Calories': mean_calories
})


final_sub.to_csv('submission_3_mean.csv', index=False)




