!pip install autogluon
!pip install scikit-learn==1.3.0


# import relevant libraries
import pandas as pd
import numpy as np
import seaborn as sns
import optuna
import matplotlib.pyplot as plt
import plotly.express as px
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import KFold, cross_val_score
# from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
import cudf
from autogluon.tabular import TabularPredictor
import category_encoders as ce


# read in data
train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
train.drop(columns = ['id'], inplace = True)


# inspect missing data 
train.isna().sum()/len(train) * 100


plt.figure(figsize=(8, 5))
sns.heatmap(train.isnull(), cmap='viridis', cbar=False, yticklabels=False,)
plt.title("Missing Values Heatmap")
plt.show()


# dropped rows with missing data
# train.dropna(how='any', inplace = True)


# build a baseline first
baseline = train.groupby(['Brand', 'Size', 'Laptop Compartment','Waterproof'])['Price'].mean().reset_index()
# submission = pd.merge(test, baseline, how = "left", on =['Brand', 'Saize', 'Laptop Compartment','Waterproof'])[['id','Price']]
# submission.fillna(train["Price"].mean(), inplace = True)
# submission.to_csv("submission.csv", index=False)
# the baseline score resulted in a poor rmse of 39.16153 and a finsih of 20 percentile 
# the baseline score didn't change after dropping the null rows


# the data appears to be mmissing at random hence we can try different approaches 
# 1. impute
# 2. drop - can't be considered as it leads to around ~15% of the data being dropped
# 3. let the algos handle the missing data first and then use 1 & 2


# FE
# 1. create weight buckets for bags
# 2. understanding relationships between different variables with price
# let's use xgboost after that


# how to create bins like if variable between 5-10 then 5-10 else 
# Define bins and labels
bins = [5, 10, 15, 20, 25, 30]  # Bin edges
labels = ["5-10", "10-15", "15-20", "20-25","25-30"]  # Labels for each bin
train['Weight Range'] = pd.cut(train['Weight Capacity (kg)'], bins=bins, labels=labels, right=False)
# sns.histplot(train['Weight Capacity (kg)'])
train


# understanding relationship between pricing of different brands
viz_df = train.groupby(['Brand','Weight Range'])['Price'].mean().reset_index()
fig = px.line(viz_df, y = 'Price', x = 'Weight Range', color = 'Brand')
fig.show()


# can we add new random features
for col in train.columns :
    print(train[col].unique())
    fig = px.histogram(train[col])
    fig.show()


# can clearly see that everything is evenly distributed except where Price > 149
# let's do a deep dive
for col in train.columns :
    fig = px.histogram(train[train['Price'] > 149][col])
    fig.show()


# there seems to be no visible relationship bw any other variable and the Price variable where Price > 149
# we can create another variable Price_high
# train['Price_high'] = (train['Price'] > 149).astype(int)
# there's no point of creating this variable as it can' be created using the test data 


# Create target encoded variables for all cat variables 
cat_col = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Style', 'Color', 'Waterproof']

for col in cat_col : 
    encoder = ce.LeaveOneOutEncoder(cols=[f'{col}'])
    # Fit and Transform
    train[f'{col}_te'] = encoder.fit_transform(train[f'{col}'], train[f'Price'])
    test[f'{col}_te'] = encoder.transform(test[f'{col}'])
    train[col] = train[col].astype('category')
    # test.drop(columns=[f'{col}'], inplace=True)
    # train.drop(columns=[f'{col}'], inplace=True)


# let's impute the values - don't need to do this with target encoding for cat_col
# cat_col = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Style', 'Color', 'Waterproof']
# for col in cat_col :
#     train[col] = train[col].astype('category') 
#     train[col] = train[col].cat.add_categories(["Unknown"])
#     train[col] = train[col].fillna("Unknown")


# Apply mode for cont variables
con_col = ['Compartments', 'Weight Capacity (kg)']
for col in con_col : 
    train[col] = train[col].fillna(train[col].mode()[0])


# let's build a model
# xgboost - without data imputation

# y = train['Price']
# train = train[train['Price'] < 149]

y = train['Price']
X = train.drop(columns = ['Price', 'Weight Range'])
X = cudf.DataFrame.from_pandas(X)
y = cudf.Series(y).to_numpy()


def objective(trial):
     
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'enable_categorical': True,
        'tree_method': 'hist',
        'device' : 'cuda'
    }
    kf = KFold(n_splits=5, shuffle=True, random_state=42, stratified = True)
    model = XGBRegressor(**params)
    scores = cross_val_score(model, X, y, cv = kf, scoring = 'neg_root_mean_squared_error')
    rmse_scores = -scores
    return np.mean(rmse_scores)


# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=25)

# # Best hyperparameters
# print("Best RMSE:", study.best_value)
# print("Best parameters:", study.best_params)

# best_params = study.best_params


# based on our optuna testing these below are the best params
# params = {'n_estimators': 500, 
#           'max_depth': 3, 
#           'learning_rate': 0.012874356543324734, 
#           'subsample': 0.729710486992358, 
#           'colsample_bytree': 0.919732683889473, 
#           'reg_alpha': 0.026475881641569287, 
#           'reg_lambda': 0.007339860585423517}


# params.update({'enable_categorical': True,
#         'tree_method': 'hist',
#         'device' : 'cuda',
#         'predictor': 'gpu_predictor'})


# # # these params have been achieved after removing all Price values > 149  
best_params = {'n_estimators': 1000, 
               'max_depth': 10, 
               'learning_rate': 0.05243983795434151, 
               'subsample': 0.9270575266283493, 
               'colsample_bytree': 0.8721558198117955, 
               'reg_alpha': 0.02202954887830654, 
               'reg_lambda': 0.6088046253648468}


best_params.update({'enable_categorical': True,
        'tree_method': 'hist',
        'device' : 'cuda',
        'predictor': 'gpu_predictor'})


# Using autogluon 
# predictor = TabularPredictor(label='Price', eval_metric = 'root_mean_squared_error').fit(train, presets='best_quality', 
#                                                                                          hyperparameters={'GBM': {'tree_method': 'gpu_hist'},  # Use GPU-optimized LightGBM
#                                                                                                           'XGB': {'device': 'cuda'}})  # Enable GPU for XGBoost


# model = XGBRegressor(**params)
model = XGBRegressor(**best_params)
model.fit(X, y)


# applying transformations before predictions
# test['Weight Range'] = pd.cut(test['Weight Capacity (kg)'], bins=bins, labels=labels, right=False)

for col in cat_col :
    test[col] = test[col].astype('category') 
#     test[col] = test[col].cat.add_categories(["Unknown"])
#     test[col] = test[col].fillna("Unknown")

for col in con_col : 
    test[col] = test[col].fillna(test[col].mode()[0])


# for col in cat_col : 
#     encoder = ce.LeaveOneOutEncoder(cols=[f'{col}'])
#     # Fit and Transform
#     test[f'{col}_te'] = encoder.transform(test[f'{col}'])
#     test.drop(columns=[f'{col}'], inplace=True)


pred = model.predict(test.drop(columns =['id']))
# Make predictions using autogluon
# pred = predictor.predict(test)


# Convert the array into a pandas Series
pred = pd.Series(pred , name='Price')

# Concatenate the DataFrame column and the Series
submission = pd.concat([test['id'], pred], axis=1)

submission.to_csv("submission.csv", index = False)

