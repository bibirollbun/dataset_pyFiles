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

train_csv = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test_csv = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
sample_submission_csv = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')


# for beautiful dataframes
from IPython.display import display

# for Heatmaps and 
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
%matplotlib inline

# for density plots and histograms
import matplotlib.pyplot as plt

# for the Fourier features
import numpy as np

# to get more datasets
import requests


train_csv.head()


# missing values in the datasets
train_null = train_csv.isnull()
test_null = test_csv.isnull()

# percentage of missing values per feature in the training set
train_not_null = (len(train_csv) - train_null.sum()) / len(train_csv) *100
test_not_null = (len(test_csv) - test_null.sum()) / len(test_csv) *100

# create a single figure with subplots for both datasets
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# heatmap for missing values in the training dataset
sns.heatmap(train_null, cmap='viridis', cbar=False, yticklabels=False, ax=axes[0])
axes[0].set_xlabel('Training features + completeness (%)', fontsize=12)
axes[0].set_ylabel('Entries, yellow=missing', fontsize=12)

# annotate training heatmap columns
for i in range(len(train_null.columns)):
    axes[0].text(i + 0.5, -0.5, f"{train_not_null.iloc[i]:.2f}", ha='center', va='bottom')

# heatmap for missing values in the test dataset
sns.heatmap(test_null, cmap='viridis', cbar=False, yticklabels=False, ax=axes[1])
axes[1].set_xlabel('Test features + completeness (%)', fontsize=12)
axes[1].set_ylabel('Entries, yellow=missing', fontsize=12)

# annotate test heatmap columns
for i in range(len(test_null.columns)):
    axes[1].text(i + 0.5, -0.5, f"{test_not_null.iloc[i]:.2f}", ha='center', va='bottom')

plt.tight_layout()
plt.show()


train_full = train_csv.dropna(subset=['num_sold'])
#train_full = train_csv.fillna(method='bfill', inplace=False)#.fillna(method='bfill', inplace=False)
train_full.isnull().sum()


test_id = test_csv['id']

train_full = train_full.drop('id', axis=1)
test_full = test_csv.drop('id', axis=1)


# create new basic time-related features
df = train_full['date'].to_frame()
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['quarter'] = df['date'].dt.quarter
df['month'] = df['date'].dt.month
df['week'] = df['date'].dt.isocalendar().week.astype('int32')
df['weekday'] = df['date'].dt.weekday
df['day'] = df['date'].dt.day

# new seasonality features
df['year_sin'] = np.sin(2 * np.pi * df['year'] / 7.0) # seasonality over the 7 years in the dataset
df['year_cos'] = np.cos(2 * np.pi * df['year'] / 7.0)
df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)  # range of sin/cos input: [0, 2pi]
df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['day_sin'] = np.sin(2 * np.pi * df['day'] / 365) # annual cycle, every 180 days
df['day_cos'] = np.cos(2 * np.pi * df['day'] / 365)
df['day_sin2'] = np.sin(4 * np.pi * df['day'] / 365) # higher order harmonic, every 90 days
df['day_cos2'] = np.cos(4 * np.pi * df['day'] / 365)
df['day_sin3'] = np.sin(6 * np.pi * df['day'] / 365) # every 60 days
df['day_cos3'] = np.cos(6 * np.pi * df['day'] / 365)
df['day_sin4'] = np.sin(8 * np.pi * df['day'] / 365) # every 45 days
df['day_cos4'] = np.cos(8 * np.pi * df['day'] / 365)
df['day_sin_0.5'] = np.sin(1 * np.pi *df['day'] / 365)
df['day_cos_0.5'] = np.cos(1 * np.pi *df['day'] / 365)
df['group'] = (df['year'] - 2020) * 48 + df['month'] * 4 + df['day'] // 7

# drop 'date' and join the results
df = df.drop('date', axis=1)
train_full = train_full.drop('date', axis=1)
train_full = train_full.join(df)


# same for the test set
df = test_full['date'].to_frame()
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['quarter'] = df['date'].dt.quarter
df['month'] = df['date'].dt.month
df['week'] = df['date'].dt.isocalendar().week.astype('int32')
df['weekday'] = df['date'].dt.weekday
df['day'] = df['date'].dt.day

df['year_sin'] = np.sin(2 * np.pi * df['year'] / 7.0)
df['year_cos'] = np.cos(2 * np.pi * df['year'] / 7.0)
df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['day_sin'] = np.sin(2 * np.pi * df['day'] / 365)
df['day_cos'] = np.cos(2 * np.pi * df['day'] / 365)
df['day_sin2'] = np.sin(4 * np.pi * df['day'] / 365)
df['day_cos2'] = np.cos(4 * np.pi * df['day'] / 365)
df['day_sin3'] = np.sin(6 * np.pi * df['day'] / 365)
df['day_cos3'] = np.cos(6 * np.pi * df['day'] / 365)
df['day_sin4'] = np.sin(8 * np.pi * df['day'] / 365)
df['day_cos4'] = np.cos(8 * np.pi * df['day'] / 365)
df['day_sin_0.5'] = np.sin(1 * np.pi *df['day'] / 365)
df['day_cos_0.5'] = np.cos(1 * np.pi *df['day'] / 365)
df['group'] = (df['year'] - 2020) * 48 + df['month'] * 4 + df['day'] // 7

df = df.drop('date', axis=1)
test_full = test_full.drop('date', axis=1)
test_full = test_full.join(df)


# get the OHEs for the following variables
cols = ['country', 'store', 'product', 'month', 'weekday']

# get the train OHEs
train_full = pd.get_dummies(
    data = train_full,
    columns=cols,
    drop_first=True,
    dtype=int
)

# get the test OHEs
test_full = pd.get_dummies(
    data = test_full,
    columns=cols,
    drop_first=True,
    dtype=int
)


train_full['num_sold'] = np.log1p(train_full['num_sold'])


from sklearn.metrics import make_scorer, mean_absolute_percentage_error

# create the scorer to pass for the cross-validation
mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)


'''from sklearn.feature_selection import RFECV
from xgboost import XGBRegressor

cols = [c for c in train_full.columns if c != 'num_sold']
X = train_full[cols]
y = train_full['num_sold']

rfecv = RFECV(estimator=XGBRegressor(n_estimators=500, random_state=42), step=1, min_features_to_select=38, cv=10, scoring=mape_scorer)
rfecv.fit(X, y)

# summarize the selection of attributes
print('Optimal number of features: %d' % rfecv.n_features_)
print('Features kept: ', X.columns[rfecv.support_])
print('Features removed: ', X.columns[~rfecv.support_])

# Plot number of features VS. average cross-validation scores with error bars
plt.figure(figsize=(10,6))
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation average score")
plt.errorbar(range(1, len(rfecv.cv_results_['mean_test_score']) + 1),
             rfecv.cv_results_['mean_test_score'],
             yerr=rfecv.cv_results_['std_test_score'],
             fmt='-o')
plt.show()'''


# train_full = train_full.drop(X.columns[~rfecv.support_], axis=1)
# test_full = test_full.drop(X.columns[~rfecv.support_], axis=1)


display(train_full.head())
display(test_full.head())


from sklearn.model_selection import cross_val_score

# train data organization for cross-validation
X = train_full.drop('num_sold', axis=1)
y = train_full['num_sold']


from sklearn.linear_model import LinearRegression

model = LinearRegression()

scores = cross_val_score(model, X, y, cv=10, scoring=mape_scorer)

print('K-fold cross-validation results:')
print(model.__class__.__name__ + " average MAPE is %.3f" % -scores.mean())


from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=500,
    random_state=42
)

scores = cross_val_score(model, X, y, cv=10, scoring=mape_scorer)

print('K-fold cross-validation results:')
print(model.__class__.__name__ + " average MAPE is %.3f" % -scores.mean())


import optuna
from sklearn.model_selection import train_test_split

# sample data split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# switch Optuna logs on/off (INFO/CRITICAL)
optuna.logging.set_verbosity(optuna.logging.CRITICAL)

# define the objective for Optuna
def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 1, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'gamma': trial.suggest_float('gamma', 0, 10),
        'lambda': trial.suggest_float('lambda', 0, 10),
        'alpha': trial.suggest_float('alpha', 0, 10)
    }

    # train the model with the sampled hyperparamters
    model = XGBRegressor(**param)
    model.fit(X_train, y_train)
    
    # evaluate the results
    preds = model.predict(X_val)
    return mean_absolute_percentage_error(y_val, preds)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

best_params = study.best_params
print("Best parameters: ", best_params)
print("Best MSE: ", study.best_value)


# visualize the results of the Optuna optimization process
optuna.visualization.plot_param_importances(study).show()


model = XGBRegressor(random_state=42, **best_params)

# cross-validation
scores = cross_val_score(model, X, y, cv=10, scoring=mape_scorer)

print('K-fold cross-validation results:')
print(model.__class__.__name__ + " average MAPE is %.3f" % -scores.mean())


%%time
model1 = XGBRegressor(random_state=42, **best_params)
model1.fit(X, y)


%%time
from lightgbm import LGBMRegressor 

model2 = LGBMRegressor(
    n_estimators=2698,
    learning_rate=0.08073011570606378,
    max_depth=14,
    reg_alpha=0.4709528600827254,
    reg_lambda=0.039462913868267044,
    min_child_samples=24,
    colsample_bytree=0.5718789118717338,
    subsample=0.7441516869335623,
    random_state=42
)
model2.fit(X, y)


%%time
from catboost import CatBoostRegressor

model3 = CatBoostRegressor(
    iterations=3126,
    learning_rate=0.03695190923929497,
    depth=8,
    l2_leaf_reg=0.024979837379690215,
    border_count=88,
    subsample=0.9327787012863534,
    random_strength=5.234080751473561,
    eval_metric='MAPE',
    random_seed=42,
    verbose=0
)
model3.fit(X, y)


'''# model instantiation
model = XGBRegressor(random_state=42, **best_params)

# train and prediction, bringing the target back to the non-logarithm state
model.fit(X, y)
test_full['num_sold'] = np.expm1(model.predict(test_full))

# submission crafting
test_full['id'] = test_id
submission = test_full[['id', 'num_sold']]

submission.to_csv("submission.csv", index=False)
submission.head()'''


pred1 = model1.predict(test_full)
pred2 = model2.predict(test_full)
pred3 = model3.predict(test_full)
predictions = (pred1+pred2+pred3) / 3

predictions = np.expm1(predictions)

submission = pd.DataFrame({
    'id': test_id,
    'num_sold': predictions
})

submission.to_csv("submission.csv", index=False)
submission.head()

