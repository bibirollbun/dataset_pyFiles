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

train_csv = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_csv = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
sample_submission_csv = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')


# for beautiful dataframes
from IPython.display import display

# for heatmaps
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
%matplotlib inline

# for density plots and histograms
import matplotlib.pyplot as plt

# for maths and other operations
import numpy as np


display(train_csv.head())
train_csv.shape, test_csv.shape


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
axes[1].set_xlabel('Test features + Completeness (%)', fontsize=12)
axes[1].set_ylabel('Entries, yellow=missing', fontsize=12)

# annotate test heatmap columns
for i in range(len(test_null.columns)):
    axes[1].text(i + 0.5, -0.5, f"{test_not_null.iloc[i]:.2f}", ha='center', va='bottom')

plt.tight_layout()
plt.show()


for col in train_csv.columns:
    print(col, ':', len(train_csv[col].unique()), '\t\t', train_csv[col].unique())


train_csv.info()


# grid for the countplots of the categorical values
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 20))

# switch between train/test
df = test_csv

# flatten axes array for easy iteration
axes = axes.flatten()

# define the categorical columns to inspect
cat_feats = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']

# create countplots
for i, col in enumerate(cat_feats):
    ax = axes[i]
    sns.countplot(x=df[col], ax=ax)

    # add numbers on top of bars
    for p in ax.patches:
        ax.annotate(str(int(p.get_height())), (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom', fontsize=10, color='black')

# remove the empty subplot
fig.delaxes(axes[7])

plt.tight_layout()
plt.show()


# grid for the countplots of the categorical values
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

# switch between train/test
df = train_csv

# flatten axes array for easy iteration
axes = axes.flatten()

# the continuous features to inspect
cont_feats = ['Compartments', 'Weight Capacity (kg)', 'Price']

# create countplots
for i, col in enumerate(cont_feats):
    ax = axes[i]

    # histogram
    df[col].hist(bins=15, density=True, stacked=True, color='teal', alpha=0.6, ax=ax)

    # density plot
    df[col].plot(kind='density', color='teal', ax=ax)

    # mean and median vertical lines
    mean = df[col].mean(skipna=True)
    median = df[col].median(skipna=True)
    ax.axvline(mean, color='r', label=f'Mean: {mean:.3f}')
    ax.axvline(median, color='y', label=f'Median: {median:.3f}')

    # labels and formatting
    ax.set(xlabel=col)
    ax.legend()

# remove the empty subplot
fig.delaxes(axes[3])

plt.tight_layout()
plt.show()


# make copies of our data
train_full = train_csv.copy()
test_full = test_csv.copy()

# infer the NaN with the most common categories
for col in cat_feats:
    test_full[col] = test_full[col].fillna(test_full[col].value_counts().idxmax())

# infer the NaN with the mean
train_full['Weight Capacity (kg)'] = train_full['Weight Capacity (kg)'].fillna(train_full['Weight Capacity (kg)'].mean(skipna=True))
test_full['Weight Capacity (kg)'] = test_full['Weight Capacity (kg)'].fillna(test_full['Weight Capacity (kg)'].mean(skipna=True))

# drop the remaining train NaN just for now to see the possible improvements by inferring them
train_full = train_full.dropna(subset=['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color'])

# final check for null values
if (train_full.isnull().sum() == 0).all() and (test_full.isnull().sum() == 0).all():
    print('SUCCESS: Datasets clear from NaN values')
    print(train_full.shape, test_full.shape)
else:
    print('ERROR: Datasets NOT clear from NaN values')


test_id = test_full['id']

train_full = train_full.drop('id', axis=1)
test_full = test_full.drop('id', axis=1)


cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']

train_full = pd.get_dummies(
    data = train_full,
    columns = cols,
    drop_first = True,
    dtype = int
)

test_full = pd.get_dummies(
    data = test_full,
    columns = cols,
    drop_first = True,
    dtype = int
)


display(train_full.head())
display(test_full.head())


import numpy as np
from sklearn.metrics import make_scorer, mean_squared_error

# for some reason, importing root_mean_squared_error was not feasible, so, we'll create our own RMSE with mean_squared_error
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_scorer = make_scorer(rmse, greater_is_better=False)


from sklearn.model_selection import cross_val_score

# train data organization
X = train_full.drop('Price', axis=1)
y = train_full['Price']


from sklearn.linear_model import LinearRegression

model = LinearRegression()

scores = cross_val_score(model, X, y, cv=10, scoring=rmse_scorer)

print('K-fold cross-validation results:')
print(model.__class__.__name__ + " average RMSE is %.3f" % -scores.mean())


from xgboost import XGBRegressor

model = XGBRegressor(n_estimators=100, random_state=42)

scores = cross_val_score(model, X, y, cv=10, scoring=rmse_scorer)

print('K-fold cross-validation results:')
print(model.__class__.__name__ + " average RMSE is %.3f" % -scores.mean())


import optuna
from sklearn.model_selection import train_test_split

# sample data split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# switch Optuna logs on/off (INFO/CRITICAL)
optuna.logging.set_verbosity(optuna.logging.CRITICAL)

# define the objective for Optuna
def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 1, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5, log=True),
        'max_depth': trial.suggest_int('max_depth', 1, 15),
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
    return rmse(y_val, preds)

study = optuna.create_study(direction='minimize')
#study.optimize(objective, n_trials=50)

#best_params = study.best_params
#print("Best parameters: ", best_params)
#print("Best RMSE: ", study.best_value)


# visualize the results of the Optuna optimization process
optuna.visualization.plot_param_importances(study).show()


# best parameters obtained from a previous Optuna run
best_params = {'n_estimators': 294, 'learning_rate': 0.429371352497821, 'max_depth': 1, 'subsample': 0.8429291408373879, 'colsample_bytree': 0.8189642515155546, 'min_child_weight': 1, 'gamma': 4.017604897244401, 'lambda': 7.502113869795271, 'alpha': 3.4486909506531522}

model = XGBRegressor(random_state=42, **best_params)

# cross-validation
scores = cross_val_score(model, X, y, cv=10, scoring=rmse_scorer)

print('K-fold cross-validation results:')
print(model.__class__.__name__ + " average RMSE is %.3f" % -scores.mean())


# linear regression
model_1 = LinearRegression()
model_1.fit(X, y)

# XGBoost
model_2 = XGBRegressor(random_state=42, **best_params)
model_2.fit(X, y)


# get all the predictions and combine them
pred_1 = model_1.predict(test_full)
pred_2 = model_2.predict(test_full)
predictions = (pred_1+pred_2) / 2

# submission data
submission = pd.DataFrame({
    'id': test_id,
    'Price': pred_2
})

submission.to_csv("submission.csv", index=False)
submission.head()

