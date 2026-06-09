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


from IPython.display import display

display(train_csv.head())
print(f"Training set shape: {train_csv.shape}\nTest set shape: {test_csv.shape}")



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_missing_heatmap(ax, data, xlabel, ylabel):
    # Compute missing mask and completeness percentages
    null_mask = data.isnull()
    completeness = ((len(data) - null_mask.sum()) / len(data) * 100).round(2)

    # Plot heatmap
    sns.heatmap(null_mask, cmap='viridis', cbar=False, yticklabels=False, ax=ax)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Set x-tick labels to include both the feature name and its completeness percentage
    new_labels = [f"{col}\n({perc}%)" for col, perc in completeness.items()]
    ax.set_xticks(np.arange(len(data.columns)) + 0.5)
    ax.set_xticklabels(new_labels, fontsize=10, rotation=45, ha='right')

# Define dataset details as a list of tuples: (DataFrame, xlabel, ylabel)
datasets = [
    (train_csv, "Training features + completeness (%)", "Entries, yellow=missing"),
    (test_csv, "Test features + Completeness (%)", "Entries, yellow=missing")
]

# Create subplots and plot each dataset
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
for ax, (data, xlabel, ylabel) in zip(axes, datasets):
    plot_missing_heatmap(ax, data, xlabel, ylabel)

plt.tight_layout()
plt.show()



# Optimized version using a one-liner list comprehension
print("\n".join(
    f"{col}: {len(train_csv[col].unique())}\t\t {train_csv[col].unique()}"
    for col in train_csv.columns
))



train_csv.info()


import matplotlib.pyplot as plt
import seaborn as sns
import math

def annotate_bars(ax):
    """Annotate each bar in the countplot with its height value."""
    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(f"{int(height)}",
                    (patch.get_x() + patch.get_width() / 2, height),
                    ha='center', va='bottom', fontsize=10, color='black')

# Choose the DataFrame (train_csv or test_csv)
df = test_csv

# Define the categorical columns to inspect
cat_feats = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']

# Dynamically calculate grid dimensions: 2 columns, rows as needed
n_cols = 2
n_rows = math.ceil(len(cat_feats) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 5))
axes = axes.flatten()

# Create countplots with annotations
for ax, col in zip(axes, cat_feats):
    sns.countplot(x=df[col], ax=ax)
    annotate_bars(ax)
    ax.set_title(col)

# Remove any unused subplots
for ax in axes[len(cat_feats):]:
    fig.delaxes(ax)

plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import math

def plot_continuous_feature(ax, data, feature):
    """Plots histogram and density curve with mean/median annotations."""
    sns.histplot(data[feature], bins=15, kde=True, color='teal', alpha=0.6, ax=ax)

    # Compute mean & median
    mean, median = data[feature].mean(), data[feature].median()
    
    # Add vertical lines
    ax.axvline(mean, color='r', linestyle="--", lw=2, label=f'Mean: {mean:.2f}')
    ax.axvline(median, color='y', linestyle="-.", lw=2, label=f'Median: {median:.2f}')
    
    ax.set_xlabel(feature, fontsize=12)
    ax.legend()

# Select DataFrame (train_csv or test_csv)
df = train_csv

# Continuous features list
features = ['Compartments', 'Weight Capacity (kg)', 'Price']

# Grid layout: 2 columns, rows calculated dynamically
cols = 2
rows = math.ceil(len(features) / cols)
fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 5))

# Flatten axes for easy iteration
axes = np.ravel(axes)

# Plot each feature
for ax, feature in zip(axes, features):
    plot_continuous_feature(ax, df, feature)

# Hide unused subplots
for ax in axes[len(features):]:
    ax.set_visible(False)

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


def one_hot_encode(df, columns, drop_first=True, dtype=int):
    """Apply one-hot encoding to the specified columns of a DataFrame."""
    # Optionally check for missing columns and warn the user
    missing = set(columns) - set(df.columns)
    if missing:
        print(f"Warning: The following columns are missing and will be ignored: {missing}")
        columns = [col for col in columns if col in df.columns]
    return pd.get_dummies(data=df, columns=columns, drop_first=drop_first, dtype=dtype)

# Define the categorical columns to encode
cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']

# Apply one-hot encoding to both datasets
train_full = one_hot_encode(train_full, cols)
test_full  = one_hot_encode(test_full,  cols)



display(train_full.head())
display(test_full.head())


import numpy as np
from sklearn.metrics import make_scorer, mean_squared_error

rmse_scorer = make_scorer(lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred)), greater_is_better=False)



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
from sklearn.model_selection import cross_val_score

model = XGBRegressor(n_estimators=100, random_state=42)
scores = cross_val_score(model, X, y, cv=10, scoring=rmse_scorer)

print("K-fold cross-validation results:")
print(f"{model.__class__.__name__} average RMSE is {-scores.mean():.3f}")



import optuna
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
optuna.logging.set_verbosity(optuna.logging.CRITICAL)

def objective(trial):
    params = {
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
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    return rmse(y_val, model.predict(X_val))

study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=50)



# visualize the results of the Optuna optimization process
optuna.visualization.plot_param_importances(study).show()


best_params = {
    'n_estimators': 294,
    'learning_rate': 0.429371352497821,
    'max_depth': 1,
    'subsample': 0.8429291408373879,
    'colsample_bytree': 0.8189642515155546,
    'min_child_weight': 1,
    'gamma': 4.017604897244401,
    'lambda': 7.502113869795271,
    'alpha': 3.4486909506531522
}

model = XGBRegressor(random_state=42, **best_params)
scores = cross_val_score(model, X, y, cv=10, scoring=rmse_scorer)
print(f"K-fold cross-validation results:\n{model.__class__.__name__} average RMSE is {-scores.mean():.3f}")



model_1 = LinearRegression().fit(X, y)
model_2 = XGBRegressor(random_state=42, **best_params).fit(X, y)


# Ensure test_id is a proper column by resetting its index
test_id = test_id.reset_index(drop=True)

submission = pd.DataFrame({
    'id': test_id,
    'Price': (model_1.predict(test_full) + model_2.predict(test_full)) / 2
})
submission.to_csv("submission.csv", index=False)
submission.head()


