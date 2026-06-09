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


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


train_df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
train_extra_df = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
test_df  = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')



train_df.head()


train_df.describe()


test_df.info()


test_df.head()


train_extra_df.shape,train_df.shape


train_df = pd.concat([train_extra_df, train_df], axis=0).reset_index(drop=True)
train_df.shape


train_df=train_df[:4000000]


import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

# Create a figure
fig = plt.figure(figsize=(15, 5))
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.4)  # Adjust spacing

# Define the subplots
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

# Plot boxplots
sns.boxplot(x=train_df["Price"], color='blue', ax=ax1)
ax1.set_title("Boxplot of Price")

sns.boxplot(x=train_df["Compartments"], color='green', ax=ax2)
ax2.set_title("Boxplot of Compartments")

sns.boxplot(x=train_df["Weight Capacity (kg)"], color='red', ax=ax3)
ax3.set_title("Boxplot of Weight Capacity")

plt.show()



categorical_features = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]

# Defining figure size
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 18))
axes = axes.flatten()  # Flattening to iterate easily

# Plot count plots for categorical variables
for i, col in enumerate(categorical_features):
    sns.countplot(x=train_df[col], palette="coolwarm", ax=axes[i])
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)
    axes[i].set_ylabel("Count")
    axes[i].set_title(f"Count of {col}")

plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
corr = train_df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()


if train_df.isnull().values.any():
    plt.figure(figsize=(12, 6))
    sns.heatmap(train_df.isnull(), cmap="viridis", cbar=False, yticklabels=False)
    plt.title("Missing Values Heatmap")
    plt.show()
else:
    print("No missing values in the dataset.")


train_df.drop(columns=['id'], inplace=True)
test_df.drop(columns=['id'], inplace=True)


test_df.isnull().sum()


train_df.isnull().sum()


import pandas as pd

def feature_engineering(df):

    # Converting Yes/No to binary
    df["Waterproof"] = df["Waterproof"].map({"Yes": 1, "No": 0})
    df["Laptop Compartment"] = df["Laptop Compartment"].map({"Yes": 1, "No": 0})

    # Creating new features based on numerical relationships
    df["Compartments_per_Weight"] = df["Compartments"] / (df["Weight Capacity (kg)"] + 1e-5)  # Avoid division by zero
    df["Weight_per_Compartment"] = df["Weight Capacity (kg)"] / (df["Compartments"] + 1e-5)

    return df

# Applying feature engineering to train_df and test_df
train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)


train_df.isnull().sum()


train_df.columns


def preprocess_data(df, median_weight):
    categorical_cols = ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment', 
                        'Waterproof', 'Style', 'Color']

    # Filling missing categorical values and converting to category type
    df[categorical_cols] = df[categorical_cols].fillna('None').astype('category')

    # Filling missing numerical values and creating a categorical version
    df['Weight Capacity (kg) categorical'] = df['Weight Capacity (kg)'].fillna(median_weight).astype(str)
    df['Weight Capacity (kg)'] = df['Weight Capacity (kg)'].fillna(median_weight).astype(float)

    return df

# Computing median once for efficiency
median_weight = train_df['Weight Capacity (kg)'].median()

# Applying preprocessing to train and test sets
train_df = preprocess_data(train_df, median_weight)
test_df = preprocess_data(test_df, median_weight)



train_df.isnull().sum()


y = train_df['Price'] 
train_df = train_df.drop(['Price'],axis=1)
X = train_df
X_test = test_df


from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from sklearn.model_selection import KFold
import gc

cat_cols = ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment','Waterproof', 'Style', 'Color', 'Weight Capacity (kg) categorical']



from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Defining RMSE function
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# CatBoost parameters
catboost_params = {
    'task_type': 'GPU',
    'learning_rate': 0.062,
    'l2_leaf_reg': 7,
    'depth': 6,
    'iterations': 500,
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'random_seed': 42,
    'early_stopping_rounds': 200,
    'verbose': 100
}

# Cross-validation setup
n_splits = 2
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Ensuring categorical features are strings
train_df[cat_cols] = train_df[cat_cols].astype(str)
test_df[cat_cols] = test_df[cat_cols].astype(str)

# Ensuringy is a numpy array
y = np.array(y)

# Creating a test pool (used for predictions across folds)
X_test_pool = Pool(test_df, cat_features=cat_cols)

def cross_validate_catboost(X, y, X_test_pool, cat_cols, params, kf):
    scores = []
    test_preds = np.zeros((X_test_pool.num_row(), kf.get_n_splits()))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):  # Starting fold from 1
        print(f"Training Fold {fold}/{n_splits}...")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_pool = Pool(X_train, y_train, cat_features=cat_cols)
        valid_pool = Pool(X_val, y_val, cat_features=cat_cols)

        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True, verbose=100)

        val_pred = model.predict(valid_pool)
        score = rmse(y_val, val_pred)
        scores.append(score)

        test_preds[:, fold - 1] = model.predict(X_test_pool)  # Store predictions

        print(f"Fold {fold} RMSE: {score:.4f}")

    return scores, test_preds

# Running cross-validation
scores, test_preds = cross_validate_catboost(train_df, y, X_test_pool, cat_cols, catboost_params, kf)

# Final evaluation
print(f'\nCross-validated RMSE: {np.mean(scores):.3f} ± {np.std(scores):.3f}')
print(f'Max RMSE: {np.max(scores):.3f}, Min RMSE: {np.min(scores):.3f}')

# Final test prediction (average across folds)
final_test_pred = test_preds.mean(axis=1)



test_preds.shape


test_preds


final_test_pred = np.mean(test_preds, axis=1) 


test_df['Price'] = final_test_pred


sub_df=pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')


sub_df.head()


sub_df['Price']=test_df['Price']
sub_df.to_csv('submission.csv', index=False)




