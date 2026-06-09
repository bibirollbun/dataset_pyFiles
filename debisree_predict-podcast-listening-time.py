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
import seaborn as sns

import shap

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelBinarizer
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV



from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor

from tqdm import tqdm 

from sklearn.metrics import mean_squared_error, r2_score

from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings

warnings.simplefilter("ignore")
pd.options.mode.chained_assignment = None  


#Reading Data:

test_filepath = "/kaggle/input/playground-series-s5e4/test.csv"
train_filepath = "/kaggle/input/playground-series-s5e4/train.csv"

train = pd.read_csv(train_filepath)
print(train.shape)

test = pd.read_csv(test_filepath)
print(test.shape)


train.head()


train.isnull().sum()


test.isnull().sum()


# id:

train.set_index('id', inplace=True)
test.set_index('id', inplace=True)

train.head()


train_= train.drop_duplicates()
print(train_.shape)

test.drop_duplicates(inplace=True)
print(test.shape)


# Target Distribution:

train['Listening_Time_minutes'].describe()


sns.histplot(train['Listening_Time_minutes'], bins=50)
plt.title("Train Data Target Distribution")
plt.show()


def missing (df, col):
    df[col] = df.groupby('Podcast_Name')[col].transform(lambda x: x.fillna(x.median()))
    return 

missing(train, 'Episode_Length_minutes')
missing(test, 'Episode_Length_minutes')


sns.histplot(train['Episode_Length_minutes'], bins=50)
plt.title("Train Data Target Distribution")
plt.show()


# train['no_guest'] = train['Guest_Popularity_percentage'].isnull().astype(int)
# test['no_guest'] = test['Guest_Popularity_percentage'].isnull().astype(int)




train['Guest_Popularity_percentage'] = train['Guest_Popularity_percentage'].fillna(0)
test['Guest_Popularity_percentage'] = test['Guest_Popularity_percentage'].fillna(0)



train.isnull().sum()


train.dropna(inplace= True)
train.isnull().sum()


test.isnull().sum()


time = train.groupby('Genre')['Listening_Time_minutes'].mean().reset_index()


#plot:
sns.barplot(x='Genre', y='Listening_Time_minutes', data = time)
plt.xticks(rotation = 45)
plt.ylabel('Mean Listening Time for each Genre')
plt.show()


time = train.groupby('Genre')['Episode_Length_minutes'].mean().reset_index()


#plot:
sns.barplot(x='Genre', y='Episode_Length_minutes', data = time)
plt.xticks(rotation = 45)
plt.ylabel('Mean Episode minutes for each Genre')
plt.show()


train['Podcast_Name'].value_counts()


train['episode_count'] =train.groupby('Podcast_Name')['Episode_Title'].transform('count')
train.head()

test['episode_count'] =test.groupby('Podcast_Name')['Episode_Title'].transform('count')



ep = train.groupby(['Genre','episode_count'] )['Listening_Time_minutes'].mean().reset_index()
#ep


#plot:
sns.scatterplot( x='episode_count', y = 'Listening_Time_minutes', hue = 'Genre', data = ep)
plt.xticks(rotation = 45)
plt.ylabel('')
plt.show()


sns.boxplot(x="Genre", y="Listening_Time_minutes", data=train, )
plt.xticks(rotation = 45)
plt.show()


sns.boxplot(x="Publication_Day", y="Listening_Time_minutes", data=train, )
plt.show()


sns.boxplot(x="Publication_Time", y="Listening_Time_minutes", data=train )
plt.show()


sns.boxplot(x="Genre", y="Number_of_Ads", data=train, )
plt.xticks(rotation = 45)
plt.show()


xx= train.groupby(['Genre', 'Publication_Day',"Publication_Time"] )["Listening_Time_minutes"].mean()
xx.head()


sns.boxplot(x="Episode_Sentiment", y="Listening_Time_minutes", data=train )
plt.xticks(rotation = 45)
plt.show()


train.head()


train_encoded = train.copy()
test_encoded = test.copy()

for col in ['Podcast_Name',   'Episode_Title', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']:
    freq = train[col].value_counts(normalize=True)
    # Map the frequency encoding for train data
    train_encoded[col] = train[col].map(freq)
    test_encoded[col] = test[col].map(freq)


train_encoded.head()


# Linear correlation:



#  Compute correlation matrix
corr_matrix = train_encoded.corr()

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix - Features with the target")
plt.show()


#Delete column:

train_encoded.drop('episode_count', axis=1, inplace = True)
test_encoded.drop('episode_count', axis=1, inplace = True)


train_encoded.head()


# train-test split

target = "Listening_Time_minutes"
X = train_encoded.drop(columns= target)


y = train_encoded[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#y_train_log = np.log1p(y_train+1)  




# Assuming X and y are your features and numerical target
kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_best_iterations = []
oof_train_rmse = []
oof_val_rmse = []

# Loop through folds and use early stopping to capture the optimal boosting rounds
for fold, (train_index, val_index) in enumerate(kf.split(X), start=1):
    X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
    y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
    
    # Define the XGBoost regressor with early stopping enabled
    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=10,
        min_child_weight=4,
        colsample_bytree=0.66,
        subsample=0.9,
        gamma=1.6,
        reg_alpha=5.5,
        reg_lambda=8,
        eval_metric="rmse",
        early_stopping_rounds=100,
        random_state=42,
        tree_method="hist",
        verbosity=0
    )
    
    # Fit the model using the validation fold for early stopping
    model.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        verbose=False
    )
    
    # Record the best iteration from early stopping
    best_iter = model.best_iteration
    fold_best_iterations.append(best_iter)
    
    # Compute RMSE on training and validation folds
    train_rmse = np.sqrt(mean_squared_error(y_train_fold, model.predict(X_train_fold)))
    val_rmse = np.sqrt(mean_squared_error(y_val_fold, model.predict(X_val_fold)))
    oof_train_rmse.append(train_rmse)
    oof_val_rmse.append(val_rmse)
    
    print(f"Fold {fold}: Best Iteration = {best_iter}, Train RMSE = {train_rmse:.4f}, Val RMSE = {val_rmse:.4f}")

# Calculate the mean best iteration from all folds
mean_best_iter = int(np.mean(fold_best_iterations))
print(f"\nMean Best Iteration from CV: {mean_best_iter}")

# Optionally, you can also review average RMSEs across folds:
print(f"Average Train RMSE: {np.mean(oof_train_rmse):.4f}")
print(f"Average Validation RMSE: {np.mean(oof_val_rmse):.4f}")

# Retrain final model on the entire dataset using the average best iteration
final_model = XGBRegressor(
    n_estimators=mean_best_iter,
    learning_rate=0.03,
    max_depth=10,
    min_child_weight=4,
    colsample_bytree=0.66,
    subsample=0.9,
    gamma=1.6,
    reg_alpha=5.5,
    reg_lambda=8,
    eval_metric="rmse",
    random_state=42,
    tree_method="hist",
    verbosity=0
)

final_model.fit(X_train, y_train)




model_rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)

model_rf.fit(X_train, y_train)


# Predict with each model
pred_xgb = final_model.predict(X_test)
pred_rf = model_rf.predict(X_test)



# Weighted average
y_pred_blend = 1.0 * pred_xgb + 0.0 * pred_rf 

rmse_blend = mean_squared_error(y_test, y_pred_blend, squared=False)
print(f"Blended Model RMSE: {rmse_blend:.4f}")


xgb_pred = final_model.predict(test_encoded)
rf_pred = model_rf.predict(test_encoded)
#test_pred = np.expm1(test_pred_log) 

# Weighted average
test_pred = 1.0 * xgb_pred + 0.0 * rf_pred 

# Clip negative values to 0
test_pred = np.maximum(0, test_pred)

submission = pd.DataFrame({'id': test.index, 'Listening_Time_minutes': test_pred})
submission.to_csv('/kaggle/working/submission.csv', index=False)
display(submission)


#Test Prediction values 


plt.figure(figsize=(6,4))
plt.hist(test_pred, bins=100)
plt.title("Test Predictions")
plt.show()


test_pred.min()




