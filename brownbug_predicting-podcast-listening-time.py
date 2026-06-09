import torch

print("Number of GPUs available:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


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


import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import optuna 

from xgboost import XGBRegressor


import warnings

warnings.simplefilter("ignore")


train_filepath = '/kaggle/input/playground-series-s5e4/train.csv'
test_filepath = '/kaggle/input/playground-series-s5e4/test.csv'

train = pd.read_csv(train_filepath)
print(train.shape)

test = pd.read_csv(test_filepath)
print(test.shape)


# train.head()
train.sample(n=10)


test.head()


train.isnull().sum()


test.isnull().sum()


# Set id as index
train.set_index('id', inplace=True)
test.set_index('id', inplace=True)

train.head()


train.describe()


train = train.drop_duplicates()
test = test.drop_duplicates()

print(train.shape)
print(test.shape)


def fill_missing(df, col):
    df[col] = df.groupby('Podcast_Name')[col].transform(lambda x:x.fillna(x.mean()))
    return

fill_missing(train, 'Episode_Length_minutes')
fill_missing(test, 'Episode_Length_minutes')



train['Guest_Popularity_percentage']=train['Guest_Popularity_percentage'].fillna(train['Guest_Popularity_percentage'].mean())
test['Guest_Popularity_percentage']=test['Guest_Popularity_percentage'].fillna(train['Guest_Popularity_percentage'].mean())


train[train['Number_of_Ads'].isna()]


train.dropna(inplace=True)


train.isnull().sum()


test.isnull().sum()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()

print(f"Using {num_gpus} GPUs for training!")


listen_mean_time = train.groupby('Genre')['Listening_Time_minutes'].mean().reset_index()

plt.figure(figsize=(8, 5))
sns.barplot(listen_mean_time, x='Genre', y='Listening_Time_minutes')
plt.xticks(rotation=45)
plt.ylabel('Mean Listening Time for each Genre')
plt.tight_layout()
plt.show()


ep_mean_time = train.groupby('Genre')['Episode_Length_minutes'].mean().reset_index()

plt.figure(figsize=(8, 5))
sns.barplot(ep_mean_time, x='Genre', y='Episode_Length_minutes')
plt.xticks(rotation=45)
plt.ylabel('Mean Episode Length minutes for each Genre')
plt.tight_layout()
plt.show()


custom_palette = ['#3498db', '#e74c3c', '#2ecc71']

def numerical_col_visuals(feature_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    plt.subplot(1, 2, 1)
    sns.boxplot(train, x=feature_name, y='Podcast_Name', palette=custom_palette)
    plt.xlabel(feature_name)
    plt.title(f"Box Plot for {feature_name} Across Datasets")
    
    plt.subplot(1, 2, 2)
    sns.histplot(train, x=feature_name, color=custom_palette[0], kde=True, bins=30, label="Train", alpha=0.6)
    sns.histplot(test, x=feature_name, color=custom_palette[1], kde=True, bins=30, label="Test", alpha=0.6)
    
    plt.xlabel(feature_name)
    plt.ylabel("Frequency")
    plt.title(f"Histogram for {feature_name} (Train vs Test)")
    plt.legend(title="Dataset")
    
    plt.tight_layout()
    plt.show()

# for coll in train.select_dtypes(include=['Number']).column.tolist():
numerical_col_visuals('Host_Popularity_percentage')
numerical_col_visuals('Guest_Popularity_percentage')


podcast_counts = train['Podcast_Name'].value_counts().reset_index()
print(podcast_counts)


def hist_plot(df, bin_size):
    plt.figure(figsize=(8, 5))
    sns.histplot(df, bins=bin_size,  kde=True)
    plt.show()

hist_plot(train['Listening_Time_minutes'], bin_size=50)


hist_plot(train['Episode_Length_minutes'], bin_size=50)


hist_plot(train['Guest_Popularity_percentage'], bin_size=30)


plt.figure(figsize=(5, 6))
sns.scatterplot(train, x='Genre', y='Number_of_Ads', hue='Genre')
plt.xticks(rotation=45)
plt.show()


def box_plot(df, x_base, y_base):
    plt.figure(figsize=(8, 5))
    sns.boxplot(train, x=x_base, y=y_base, hue=x_base)
    plt.xticks(rotation=45)
    plt.show()

box_plot(train, x_base='Genre', y_base='Number_of_Ads')


box_plot(train, x_base='Genre', y_base='Listening_Time_minutes')


box_plot(train, x_base='Episode_Sentiment', y_base='Listening_Time_minutes')


gen_ep_grp = train.groupby(['Genre', 'Episode_Sentiment', 'Episode_Length_minutes'])['Listening_Time_minutes'].mean().reset_index()

box_plot(gen_ep_grp, x_base='Genre', y_base='Listening_Time_minutes')


pd_grp = train.groupby(['Genre', 'Publication_Day', 'Publication_Time'])['Listening_Time_minutes'].mean().reset_index()
print(pd_grp)
box_plot(pd_grp, x_base='Genre', y_base='Listening_Time_minutes')


num_col = train.select_dtypes(include=['number']).columns.tolist()
cat_col = train.select_dtypes(exclude=['number']).columns.tolist()

print(num_col)
print(cat_col)


# frequency encoding
train_encoded = train.copy()
test_encoded = test.copy()

for col in cat_col:
    train_f=train[col].value_counts(normalize=True)
    test_f=test[col].value_counts(normalize=True)

    train_encoded[col] = train[col].map(train_f)
    test_encoded[col] = test[col].map(test_f)


train_encoded.head()


# log transform
# train_encoded['Listening_Time_minutes'] = np.log1p(train_encoded['Listening_Time_minutes'])
# train_encoded.head()


corr_mat=train_encoded.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_mat, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix - Features with the target")
plt.show()


# train test split
target_column='Listening_Time_minutes'

X = train_encoded.drop(columns=target_column)
y = train_encoded[target_column]

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42)


X_train.head()


def objective(trial):
    params = {
        "n_estimators": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        'random_state': 42,
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'n_jobs': -1
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    val_scores = []

    for train_idx, val_idx in kf.split(X):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBRegressor(**params)
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            early_stopping_rounds=50,
            verbose=False
        )

        preds = model.predict(X_val_fold)
        rmse = mean_squared_error(y_val_fold, preds, squared=False)
        val_scores.append(rmse)
        print(val_scores)
        print(np.mean(val_scores))

    return np.mean(val_scores)


# Creating study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)

# Best result
print("Best parameters:", study.best_params)
print("Best RMSE:", study.best_value ** 0.5)



# final training
final_model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.0313,
    max_depth=13,
    subsample= 0.8702,
    colsample_bytree=0.9561,
    reg_alpha= 0.1,
    reg_lambda=6.39,
    min_child_weight=10,
    random_state=42,
    tree_method='gpu_hist',
    predictor='gpu_predictor',
    n_jobs=-1
)

final_model.fit(X_train, y_train)


test_pred = final_model.predict(test_encoded)
test_pred


submission = pd.DataFrame({'id': test.index, 'Listening_Time_minutes': test_pred})
submission.to_csv('/kaggle/working/submission.csv', index=False)
display(submission)




