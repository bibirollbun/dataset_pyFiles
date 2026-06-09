import time
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.model_selection import RandomizedSearchCV, cross_validate, KFold, train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from lightgbm import LGBMRegressor

import optuna

import warnings
warnings.filterwarnings("ignore")


train_data = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv', index_col='id')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv', index_col='id')

train_data.head()


original_data = pd.read_csv('/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv')
original_data.head()


train_data.info()


train_data.describe()


num_columns = train_data.select_dtypes('float64').columns

plt.figure(figsize=(12, 8))
for i, col in enumerate(num_columns):
    plt.subplot(3, 2, i+1)
    sns.histplot(train_data[col], bins=30)

plt.tight_layout()
plt.show()


train_data['Number_of_Ads'].value_counts()


original_data['Number_of_Ads'].value_counts()


test_data.describe()


test_data['Episode_Length_minutes'].quantile(0.75), test_data['Episode_Length_minutes'].max()


test_data.sort_values(by='Episode_Length_minutes', ascending=False).head(5)


test_data.loc[806597, 'Episode_Length_minutes'] = np.nan
test_data.loc[804434, 'Episode_Length_minutes'] = np.nan


train_data[['Host_Popularity_percentage', 'Guest_Popularity_percentage']].describe()


cat_columns = train_data.select_dtypes('object').columns.tolist()
cat_columns.remove('Podcast_Name')
cat_columns.remove('Episode_Title')

plt.figure(figsize=(12, 8))
for i, col in enumerate(cat_columns):
    plt.subplot(2, 2, i+1)
    plt.xticks(rotation=45)
    sns.countplot(x=train_data[col])

plt.tight_layout()
plt.show()


sns.heatmap(train_data[num_columns].corr(), annot=True)


target = 'Listening_Time_minutes'

plt.figure(figsize=(12, 8))
for i, col in enumerate(num_columns):
    if col == target:
        continue
    plt.subplot(2, 2, i+1)
    sns.scatterplot(data=train_data, x=col, y=target)

plt.tight_layout()
plt.show()


listening_time_gt_lenght = (train_data[target] > train_data['Episode_Length_minutes'])
print(f'Number of samples for which the listening time is greater than episode length: {listening_time_gt_lenght.sum()}')


num_columns = ['Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads']

plt.figure(figsize=(6, 8))
for i, col in enumerate(num_columns):
    plt.subplot(3, 1, i+1)
    plt.xlabel(col)
    plt.ylabel(target)
    plt.hexbin(train_data[col], train_data[target], gridsize=25, cmap='Blues')
    plt.colorbar(label='Density')

plt.tight_layout()
plt.show()


number_of_ads_cat = train_data['Number_of_Ads'].apply(lambda x: x if x <= 3.0 else 'Other')
number_of_ads_cat.value_counts()

sns.boxplot(x=number_of_ads_cat, y=train_data[target])


plt.figure(figsize=(12, 8))
for i, col in enumerate(cat_columns):
    plt.subplot(2, 2, i+1)
    plt.xticks(rotation=45)
    sns.boxplot(data=train_data, x=col, y=target)

plt.tight_layout()
plt.show()


train_data['Episode_Number'] = train_data['Episode_Title'].str.split(' ', expand=True)[1].astype('float64')
test_data['Episode_Number'] = test_data['Episode_Title'].str.split(' ', expand=True)[1].astype('float64')

podcast_name_encoding = train_data.groupby('Podcast_Name')['Listening_Time_minutes'].mean()
train_data['Podcast_Name_target_encoded'] = train_data['Podcast_Name'].map(podcast_name_encoding)
test_data['Podcast_Name_target_encoded'] = test_data['Podcast_Name'].map(podcast_name_encoding)

train_data.loc[train_data['Number_of_Ads'] > 3., 'Number_of_Ads'] = np.nan
test_data.loc[test_data['Number_of_Ads'] > 3., 'Number_of_Ads'] = np.nan

train_data_dropped = train_data.drop(columns=['Podcast_Name', 'Episode_Title'])
test_data_dropped = test_data.drop(columns=['Podcast_Name', 'Episode_Title'])


X = train_data_dropped.drop('Listening_Time_minutes', axis=1)
y = train_data_dropped['Listening_Time_minutes']
X_test = test_data_dropped

num_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='mean')),
    ('scale', StandardScaler())
])

cat_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('encode', OneHotEncoder(drop='first', sparse_output=False))
])

num_columns = X.select_dtypes('float64').columns
cat_columns = X.select_dtypes('object').columns

preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, num_columns),
    ('cat', cat_pipeline, cat_columns)],
    remainder='passthrough'
)

preprocessor.fit(X, y)
X = preprocessor.transform(X)
X_test = preprocessor.transform(X_test)


TUNE_PARAMS = False
CROSS_VALIDATE = False

def cross_validate_model(model, X, y):
    cv = KFold(n_splits=4, shuffle=True, random_state=42)
    
    scores = cross_validate(model, X, y, return_train_score=True, cv=cv,
                       scoring=['neg_root_mean_squared_error', 'r2'], verbose=10)

    train_rmse = -np.mean(scores['train_neg_root_mean_squared_error'])
    test_rmse = -np.mean(scores['test_neg_root_mean_squared_error'])
    train_r2 = np.mean(scores['train_r2'])
    test_r2 = np.mean(scores['test_r2'])
    
    print(f'Train RMSE: {train_rmse:.4f}')
    print(f' Test RMSE: {test_rmse:.4f}')
    print(f'Train R2: {train_r2:.4f}')
    print(f' Test R2: {test_r2:.4f}')


subset_idx = np.random.choice(X.shape[0], 150_000, replace=False)
X_subset = X[subset_idx]
y_subset = y[subset_idx]

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 500, 1500)
    learning_rate = trial.suggest_float('learning_rate', 0.005, 0.1)
    max_depth = trial.suggest_int('max_depth', 5, 16)
    num_leaves = trial.suggest_int('num_leaves', 2, 2**max_depth)
    reg_alpha = trial.suggest_float('reg_alpha', 0., 1.)
    reg_lambda = trial.suggest_float('reg_lambda', 0., 5.)
    min_split_gain = trial.suggest_float('min_split_gain', 0., 5.)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    subsample_freq = trial.suggest_int('subsample_freq', 0, 5)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.3, 1.0)
    objective = trial.suggest_categorical('objective', ['regression', 'regression_l1'])

    model = LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_split_gain=min_split_gain,
        subsample=subsample,
        subsample_freq=subsample_freq,
        colsample_bytree=colsample_bytree,
        objective=objective,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )

    start = time.time()
    cv = KFold(n_splits=4, shuffle=True, random_state=42)
    score = cross_val_score(model, X_subset, y_subset, cv=cv, scoring='neg_root_mean_squared_error').mean()
    print(time.time() - start)
    return -1 * score

if TUNE_PARAMS:
    study = optuna.create_study(direction='minimize', study_name='LGBMRegressor')
    study.optimize(objective, n_trials=100)
    
    print("Best trial:")
    print(study.best_trial.params)
    best_params = study.best_trial.params
else:
    best_params = {'n_estimators': 5000, 'learning_rate': 0.01, 'max_depth': 12,
                   'num_leaves': 2000, 'reg_alpha': 0.35, 'reg_lambda': 3,
                   'min_split_gain': 0.5, 'subsample': 0.75, 'subsample_freq': 1,
                   'colsample_bytree': 0.75, 'objective': 'regression'}


if CROSS_VALIDATE:
    model = LGBMRegressor(**best_params, n_jobs=-1, random_state=42, verbose=-1)
    cross_validate_model(model, X, y)

gc.collect()


lgbm = LGBMRegressor(**best_params, n_jobs=-1, verbose=-1)
lgbm.fit(X, y)
predictions_lgbm = lgbm.predict(X_test)
predictions_lgbm[predictions_lgbm < 0] = 0 # make sure predictions are strictly positive
    
submission = pd.DataFrame({'id': test_data.index, 'Listening_Time_minutes': predictions_lgbm})
submission.to_csv('/kaggle/working/submission_lgbm_1.csv', index=False)


feature_importances = pd.DataFrame({'feature_name': preprocessor.get_feature_names_out(),
                                    'feature_importance': lgbm.feature_importances_}).sort_values(by='feature_importance', ascending=False)

sns.barplot(data=feature_importances, y='feature_name', x='feature_importance', orient='h')


plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.title('Target (training data)')
sns.histplot(y, bins=30)

plt.subplot(1, 2, 2)
plt.title('Target (test predictions)')
sns.histplot(predictions_lgbm, bins=30)

plt.tight_layout()
plt.show()

