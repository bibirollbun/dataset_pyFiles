import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import optuna
from optuna import create_study

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


import warnings
warnings.filterwarnings('ignore')


df_train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
df_sub = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


df_train.head()


df_train.describe()


df_train.info()


df_test.info()


# Check missing values in training dataset
df_train.isna().sum()


# Check missing values in test dataset
df_test.isna().sum()


df_numerical = df_train.select_dtypes(include=[np.number])

plt.figure(figsize=(12, 10))
for idx, col in enumerate(df_numerical.columns[1:], 1):
    plt.subplot(3, 2, idx)
    sns.histplot(df_numerical[col], kde=True)
    plt.title(col)

plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))

df_corr = df_numerical.iloc[:, 1:].corr()
mask = np.triu(np.ones_like(df_corr))
sns.heatmap(df_corr, mask=mask, annot=True, 
            cmap='coolwarm',linewidth=0.5, 
            fmt='.2f', square=True)


df_categorical = df_train.select_dtypes(exclude=[np.number])


plt.figure(figsize=(12, 10))

for idx, col in enumerate (df_categorical.columns, 1):
    plt.subplot(3, 2, idx)
    sns.barplot(x=df_categorical[col].value_counts().index, y=df_categorical[col].value_counts().values)
    plt.xticks(rotation=90)
    
    plt.title(col)

plt.tight_layout()
plt.show()


# check Episode_Title unique values

df_train['Episode_Title'].unique()


class feature_eng(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.original_columns = X.columns.tolist()
        return self
    
    def transform(self, X, y=None):
        X_transformed = X.copy()

        X_transformed['Number_of_Ads'] = X_transformed['Number_of_Ads'].fillna(X_transformed['Number_of_Ads'].median())

        X_transformed['is_weekend'] = X_transformed['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)

        X_transformed['long_episode'] = (X_transformed['Episode_Length_minutes'] > 75).astype(int)

        X_transformed['with_popular_host'] = (X_transformed['Host_Popularity_percentage'] > 75).astype(int)

        X_transformed['with_popular_guest'] = (X_transformed['Guest_Popularity_percentage'] > 75).astype(int)

        X_transformed['host_guest_popularity_ratio'] = X_transformed['Host_Popularity_percentage'] / (X_transformed['Guest_Popularity_percentage'] + 1e-5)

        X_transformed['popularity_index'] = (X_transformed['Host_Popularity_percentage'] * X_transformed['Guest_Popularity_percentage']) / 100

        X_transformed['avg_popularity_score'] = (X_transformed['Host_Popularity_percentage'] + X_transformed['Guest_Popularity_percentage']) / 2

        X_transformed['ads_per_min'] = X_transformed['Number_of_Ads'] / (X_transformed['Episode_Length_minutes'] + 1e-5)

        # try sine and cosine transformation for cyclical features
        X_transformed['Episode_Title'] = X_transformed['Episode_Title'].apply(lambda x:int(x[len('Episode '):]) if isinstance(x, str) else np.nan).astype('Int64')
        X_transformed['Episode_Title_sin'] = np.sin(2*np.pi*X_transformed['Episode_Title']/12)
        X_transformed['Episode_Title_cos'] = np.cos(2*np.pi*X_transformed['Episode_Title']/12)

        X_transformed['Episode_Length_minutes_sin'] = np.sin(2*np.pi*X_transformed['Episode_Length_minutes']/60)
        X_transformed['Episode_Length_minutes_cos'] = np.cos(2*np.pi*X_transformed['Episode_Length_minutes']/60)

        return X_transformed

    def get_feature_names_out(self):

        new_features = ['Number_of_Ads', 'is_weekend', 'long_episode', 'with_popular_host',
                        'with_popular_guest', 'host_guest_popularity_ratio', 'popularity_index',
                        'avg_popularity_score', 'ads_per_min', 'Episode_Title_sin', 'Episode_Title_cos',
                        'Episode_Length_minutes_sin', 'Episode_Length_minutes_cos']
        
        final_features = self.original_columns + new_features

        return np.array(final_features)



X = df_train.drop(['id', 'Listening_Time_minutes'], axis=1)
y = df_train['Listening_Time_minutes']

X_sub = df_test.drop(['id'], axis=1)

X_eng = feature_eng().fit_transform(X)
X_sub_eng = feature_eng().transform(X_sub)


X_train, X_test, y_train, y_test = train_test_split(X_eng, y, test_size=0.2, random_state=42)
k_fold = KFold(n_splits=5, shuffle=True, random_state=42)


num_cols = X_eng.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_eng.select_dtypes(exclude=[np.number]).columns.tolist()


preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', KNNImputer(weights='distance')),
        #('scaler', StandardScaler())
    ]), num_cols),
    
    ('cat', Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]), cat_cols)
])

pipe = Pipeline([
    ('preprocess', preprocessor),
    #('featureGenerator', PolynomialFeatures(degree=2, interaction_only=True)),
    ('selector', VarianceThreshold(0.001))
])


#X_train_transformed = pipe.fit_transform(X_train)
#X_test_transformed = pipe.transform(X_test)
#X_sub_transformed = pipe.transform(X_sub)

#all_cols = pipe.named_steps['preprocess'].get_feature_names_out()

#X_train_transformed_with_names = pd.DataFrame(X_train_transformed, columns=all_cols)
#X_test_transformed_with_names = pd.DataFrame(X_test_transformed, columns=all_cols)
#X_sub_transformed_with_names = pd.DataFrame(X_sub_transformed, columns=all_cols)

X_train_transformed = pd.read_csv('/kaggle/input/transformed-data-with-names/X_train_transformed_with_names.csv')
X_test_transformed = pd.read_csv('/kaggle/input/transformed-data-with-names/X_test_transformed_with_names.csv')
X_sub_transformed = pd.read_csv('/kaggle/input/transformed-data-with-names/X_sub_transformed_with_names.csv')


X_train_transformed.head()


X_train_transformed.info()


xgb_params = {
    'tree_method': 'hist',
    'n_estimators': 50_000,
    'objective': 'reg:squarederror',
    'random_state': 42,
    'enable_categorical': True,
    'verbosity': 0,
    'early_stopping_rounds': 200,
    'eval_metric': 'rmse',
    'booster': 'gbtree',
    "device": "cuda",
    'max_depth': 15,
    'min_child_weight': 17, 
    'subsample': 0.8988920126079047,
    'reg_alpha': 0.20297150491417332, 
    'reg_lambda': 0.797095587096917, 
    'colsample_bytree': 0.5005221169171277,
    'n_jobs': -1,
    'learning_rate': 0.01,
}



rmse_scores=[]
oof_pred = np.zeros(X_train.shape[0])
test_pred = np.zeros(X_test.shape[0])
sub_pred = np.zeros(X_sub.shape[0])
n_splits = k_fold.get_n_splits()

for fold_idx, (train_idx, val_idx) in enumerate(k_fold.split(X_train_transformed, y_train)):
    X_train_fold, X_val_fold = X_train_transformed.iloc[train_idx], X_train_transformed.iloc[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

    xgbreg = XGBRegressor(**xgb_params)
    
    print(f'Training Fold {fold_idx + 1}')
    
    xgbreg.fit(X_train_fold, y_train_fold,
              eval_set=[(X_val_fold, y_val_fold)],
              verbose = 500)
    y_val_pred = xgbreg.predict(X_val_fold)

    oof_pred[val_idx] = y_val_pred
    test_pred += xgbreg.predict(X_test_transformed) / n_splits
    sub_pred += xgbreg.predict(X_sub_transformed) / n_splits
    
    rmse_score = np.sqrt(mean_squared_error(y_val_fold, y_val_pred))
    rmse_scores.append(rmse_score)
    
    print(f'RMSE: {rmse_score:.4f}')

print('Finish Training !!!!')
    


df_final_sub = df_sub.copy()
df_final_sub['Listening_Time_minutes'] = sub_pred
df_final_sub.head()


df_final_sub.to_csv('submission.csv', index=False)


summray_rmse = pd.DataFrame({
    'training_rmse': [np.mean(rmse_scores)],
    'oof_pred_rmse': [np.sqrt(mean_squared_error(y_train, oof_pred))],
    'test_pred_rmse': [np.sqrt(mean_squared_error(y_test, test_pred))]
}, index=['summary'])

summray_rmse











































