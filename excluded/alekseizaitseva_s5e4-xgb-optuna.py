#Basic part
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

#ML part
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy import stats


train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test =  pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


train.head()


train.info()


train.isna().sum()


test.isna().sum()


train.describe().applymap('{:,.2f}'.format)


train.describe(include = 'O')


y_train = train["Listening_Time_minutes"]
X_train = train.drop(columns=["Listening_Time_minutes"])
X_test = test


num_col = X_train.select_dtypes(include=['number']).columns.tolist()
cat_col = X_train.select_dtypes(exclude=['number']).columns.tolist()

print(num_col)
print(cat_col)


def create_features(df):
    df = df.copy()
    num_col = df.select_dtypes(include=['number']).columns.tolist()
    cat_col = df.select_dtypes(exclude=['number']).columns.tolist()

    df[num_col] = df[num_col].fillna(df[num_col].median())
    df[cat_col] = df[cat_col].fillna(df[cat_col].mode().iloc[0])

    return df
    


X_train = create_features(X_train)
X_test = create_features(X_test)

print(X_train.isnull().sum())
print(X_test.isnull().sum())



import seaborn as sns
import matplotlib.pyplot as plt


sns.heatmap(X_train[num_col].corr(), annot = True, cmap = 'viridis')


for col in cat_col:
    sns.boxplot(x=col, y='Listening_Time_minutes', data=train)
    plt.title(f"Listening Time vs {col}")
    plt.xticks(rotation = 90)
    plt.tight_layout()
    plt.show()


X_train['is_weekend'] = X_train['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)
X_train.head()


def feature_engineering(df):
    df = df.copy()

    df['is_weekend'] = df['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)
    
    df['episode_number'] = df['Episode_Title'].str.extract(r'(\d+)')[0].astype(int)

    df.drop(columns = 'Episode_Title', inplace = True)
    
    df['Number_of_Ads'] = df['Number_of_Ads'].fillna(0)
    
    df['ads_per_minute'] = (
    df.apply(lambda row: row['Number_of_Ads'] / (row['Episode_Length_minutes'] + 0.001)
            if pd.notna(row['Number_of_Ads']) and pd.notna(row['Episode_Length_minutes'])
            else 0.0,
            axis=1)
) # due to some data got 0 episode length

    df['is_weekend'] = df['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)

    df['length'] = pd.cut(df['Episode_Length_minutes'], bins=[0, 30, 60, 100, 200],
                                       labels=['short', 'average', 'long', 'extremely_long'])

    df['genre_sentiment'] = df['Genre'].astype(str) + "_" + df['Episode_Sentiment'].astype(str)

    
    for col in ['Episode_Length_minutes', 'Guest_Popularity_percentage']:
        df[col] = df.groupby(['Genre', 'genre_sentiment'])[col].transform(
        lambda x: x.fillna(x.mean()))

    cat_col = ['Podcast_Name', 'Genre', 'Publication_Day',
                        'Publication_Time', 'Episode_Sentiment', 'length', 'genre_sentiment']

    for col in cat_col:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    return df


y_train = train["Listening_Time_minutes"]
X_train = train.drop(columns=["Listening_Time_minutes"])
X_test = test

combined_df = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)

combined_df = feature_engineering(combined_df)

X_train = combined_df.iloc[:len(train)].reset_index(drop=True)
X_test = combined_df.iloc[len(train):].reset_index(drop=True) 


X_train.head()


X = X_train
y = y_train
X_test = X_test


# import optuna
# from xgboost import XGBRegressor
# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_error
# import numpy as np

# def objective(trial):
#     params = {
#         'n_estimators': 5000,
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
#         'max_depth': trial.suggest_int('max_depth', 3, 15),
#         'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#         'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
#         'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
#         'random_state': 42,
#         'tree_method': 'hist', 
#     }
#     cv_rmse = []
#     kf = KFold(n_splits=5, shuffle=True, random_state=42)
#     for train_idx, val_idx in kf.split(X):
#         X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
#         y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
        
#         model = XGBRegressor(**params, early_stopping_rounds=50)
#         model.fit(
#             X_train_cv, y_train_cv,
#             eval_set=[(X_val_cv, y_val_cv)],
#             verbose=False
#         )
#         preds = model.predict(X_val_cv)
#         rmse_score = np.sqrt(mean_squared_error(y_val_cv, preds))
#         cv_rmse.append(rmse_score)
#     return np.mean(cv_rmse)

# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=50)

# print('Лучшие параметры:', study.best_params) 
# print('Лучшее значение RMSE:', study.best_value)


#from optuna
best_params = {
    'n_estimators': 5000,
    'max_depth': 15,
    'learning_rate': 0.051564535401996674,
    'subsample': 0.6816345671807827,
    'colsample_bytree': 0.9977810444050708,
    'gamma': 1.4032650461122345,
    'reg_alpha': 2.7815627866713517,
    'reg_lambda': 3.780137117381534,
    'random_state': 42, 
}


from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt


# Initialize model
#model = XGBRegressor(**study.best_params)
model = XGBRegressor(**best_params)
# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rmse = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
    X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
    y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
    
    model.fit(X_train_cv, y_train_cv, 
              eval_set=[(X_val_cv, y_val_cv)],
              early_stopping_rounds=50,
              verbose=False)
    
    preds = model.predict(X_val_cv)
    rmse_score = np.sqrt(mean_squared_error(y_val_cv, preds))
    cv_rmse.append(rmse_score)

    print(f"Fold {fold} RMSE: {rmse_score:.4f}")

print(f"\nAverage CV RMSE: {np.mean(cv_rmse):.4f}")




plt.figure(figsize=(12, 6))
plot_importance(model, max_num_features=20, importance_type='gain')
plt.title("Top 20 Feature Importances (by Gain)")
plt.show()


submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv', index_col='id')
submission['Listening_Time_minutes'] = model.predict(X_test)
submission.to_csv('submission.csv')

