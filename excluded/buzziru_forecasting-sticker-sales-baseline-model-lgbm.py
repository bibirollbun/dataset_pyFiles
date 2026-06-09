import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

import optuna
from lightgbm import LGBMRegressor, early_stopping

import warnings

warnings.filterwarnings("ignore")


%matplotlib inline
plt.style.use('ggplot')

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 50)


SEED=42
np.random.seed(SEED)
random.seed(SEED)


train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')

all_df = pd.concat([train, test], axis=0, ignore_index=True)


train.head()


train.shape, test.shape


train.info()


all_df['date'] = pd.to_datetime(all_df['date'])


removed = ["id", "num_sold"]
features = [c for c in all_df.columns if not c in removed]

cat_cols = []
high_cardinality = []
print(f"THE {len(features)} BASIC FEATURES ARE:")

for c in features:
    ftype = str(all_df[c].dtype)
    if all_df[c].dtype=="object":
        cat_cols.append(c)
        ftype = "categorical"
        
    n = all_df[c].nunique()
    print(f"{c} ({ftype}) with {n} unique values")
    if n>=7: high_cardinality.append(c)
    
train = all_df.iloc[:len(train)].copy()
test = all_df.iloc[len(train):].reset_index(drop=True).copy()

print("\nTHE FOLLOWING HAVE 7 OR MORE UNIQUE VALUES:", high_cardinality )


for col in cat_cols:
    print(f'\nTop value counts in "{col}" : \n{train[col].value_counts()}')


def plot_cat_col(df, col):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(30, 10))

    sns.countplot(data=df, x=col, ax=ax[0], hue=col)
    ax[0].set_title(f'{col} Distribution')
    ax[0].set_xlabel(f'{col}')
    ax[0].set_ylabel('Count')
    
    sns.boxplot(data=df, x=col, y='num_sold', ax=ax[1], hue=col)
    ax[1].set_title(f'{col} Boxplot')
    ax[1].set_xlabel(f'{col}')

    plt.tight_layout()
    plt.show()


for col in cat_cols:
    plot_cat_col(train, col)


plt.figure(figsize=(15,9))
plt.title("Visualizing Missing Values")
sns.heatmap(train.isnull(), cbar=False, cmap=sns.color_palette('magma'), yticklabels=False);
plt.show()


df_nan = train[train['num_sold'].isnull()]
df_nan.head()


for col in cat_cols:
    print(f'\nTop value counts in "{col}" : \n{df_nan[col].value_counts()}')


print('nan count of "num_sold"', train.loc[(train['country']=='Kenya')&(train['store']=='Discount Stickers')&(train['product']=='Holographic Goose')]['num_sold'].isnull().sum())
print('Total Count', len(train.loc[(train['country']=='Kenya')&(train['store']=='Discount Stickers')&(train['product']=='Holographic Goose')]))


nan_summary = (
    train.groupby(['country', 'store', 'product'])
         .agg(
             NaN_Count=('num_sold', lambda x: x.isnull().sum()),  
             Total_Count=('num_sold', 'size')                     
         )
         .assign(NaN_Ratio=lambda x: (x['NaN_Count'] / x['Total_Count']) * 100)  
         .sort_values(by='NaN_Ratio', ascending=False)  
         .reset_index()
)

nan_summary.loc[nan_summary['NaN_Ratio']>0]


def plot_transform(df, col):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    sns.histplot(df[col], kde=True, ax=ax[0])
    ax[0].set_title(f'{col} Distribution Before Log Transformation', fontsize=14)
    skew_before = df[col].skew()
    ax[0].text(0.7, 0.9, f'Skewness: {skew_before:.2f}', transform=ax[0].transAxes, fontsize=12, color='red')

    sns.histplot(np.log1p(df[col]), kde=True, ax=ax[1])
    ax[1].set_title(f'{col} Distribution After Log Transformation', fontsize=14)
    skew_after = np.log1p(df[col]).skew()
    ax[1].text(0.7, 0.9, f'Skewness: {skew_after:.2f}', transform=ax[1].transAxes, fontsize=12, color='red')
    plt.show()


plot_transform(train, 'num_sold')


Q1 = train['num_sold'].quantile(0.25)
Q3 = train['num_sold'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = train[(train['num_sold'] < lower_bound) | (train['num_sold'] > upper_bound)]
print(f'Number of Outliers: {len(outliers)}')
print(f"Ratio of Outliers: {(len(outliers) / len(train['num_sold'])) * 100}%")


def date(df):
    df['Year'] = df['date'].dt.year
    df['Day'] = df['date'].dt.day
    df['Month'] = df['date'].dt.month
    # df['Month_name'] = df['date'].dt.month_name()
    df['Day_of_week'] = df['date'].dt.day_name()
    df['Week'] = df['date'].dt.isocalendar().week
    df['Year_sin'] = np.sin(2 * np.pi * df['Year'])
    df['Year_cos'] = np.cos(2 * np.pi * df['Year'])
    min_year = df['Year'].min()
    max_year = df['Year'].max()
    df['Year_sin'] = np.sin(2 * np.pi * (df['Year'] - min_year) / (max_year - min_year))
    df['Year_cos'] = np.cos(2 * np.pi * (df['Year'] - min_year) / (max_year - min_year))
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12) 
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['Day_sin'] = np.sin(2 * np.pi * df['Day'] / 31)  
    df['Day_cos'] = np.cos(2 * np.pi * df['Day'] / 31)
    df['Group']=(df['Year']-2010)*48+df['Month']*4+df['Day']//7
    # df.drop('date', axis=1, inplace=True)
    return df

train = date(train)
test = date(test)
train.head()


removed = ['id', 'date', 'num_sold']
features = [c for c in train.columns if not c in removed]
features


train = train.dropna()
train.shape


cat_cols.append('Day_of_week')
label_encoders = {col: LabelEncoder() for col in cat_cols}

for col in cat_cols:
    train[col] = label_encoders[col].fit_transform(train[col])
    test[col] = label_encoders[col].transform(test[col])


test.head()


x = train.drop(['id', 'date', 'num_sold'], axis=1)
y = train['num_sold']
x_test = test.drop(['id', 'date', 'num_sold'], axis=1)
y_log = np.log1p(y)


# %%time
# def objective(trial):
#     params = {
#         "n_estimators": 300,
#         "objective": "regression",
#         "metric": "mape",
#         "boosting_type": "gbdt",
#         "num_leaves": trial.suggest_int("num_leaves", 10, 200),
#         "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
#         "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
#         "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
#         "bagging_freq": trial.suggest_int("bagging_freq", 5, 12),
#         "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
#         "max_depth": trial.suggest_int("max_depth", -1, 12),
#         "lambda_l1": trial.suggest_float("lambda_l1", 1e-4, 10.0, log=True),
#         "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, 10.0, log=True),
#         "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.001, 0.1),
#         "device_type": "cpu",
#         "seed" : SEED,
#         'verbose':-1

#     }
    
#     model = LGBMRegressor(**params)
#     folds = KFold(n_splits=5, shuffle=True, random_state=SEED)
#     scores = []

#     for train_idx, val_idx in folds.split(x):
#         x_train, x_val = x.iloc[train_idx], x.iloc[val_idx]
#         y_train, y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]

#         model.fit(
#             x_train, y_train,
#             eval_set=[(x_val, y_val)],
#             eval_metric='mape',
#             callbacks=[early_stopping(100)]
#         )
#         preds = model.predict(x_val)
#         mape = mean_absolute_percentage_error(y_val, preds)
#         scores.append(mape)

#     return np.mean(scores)

# study = optuna.create_study(
#     direction="minimize",
#     sampler=optuna.samplers.TPESampler(seed=SEED)
#     )
# study.optimize(objective, n_trials=5)


# study.best_params, study.best_value


%%time
best_params = {
'n_estimators': 1000,
'num_leaves': 81,
'learning_rate': 0.07114476009343425,
'feature_fraction': 0.892797576724562,
'bagging_fraction': 0.8394633936788146,
'bagging_freq': 6,
'min_data_in_leaf': 24,
'max_depth': -1,
'lambda_l1': 2.1423021757741068,
'lambda_l2': 0.10129197956845731,
'min_gain_to_split': 0.07109918520180851,
'random_state': SEED,
'verbose': -1,
'n_jobs': -1
}

models_lgb = []
lgbm_OOF = np.zeros(len(x))
lgbm_preds = np.zeros(len(x_test))
group_col = x['Year']
folds = GroupKFold(n_splits=5)

for train_idx, val_idx in folds.split(x, y_log, groups=group_col):
    x_train, x_val = x.iloc[train_idx], x.iloc[val_idx]
    y_train, y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]

    model = LGBMRegressor(**best_params)
    model.fit(
        x_train, y_train, 
        eval_set=[(x_val, y_val)],
        eval_metric='mape',
        callbacks=[early_stopping(100)])

    lgbm_OOF[val_idx] += model.predict(x_val)
    lgbm_preds += model.predict(x_test) / folds.n_splits
    models_lgb.append(model)


mape = mean_absolute_percentage_error(np.expm1(y_log), np.expm1(lgbm_OOF))
mae = mean_absolute_error(y_log, lgbm_OOF)

print(f"\nPerformance Metrics:\n{'-'*25}")
print(f"MAPE: {mape:.4f}")
print(f"MAE: {mae:.4f}")


feature_names = models_lgb[0].booster_.feature_name()

importances_gain = np.zeros(len(feature_names)) 
importances_split = np.zeros(len(feature_names))  

for model in models_lgb:
    importances_gain += model.booster_.feature_importance(importance_type='gain')
    importances_split += model.booster_.feature_importance(importance_type='split')

importances_gain /= len(models_lgb)
importances_split /= len(models_lgb)
 
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Gain': importances_gain,
    'Split': importances_split
})

importance_df = importance_df.sort_values(by='Gain', ascending=False).reset_index(drop=True)
top_features = importance_df

fig, axes = plt.subplots(1, 2, figsize=(16, 8)) 
plt.subplots_adjust(wspace=0.4) 

sns.barplot(data=top_features, x='Gain', y='Feature', ax=axes[0], palette="coolwarm")
axes[0].set_title("Top Feature Importances (Gain)", fontsize=16, fontweight='bold')
axes[0].set_xlabel("Feature Importance (Gain)", fontsize=12)
axes[0].set_ylabel("Feature", fontsize=12)

sns.barplot(data=top_features, x='Split', y='Feature', ax=axes[1], palette="viridis")
axes[1].set_title("Top Feature Importances (Split)", fontsize=16, fontweight='bold')
axes[1].set_xlabel("Feature Importance (Split)", fontsize=12)
axes[1].set_ylabel("Feature", fontsize=12)

plt.tight_layout()
plt.show()


submission = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')
submission.head()


preds = np.expm1(lgbm_preds)
submission['num_sold'] = preds
submission.head()


submission.to_csv('/kaggle/working/submission.csv', index=False)




