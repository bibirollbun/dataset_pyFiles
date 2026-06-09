import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import gc
import catboost as cb
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold


train = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
test = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')

# downgrade to float32 for memory management
train = train.astype('float32')
test = test.astype('float32')


print(f"This dataset has {train.shape[0]} rows and {train.shape[1]} columns.")
print(f"There are {train.isna().sum().sum()} NA's in the dataset.")
print(f"There are {str(train.duplicated().sum())} duplicates in the dataset.")
print(f"There are {np.isinf(train).sum().sum()} infinite values in the dataset.")

# quick look at the data
train.head(3)

# We notice we don't have any NA's in this dataset, but we do have infinite values amounting to ~ 2.3% of the dataset we will need to work with.


train = train.replace([np.inf, -np.inf], np.nan)

nan_cols = train.columns[train.isna().any()].tolist() # get list of nan columns

print('The columns with NaNs and their NaN count:')
print(train[nan_cols].isna().sum()) # print NaN count per column

# It appears these 21 columns only contain NaN values, so for now we will remove them from our dataset (both train & test)
train = train.drop(nan_cols, axis=1)
test = test.drop(nan_cols, axis=1)

gc.collect()


# Drop columns have exactly 1 value
NUNIQUE1=[c for c in train.columns if train[c].nunique()==1]
train.drop(NUNIQUE1,axis=1,inplace=True)
test.drop(NUNIQUE1,axis=1,inplace=True)


target = 'label'
categorical_columns = []
numerical_columns = [col for col in train.columns if col != target]


skewness_threshold = .5 # can tune / experiment with this value
skewed_cols = [col for col in numerical_columns if train[col].skew() > skewness_threshold]

print(f'There are {len(skewed_cols)} skewed columns: {str(skewed_cols)}')

# According to our threshold of .5, all of the original features (bid_qty, ask_qty, buy_qty, sell_qty, & volume are skewed, in addition to 262 of the proprietary, derived features.


train_unique_cols = [x for x in train.drop([target],axis=1).columns if x not in test.columns]
print('All columns in train exist in test.') if not train_unique_cols else print(f'The following train columns are not in test: {train_unique_cols}')

test_unique_cols = [x for x in test.columns if x not in train.columns]
print('All columns in test exist in train.') if not test_unique_cols else print(f'The following test columns are not in train: {test_unique_cols}')

# Perfect, we can use all train columns in our modelling efforts.


train_samp = train[['bid_qty','ask_qty', 'buy_qty', 'sell_qty',	'volume']].sample(frac=0.1, replace=False, random_state=1)
train_samp = train_samp[~(train_samp == 0).any(axis=1)] # we will be taking the logarithmic of these columns for plotting purposes; removing rows w/ zeroes

for col in ['bid_qty','ask_qty', 'buy_qty', 'sell_qty',	'volume']:
    _, axes = plt.subplots(1,2,figsize=(8,4),sharex=False,sharey=False)
   
    sns.histplot(data=train_samp.dropna(), x=col,bins=10,ax=axes[0],log_scale=True, color = 'lightblue')
    axes[0].set_title(f'Log of {col}')

    sns.boxplot(data=train_samp.dropna(),x=col,ax=axes[1],showfliers=False, color = 'honeydew')
    axes[1].set_title(f'{col} (no outliers)')

    plt.tight_layout()
    plt.show()

del train_samp
gc.collect()


print(f"This dataset has {test.shape[0]} rows and {test.shape[1]} columns.")
print(f"There are {test.isna().sum().sum()} NA's in the dataset.")
print(f"There are {str(test.duplicated().sum())} duplicates in the dataset.")
print(f"There are {np.isinf(test).sum().sum()} infinite values in the dataset.")

# quick look at the data
test.head(3)

# Notice there are no infinite values in test because we processed them earlier with train.


train_samp = pd.DataFrame(train[target]).sample(frac=0.1, replace=False, random_state=1).sort_index()

# plot target histplot & boxplot
_, axes = plt.subplots(1,2,figsize=(8,4),sharex=False,sharey=False)
sns.histplot(data=train_samp.dropna(), x=target,bins=10,ax=axes[0],log_scale=False, color = 'lightblue')
axes[0].set_title(f'{target}')
sns.boxplot(data=train_samp.dropna(),x=target,ax=axes[1],showfliers=False, color = 'honeydew')
axes[1].set_title(f'{target} (no outliers)')
plt.tight_layout()
plt.show()

# plot target for entire timeframe
_, axes = plt.subplots(1,1,figsize=(8,6),sharex=False,sharey=False)
sns.lineplot(data=train_samp.dropna(),x=train_samp.index, y=target, color = 'mediumslateblue')
axes.set_title(f'{target} time series plot (all timeframes)')
plt.tight_layout()
plt.show()

# plot target for month of 2023-03
train_samp = train_samp.filter(regex='^2023-03', axis=0)
_, axes = plt.subplots(1,1,figsize=(8,6),sharex=False,sharey=False)
sns.lineplot(data=train_samp.dropna(),x=train_samp.index, y=target, color = 'navy')
axes.set_title(f'{target} time series plot (2023-03 timeframe)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

del train_samp
gc.collect()



X = train.drop([target],axis=1)
y = train[target]
cv_method = KFold(n_splits=5, shuffle=True, random_state=1)
X_test = test.drop([target],axis=1)
test_ids = test.index
del train, test
gc.collect()


scores = []
oof = np.zeros(len(y), dtype=float)
preds = np.zeros(X_test.shape[0],dtype=float)

params = {'iterations': 3077, 'learning_rate': 0.023604883174757015, 'depth': 9, 'l2_leaf_reg': 1.1645944960977261, 
          'subsample': 0.7626370991977192, 'colsample_bylevel': 0.9894180344611699, 'min_data_in_leaf': 75}

for fold, (idx_tr, idx_va) in enumerate(cv_method.split(X, y), start=1):
    X_tr = X.iloc[idx_tr]
    X_va = X.iloc[idx_va]
    y_tr = y.iloc[idx_tr]
    y_va = y.iloc[idx_va]
    
    model = cb.CatBoostRegressor(**params, boosting_type='Plain', task_type='CPU', random_state=1,cat_features = categorical_columns)
    model.fit(X_tr, y_tr,verbose=False)
    
    y_pred = model.predict(X_va)
    preds += model.predict(X_test)
    
    score = pearsonr(y_va, y_pred)[0]
    print(f"# Fold {fold}: {score=:.5f}")
    
    scores.append(score)
    oof[idx_va] = y_pred
    score = np.mean(scores)
    
print(f"# XGBoost_gbtree Overall score: {score}: +/- {np.std(scores)}")

preds /= 5


submission = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')
submission.head(3)


submission['prediction'] = preds
submission.head(3)


submission.to_csv('submission.csv',index=False)

