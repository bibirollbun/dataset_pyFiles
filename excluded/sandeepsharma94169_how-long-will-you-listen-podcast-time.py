# Thanks to Masaya Kawamata. I am giving credit of this notebook to him. 


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy.stats import f_oneway
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBRegressor 
from sklearn.model_selection import KFold,train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from itertools import combinations
from tqdm import tqdm
from category_encoders import TargetEncoder
import lightgbm as lgb
import warnings 
warnings.filterwarnings('ignore')


train = pd.read_csv(r'/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv(r'/kaggle/input/playground-series-s5e4/test.csv')
train.head()


original_data = pd.read_csv(r'/kaggle/input/original-podcast-data/podcast_dataset.csv')
original_data.head()


print(train.shape)
print(test.shape)
print(original_data.shape)


original_clean = original_data.dropna(subset='Listening_Time_minutes').drop_duplicates()
train = pd.concat([train,original_clean],axis=0,ignore_index=True)


train.shape


train.head()


def feat_eng(data):
    df=data.copy()
    df['Episode_Num'] = df['Episode_Title'].str.extract(r'(\d+)')
    df['is_weekend']   = df['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)
    return df.drop(columns=['Episode_Title'])

train = feat_eng(train)
test = feat_eng(test)


elm = []
for k in range(3):
    col_name = f'ELm_r{k}'
    train[col_name] = train['Episode_Length_minutes'].round(k)
    test[col_name] = test['Episode_Length_minutes'].round(k)
    elm.append(col_name)


train.head()


# From this part reffered this notebook 
# https://www.kaggle.com/code/masayakawamata/single-xgboost-add-selected-features
encoded_columns = []

selected_comb = [
    # 2-interaction
    ['Episode_Length_minutes', 'Host_Popularity_percentage'],
    ['Episode_Length_minutes', 'Guest_Popularity_percentage'],
    ['Episode_Length_minutes', 'Number_of_Ads'],
    ['Episode_Num', 'Host_Popularity_percentage'],
    ['Episode_Num', 'Guest_Popularity_percentage'],
    ['Episode_Num', 'Number_of_Ads'],    
    ['Host_Popularity_percentage', 'Guest_Popularity_percentage'],
    ['Host_Popularity_percentage', 'Number_of_Ads'],
    ['Host_Popularity_percentage', 'Episode_Sentiment'],
    ['Episode_Length_minutes', 'Podcast_Name'],
    ['Episode_Num', 'Podcast_Name'],  
    ['Guest_Popularity_percentage', 'Podcast_Name'],
    ['ELm_r1', 'Episode_Num'],
    ['ELm_r1', 'Host_Popularity_percentage'], 
    ['ELm_r1', 'Guest_Popularity_percentage'],
    ['ELm_r2', 'Episode_Num'],
    ['ELm_r2', 'Episode_Sentiment'],
    ['ELm_r2', 'Publication_Day'],

    
    # 3-interaction
    ['Episode_Length_minutes', 'Episode_Num', 'Host_Popularity_percentage'],
    ['Episode_Length_minutes', 'Episode_Num', 'Guest_Popularity_percentage'],
    ['Episode_Length_minutes', 'Episode_Num', 'Number_of_Ads'],
    ['Episode_Length_minutes', 'Episode_Num', 'Episode_Sentiment'],
    ['Episode_Length_minutes', 'Episode_Num', 'Publication_Day'],
    ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage'],
    ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Number_of_Ads'],
    ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Episode_Sentiment'],
    ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Publication_Day'],
    ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Publication_Time'],
    ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads'],
    ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Publication_Day'],
    ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Publication_Time'],
    ['Episode_Length_minutes', 'Number_of_Ads', 'Episode_Sentiment'],
    ['Episode_Length_minutes', 'Number_of_Ads', 'Publication_Day'],
    ['Episode_Length_minutes', 'Episode_Sentiment', 'Publication_Time'],
    ['Episode_Num', 'Host_Popularity_percentage', 'Guest_Popularity_percentage'],
    ['Episode_Num', 'Host_Popularity_percentage', 'Number_of_Ads'],
    ['Episode_Num', 'Host_Popularity_percentage', 'Episode_Sentiment'],
    ['Episode_Num', 'Host_Popularity_percentage', 'Publication_Day'],
    ['Episode_Num', 'Host_Popularity_percentage', 'Publication_Time'],
    ['Episode_Num', 'Host_Popularity_percentage', 'Genre'],
    ['Episode_Num', 'Guest_Popularity_percentage', 'Number_of_Ads'],
    ['Episode_Num', 'Guest_Popularity_percentage', 'Episode_Sentiment'],
    ['Episode_Num', 'Guest_Popularity_percentage', 'Publication_Day'],
    ['Episode_Num', 'Guest_Popularity_percentage', 'Publication_Time'],
    ['Episode_Num', 'Guest_Popularity_percentage', 'Genre'],
    ['Episode_Num', 'Number_of_Ads', 'Episode_Sentiment'],
    ['Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads'],
    ['Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Episode_Sentiment'],
    ['Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Publication_Day'],
    ['Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Publication_Time'],
    ['Host_Popularity_percentage', 'Number_of_Ads', 'Publication_Day'],

    ['Guest_Popularity_percentage', 'Number_of_Ads', 'Episode_Sentiment'],
    ['Guest_Popularity_percentage', 'Number_of_Ads', 'Genre'],   
    ['ELm_r1', 'Number_of_Ads', 'Episode_Sentiment'],
    ['ELm_r2', 'Number_of_Ads', 'Podcast_Name'],
    
    # 4-interaction
    ['Episode_Length_minutes', 'Episode_Num', 'Host_Popularity_percentage', 'Guest_Popularity_percentage'],
    ['Episode_Length_minutes', 'Episode_Num', 'Host_Popularity_percentage', 'Number_of_Ads'],
    ['Episode_Length_minutes', 'Episode_Num', 'Host_Popularity_percentage', 'Episode_Sentiment'],
    ['Episode_Length_minutes', 'Episode_Num', 'Host_Popularity_percentage', 'Publication_Day'],
    ['Episode_Length_minutes', 'Episode_Num', 'Host_Popularity_percentage', 'Publication_Time'],
    ['Episode_Length_minutes', 'Episode_Num', 'Host_Popularity_percentage', 'Genre'],
    ['Episode_Length_minutes', 'Episode_Num', 'Guest_Popularity_percentage', 'Number_of_Ads'],
    ['Episode_Length_minutes', 'Episode_Num', 'Guest_Popularity_percentage', 'Episode_Sentiment'],
    ['Episode_Length_minutes', 'Episode_Num', 'Guest_Popularity_percentage', 'Publication_Day'],
    ['Episode_Length_minutes', 'Episode_Num', 'Guest_Popularity_percentage', 'Publication_Time'],
    ['Episode_Length_minutes', 'Episode_Num', 'Number_of_Ads', 'Episode_Sentiment'],
    ['Episode_Length_minutes', 'Episode_Num', 'Number_of_Ads', 'Publication_Day'],
    ['Episode_Length_minutes', 'Episode_Num', 'Number_of_Ads', 'Publication_Time'],
    ['Episode_Length_minutes', 'Episode_Num', 'Publication_Day', 'Publication_Time'],
    ['Episode_Length_minutes', 'Episode_Num', 'Publication_Day', 'Genre'],    
    ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads'],
    ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Episode_Sentiment'],
    ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Publication_Day'],
    ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Publication_Time'],
    ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Number_of_Ads', 'Episode_Sentiment'],
    ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Number_of_Ads', 'Publication_Day'],
    ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Publication_Day', 'Publication_Time'],
    ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Publication_Day', 'Genre'],
    ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Episode_Sentiment'],
    ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Publication_Day'],
    ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Publication_Time'],
    ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Genre'],
    ['Episode_Length_minutes', 'Episode_Num', 'Publication_Time', 'Podcast_Name'],
    
    ['Episode_Num', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads'],
    ['Episode_Num', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Episode_Sentiment'],
    ['Episode_Num', 'Host_Popularity_percentage', 'Number_of_Ads', 'Publication_Day'],
    ['Episode_Num', 'Host_Popularity_percentage', 'Number_of_Ads', 'Publication_Time'],
    ['Episode_Num', 'Host_Popularity_percentage', 'Episode_Sentiment', 'Publication_Day'],
    ['Episode_Num', 'Host_Popularity_percentage', 'Episode_Sentiment', 'Publication_Time'],
    ['Episode_Num', 'Host_Popularity_percentage', 'Episode_Sentiment', 'Genre'],
    ['Episode_Num', 'Host_Popularity_percentage', 'Publication_Day', 'Publication_Time'],
    ['Episode_Num', 'Host_Popularity_percentage', 'Publication_Time', 'Genre'],
    ['Episode_Num', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Episode_Sentiment'],
    ['Episode_Num', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Genre'],
    ['Episode_Num', 'Host_Popularity_percentage', 'Episode_Sentiment', 'Podcast_Name'],
    ['Host_Popularity_percentage', 'Number_of_Ads', 'Episode_Sentiment', 'Podcast_Name'],
    ['Host_Popularity_percentage', 'Number_of_Ads', 'Publication_Day', 'Podcast_Name'],
    ['Host_Popularity_percentage', 'Number_of_Ads', 'Publication_Time', 'Podcast_Name'],
    
]

for comb in selected_comb:
    name = '_'.join(comb)
    if len(comb)==2:
        train[name] = train[comb[0]].astype(str) + '_' + train[comb[1]].astype(str)
        test[name] = test[comb[0]].astype(str) + '_' + test[comb[1]].astype(str)
        
    elif len(comb) == 3:
        train[name] = (train[comb[0]].astype(str) + '_' +
                       train[comb[1]].astype(str) + '_' +
                       train[comb[2]].astype(str))
        test[name] = (test[comb[0]].astype(str) + '_' +
                      test[comb[1]].astype(str) + '_' +
                      test[comb[2]].astype(str))
        
    elif len(comb) == 4:
        train[name] = (train[comb[0]].astype(str) + '_' +
                       train[comb[1]].astype(str) + '_' +
                       train[comb[2]].astype(str) + '_' +
                       train[comb[3]].astype(str))
        test[name] = (test[comb[0]].astype(str) + '_' +
                      test[comb[1]].astype(str) + '_' +
                      test[comb[2]].astype(str) + '_' +
                      test[comb[3]].astype(str))
    
    encoded_columns.append(name)

train[encoded_columns] = train[encoded_columns].astype('category')
test[encoded_columns] = test[encoded_columns].astype('category')



CATS = ['Podcast_Name', 'Episode_Num', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
NUMS = ['Episode_Length_minutes', 'Host_Popularity_percentage', 
        'Guest_Popularity_percentage', 'Number_of_Ads']


train[NUMS] = train[NUMS].fillna(train[NUMS].median())
test[NUMS] = test[NUMS].fillna(train[NUMS].median())


FEATURES = NUMS + CATS + encoded_columns

print(f"Train Shape: {train.shape}")
print(f"Test  Shape: {test.shape}")
train.head(3)


from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import gc


def target_encoder(df_train,df_val,col,stats='mean'):
    df = df_val.copy()
    agg = df_train.groupby(col)['Listening_Time_minutes'].agg(stats)

    col_name = f'te_{col}_mean'
    df[col_name] = df[col].map(agg).astype('float')
    df[col_name].fillna(agg.mean(),inplace=True)
    return df    


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin

class OrderedTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Outâ€‘ofâ€‘fold **mean-rank** encoder with optional smoothing.
    â€¢ Encodes each category by the *rank* of its target mean within a fold.
    â€¢ Unseen categories get the global mean rank (or âˆ’1 if not found).
    """

    def __init__(self, cat_cols=None, n_splits=5, smoothing=0):
        self.cat_cols   = cat_cols
        self.n_splits   = n_splits
        self.smoothing  = smoothing       # 0 = no smoothing
        self.maps_      = {}              # perâ€‘fold maps for CV
        self.global_map = {}              # fit on full data for inference

    def _make_fold_map(self, X_col, y):
        df_temp = pd.DataFrame({ "col": X_col, "target": y })

        means = df_temp.groupby("col")["target"].mean()
        if self.smoothing > 0:
            counts = df_temp.groupby("col")["target"].count()
            smooth = (counts * means + self.smoothing * y.mean()) / (counts + self.smoothing)
            means = smooth

        # Return dict where each category is mapped to a rank (0, 1, 2, ...)
        return {k: r for r, k in enumerate(means.sort_values().index)}

    def fit(self, X, y):
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)

        # Auto-detect categorical columns if not provided
        if self.cat_cols is None:
            self.cat_cols = X.select_dtypes(include='object').columns.tolist()

        kf = KFold(self.n_splits, shuffle=True, random_state=42)
        self.maps_ = {col: [None]*self.n_splits for col in self.cat_cols}

        # Create fold-specific maps to avoid leakage
        for fold, (tr_idx, _) in enumerate(kf.split(X)):
            x_train, y_train = X.loc[tr_idx], y.loc[tr_idx]
            for col in self.cat_cols:
                self.maps_[col][fold] = self._make_fold_map(X[col], y)

        # Create global map (used at inference when no fold is available)
        for col in self.cat_cols:
            self.global_map[col] = self._make_fold_map(X[col], y)

        return self

    def transform(self, X, y=None, fold=None):
        """
        Transform method
        â€¢ During CV: pass fold index to use foldâ€‘specific maps (leakâ€‘free).
        â€¢ At inference time (fold=None) uses global map.
        """
        X = X.copy()
        tgt_maps = {
            col: (self.global_map[col] if fold is None else self.maps_[col][fold])
            for col in self.cat_cols
        }

        for col, mapping in tgt_maps.items():
            X[col] = X[col].map(mapping).fillna(-1).astype(int)

        return X



folds = 10 
outer_kf = KFold(n_splits=folds, shuffle=True, random_state=42)
oof = np.zeros(len(train))
preds = np.zeros(len(test))
TARGET = 'Listening_Time_minutes'


for fold, (train_idx,val_idx) in enumerate(outer_kf.split(train),1):
    print(f'outer fold {fold}')
    x_train_raw = train.loc[train_idx,FEATURES].reset_index(drop=True)
    y_train_raw = train.loc[train_idx,'Listening_Time_minutes'].reset_index(drop=True)
    x_val_raw = train.loc[val_idx,FEATURES].reset_index(drop=True)
    y_val_raw = train.loc[val_idx,'Listening_Time_minutes'].reset_index(drop=True)

    x_test_raw = test[FEATURES].copy()

    x_train,x_val,x_test = x_train_raw.copy(),x_val_raw.copy(),x_test_raw.copy()

    inner_kf = KFold(n_splits=folds,shuffle = True, random_state = 42)

    for _,(in_train_idx,in_val_idx) in enumerate(inner_kf.split(x_train),1):
        inner_train = pd.concat([x_train_raw.loc[in_train_idx],y_train_raw.loc[in_train_idx]],axis=1)
        inner_val = x_train_raw.loc[in_val_idx].reset_index(drop=True)

        for col in encoded_columns:
            te_temp = target_encoder(inner_train,inner_val,col,stats='mean')
            te_col = f'te_{col}_mean'
            x_train.loc[in_val_idx,te_col] = te_temp[te_col].values
    
    train_with_y = pd.concat([x_train_raw, y_train_raw], axis=1)

    for col in encoded_columns:
        te_col = f'te_{col}_mean'
        x_val = target_encoder(train_with_y,x_val,col,stats='mean')
        x_test = target_encoder(train_with_y,x_test,col,stats='mean')

    x_train.drop(encoded_columns, axis=1, inplace=True)
    x_val.drop(encoded_columns, axis=1, inplace=True)
    x_test.drop(encoded_columns, axis=1, inplace=True)  

    enc = OrderedTargetEncoder(
        cat_cols=CATS,
        n_splits=folds,
        smoothing=20
    ).fit(x_train, y_train_raw)

    x_train[CATS] = enc.transform(x_train[CATS], fold=None)[CATS]
    x_val[CATS] = enc.transform(x_val[CATS], fold=None)[CATS]
    x_test[CATS] = enc.transform(x_test[CATS], fold=None)[CATS]

    model = XGBRegressor(
        tree_method='hist',
        max_depth=14,
        colsample_bytree=0.5,
        subsample=0.9,
        n_estimators=50_000,
        learning_rate=0.02,
        enable_categorical=True,
        min_child_weight=10,
        early_stopping_rounds=150,
    )

    model.fit(
        x_train, y_train_raw,
        eval_set=[(x_val, y_val_raw)],
        verbose=500
    )

    oof[val_idx]  = model.predict(x_val)
    preds += model.predict(x_test)

    del x_train_raw, x_val_raw, x_test_raw, x_train, x_val, x_test, y_train_raw, y_val_raw
    if fold != folds:
        del model
    gc.collect()

preds /= folds
rmse = mean_squared_error(train[TARGET], oof, squared=False)
print(f"Final OOF RMSE (XGB): {rmse:.5f}")


submit_df = pd.read_csv(r'/kaggle/input/playground-series-s5e4/sample_submission.csv')
submit_df['Listening_Time_minutes'] = preds
submit_df


submit_df.to_csv(r'/kaggle/working/results.csv',index=False)




