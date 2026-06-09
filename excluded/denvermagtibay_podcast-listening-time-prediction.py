# === Setup ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from category_encoders import TargetEncoder
import gc, warnings; warnings.filterwarnings('ignore')


# === Load Data ===
train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
extra_data = pd.read_csv('/kaggle/input/original-podcast-data/podcast_dataset.csv')


# === Append Extra Data to Training Set ===
extra_clean = extra_data.dropna(subset=['Listening_Time_minutes']).drop_duplicates()
train = pd.concat([train, extra_clean], ignore_index=True)


# === Feature Engineering ===
def feat_eng(df):
    df = df.copy()
    df['Episode_Num'] = df['Episode_Title'].str.extract(r'(\d+)')
    df['is_weekend'] = df['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)
    return df.drop(columns=['Episode_Title'])

train = feat_eng(train)
test = feat_eng(test)

for k in range(3):
    train[f'ELm_r{k}'] = train['Episode_Length_minutes'].round(k)
    test[f'ELm_r{k}'] = test['Episode_Length_minutes'].round(k)


# === Interaction Features ===
interaction_cols = [
    ['Episode_Length_minutes', 'Host_Popularity_percentage'],
    ['Episode_Num', 'Guest_Popularity_percentage'],
    ['Host_Popularity_percentage', 'Episode_Sentiment']
]
encoded_columns = []

for cols in interaction_cols:
    name = '_'.join(cols)
    train[name] = train[cols[0]].astype(str) + '_' + train[cols[1]].astype(str)
    test[name] = test[cols[0]].astype(str) + '_' + test[cols[1]].astype(str)
    encoded_columns.append(name)

train[encoded_columns] = train[encoded_columns].astype('category')
test[encoded_columns] = test[encoded_columns].astype('category')


# === Feature Types ===
CATS = ['Podcast_Name', 'Episode_Num', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
NUMS = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads']
train[NUMS] = train[NUMS].fillna(train[NUMS].median())
test[NUMS] = test[NUMS].fillna(train[NUMS].median())

FEATURES = NUMS + CATS + encoded_columns
TARGET = 'Listening_Time_minutes'


# === Target Encoding Function ===
def target_encoder(df_train, df_val, col):
    agg = df_train.groupby(col)[TARGET].mean()
    col_name = f'te_{col}_mean'
    df_val[col_name] = df_val[col].map(agg).fillna(agg.mean())
    return df_val


# === Custom Rank-Based Encoder ===
from sklearn.base import BaseEstimator, TransformerMixin

class OrderedTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cat_cols=None, n_splits=5, smoothing=0):
        self.cat_cols = cat_cols
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.maps_ = {}
        self.global_map = {}

    def _make_fold_map(self, X_col, y):
        df_temp = pd.DataFrame({"col": X_col, "target": y})
        means = df_temp.groupby("col")["target"].mean()
        if self.smoothing > 0:
            counts = df_temp.groupby("col")["target"].count()
            means = (counts * means + self.smoothing * y.mean()) / (counts + self.smoothing)
        return {k: r for r, k in enumerate(means.sort_values().index)}

    def fit(self, X, y):
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)
        if self.cat_cols is None:
            self.cat_cols = X.select_dtypes(include='object').columns.tolist()
        kf = KFold(self.n_splits, shuffle=True, random_state=42)
        self.maps_ = {col: [None]*self.n_splits for col in self.cat_cols}
        for fold, (tr_idx, _) in enumerate(kf.split(X)):
            x_train, y_train = X.loc[tr_idx], y.loc[tr_idx]
            for col in self.cat_cols:
                self.maps_[col][fold] = self._make_fold_map(x_train[col], y_train)
        for col in self.cat_cols:
            self.global_map[col] = self._make_fold_map(X[col], y)
        return self

    def transform(self, X, y=None, fold=None):
        X = X.copy()
        tgt_maps = {
            col: (self.global_map[col] if fold is None else self.maps_[col][fold])
            for col in self.cat_cols
        }
        for col, mapping in tgt_maps.items():
            X[col] = X[col].map(mapping).fillna(-1).astype(int)
        return X


# === Cross-Validation and Training ===
FOLDS = 10
outer_kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
oof = np.zeros(len(train))
preds = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(outer_kf.split(train), 1):
    print(f'\nğŸ“¦ Fold {fold}')
    
    x_train_raw = train.loc[train_idx, FEATURES].reset_index(drop=True)
    y_train = train.loc[train_idx, TARGET].reset_index(drop=True)
    x_val_raw = train.loc[val_idx, FEATURES].reset_index(drop=True)
    y_val = train.loc[val_idx, TARGET].reset_index(drop=True)
    x_test_raw = test[FEATURES].copy()
    
    x_train, x_val, x_test = x_train_raw.copy(), x_val_raw.copy(), x_test_raw.copy()
    
    inner_kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    for _, (in_train_idx, in_val_idx) in enumerate(inner_kf.split(x_train), 1):
        inner_train = pd.concat([x_train_raw.loc[in_train_idx], y_train.loc[in_train_idx]], axis=1)
        inner_val = x_train_raw.loc[in_val_idx].reset_index(drop=True)
        for col in encoded_columns:
            te_temp = target_encoder(inner_train, inner_val, col)
            x_train.loc[in_val_idx, f'te_{col}_mean'] = te_temp[f'te_{col}_mean'].values

    train_with_y = pd.concat([x_train_raw, y_train], axis=1)
    for col in encoded_columns:
        x_val = target_encoder(train_with_y, x_val, col)
        x_test = target_encoder(train_with_y, x_test, col)

    x_train.drop(encoded_columns, axis=1, inplace=True)
    x_val.drop(encoded_columns, axis=1, inplace=True)
    x_test.drop(encoded_columns, axis=1, inplace=True)

    encoder = OrderedTargetEncoder(cat_cols=CATS, n_splits=FOLDS, smoothing=20).fit(x_train, y_train)
    x_train[CATS] = encoder.transform(x_train[CATS], fold=None)[CATS]
    x_val[CATS] = encoder.transform(x_val[CATS], fold=None)[CATS]
    x_test[CATS] = encoder.transform(x_test[CATS], fold=None)[CATS]

    model = XGBRegressor(
        tree_method='hist',
        max_depth=14,
        colsample_bytree=0.5,
        subsample=0.9,
        n_estimators=50000,
        learning_rate=0.02,
        min_child_weight=10,
        early_stopping_rounds=150
    )

    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=500)
    oof[val_idx] = model.predict(x_val)
    preds += model.predict(x_test)

    del x_train_raw, x_val_raw, x_test_raw, x_train, x_val, x_test, y_train, y_val
    if fold != FOLDS:
        del model
    gc.collect()


# === Final Evaluation ===
preds /= FOLDS
final_rmse = mean_squared_error(train[TARGET], oof, squared=False)
print(f'\nğŸ“Š Final OOF RMSE: {final_rmse:.5f}')


# === Submission File ===
sample_sub['Listening_Time_minutes'] = preds
sample_sub.to_csv('/kaggle/working/submission.csv', index=False)
sample_sub.head()

