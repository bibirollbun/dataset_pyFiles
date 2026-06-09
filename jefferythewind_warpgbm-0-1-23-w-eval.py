import kagglehub
kagglehub.login()


# Upgrade Torch to 2.6.0+CUDA 12.4
!pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Confirm torch version
import torch
print("Torch version:", torch.__version__)
print("Torch CUDA version:", torch.version.cuda)



import torch
print(torch.__version__)
print(torch.version.cuda)
!nvidia-smi


!pip install warpgbm --no-build-isolation


from warpgbm import WarpGBM



PATH = kagglehub.competition_download('playground-series-s5e4')
OG_PATH = kagglehub.dataset_download('ysthehurricane/podcast-listening-time-prediction-dataset')



import pandas as pd
import numpy as np
from itertools import combinations
from joblib import Parallel, delayed
from tqdm.auto import tqdm


# ─── Helper for combo‐factorization ─────────────────────────────────────────
def make_combo_codes(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """
    Build a structured array from `cols` and factorize in C.
    Returns a 1-based int32 array of codes.
    """
    rec, _ = pd.factorize(df[cols].to_records(index=False))
    return rec.astype(np.int32) + 1

# ─── 1) Load & concat ────────────────────────────────────────────────────────
df_train    = pd.read_csv(f"{PATH}/train.csv")
df_original = pd.read_csv(f"{OG_PATH}/podcast_dataset.csv")
df_test     = pd.read_csv(f"{PATH}/test.csv")

df = pd.concat([df_train, df_original, df_test], ignore_index=True)
df.drop(columns=['id'], inplace=True)
# df.drop_duplicates(inplace=True)

# ─── 2) Outlier clipping (vectorized) ───────────────────────────────────────
#df['Episode_Length_minutes']      = df['Episode_Length_minutes'].clip(0, 120)
#df['Host_Popularity_percentage']  = df['Host_Popularity_percentage'].clip(20, 100)
#df['Guest_Popularity_percentage'] = df['Guest_Popularity_percentage'].clip(0, 100)
#df.loc[df['Number_of_Ads'] > 3, 'Number_of_Ads'] = 0

# ─── 3) Map days / times / sentiment ────────────────────────────────────────
day_map  = {'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,
            'Friday':5,'Saturday':6,'Sunday':7}
time_map = {'Morning':1,'Afternoon':2,'Evening':3,'Night':4}
sent_map = {'Negative':1,'Neutral':2,'Positive':3}

df['Publication_Day']    = df['Publication_Day'].map(day_map).astype(np.int32)
df['Publication_Time']   = df['Publication_Time'].map(time_map).astype(np.int32)
df['Episode_Sentiment']  = df['Episode_Sentiment'].map(sent_map).astype(np.int32)

df['Episode_Title'] = (
    df['Episode_Title']
      .str.replace(r'Episode\s*', '', regex=True)
      .astype(int)
      .astype(np.int32)
)

# ─── 4) Encode remaining object cols ─────────────────────────────────────────
obj_cols = list(df.select_dtypes(include=['object']).columns)
for col in tqdm(obj_cols, desc="Categorical → codes"):
    df[col] = df[col].astype('category').cat.codes.astype(np.int32)

# ─── 5) Numeric transforms ─────────────────────────────────────────────────
for col in tqdm(['Episode_Length_minutes'], desc="Numeric feats"):
    df[f'{col}_sqrt']    = np.sqrt(df[col])
    df[f'{col}_squared'] = df[col] ** 2

# ─── 6) Group‐mean EP features ───────────────────────────────────────────────
ep_cols = [
    'Episode_Sentiment','Genre','Publication_Day',
    'Podcast_Name','Episode_Title',
    'Guest_Popularity_percentage','Host_Popularity_percentage',
    'Number_of_Ads'
]
for c in tqdm(ep_cols, desc="Engineering EP features"):
    df[f'{c}_EP'] = (
        df.groupby(c)['Episode_Length_minutes']
          .transform('mean')
          .astype(np.float32)
    )

# ─── 7) High‐order combos via batch insert ──────────────────────────────────
comb_cols  = [
    'Episode_Length_minutes','Episode_Title','Publication_Time',
    'Host_Popularity_percentage','Number_of_Ads','Episode_Sentiment',
    'Publication_Day','Podcast_Name','Genre','Guest_Popularity_percentage'
]
pair_sizes = [2, 3, 5, 7]

# Collect all new features in a dict to avoid fragmentation
new_feats: dict[str, np.ndarray] = {}

for r in tqdm(pair_sizes, desc="Combo sizes"):
    combos = list(combinations(comb_cols, r))
    # compute codes in parallel
    codes_list = Parallel(n_jobs=-1)(
        delayed(make_combo_codes)(df, list(c)) for c in combos
    )
    for combo, arr in zip(combos, codes_list):
        new_feats['+'.join(combo)] = arr

# one-time concat of all new columns
new_df = pd.DataFrame(new_feats, index=df.index)
df = pd.concat([df, new_df], axis=1)

# ─── 8) Split back & finalize ───────────────────────────────────────────────
n_test        = len(df_test)
df_test_final = df.iloc[-n_test:].reset_index(drop=True)
df_train_final= df.iloc[:-n_test].reset_index(drop=True)
df_train_final= df_train_final[df_train_final['Listening_Time_minutes'].notnull()]

y_train       = df_train_final.pop('Listening_Time_minutes').astype(np.float32)
df_test_final = df_test_final.drop(columns=['Listening_Time_minutes'])

# cast all to float32
df_train_final = df_train_final.astype(np.float32)
df_test_final  = df_test_final.astype(np.float32)

print("Train shape:", df_train_final.shape)
print("Test  shape:", df_test_final.shape)



import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GroupKFold
from cuml.preprocessing import TargetEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm
from warpgbm import WarpGBM

overwrite = True

# ─── 0) assume df_train_final, df_test_final, y_train already exist ─────────
FOLDS = 7
groups = None  # or your grouping series
cv = GroupKFold(n_splits=FOLDS) if groups is not None else KFold(n_splits=FOLDS)

oof_preds  = np.zeros(len(df_train_final), dtype=np.float32)
test_preds = np.zeros(len(df_test_final),  dtype=np.float32)

# TE cols = everything from column #20 onward
te_cols     = df_train_final.columns.tolist()[20:]
static_cols = [c for c in df_train_final.columns if c not in te_cols]

test_preds = []

# ─── CV LOOP ────────────────────────────────────────────────────────────────
for fold, (tr_idx, val_idx) in tqdm(
    enumerate(cv.split(df_train_final, y_train, groups), start=1),
    total=FOLDS,
    desc="CV folds"
):
    # — split out static vs TE parts
    X_full_tr = df_train_final.iloc[tr_idx].reset_index(drop=True)
    X_full_va = df_train_final.iloc[val_idx].reset_index(drop=True)
    X_full_te = df_test_final.reset_index(drop=True)
    y_tr = y_train.iloc[tr_idx].reset_index(drop=True)
    y_va = y_train.iloc[val_idx].reset_index(drop=True)

    X_tr_static = X_full_tr[static_cols]
    X_va_static = X_full_va[static_cols]
    X_te_static = X_full_te[static_cols]

    # — prepare paths for cached files
    fold_dir = f"fold_{fold}"
    os.makedirs(fold_dir, exist_ok=True)
    tr_te_path = os.path.join(fold_dir, "train_te.parquet")
    va_te_path = os.path.join(fold_dir, "val_te.parquet")
    te_te_path = os.path.join(fold_dir, "test_te.parquet")

    # — load if exist, else compute and save
    if all(os.path.exists(p) for p in [tr_te_path, va_te_path, te_te_path]) and overwrite == False:
        df_tr_te = pd.read_parquet(tr_te_path)
        df_va_te = pd.read_parquet(va_te_path)
        df_te_te = pd.read_parquet(te_te_path)
        print(f"Loaded cached target encodings for fold {fold}")
    else:
        df_tr_te = pd.DataFrame(index=X_tr_static.index, columns=te_cols)
        df_va_te = pd.DataFrame(index=X_va_static.index, columns=te_cols)
        df_te_te = pd.DataFrame(index=X_te_static.index, columns=te_cols)

        for col in tqdm(te_cols, desc=f"Fold {fold} TE"):
            te = TargetEncoder(n_folds=7)
            tr_enc = te.fit_transform(X_full_tr[[col]], y_tr).astype(np.float32)
            va_enc = te.transform(X_full_va[[col]]).astype(np.float32)
            te_enc = te.transform(X_full_te[[col]]).astype(np.float32)

            df_tr_te[col] = tr_enc
            df_va_te[col] = va_enc
            df_te_te[col] = te_enc

        # Save computed TEs
        df_tr_te.to_parquet(tr_te_path)
        df_va_te.to_parquet(va_te_path)
        df_te_te.to_parquet(te_te_path)
        print(f"Saved target encodings for fold {fold}")

    # — now one‐time concat per fold
    X_tr = pd.concat([X_tr_static, df_tr_te], axis=1)
    X_va = pd.concat([X_va_static, df_va_te], axis=1)
    X_te = pd.concat([X_te_static, df_te_te], axis=1)

    # ─── train & predict (same as before) ─────────────────────────────────────
    # model = XGBRegressor(
    #     tree_method="hist",
    #     device="cuda",
    #     max_depth=16,
    #     learning_rate=.04,
    #     n_estimators=2000,
    #     colsample_bytree=1,
    #     subsample=1.0,
    #     early_stopping_rounds=100,
    #     eval_metric="rmse",
    #     n_jobs=-1,
    #     verbosity=1,
    #     min_child_weight=300,
    #     max_bin=160,
    #     enable_categorical=True
    # )
    # model.fit(
    #     X_tr, y_tr,
    #     eval_set=[(X_va, y_va)],
    #     verbose=10
    # )

    model = WarpGBM(
        max_depth=18,
        num_bins=127,
        n_estimators=5000,
        learning_rate=0.03,
        threads_per_block=64,
        rows_per_thread=4,
        min_split_gain=0,
        min_child_weight=100,
        # L2_reg=1
    )
    
    model.fit(
        X_tr.values, y_tr.values,
        X_eval=X_va.values, y_eval=y_va.values,
        eval_every_n_trees = 10,
        early_stopping_rounds = 3
    )

    oof_preds[val_idx] = model.predict(X_va.values)
    test_preds.append( model.predict(X_te.values) )

# model.fit( 
#     pd.concat([X_tr, X_va]).values,
#     pd.concat([y_tr, y_va]).values
# )
# test_preds += model.predict(X_te.values)

# ─── finalize (same as before) ───────────────────────────────────────────────
rmse = np.sqrt( mean_squared_error(y_train, oof_preds) )
print(f"OOF RMSE: {rmse:.4f}")



final_preds = np.mean( np.stack(test_preds[:6]), axis=0 )

sub = pd.read_csv(f'{PATH}/sample_submission.csv')
sub["Listening_Time_minutes"] = final_preds
sub.to_csv('submission.csv', index=False)
sub.head(3)

