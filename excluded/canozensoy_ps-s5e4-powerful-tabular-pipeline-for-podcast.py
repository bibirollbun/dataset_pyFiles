# ======================= Imports & Global Settings ======================= #
import os
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from cuml.preprocessing import TargetEncoder
from tqdm.auto import tqdm
from itertools import combinations

warnings.simplefilter('ignore')
pd.options.mode.copy_on_write = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

SEED = 42
N_FOLDS = 7
DEVICE = 'cuda'

np.random.seed(SEED)


# ======================= Utility Functions ======================= #
def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

seed_everything(SEED)


# ======================= Load Data ======================= #
def load_data():
    train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
    test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
    original = pd.read_csv("/kaggle/input/original-podcast-dataset/podcast_dataset.csv")
    
    df = pd.concat([train, original, test], axis=0, ignore_index=True)
    df.drop(columns=['id'], inplace=True)
    df = df.drop_duplicates()
    return df, len(test)


# ======================= Preprocessing ======================= #
def preprocess(df):
    df['Episode_Length_minutes'] = np.clip(df['Episode_Length_minutes'], 0, 120)
    df['Host_Popularity_percentage'] = np.clip(df['Host_Popularity_percentage'], 20, 100)
    df['Guest_Popularity_percentage'] = np.clip(df['Guest_Popularity_percentage'], 0, 100)
    df.loc[df['Number_of_Ads'] > 3, 'Number_of_Ads'] = 0

    day_mapping = {'Monday':1, 'Tuesday':2, 'Wednesday':3, 'Thursday':4, 'Friday':5, 'Saturday':6, 'Sunday':7}
    time_mapping = {'Morning':1, 'Afternoon':2, 'Evening':3, 'Night':4}
    sentiment_map = {'Negative':1, 'Neutral':2, 'Positive':3}
    
    df['Publication_Day'] = df['Publication_Day'].map(day_mapping)
    df['Publication_Time'] = df['Publication_Time'].map(time_mapping)
    df['Episode_Sentiment'] = df['Episode_Sentiment'].map(sentiment_map)
    
    df['Episode_Title'] = df['Episode_Title'].str.replace('Episode ', '', regex=True).astype(int)
    
    le = LabelEncoder()
    for col in df.select_dtypes('object').columns:
        df[col] = le.fit_transform(df[col]) + 1

    return df


# ======================= Feature Engineering ======================= #
def feature_engineering(df):
    for col in ['Episode_Length_minutes']:
        df[col + '_sqrt'] = np.sqrt(df[col])
        df[col + '_squared'] = df[col] ** 2

    group_cols = [
        'Episode_Sentiment', 'Genre', 'Publication_Day', 'Podcast_Name',
        'Episode_Title', 'Guest_Popularity_percentage', 'Host_Popularity_percentage', 'Number_of_Ads'
    ]
    
    for col in tqdm(group_cols, desc="Group Mean Encoding"):
        df[col + '_EP'] = df.groupby(col)['Episode_Length_minutes'].transform('mean')
    
    return df


# ======================= Combination Feature Creation ======================= #
def process_combinations_fast(df, columns_to_encode, pair_sizes, max_batch_size=2000):
    str_df = df[columns_to_encode].astype(str)
    le = LabelEncoder()
    
    if isinstance(pair_sizes, int):
        pair_sizes = [pair_sizes]

    for r in pair_sizes:
        print(f"\nProcessing {r}-combinations...")
        combos_iter = combinations(columns_to_encode, r)
        n_combinations = np.math.comb(len(columns_to_encode), r)
        print(f"Total {r}-combinations to process: {n_combinations}")

        batch_cols = []
        batch_names = []

        with tqdm(total=n_combinations, desc=f"{r}-combinations") as pbar:
            while True:
                batch_cols.clear()
                batch_names.clear()

                for _ in range(max_batch_size):
                    try:
                        cols = next(combos_iter)
                        batch_cols.append(list(cols))
                        batch_names.append('+'.join(cols))
                    except StopIteration:
                        break

                if not batch_cols:
                    break

                for cols, new_name in zip(batch_cols, batch_names):
                    result = str_df[cols[0]].copy()
                    for col in cols[1:]:
                        result += str_df[col]
                    df[new_name] = le.fit_transform(result) + 1
                    pbar.update(1)

        print(f"Completed {r}-combinations. Total columns now: {len(df.columns)}")
    return df


# ======================= Target Encoding ======================= #
def target_encode(train_df, valid_df, test_df, target, early_features=20):
    features = train_df.columns
    encoder = TargetEncoder(n_folds=5, seed=SEED, stat="mean")
    
    for col in tqdm(features[:early_features], desc="Target Encoding (new cols)"):
        train_df[col+'_te1'] = encoder.fit_transform(train_df[[col]], target)
        valid_df[col+'_te1'] = encoder.transform(valid_df[[col]])
        test_df[col+'_te1'] = encoder.transform(test_df[[col]])
    
    for col in tqdm(features[early_features:], desc="Target Encoding (overwrite)"):
        train_df[col] = encoder.fit_transform(train_df[[col]], target)
        valid_df[col] = encoder.transform(valid_df[[col]])
        test_df[col] = encoder.transform(test_df[[col]])

    return train_df, valid_df, test_df


# ======================= Data Split ======================= #
def prepare_train_test(df, test_len):
    df_train = df.iloc[:-test_len]
    df_test = df.iloc[-test_len:].reset_index(drop=True)

    df_train = df_train[df_train['Listening_Time_minutes'].notnull()]
    target = df_train.pop('Listening_Time_minutes')
    df_test.drop(columns=['Listening_Time_minutes'], inplace=True)

    return df_train, df_test, target


# ======================= Learning Rate Scheduler ======================= #
def lr_decay(epoch):
    if epoch < 100:
        return 0.035
    else:
        return 0.015


# ======================= Model Training Functions ======================= #
def train_xgb_model(X, y, X_test, folds):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'seed': SEED,
        'max_depth': 18,
        'learning_rate': 0.035,
        'min_child_weight': 60,
        'reg_alpha': 4,
        'reg_lambda': 2,
        'subsample': 0.85,
        'colsample_bytree': 0.7,
        'colsample_bynode': 0.5,
        'device': DEVICE
    }

    callbacks = [xgb.callback.LearningRateScheduler(lr_decay)]
    
    test_preds = np.zeros(len(X_test))
    
    for fold_idx, (train_idx, valid_idx) in enumerate(folds.split(X), 1):
        print(f"\nðŸ”µ Fold {fold_idx} XGBoost Training...")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
        X_test_fold = X_test[X_train.columns].copy()

        X_train, X_valid, X_test_fold = target_encode(X_train, X_valid, X_test_fold, y_train)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)
        dtest = xgb.DMatrix(X_test_fold)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=100000,
            evals=[(dtrain, 'train'), (dvalid, 'valid')],
            early_stopping_rounds=30,
            verbose_eval=500,
            callbacks=callbacks
        )

        preds = np.clip(model.predict(dtest), 0, 120)
        test_preds += preds

    test_preds /= N_FOLDS
    return test_preds

def train_lgb_model(X, y, X_test, folds):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.035,
        'num_leaves': 512,
        'min_child_samples': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'reg_alpha': 4,
        'reg_lambda': 2,
        'random_state': SEED,
        'device': 'gpu'
    }

    test_preds = np.zeros(len(X_test))
    
    for fold_idx, (train_idx, valid_idx) in enumerate(folds.split(X), 1):
        print(f"\nðŸŸ¢ Fold {fold_idx} LightGBM Training...")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
        X_test_fold = X_test[X_train.columns].copy()

        X_train, X_valid, X_test_fold = target_encode(X_train, X_valid, X_test_fold, y_train)

        train_dataset = lgb.Dataset(X_train, y_train)
        valid_dataset = lgb.Dataset(X_valid, y_valid, reference=train_dataset)

        model = lgb.train(
            params,
            train_dataset,
            num_boost_round=100000,
            valid_sets=[train_dataset, valid_dataset],
            callbacks=[
                lgb.early_stopping(30),
                lgb.log_evaluation(500)
            ]
        )

        preds = np.clip(model.predict(X_test_fold), 0, 120)
        test_preds += preds

    test_preds /= N_FOLDS
    return test_preds

def train_cat_model(X, y, X_test, folds):
    params = {
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'random_seed': SEED,
        'depth': 8,
        'learning_rate': 0.035,
        'l2_leaf_reg': 3,
        'border_count': 32,
        'verbose': 500,
        'task_type': 'GPU' if DEVICE == 'cuda' else 'CPU'
    }

    test_preds = np.zeros(len(X_test))

    for fold_idx, (train_idx, valid_idx) in enumerate(folds.split(X), 1):
        print(f"\nðŸ’œ Fold {fold_idx} CatBoost Training...")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
        X_test_fold = X_test[X_train.columns].copy()

        X_train, X_valid, X_test_fold = target_encode(X_train, X_valid, X_test_fold, y_train)

        model = CatBoostRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            early_stopping_rounds=30,
            verbose=500
        )

        preds = np.clip(model.predict(X_test_fold), 0, 120)
        test_preds += preds

    test_preds /= N_FOLDS
    return test_preds


# ======================= Blending ======================= #
def blend_predictions(xgb_preds, lgb_preds, cat_preds, weights=(0.5, 0.3, 0.2)):
    w_xgb, w_lgb, w_cat = weights
    return (w_xgb * xgb_preds) + (w_lgb * lgb_preds) + (w_cat * cat_preds)


# ======================= Save Submission ======================= #
def save_submission(preds, filename="submission.csv"):
    sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")
    sample_submission['Listening_Time_minutes'] = preds
    sample_submission.to_csv(filename, index=False)
    print(f"ðŸš€ Submission file saved as: {filename}")


# ======================= Main ======================= #
if __name__ == "__main__":
    df, test_len = load_data()
    df = preprocess(df)
    df = feature_engineering(df)
    columns_to_combine = [
        'Episode_Length_minutes', 'Episode_Title', 'Publication_Time', 'Host_Popularity_percentage', 
        'Number_of_Ads', 'Episode_Sentiment', 'Publication_Day', 'Podcast_Name', 'Genre', 'Guest_Popularity_percentage'
    ]
    df = process_combinations_fast(df, columns_to_combine, pair_sizes=[2,3,5,7], max_batch_size=1000)
    df = df.astype('float32')

    df_train, df_test, target = prepare_train_test(df, test_len)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    xgb_preds = train_xgb_model(df_train, target, df_test, kf)
    lgb_preds = train_lgb_model(df_train, target, df_test, kf)
    cat_preds = train_cat_model(df_train, target, df_test, kf)

    final_preds = blend_predictions(xgb_preds, lgb_preds, cat_preds, weights=(0.5, 0.3, 0.2))

    save_submission(final_preds)

