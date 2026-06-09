%load_ext cudf.pandas

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
import gc


from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingClassifier
from sklearn.decomposition import PCA
from itertools import combinations
from scipy.stats import zscore


from cuml import Ridge, Lasso, SVR, LinearRegression, KNeighborsRegressor, RandomForestRegressor, ElasticNet

import xgboost as xgb
import lightgbm as lgb

import warnings
warnings.simplefilter('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv", index_col="id")
test  = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv", index_col="id")

submission = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")




train.shape, test.shape


train.head()





def reduce_mem_usage(df, verbose=True):
    mem_before = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f"Memory usage of dataframe is {mem_before:.2f} MB")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if str(col_type)[:3] == 'int':
                c_min = df[col].min()
                c_max = df[col].max()
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)

                        
            elif str(col_type)[:5] == 'float':
                c_min = df[col].min()
                c_max = df[col].max()
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float64).min and c_max < np.finfo(np.float64).max:
                    df[col] = df[col].astype(np.float64)

    
    mem_after = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f"Memory usage after optimization is: {mem_after:.2f} MB")
        print(f"Decreased by {(100 * (mem_before - mem_after) / mem_before):.1f}%")
        
    return df


def create_binned_features(df, numerical_cols, bin_sizes=[10, 15, 20]):

    df_copy = df.copy()
    for col in numerical_cols:
        for size in bin_sizes:

            bins = np.arange(df_copy[col].min(), df_copy[col].max() + size, size)

            df_copy[f'{col}_bin_{size}'] = pd.cut(df_copy[col], bins=bins, labels=False, include_lowest=True)

            df_copy[f'{col}_bin_{size}'] = df_copy[f'{col}_bin_{size}'].fillna(-1).astype(int)
    return df_copy

def create_interaction_features(df, numerical_cols):

    df_copy = df.copy()

    col_pairs = list(combinations(numerical_cols, 2))

    for c1, c2 in col_pairs:
        # Create the new interaction feature
        df_copy[f"{c1}_plus_{c2}"] = df_copy[c1] + df_copy[c2]
        df_copy[f"{c1}_minus_{c2}"] = df_copy[c1] - df_copy[c2]
        # df_copy[f"{c2}_minus_{c1}"] = df_copy[c2] - df_copy[c1]
        df_copy[f"{c1}_times_{c2}"] = df_copy[c1] * df_copy[c2]
        df_copy[f"{c1}_div_{c2}"] = df_copy[c1] / (df_copy[c2] + 1e-6)
        # df_copy[f"{c2}_div_{c1}"] = df_copy[c2] / (df_copy[c1] + 1e-6)
    
    return df_copy


# took from the notebook: https://www.kaggle.com/code/conradkleykamp/song-bpm-prediction-lgbm-kfold-simple

def preprocess_and_engineer(df, method='zscore', is_train=True):
    
    # Making a fresh copy of the data
    df = df.copy()
    
    # --- Handling missing values (median imputation) ---
    # Not needed for this particular dataset but handy to have in general
    for col in df.columns:
        if df[col].dtype in ['float64']:
            df[col] = df[col].fillna(df[col].median())
    
    # --- Outlier handling (IQR or Z-score) ---
    # TRAIN ONLY, DO NOT APPLY TO TEST SET
    if is_train:
        # IQR method
        if method == 'iqr':
            num_cols = df.select_dtypes(include=['float64']).columns
            for col in num_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        # Zscore method
        elif method == 'zscore':
            num_cols = df.select_dtypes(include=['float64']).columns
            z_scores = df[num_cols].apply(zscore)
            df = df[(z_scores.abs() <= 3).all(axis=1)]
        # No outlier removal method
        else:
            pass

    # --- Transformations of features with high skewness ---
    # 'LogTrackDuration' --> Skew: 1.0363
    df['TrackDurationMs'] = np.log1p(df['TrackDurationMs'])
    # 'VocalContent' --> Skew: 0.7891
    df['VocalContent'] = np.sqrt(df['VocalContent'])
    # 'AcousticQuality' --> 0.7860
    df['AcousticQuality'] = np.sqrt(df['AcousticQuality'])
    
    # --- New features ---
    # Time-based features
    df['TrackDurationMin'] = df['TrackDurationMs'] / 60000
    df['RhythmPerMs'] = df['RhythmScore'] / (df['TrackDurationMs'] + 1)
    # Interactions
    df['LoudnessEnergy'] = df['AudioLoudness'] * df['Energy']
    df['RhythmEnergy'] = df['RhythmScore'] * df['Energy']
    df['AcousticVocal'] = df['AcousticQuality'] * df['VocalContent']
    df['MoodLive'] = df['MoodScore'] * df['LivePerformanceLikelihood'] 
    # Ratios
    # Establishing min_val to preserve magnitude of ratios
    min_val = 1e-3
    df['InstrVocalRatio'] = df['InstrumentalScore'] / df['VocalContent'].clip(lower=min_val)
    df['EnergyRhythmRatio'] = df['Energy'] / df['RhythmScore'].clip(lower=min_val)
    df['VocalAcousticRatio'] = df['VocalContent'] / df['AcousticQuality'].clip(lower=min_val)
    
    # --- Quantile rank features ---
    for col in ['Energy', 'MoodScore', 'TrackDurationMs']:
        df[f'{col}_Rank'] = df[col].rank(pct=True)
    
    return df



CATS = []
NUMS = []
for c in train.columns[:-1]:
    t = "CAT"
    if train[c].dtype=='object':
        CATS.append(c)
    else:
        NUMS.append(c)
        t = "NUM"
    n = train[c].nunique()
    na = train[c].isna().sum()
    print(f"[{t}] {c} has {n} unique and {na} NA")
print("CATS:", CATS )
print("NUMS:", NUMS )


# train_binned = create_binned_features(train, NUMS)
# test_binned = create_binned_features(test, NUMS)

# train_new = create_interaction_features(train, NUMS)
# test_new = create_interaction_features(test, NUMS)

# Applying function to df_train and df_test
train_new = preprocess_and_engineer(train, method='zscore', is_train=True)
test_new = preprocess_and_engineer(test, method='zscore', is_train=False)


train = reduce_mem_usage(train_new)
test = reduce_mem_usage(test_new)


def plot_binned_features(df, numerical_cols, bin_sizes=[10, 15, 20]):

    n_features = len(numerical_cols)
    n_bins = len(bin_sizes)
    
    # Create the Nx4 subplot grid
    fig, axes = plt.subplots(n_features, n_bins + 1, figsize=(16, 4 * n_features), tight_layout=True)
    fig.suptitle('Distribution of Original vs. Binned Features', fontsize=18, y=1.02)
    
    for i, col in enumerate(numerical_cols):
        # Plot the original feature distribution
        ax_orig = axes[i, 0]
        sns.histplot(data=df, x=col, kde=True, ax=ax_orig, color='skyblue')
        ax_orig.set_title(f'Original: {col}')
        ax_orig.set_xlabel('Value')
        ax_orig.set_ylabel('Frequency')

        # Plot each binned feature distribution
        for j, size in enumerate(bin_sizes):
            binned_col = f'{col}_bin_{size}'
            ax_bin = axes[i, j + 1]
            sns.histplot(data=df, x=binned_col, discrete=True, ax=ax_bin, color='salmon')
            ax_bin.set_title(f'Binned: {binned_col}')
            ax_bin.set_xlabel('Bin ID')
            ax_bin.set_ylabel('Count')

    plt.show()



def plot_numerical_features(train_df, test_df, numerical_cols):
    n_features = len(numerical_cols)
    
    # Create an Nx2 subplot grid (N features x 2 datasets)
    fig, axes = plt.subplots(n_features, 2, figsize=(16, 4 * n_features), tight_layout=True)
    fig.suptitle('Distribution of Numerical Features (Train vs. Test)', fontsize=16, y=1.02)
    
    # Ensure axes is a 2D array even if there is only one feature
    if n_features == 1:
        axes = np.array([axes])
        
    for i, col in enumerate(numerical_cols):
        # Plot the training data
        ax_train = axes[i, 0]
        sns.histplot(data=train_df, x=col, kde=True, ax=ax_train, color='skyblue')
        ax_train.set_title(f'Train: {col}')
        ax_train.set_xlabel('Value')
        ax_train.set_ylabel('Frequency')
        
        # Plot the testing data
        ax_test = axes[i, 1]
        sns.histplot(data=test_df, x=col, kde=True, ax=ax_test, color='salmon')
        ax_test.set_title(f'Test: {col}')
        ax_test.set_xlabel('Value')
        ax_test.set_ylabel('Frequency')

    plt.show()


# plot_numerical_features(train, test, NUMS)


X = train.drop(columns=["BeatsPerMinute"])
y = train["BeatsPerMinute"]


if len(train.columns) - 1 > 50:
    n_pca_components = min(len(train.columns) - 1, 50)
    pca = PCA(n_components=n_pca_components, random_state=42)
    
    X = pca.fit_transform(X)
    test = pca.transform(test)
    
    columns = [f"Component_{i}" for i in range(1, 51)]
    
    X = pd.DataFrame(X, columns=columns)
    test = pd.DataFrame(test, columns=columns)
    
    train = X.copy(deep=True)
    train["BeatsPerMinute"] = y.values
    
    train = reduce_mem_usage(train)
    test = reduce_mem_usage(test)

FEATURES = X.columns.to_list()


class IterLoadForDMatrix(xgb.core.DataIter):
    def __init__(self, df=None, features=None, target=None, batch_size=256*1024):
        self.features = features
        self.target = target
        self.df = df
        self.it = 0 
        self.batch_size = batch_size
        self.batches = int( np.ceil( len(df) / self.batch_size ) )
        super().__init__()

    def reset(self):
        '''Reset the iterator'''
        self.it = 0

    def next(self, input_data):
        '''Yield next batch of data.'''
        if self.it == self.batches:
            return 0 # Return 0 when there's no more batch.
        
        a = self.it * self.batch_size
        b = min( (self.it + 1) * self.batch_size, len(self.df) )
        #dt = cudf.DataFrame(self.df.iloc[a:b])
        dt = self.df.iloc[a:b]
        input_data(data=dt[self.features], label=dt[self.target]) 
        self.it += 1
        return 1


best_params_lgb = {
        'n_estimators': 20000,
        'learning_rate': 0.0005,
        'num_leaves': 100,
        'max_depth': 11,
        'min_child_samples': 10,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'reg_alpha': 2.5,
        'reg_lambda': 1.0,
        'random_state': 42,
        'verbosity': -1,
        
        'metric': 'rmse',
        'objective': 'regression',
        'boosting_type' :'gbdt',
        'seed': 42,
        'feature_pre_filter': False,
        'device': 'gpu',
    }

best_params_xgb = {        
        'learning_rate': 0.0005, 
        'max_depth': 7, 
        'min_child_weight': 9, 
        'gamma': 0.13, 
        'subsample': 0.7, 
        'colsample_bytree': 0.8, 
        'lambda': 2.0, 
        'alpha': 1.0,
    
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'seed': 42,
        'device':'cuda',
        'tree_method': 'hist',
    }


n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

y_preds_lgb = np.zeros(len(test))
y_preds_xgb = np.zeros(len(test))

y_train_lgb = np.zeros(len(train))
y_train_xgb = np.zeros(len(train))

models = []
val_rmses = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\nTraining fold {fold + 1}/{n_splits} >>>")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    model_lgb = lgb.LGBMRegressor(**best_params_lgb, verbose=500,)
    
    model_lgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(period=100)
        ],
    )
    
    models.append(model_lgb)
    
    y_preds_lgb += model_lgb.predict(test) / n_splits
    
    val_pred = model_lgb.predict(X_val)
    y_train_lgb[val_idx] += val_pred
    
    val_rmse = mean_squared_error(y_val, val_pred, squared=False)
    val_rmses.append(val_rmse)

    print(f"\nXGBoost Training started for {fold + 1}/{n_splits}\n")
    Xy_train = train.iloc[train_idx][ FEATURES+['BeatsPerMinute'] ].copy()
    
    Xy_train = IterLoadForDMatrix(Xy_train, FEATURES, 'BeatsPerMinute')
    dtrain = xgb.QuantileDMatrix(Xy_train, max_bin=256)
    dval   = xgb.DMatrix(X_val, label=y_val)
    dtest  = xgb.DMatrix(test)

    model_xgb = xgb.train(
        params=best_params_xgb,
        dtrain=dtrain,
        num_boost_round=5000,
        evals=[(dtrain, 'train'), (dval, "valid")],
        early_stopping_rounds=250,
        verbose_eval=500
    )
    
    y_preds_xgb += model_xgb.predict(dtest) / n_splits
    y_train_xgb[val_idx] += model_xgb.predict(dval)


print(f"\nMean LGBM RMSE: {np.mean(val_rmses):.6f}")


submission["BeatsPerMinute"] = (0.7*y_preds_lgb + 0.3*y_preds_xgb)
submission.to_csv("submission.csv", index=False)
submission.head(10)


submission["BeatsPerMinute"] = y_preds_lgb
submission.to_csv("submission_lightgbm_optuna.csv", index=False)

submission["BeatsPerMinute"] = y_preds_xgb
submission.to_csv("submission_xgboost_optuna.csv", index=False)


from tqdm import tqdm

cuml_models = {

    "ElasticNet" : ElasticNet(alpha=1.0, l1_ratio=0.8, fit_intercept=True, normalize=False, max_iter=3000, tol=0.001, solver='cd', 
                    selection='cyclic'),
    
    "ElasticNet2" : ElasticNet(alpha=1.0, l1_ratio=0.6, fit_intercept=True, normalize=False, max_iter=5000, tol=0.0001, solver='cd', 
                    selection='cyclic'),

    "KNeighborsRegressor" : KNeighborsRegressor(n_neighbors=23, weights='uniform'),
    
    "KNeighborsRegressor2" : KNeighborsRegressor(n_neighbors=37, weights='uniform'),
    
    "RandomForestRegressor": RandomForestRegressor(n_estimators=5000, max_depth=9, 
                                                   min_samples_split=6, min_samples_leaf=3, 
                                                   bootstrap=True, random_state=42),
    
    "RandomForestRegressor": RandomForestRegressor(n_estimators=3000, max_depth=7, 
                                                   min_samples_split=8, min_samples_leaf=5, 
                                                   bootstrap=True, random_state=42)
}

results = {}
results_test = {}
results_train = {}

print("--- Starting Training and Evaluation with Parameters ---")
for model_name, model in tqdm(cuml_models.items(), desc="Training Models"):
    print(f"Training {model_name}...")
    
    model.fit(X, y)
    y_train_pred = model.predict(X)
    y_pred = model.predict(test)

    results_test[model_name] = y_pred
    results_train[model_name] = y_train_pred

    rmse_train = np.sqrt(mean_squared_error(y, y_train_pred))
    
    results[model_name] = rmse_train
    print(f"{model_name} RMSE train: {rmse_train:.4f}\n")

print("--- Final Results ---")
for model_name, rmse_value in results.items():
    print(f"{model_name}: {rmse_value:.4f}")


oof_pred_probs ={
    "LGBM":y_train_lgb,
    "XGB":y_train_xgb,
    "combined": (0.7*y_train_lgb + 0.3*y_train_xgb),
}


test_pred_probs = {
    "LGBM":y_preds_lgb,
    "XGB":y_preds_xgb,
    "combined": (0.7*y_preds_lgb + 0.3*y_preds_xgb),
}

oof_pred_probs.update(results_train)
test_pred_probs.update(results_test)

oof_pred_df = pd.DataFrame(oof_pred_probs)
test_pred_df = pd.DataFrame(test_pred_probs)

oof_pred_df.head()


meta_model = RandomForestRegressor(n_estimators=4000, max_depth=7, 
                                                   min_samples_split=5, min_samples_leaf=3, 
                                                   max_features='sqrt', bootstrap=True, random_state=42)

meta_model.fit(oof_pred_df, y)

meta_oof = meta_model.predict(oof_pred_df)
meta_test_preds = meta_model.predict(test_pred_df)
print("Stacking RF RMSE:", np.sqrt(mean_squared_error(y.to_numpy(), y_train_pred.to_numpy())))


oof_pred_probs["meta_model"] = meta_oof
test_pred_probs["meta_model"] = meta_test_preds


def objective(trial):
    weights = np.array([trial.suggest_float(f'w{m}', -1, 1) for m in oof_pred_probs.keys()])
    weights /= np.sum(weights)

    preds = np.zeros(len(y))
    for m, weight in zip(oof_pred_probs.keys(), weights):
        preds += oof_pred_probs[m] * weight

    return np.sqrt(mean_squared_error(y.to_numpy(), preds.to_numpy()))

sampler = optuna.samplers.TPESampler(seed=42, multivariate=True, n_startup_trials=5)
study = optuna.create_study(direction='minimize', sampler=sampler)
study.optimize(objective, n_trials=100, n_jobs=-1)


import json

best_params = study.best_params

best_weights = np.array([study.best_params[f'w{m}'] for m in oof_pred_probs.keys()])
best_weights /= np.sum(best_weights)

best_weights = {
    model: weight for model, weight in sorted(
        zip(oof_pred_probs.keys(), best_weights),
        key=lambda x: x[1],
        reverse=True
    )
}
print(json.dumps(best_weights, indent=2))


weighted_test_preds = np.zeros(len(test_pred_probs["XGB"]))
weighted_train_preds = np.zeros(len(oof_pred_probs["XGB"]))

for m, weight in best_weights.items():
    weighted_test_preds += test_pred_probs[m] * weight
    weighted_train_preds += oof_pred_probs[m] * weight

print(f"Weighted oof train RMSE: {np.sqrt(mean_squared_error(y, weighted_train_preds)):.5f}")


submission['BeatsPerMinute'] = weighted_test_preds.to_numpy()
submission.to_csv('submission_weighted.csv', index=False)
submission.head(15)




