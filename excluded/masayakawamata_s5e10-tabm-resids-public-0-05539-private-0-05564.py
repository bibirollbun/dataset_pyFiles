!pip install -qq pytabkit==1.5.2


import warnings
warnings.simplefilter('ignore')


import numpy as np, pandas as pd

train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
orig = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_100k.csv')
orig_2 = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_10k.csv')
orig_3 = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_2k.csv')

orig = pd.concat([orig, orig_2, orig_3])

print('Train Shape:', train.shape)
print('Test Shape:', test.shape)
print('Orig Shape:', orig.shape)

train.head(3)


TARGET = 'accident_risk'
y_true = train[TARGET]
BASE = [col for col in train.columns if col not in ['id', TARGET]]
CATS = ['road_type', 'lighting', 'weather', 'road_signs_present', 'public_road', 'time_of_day', 'holiday', 'school_season']

print(f'{len(BASE)} Base Features:{BASE}')


DIGIT = []

for df in [train, test, orig]:
    df['curvature_d1'] = (df['curvature'] * 10).astype(int) % 10
    df['curvature_d2'] = (df['curvature'] * 100).astype(int) % 10

DIGIT.extend(['curvature_d1', 'curvature_d2'])
train[['curvature', 'curvature_d1', 'curvature_d2']].head(3)


ORIG = []

for col in BASE:
    # MEAN
    mean_map = orig.groupby(col)[TARGET].mean()
    new_mean_col_name = f"orig_mean_{col}"
    mean_map.name = new_mean_col_name
    
    train = train.merge(mean_map, on=col, how='left')
    test = test.merge(mean_map, on=col, how='left')
    ORIG.append(new_mean_col_name)

    # MIN
    min_map = orig.groupby(col)[TARGET].min()
    new_min_col_name = f"orig_min_{col}"
    min_map.name = new_min_col_name
    
    train = train.merge(min_map, on=col, how='left')
    test = test.merge(min_map, on=col, how='left')
    ORIG.append(new_min_col_name)

print(len(ORIG), 'Orig Features Created!!')


from scipy.stats import norm

def f(X):
    return \
    0.304512 * X["curvature"] + \
    0.189685 * (X["lighting"] == "night").astype(int) + \
    0.092958 * (X["weather"] != "clear").astype(int) + \
    0.186197 * (X["speed_limit"] >= 60).astype(int) + \
    0.096676 * (X["num_reported_accidents"] > 2).astype(int)

def clip(f):
    def clip_f(X):
        sigma = 0.05
        mu = f(X)
        a = -mu / sigma
        b = (1 - mu) / sigma
        
        Phi_a = norm.cdf(a)
        Phi_b = norm.cdf(b)
        phi_a = norm.pdf(a)
        phi_b = norm.pdf(b)
        
        return mu * (Phi_b - Phi_a) + sigma * (phi_a - phi_b) + 1 - Phi_b
        
    return clip_f

clipped_f = clip(f)
train['y'] = clipped_f(train)
test['y'] = clipped_f(test)
orig['y'] = clipped_f(orig)


train['orig_mean_curvature'] = train['orig_mean_curvature'].fillna(orig[TARGET].mean())
test['orig_mean_curvature'] = test['orig_mean_curvature'].fillna(orig[TARGET].mean())

train['orig_min_curvature'] = train['orig_min_curvature'].fillna(orig[TARGET].mean())
test['orig_min_curvature'] = test['orig_min_curvature'].fillna(orig[TARGET].mean())


FEATURES = BASE + ['y'] + ORIG + DIGIT
print(len(FEATURES), 'Features.')


X = train[FEATURES]
y = train[TARGET] - train['y']


from sklearn.model_selection import KFold

N_SPLITS = 15
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)


# from pytabkit import TabM_HPO_Regressor

# model = TabM_HPO_Regressor(
#         device='cuda',
#         random_state=42,
#         n_cv=1,
#         # n_refit=0,
#         val_metric_name='rmse',
#         verbosity=2
#     )
    
# model.fit(X, y, cat_col_names=CATS)


from pytabkit import TabM_D_Regressor
from sklearn.metrics import root_mean_squared_error


from sklearn.base import BaseEstimator, TransformerMixin

class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target Encoder that supports multiple aggregation functions,
    internal cross-validation for leakage prevention, and smoothing.

    Parameters
    ----------
    cols_to_encode : list of str
        List of column names to be target encoded.

    aggs : list of str, default=['mean']
        List of aggregation functions to apply. Any function accepted by
        pandas' `.agg()` method is supported, such as:
        'mean', 'std', 'var', 'min', 'max', 'skew', 'nunique', 
        'count', 'sum', 'median'.
        Smoothing is applied only to the 'mean' aggregation.

    cv : int, default=5
        Number of folds for cross-validation in fit_transform.

    smooth : float or 'auto', default='auto'
        The smoothing parameter `m`. A larger value puts more weight on the 
        global mean. If 'auto', an empirical Bayes estimate is used.
        
    drop_original : bool, default=False
        If True, the original columns to be encoded are dropped.
    """
    def __init__(self, cols_to_encode, aggs=['mean'], cv=5, smooth='auto', drop_original=False):
        self.cols_to_encode = cols_to_encode
        self.aggs = aggs
        self.cv = cv
        self.smooth = smooth
        self.drop_original = drop_original
        self.mappings_ = {}
        self.global_stats_ = {}

    def fit(self, X, y):
        """
        Learn mappings from the entire dataset.
        These mappings are used for the transform method on validation/test data.
        """
        temp_df = X.copy()
        temp_df['target'] = y

        # Learn global statistics for each aggregation
        for agg_func in self.aggs:
            self.global_stats_[agg_func] = y.agg(agg_func)

        # Learn category-specific mappings
        for col in self.cols_to_encode:
            self.mappings_[col] = {}
            for agg_func in self.aggs:
                mapping = temp_df.groupby(col)['target'].agg(agg_func)
                self.mappings_[col][agg_func] = mapping
        
        return self

    def transform(self, X):
        """
        Apply learned mappings to the data.
        Unseen categories are filled with global statistics.
        """
        X_transformed = X.copy()
        for col in self.cols_to_encode:
            for agg_func in self.aggs:
                new_col_name = f'TE_{col}_{agg_func}'
                map_series = self.mappings_[col][agg_func]
                X_transformed[new_col_name] = X[col].map(map_series)
                X_transformed[new_col_name].fillna(self.global_stats_[agg_func], inplace=True)
        
        if self.drop_original:
            X_transformed.drop(columns=self.cols_to_encode, inplace=True)
            
        return X_transformed

    def fit_transform(self, X, y):
        """
        Fit and transform the data using internal cross-validation to prevent leakage.
        """
        # First, fit on the entire dataset to get global mappings for transform method
        self.fit(X, y)

        # Initialize an empty DataFrame to store encoded features
        encoded_features = pd.DataFrame(index=X.index)
        
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            
            temp_df_train = X_train.copy()
            temp_df_train['target'] = y_train

            for col in self.cols_to_encode:
                # --- Calculate mappings only on the training part of the fold ---
                for agg_func in self.aggs:
                    new_col_name = f'TE_{col}_{agg_func}'
                    
                    # Calculate global stat for this fold
                    fold_global_stat = y_train.agg(agg_func)
                    
                    # Calculate category stats for this fold
                    mapping = temp_df_train.groupby(col)['target'].agg(agg_func)

                    # --- Apply smoothing only for 'mean' aggregation ---
                    if agg_func == 'mean':
                        counts = temp_df_train.groupby(col)['target'].count()
                        
                        m = self.smooth
                        if self.smooth == 'auto':
                            # Empirical Bayes smoothing
                            variance_between = mapping.var()
                            avg_variance_within = temp_df_train.groupby(col)['target'].var().mean()
                            if variance_between > 0:
                                m = avg_variance_within / variance_between
                            else:
                                m = 0  # No smoothing if no variance between groups
                        
                        # Apply smoothing formula
                        smoothed_mapping = (counts * mapping + m * fold_global_stat) / (counts + m)
                        encoded_values = X_val[col].map(smoothed_mapping)
                    else:
                        encoded_values = X_val[col].map(mapping)
                    
                    # Store encoded values for the validation fold
                    encoded_features.loc[X_val.index, new_col_name] = encoded_values.fillna(fold_global_stat)

        # Merge with original DataFrame
        X_transformed = X.copy()
        for col in encoded_features.columns:
            X_transformed[col] = encoded_features[col]
            
        if self.drop_original:
            X_transformed.drop(columns=self.cols_to_encode, inplace=True)
            
        return X_transformed


params = {'batch_size': 'auto',
          'n_epochs': 15,
          'patience': 3,
          'allow_amp': True,
          'arch_type': 'tabm-mini',
          'tabm_k': 40,
          'gradient_clipping_norm': 1.0,
          'share_training_batches': False,
          'lr': 0.0013297554090738672,
          'weight_decay': 0.003029610442821251,
          'n_blocks': 4,
          'd_block': 512,
          'dropout': 0.015714592843367126,
          'num_emb_type': 'pwl',
          'd_embedding': 44,
          'num_emb_n_bins': 42,
          'verbosity': 2,
         }

SEEDS = [42]

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f'--- Fold {fold+1}/{N_SPLITS} ---')
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    X_test = test[FEATURES].copy()

    # Target Encoding
    TE = TargetEncoder(cols_to_encode=['curvature'], cv=5, smooth='auto', aggs=['std'], drop_original=False)
    X_train = TE.fit_transform(X_train, y_true.iloc[train_idx])
    X_val = TE.transform(X_val)
    X_test = TE.transform(X_test)

    oof_preds_fold = np.zeros(len(val_idx))
    test_preds_fold = np.zeros(len(test))

    for i, seed in enumerate(SEEDS):
        print(f'  - Seed {seed} ({i+1}/{len(SEEDS)})')
        model = TabM_D_Regressor(**params, random_state=seed+fold)
        model.fit(X_train, y_train, X_val, y_val, cat_col_names=CATS)
        
        oof_preds_fold += model.predict(X_val)
        test_preds_fold += model.predict(X_test)
        
    oof_preds[val_idx] = oof_preds_fold / len(SEEDS)
    test_preds += test_preds_fold / len(SEEDS)
    
    print(f"Fold {fold+1} RMSE: {root_mean_squared_error(y_val+train.iloc[val_idx].y, oof_preds[val_idx]+train.iloc[val_idx].y):.5f}")
    print("-" * 30)

test_preds /= N_SPLITS

y_true_final = y.to_numpy() + train.y.to_numpy()
y_pred_final = oof_preds + train.y.to_numpy()

print(f"Overall OOF RMSE with seed averaging: {root_mean_squared_error(y_true_final, y_pred_final):.5f}")


pd.DataFrame({'id': train.id, TARGET: oof_preds+train['y'].to_numpy()}).to_csv('oof_tabm_residual.csv', index=False)
pd.DataFrame({'id': test.id, TARGET: test_preds+test['y'].to_numpy()}).to_csv('test_tabm_residual.csv', index=False)




