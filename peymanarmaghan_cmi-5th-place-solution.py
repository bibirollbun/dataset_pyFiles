# =============================================================================
# IMPORTS & SETTINGS
# =============================================================================
import os
import random
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import catboost as ctb
import lightgbm as lgb
import xgboost as xgb

import tensorflow as tf
import keras
from keras import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras import Model as keras_model
from tensorflow.keras.layers import (
    Embedding, Concatenate, Flatten, Dropout, BatchNormalization
)

from sklearn.model_selection import (
    StratifiedKFold, KFold, train_test_split, RepeatedStratifiedKFold
)
from sklearn.metrics import (
    cohen_kappa_score, mean_squared_error
)
from sklearn.preprocessing import (
    LabelEncoder, StandardScaler, OneHotEncoder, MinMaxScaler
)
from sklearn.impute import KNNImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression, Lasso
from sklearn.ensemble import (
    HistGradientBoostingClassifier, RandomForestClassifier
)
from keras.utils import set_random_seed

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# =============================================================================
# SEED & WARNINGS
# =============================================================================
RND_SEED = 42
warnings.filterwarnings('ignore')
random.seed(RND_SEED)
np.random.seed(RND_SEED)
tf.random.set_seed(RND_SEED)
set_random_seed(RND_SEED)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = str(RND_SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.config.experimental.enable_op_determinism() 


# =============================================================================
# DATA READING
# =============================================================================
train = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/train.csv')
test = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/test.csv')
sample = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/sample_submission.csv')


# =============================================================================
# CONSTANTS
# =============================================================================
TARGET_sii = 'sii'
TARGET_PC = 'PCIAT-PCIAT_Total'
TARGET_binned = 'binned'
TARGET_cols = [TARGET_sii, TARGET_binned, TARGET_PC]

THRESHOLD = [31, 50, 80 ]
# Normalized thresholds for the “binned” version of PCIAT_Total
THRESHOLD_1 = [31*9/93, 50*9/93, 80*9/93]



# Identify numeric and categorical columns
all_cols = train.columns.to_list()
feature_cols = [
    col for col in all_cols
    if ('PCIAT' not in col) and (col not in ['sii', 'id', 'binned'])
]
num_cols = train[feature_cols].select_dtypes(['number']).columns.to_list()
cat_cols = [col for col in feature_cols if 'Season' in col]

# Convert seasonal columns to numeric categories
train[cat_cols] = train[cat_cols].replace({'Spring':1,'Summer':2,'Fall':3,'Winter':4})
test[cat_cols] = test[cat_cols].replace({'Spring':1,'Summer':2,'Fall':3,'Winter':4})

train[cat_cols] = train[cat_cols].fillna(0).astype('int').astype('category')
test[cat_cols] = test[cat_cols].fillna(0).astype('int').astype('category')


# =============================================================================
# DATA ENCODING & SCALING FUNCTIONS
# =============================================================================
def label_encode(df_train, df_test, cols):
    train_le = df_train.copy()
    test_le = df_test.copy()
    cardinality = {}
    encoder = {}
    for col in cols:
        le = LabelEncoder()
        le.fit(train_le[col])
        train_le[col] = le.transform(train_le[col])
        test_le[col] = le.transform(test_le[col])
        cardinality[col] = len(le.classes_)
        encoder[col] = le
    return train_le, test_le, cardinality, encoder

def scaler_encode(df_train, df_test, cols):
    tr = df_train.copy()
    ts = df_test.copy()
    scaler = StandardScaler()
    tr[cols] = scaler.fit_transform(tr[cols])
    ts[cols] = scaler.transform(ts[cols])
    return tr, ts
def one_hot_encode(df_train, df_test, cols):
    df_oh = df_train.copy()
    ts_oh = df_test.copy()
    bin_cols = []
    for col in cols:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        temp_df = ohe.fit_transform(df_oh[[col]])
        temp_ts = ohe.transform(ts_oh[[col]])
        feature_names = [f"{col}_{category}" for category in ohe.categories_[0]]
        enc_df = pd.DataFrame(temp_df, columns=feature_names)
        enc_ts = pd.DataFrame(temp_ts, columns=feature_names)
        df_oh = pd.concat([enc_df, df_oh], axis=1).drop(columns=[col])
        ts_oh = pd.concat([enc_ts, ts_oh], axis=1).drop(columns=[col])      
        bin_cols.extend(feature_names)
    return df_oh, ts_oh, bin_cols



# =============================================================================
# PARQUET READING & PROCESSING FOR TIME SERIES
# =============================================================================
'''
Below, we read multiple parquet files. We first figure out the variance of time series columns
then pick those with higher variance to train KMeans. We cluster new data using that model.
'''


def process_file_whole(filename, dirname):
    """
    Reads a parquet file, removes null values, 
    and returns time series variance and other stats in a dict.
    """
    df = pd.read_parquet(os.path.join(dirname, filename, 'part-0.parquet'))
    df.drop('step', axis=1, inplace=True)

    # For collection == 0: gather stats
    if collection == 0:
        df.dropna(inplace=True)
        n_ts = {
            'size': len(df),
            'var': df['X'].var() + df['Y'].var() + df['Z'].var() + df['enmo'].var(),
            'non_wear': df['non-wear_flag'].sum(),
            'id': filename.split('=')[1]
        }
        return n_ts
    
    # For collection == 1: read raw data
    if collection == 1:
        return df

def load_time_series_whole(dirname):
    """
    If collection == 0, returns a dataframe of variance stats from all IDs.
    If collection == 1, merges the selected IDs (id_call) data.
    """
    ids = os.listdir(dirname)

    if collection == 0:
        with ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(
                lambda fname: process_file_whole(fname, dirname), 
                ids), total=len(ids))
            )
        return pd.DataFrame(results)

    elif collection == 1:
        data_id = []
        for id_ in id_call:
            data_id.append(process_file_whole('id='+id_, dirname))
        return pd.concat(data_id, ignore_index=True)

# First pass: figure out variance
collection = 0
pr_data = load_time_series_whole("/kaggle/input/child-mind-institute-problematic-internet-use/series_train.parquet")
pr_data = pd.merge(pr_data, train[['sii','id']], how='left', on='id')
pr_data['non-wear_size'] = pr_data['non_wear'] / pr_data['size']

# Pick top variance from each SII=0..3
id_call = []
for j in range(4):
    index_var = pr_data[pr_data['sii'] == j]['var'].nlargest(3).index
    id_call.extend(pr_data.loc[index_var]['id'].values)

# Second pass: read in those high variance files
collection = 1
prq_train = load_time_series_whole("/kaggle/input/child-mind-institute-problematic-internet-use/series_train.parquet")

# =============================================================================
# KMeans CLUSTERING
# =============================================================================
# Prepare data for clustering
prq_train['time-diff'] = (
    prq_train['time_of_day'] - prq_train['time_of_day'].shift(1)
).where(
    prq_train['relative_date_PCIAT'] == prq_train['relative_date_PCIAT'].shift(1), 
    0
)

movment_cols = ['X','Y','Z','enmo','anglez','light','time-diff']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(prq_train[movment_cols])

k = 15
kmeans = KMeans(n_clusters=k, random_state=RND_SEED)
kmeans.fit(X_scaled)


def extract_movment(parquet_data):
    """
    Applies the trained KMeans model on a parquet dataframe,
    returns the proportion of each cluster's 'time-diff' sum to the entire 'time-diff'.
    """
    pq = parquet_data.copy()
    pq['time-diff'] = (
        pq['time_of_day'] - pq['time_of_day'].shift(1)
    ).where(
        pq['relative_date_PCIAT'] == pq['relative_date_PCIAT'].shift(1),
        0
    )
    X = StandardScaler().fit_transform(pq[movment_cols])
    clusters = kmeans.predict(X)
    pq['cluster'] = clusters

    mov = {}
    for i in range(k):
        cluster_df = pq[pq['cluster'] == i]
        # Ratio of that cluster’s time-diff sum to the entire time-diff sum
        mov[f'movement_{i+1}_mean'] = cluster_df['time-diff'].sum() / pq['time-diff'].sum()
    return mov


def process_file(filename, dirname):
    """
    Reads a parquet file, 
    extracts movement clusters with extract_movment().
    """
    df = pd.read_parquet(os.path.join(dirname, filename, 'part-0.parquet'))
    df.drop('step', axis=1, inplace=True)
    df.dropna(inplace=True)
    n_ts = extract_movment(df)
    n_ts['id'] = filename.split('=')[1]
    return n_ts

def load_time_series(dirname):
    """
    Loads all parquet files in a folder, 
    extracts cluster-based movement features for each, 
    and returns them in a dataframe.
    """
    ids = os.listdir(dirname)
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(
            lambda fname: process_file(fname, dirname),
            ids), total=len(ids))
        )
    return pd.DataFrame(results)

# Load and merge with train/test data
train_ts = load_time_series("/kaggle/input/child-mind-institute-problematic-internet-use/series_train.parquet")
test_ts = load_time_series("/kaggle/input/child-mind-institute-problematic-internet-use/series_test.parquet")

train_ts.fillna(0, inplace=True)
test_ts.fillna(0, inplace=True)
train_ts = pd.merge(train, train_ts, how="left", on='id')
test_ts = pd.merge(test, test_ts, how="left", on='id')

# Time-series feature columns
ts_cols = [col for col in train_ts.columns if 'movement' in col]
num_cols_ts = num_cols + ts_cols
feature_cols_ts = feature_cols + ts_cols


# =============================================================================
# CASTING CATEGORICAL & IMPUTATION
# =============================================================================
train[cat_cols] = train[cat_cols].astype('int').astype('category')
test[cat_cols] = test[cat_cols].astype('int').astype('category')

# Impute numeric features with KNN
imputer = KNNImputer(n_neighbors=10)
train_imputed = imputer.fit_transform(train[num_cols])
test_imputed = imputer.transform(test[num_cols])

train[num_cols] = pd.DataFrame(train_imputed, columns=num_cols)
test[num_cols] = pd.DataFrame(test_imputed, columns=num_cols)

# Impute numeric features for TS-extended data
imputer_ts = KNNImputer(n_neighbors=10)
train_imputed_ts = imputer_ts.fit_transform(train_ts[num_cols_ts])
test_imputed_ts = imputer_ts.transform(test_ts[num_cols_ts])

train_ts[num_cols_ts] = pd.DataFrame(train_imputed_ts, columns=num_cols_ts)
test_ts[num_cols_ts] = pd.DataFrame(test_imputed_ts, columns=num_cols_ts)



# =============================================================================
# SPLIT TRAIN INTO LABELED & UNLABELED
# =============================================================================
train_missing = train[train[TARGET_sii].isna()].copy().reset_index(drop=True)
train_ts = train_ts.dropna(subset=[TARGET_sii]).reset_index(drop=True)
train = train.dropna(subset=[TARGET_sii]).reset_index(drop=True)

# Create “binned” column for the train
train[TARGET_binned] = pd.cut(train[TARGET_PC], bins=10, labels=False)



# using sample weight idea from :https://www.kaggle.com/code/lennarthaupts/cmi-detecting-problematic-digital-behavior.
def label_weight(data):
    bins = pd.cut(data, bins=10, labels=False)
    bin_counts = bins.value_counts(normalize=True)
    weight_map = (1 / bin_counts).to_dict()
    weights = bins.map(weight_map)
    return weights / weights.mean()


def quadratic_weighted_kappa(target, preds):
    return cohen_kappa_score(target, preds, weights="quadratic")

def threshold_Rounder(oof_non_rounded, thresholds):
    return np.where(oof_non_rounded < thresholds[0], 0,
           np.where(oof_non_rounded < thresholds[1], 1,
           np.where(oof_non_rounded < thresholds[2], 2, 3)))

def multiple_rmse_scores(actual, predicted):
    errors = []
    for i in range(predicted.shape[1]):
        errors.append(mean_squared_error(actual, predicted[:, i]))
    return errors



# =============================================================================
# MODEL INIT CLASS
# =============================================================================
'''
This class performs cross-validation training for different models 
and returns out-of-fold predictions (oof) plus test predictions.
'''
class model_init:
    def __init__(self, model_class, params):
        self.model_class = model_class
        self.model_params = params
    
    def model_build(self):
        """
        Builds a fresh model from the class and params given.
        If it's a Keras model, we handle it differently.
        """
        if (self.model_class != keras_model):
            self.model = self.model_class(**self.model_params)
        else:
            self.model = self.model_class()
    
    def Train(self, df_train, df_test, final_target, n_split):
        """
        Manages cross-validation for the specified model.
        Returns OOF predictions, test predictions, overall QWK, and the model itself.
        """
        if 'original' in df_train.columns:
            self.train = df_train[df_train['original'] == 1].drop('original', axis=1).reset_index(drop=True)
            self.plus = df_train[df_train['original'] == 0].drop('original', axis=1).reset_index(drop=True)
        else:
            self.train = df_train
            self.plus = None
        
        self.test = df_test
        self.target_col = final_target
        self.n_split = n_split

        return self.fit_cv()
    
    def fit(self, X_train, y_train, X_val, y_val):
        """
        Fits a single fold of data.
        Uses sample weighting for certain runs.
        """
        if WH == 1:
            weights = label_weight(y_train)
        else:
            weights = np.ones(len(y_train))

        # Keras model
        if isinstance(self.model, keras_model):
            self.model = build_nn_model()  # builds the structure
            self.model.compile(
                optimizer=keras.optimizers.AdamW(
                    learning_rate=learning_rate,
                    weight_decay=0.01,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-07
                ),
                loss='mse',
                metrics=['mse']
            )
            self.model.fit(
                nn_enc(X_train),
                y_train,
                batch_size=BS,
                epochs=epochs,
                sample_weight=weights,
                validation_data=(nn_enc(X_val), y_val),
                verbose=0
            )
        else:
            # Classic ML
            self.model.fit(X_train, y_train, sample_weight=weights)

    def predict(self, X):
        """
        Predict either with a Keras model or a scikit-learn model.
        """
        if isinstance(self.model, keras_model):
            return self.model.predict(nn_enc(X), verbose=0, batch_size=BS).flatten()
        else:
            return self.model.predict(X)

    def fit_cv(self):
        """
        K-fold cross-validation: returns OOF predictions, test predictions, QWK, and model list.
        """
        oof_pred = np.zeros(len(self.train))
        test_pred = np.zeros(len(self.test))

        kfold = StratifiedKFold(
            n_splits=self.n_split, shuffle=True, random_state=RND_SEED
        )

        for folds, (idx_train, idx_val) in enumerate(
            kfold.split(self.train, self.train[TARGET_sii])
        ):
            model_fold = []

            if self.plus is None:
                tr = self.train.loc[idx_train]
            else:
                tr = pd.concat(
                    [self.train.loc[idx_train], self.plus],
                    ignore_index=True
                )
            
            X_train = tr.drop(TARGET_cols, axis=1)
            y_train = tr[self.target_col]
            X_val = self.train.loc[idx_val].drop(TARGET_cols, axis=1)
            y_val = self.train.loc[idx_val][TARGET_sii]

            self.model_build()
            self.fit(X_train, y_train, X_val, y_val)
            oof_pred[idx_val] = self.predict(X_val)
            test_pred += self.predict(self.test) / self.n_split

            # Round OOF predictions using threshold
            rounded_oof_fold = threshold_Rounder(oof_pred[idx_val], THRESHOLD_1)
            fold_scores_qwk = quadratic_weighted_kappa(y_val, rounded_oof_fold)
            print(f"\nFold No:{folds+1} QWK_metrics : {fold_scores_qwk:.5f}")

            model_fold.append(self.model)

        # Evaluate overall metrics
        rounded_oof = threshold_Rounder(oof_pred, THRESHOLD_1)
        overal_score = quadratic_weighted_kappa(self.train[TARGET_sii], rounded_oof)
        overal_score_rmse = mean_squared_error(self.train[self.target_col], oof_pred)
        print(f"\n-------Overall QWK Score: {overal_score:.5f}")
        print(f"-------Overall rsme Score: {overal_score_rmse:.5f}\n")

        return oof_pred, test_pred, overal_score, model_fold


# =============================================================================
# NEURAL NETWORK UTILS
# =============================================================================
def build_nn_model():
    """
    Builds a base Keras model with embeddings for categorical features.
    """
    if nn_model_type == 'base model':
        input_categorical = []
        embedding_layers = []

        # Numeric inputs
        cont_inputs = Input(shape=(len(num_cols_nn),))
        keras.utils.set_random_seed(RND_SEED)

        # Categorical embeddings
        for col in cat_cols:
            input_cat = Input(shape=(1,))
            card = cat_card[col]
            embed_dim = min(8, card // 2)  # rule of thumb
            embedding = Embedding(input_dim=card, output_dim=embed_dim)(input_cat)
            embedding = Flatten()(embedding)
            input_categorical.append(input_cat)
            embedding_layers.append(embedding)

        # Combine all features
        all_features = Concatenate()(
            embedding_layers + [cont_inputs]
        )

        # Dense layers
        nn_layer = Dense(256, activation='relu')(all_features)
        nn_layer = Dropout(0.2, seed=RND_SEED)(nn_layer)
        nn_layer = Dense(128, activation='relu')(nn_layer)
        nn_layer = Dropout(0.2, seed=RND_SEED)(nn_layer)
        nn_layer = Dense(64, activation='relu')(nn_layer)

        # Final output
        output = Dense(1)(nn_layer)
        model = Model(inputs=[cont_inputs] + input_categorical, outputs=output)
        return model

def nn_enc(df):
    """
    Transforms the dataset into the input format expected by the base Keras model.
    Returns a list: [numerical_cols] + [cat_col1, cat_col2, ...]
    """
    return [df[num_cols_nn].values] + [df[col].values for col in cat_cols]



# =============================================================================
# HILL CLIMBING FOR ENSEMBLING
# =============================================================================
class hill_climbing:
    """
    Performs a hill-climbing approach to ensemble multiple model predictions 
    by searching for the best linear combination that lowers RMSE.
    """

    def __init__(self, eval_metric, tol=1e-9, max_model=1000):
        self.max_model = max_model
        self.tol = tol
        self.metric = eval_metric

    def fit(self, train, target, score):
        """
        Train: data frame with out-of-fold model predictions as columns
        Target: real labels
        Score: current best score
        """
        best_score = 0
        best_index = -1
        indices = [best_index]
        old_best_score = best_score
        self.train = np.array(train)
        self.target = np.array(target.values)

        # Weight steps
        start = -0.50
        ww = np.arange(start, 0.51, 0.01)
        nn = len(ww)
        files = train.keys().to_list()

        self.models = [best_index]
        self.weights = []
        metrics = [best_score]

        # The final ensemble after each addition
        best_ensemble = self.train[:, best_index]

        for kk in range(10000):
            best_score = score
            best_index = -1
            best_weight = 0

            # Try adding each model and check improvement
            for k, ff in enumerate(files):
                new_model = self.train[:, k]  
                m1 = np.repeat(best_ensemble[:, np.newaxis], nn, axis=1) * (1 - ww)
                m2 = np.repeat(new_model[:, np.newaxis], nn, axis=1) * ww
                mm = m1 + m2
                new_rmse = self.metric(self.target, mm)
                new_score = np.min(new_rmse).item()

                if new_score < best_score:
                    best_score = new_score
                    best_index = k
                    ii = np.argmin(new_rmse).item()
                    best_weight = ww[ii].item()
                    potential_ensemble = mm[:, ii]

            # STOP CRITERIA
            indices.append(best_index)
            indices = list(np.unique(indices))
            if len(indices) > self.max_model:
                print(f'=> Reached {self.max_model} models')
                indices = indices[:-1]
                break
            if abs(best_score - old_best_score) < self.tol:
                print(f'=> Reached tolerance {self.tol}')
                break

            # Record new results
            if best_index != -1:
                print(
                    kk,
                    'New best rmse', best_score,
                    f'adding "{files[best_index]}"',
                    'weight', f'{best_weight:0.3f}'
                )
                self.models.append(best_index)
                self.weights.append(best_weight)
                metrics.append(best_score)
                best_ensemble = potential_ensemble
                old_best_score = best_score

        # Combine weights
        wgt = np.array([1])
        for w in self.weights:
            wgt = wgt * (1 - w)
            wgt = np.concatenate([wgt, np.array([w])])

        # Show final model weights
        rows = []
        t = 0
        for m, w, s in zip(self.models, wgt, metrics):
            if m == -1:
                continue
            name = files[m]
            dd = {'weight': w, 'model': name}
            rows.append(dd)
            t += float(f'{w:.3f}')
        self.cl_output = pd.DataFrame(rows)
        self.cl_output = (
            self.cl_output.groupby('model')
            .agg('sum')
            .reset_index()
            .sort_values('weight', ascending=False)
            .reset_index(drop=True)
        )

    def predict(self, test):
        """
        Applies the discovered weights to the test set predictions.
        """
        self.test_final = 0
        for i, j in enumerate(self.cl_output.model):
            self.test_final += test[j] * self.cl_output.loc[i]['weight']
        return self.test_final


# =============================================================================
# SEMI-SUPERVISED LEARNING STEP 
# =============================================================================
'''
Below, we do a semi-supervised approach:
1) Train multiple models on labeled data
2) Use hill-climbing ensemble to label unlabeled data
3) Merge them back
4) Train again on the combined dataset
'''



train_nn = train[feature_cols].copy()
test_nn = train_missing[feature_cols].copy()
train_nn[TARGET_cols] = train[TARGET_cols]

train_nn, test_nn, cat_card, _ = label_encode(train_nn, test_nn, cat_cols)
train_nn, test_nn = scaler_encode(train_nn, test_nn, num_cols)
num_cols_nn = num_cols

# Keras model example
WH = 0
BS = 256
epochs = 6
learning_rate = 3e-3
nn_model_type = 'base model'
nn_instance = keras.Model()

model_nn = model_init(keras_model, None)
oof_nn, test_missing_nn, acc_nn, model_NN = model_nn.Train(
    train_nn, test_nn, TARGET_PC, 5
)

# CatBoost example
train_gb = train[feature_cols].copy()
test_gb = train_missing[feature_cols].copy()
train_gb[TARGET_cols] = train[TARGET_cols]

param_cat = {
    'objective': 'Tweedie:variance_power=1.5', 
    'iterations': 273, 
    'depth': 6,
    'learning_rate': 0.03347776308515933,
    'l2_leaf_reg': 0.0005342937261279777, 
    'subsample': 0.645614570099021, 
    'bagging_temperature': 0.6118528947223795, 
    'random_strength': 1.3957991126597662, 
    'colsample_bylevel': 0.6460723242676091,
    'min_data_in_leaf': 37,
    'random_state': RND_SEED,
    'cat_features': cat_cols,
    'verbose': 0
}
model_cat = model_init(ctb.CatBoostRegressor, param_cat)
oof_cat, test_missing_cat, acc_cat, model_cat = model_cat.Train(
    train_gb, test_gb, TARGET_PC, n_split=5
)

# LightGBM example
param_lgb = {
    'objective': 'tweedie', 
    'n_estimators': 597, 
    'max_depth': 3, 
    'learning_rate': 0.01019160829182289,
    'subsample': 0.5261976292373335,
    'colsample_bytree': 0.5351784713007832,
    'tweedie_variance_power': 1.152466250299122,
    'random_state': RND_SEED,
    'verbosity': -1
}
model_lgb = model_init(lgb.LGBMRegressor, param_lgb)
oof_lgb, test_missing_lgb, acc_lgb, model_lgb = model_lgb.Train(
    train_gb, test_gb, TARGET_PC, n_split=5
)

# Lasso example
param_lasso = {
    'fit_intercept': True,
    'precompute': True, 
    'copy_X': True, 
    'warm_start': True,
    'selection': 'random',
    'max_iter': 531, 
    'alpha': 0.0004496880541220534, 
    'tol': 1.4872791031412706e-05,
    'random_state': RND_SEED
}
model_lasso = model_init(Lasso, param_lasso)
oof_lasso, test_missing_lasso, _, model_lasso = model_lasso.Train(
    train_gb, test_gb, TARGET_PC, n_split=5
)

# XGBoost example
param_xgb = {
    'objective': 'reg:squarederror',
    'n_estimators': 327,
    'max_depth': 3,
    'learning_rate': 0.020155904604737717,
    'subsample': 0.7712561640659066, 
    'colsample_bytree': 0.5466806743991856,
    'gamma': 2.6288923622161713,
    'reg_alpha': 9.468184374972093e-05,
    'reg_lambda': 0.0003582157866941332,
    'tweedie_variance_power': 1.1581422325461046,
    'enable_categorical': True,
    'random_state': RND_SEED,
    'verbosity': 0
}
model_xgb = model_init(xgb.XGBRegressor, param_xgb)
oof_xgb, test_missing_xgb, acc_xgb, model_xgb = model_xgb.Train(
    train_gb, test_gb, TARGET_PC, n_split=5
)

# Combine OOF predictions
oof_preds = pd.DataFrame({
    'cat': oof_cat,
    'lgb': oof_lgb,
    'xgb': oof_xgb,
    'nn': oof_nn,
    'lasso': oof_lasso
})

missing_pred = pd.DataFrame({
    'cat': test_missing_cat,
    'lgb': test_missing_lgb,
    'xgb': test_missing_xgb,
    'nn': test_missing_nn,
    'lasso': test_missing_lasso
})

# Hill climbing ensemble on unlabeled portion
hc = hill_climbing(eval_metric=multiple_rmse_scores)
hc.fit(oof_preds, train[TARGET_PC], 400)
ens_missing_pred = hc.predict(missing_pred)

# Convert predictions to integer SII classes
tuned_ens_missing = threshold_Rounder(ens_missing_pred, THRESHOLD)

# Merge back and re-train
train_missing[TARGET_sii] = tuned_ens_missing.round().astype('int')
train_missing[TARGET_PC] = ens_missing_pred
train_missing[TARGET_binned] = pd.cut(ens_missing_pred.round(), bins=10, labels=False)
train_missing['original'] = 0

train['original'] = 1
train_plus = pd.concat([train_missing, train], ignore_index=True)



# =============================================================================
# FINAL VOTING EXAMPLE
# =============================================================================
'''
At the end, we do a final stage of training with multiple models 
(including time series features for some) and ensemble them via hard voting.
'''


train_nn = train_plus.copy()
test_nn = test.copy()

train_nn,test_nn,cat_card,_ = label_encode(train_nn,test_nn, cat_cols)
train_nn,test_nn = scaler_encode(train_nn,test_nn,num_cols)
num_cols_nn = num_cols
WH = 1
BS=256
epochs = 6
learning_rate = 3e-3
nn_model_type = 'base model'
nn_instance = keras.Model()
model_nn = model_init(keras_model,None)
oof_pred_nn,test_pred_nn,acc_nn,model_NN=model_nn.Train(train_nn,test_nn,TARGET_binned,5)


#Preparing data with and without timeseries features

train_gb_plus = train_plus[feature_cols].copy()
train_gb_plus['original'] = train_plus['original']
train_gb_plus[TARGET_cols] = train_plus[TARGET_cols]
test_gb = test[feature_cols].copy()

train_gb_ts = train_ts[feature_cols_ts].copy()
train_gb_ts[TARGET_cols] = train[TARGET_cols]
test_gb_ts = test_ts[feature_cols_ts]


WH = 1
train_dt,test_dt,_ = one_hot_encode(train_gb_plus,test_gb,cat_cols)
param_dt={
          'criterion': 'squared_error', 
          'splitter': 'best',
          'max_depth': 5,
          'min_samples_split': 100, 
          'min_samples_leaf': 5,
          'min_impurity_decrease': 0.02,
           'ccp_alpha': 0.01,
          'random_state' : RND_SEED,
          }

model_dt = model_init(DecisionTreeRegressor,param_dt)
oof_pred_dt_plus,test_pred_dt_plus,acc_dt,model_dt_plus=model_dt.Train(train_dt,test_dt,TARGET_binned,n_split = 5)


train_lr = train_gb_plus.copy()
test_lr = test_gb.copy()
train_lr,test_lr = scaler_encode(train_lr,test_lr ,num_cols)
WH=0
param_lr= {
         'solver': 'newton-cg',
          'penalty':'l2',
         'class_weight' :'balanced',
         'multi_class': 'multinomial',
         'max_iter': 500,
         'warm_start': True,
        'verbose':0, 
    }
        
model_lr = model_init(LogisticRegression,param_lr)
oof_pred_lr_plus,test_pred_lr_plus,overal_score,model_lr_plus=model_lr.Train(train_lr,test_lr,TARGET_binned,n_split = 5)


WH = 1
param_lgb={'objective': 'tweedie',
           'n_estimators': 576, 
           'max_depth': 3, 
           'learning_rate': 0.014504865871703278,
           'subsample': 0.7724866369943665, 
           'colsample_bytree': 0.5717189909747162,
           'random_state' : RND_SEED,
           'verbosity': -1,
          }

model_lgb = model_init(lgb.LGBMRegressor,param_lgb)
oof_pred_lgb_plus,test_pred_lgb_plus,acc_lgb,model_lgb_plus=model_lgb.Train(train_gb_plus,test_gb,TARGET_binned,n_split = 5)


WH = 1
param_cat={'objective': 'Tweedie:variance_power=1.5',
           'iterations': 292,
           'depth': 6,
           'learning_rate': 0.0624982363237828, 
           'l2_leaf_reg': 0.019685934698040447, 
           'subsample': 0.5018495996455012,
           'bagging_temperature': 0.8627252427715155,
           'random_strength': 5.015068697208244, 
           'colsample_bylevel': 0.6124379138359408,
           'min_data_in_leaf': 89,
           'random_state': RND_SEED,
            'cat_features' : cat_cols,
            'verbose': 0
            }

model_cat = model_init(ctb.CatBoostRegressor,param_cat)
oof_pred_cat_plus,test_pred_cat_plus,acc_cat,model_cat_plus=model_cat.Train(train_gb_plus,test_gb,TARGET_binned,n_split = 5)


WH = 1
param_lgb={'iterations': 1000,
             'objective':'tweedie',
            'max_depth': 10,
            # # 'cat_features' : cat_cols,
            'learning_rate': 0.04,
            # 'bagging_fraction': 0.78,
            # 'bagging_freq' : 20,
            #  # 'border_count': 60,
             'verbosity' : -1}

model_lgb = model_init(lgb.LGBMRegressor,param_lgb)
oof_pred_lgb_ts,test_pred_lgb_ts,acc_lgb,model_lgb_ts=model_lgb.Train(train_gb_ts,test_gb_ts,TARGET_binned,n_split = 5)


WH = 1
param_cat={'iterations': 500,
            'depth': 6,
            'learning_rate': 0.015,
            'l2_leaf_reg': 0.002,
            'random_strength': 0.46,
            'cat_features' : cat_cols,
            # 'border_count': 51,
            # 'min_data_in_leaf': 13,
            'verbose' : 0}
model_cat = model_init(ctb.CatBoostRegressor,param_cat)
oof_pred_cat_ts,test_pred_cat_ts,acc_cat,model_cat=model_cat.Train(train_gb_ts,test_gb_ts,TARGET_binned,n_split = 5)


WH = 1
param_xgb={'objective': 'reg:tweedie',
           'n_estimators': 480,
           'max_depth': 3, 
           'learning_rate': 0.025707833957860277,
           'subsample': 0.5290418060840998,
           'colsample_bytree': 0.9330880728874675,
           'gamma': 3.005575058716044,
           'reg_alpha': 0.006796578090758156,
           'reg_lambda': 1.2087541473056957e-05,
           'tweedie_variance_power': 1.9699098521619942,
           'random_state' : RND_SEED,
           'enable_categorical' : True,
            'verbosity' : 0}

model_xgb = model_init(xgb.XGBRegressor,param_xgb)
oof_pred_xgb_plus,test_pred_xgb_plus,acc_xgb,model_xgb_plus=model_xgb.Train(train_gb_plus,test_gb,TARGET_binned,n_split = 5)


WH = 1
param_xgb={'iterations': 500,
            'max_depth': 3,
            'eta': 0.054,
           'objective' : 'reg:tweedie',
           'tweedie_variance_power':1.3,
            'bagging_fraction': 0.8,
           'enable_categorical' : True,
            # 'bagging_freq' : 10,
            # 'random_strength': 0.5482698471489474,
             'border_count': 48,
           'random_seed' : 4,
            'verbosity' : 0}

model_xgb = model_init(xgb.XGBRegressor,param_xgb)
oof_pred_xgb_ts,test_pred_xgb_ts,acc_xgb,model_xgb_ts=model_xgb.Train(train_gb_ts,test_gb_ts,TARGET_binned,n_split = 5)


WH = 1
param_lasso = { 'fit_intercept': False,
               'precompute': True,
               'copy_X': False,
               'warm_start': False,
               'selection': 'cyclic',
               'max_iter': 1022,
               'alpha': 0.0005076318790591836, 
               'tol': 8.154903080917285e-05,
               'random_state' : RND_SEED

              }

model_lasso = model_init(Lasso,param_lasso)
oof_pred_lasso,test_pred_lasso,acc_lgb,model_lasso=model_lasso.Train(train_gb_plus,test_gb,TARGET_binned,n_split = 5)


ens_preds = pd.DataFrame()
test_preds = pd.DataFrame()

# Example collection of out-of-fold predictions
ens_preds['cat_plus'] = oof_pred_cat_plus
ens_preds['cat_ts'] = oof_pred_cat_ts
ens_preds['lgb_plus'] = oof_pred_lgb_plus
ens_preds['lgb_ts'] = oof_pred_lgb_ts
ens_preds['xgb_plus'] = oof_pred_xgb_plus
ens_preds['xgb_ts'] = oof_pred_xgb_ts
ens_preds['nn'] = oof_pred_nn
ens_preds['lasso'] = oof_pred_lasso
ens_preds['dt_plus'] = oof_pred_dt_plus
ens_preds['lr_plus'] = oof_pred_lr_plus

test_preds['cat_plus'] = test_pred_cat_plus
test_preds['cat_ts'] = test_pred_cat_ts
test_preds['lgb_plus'] = test_pred_lgb_plus
test_preds['lgb_ts'] = test_pred_lgb_ts
test_preds['xgb_plus'] = test_pred_xgb_plus
test_preds['xgb_ts'] = test_pred_xgb_ts
test_preds['nn'] = test_pred_nn
test_preds['lasso'] = test_pred_lasso
test_preds['dt_plus'] = test_pred_dt_plus
test_preds['lr_plus'] = test_pred_lr_plus

# Threshold them
tuned_ens_preds = pd.DataFrame()
tuned_test_preds = pd.DataFrame()

for col in ens_preds.keys():
    tuned_ens_preds[col] = threshold_Rounder(ens_preds[col], THRESHOLD_1)
    tuned_test_preds[col] = threshold_Rounder(test_preds[col], THRESHOLD_1)

# Hard-voting (mode) ensemble
voted_oof = tuned_ens_preds.mode(axis=1).iloc[:, 0]
final_test = tuned_test_preds.mode(axis=1).iloc[:, 0]

# QWK
kappa_score = cohen_kappa_score(train[TARGET_sii], voted_oof, weights='quadratic')
print(kappa_score)

# Make final submission
submission = pd.DataFrame({
    'id': sample['id'],
    'sii': final_test
})
submission.to_csv('submission.csv', index=False)

