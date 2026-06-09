import os
if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
    print("Running on Kaggle kernel")
    import xgboost
    print("XGBoost using:", xgboost.__version__)
    
from sklearnex import patch_sklearn
patch_sklearn()

import numpy as np 
import pandas as pd 
import time

from itertools import combinations
from scipy import stats
from collections import Counter

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, ParameterSampler
from sklearn.metrics import log_loss, make_scorer
from xgboost import XGBClassifier

import ipywidgets as widgets
from IPython.display import display

import gc
from joblib import Parallel, delayed

import logging
logging.getLogger("sklearnex").setLevel(logging.ERROR)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


INPUT_DIR = "/kaggle/input/playground-series-s5e6/" 
# INPUT_DIR = "" 

train_df = pd.read_csv(f"{INPUT_DIR}train.csv")
test_df  = pd.read_csv(f"{INPUT_DIR}test.csv")


# Display basic info
print("Training data overview:", train_df.info())
print("Test data overview:", test_df.info())
print("\nFirst few training data rows:")
print(train_df.head(3))

target = 'Fertilizer Name'


numeric_vars = [cname for cname in train_df.columns if train_df[cname].dtype in ['int64', 'float64'] and cname != 'id']
environment_vars = ['Temparature', 'Humidity', 'Moisture']
nutrient_vars = ['Nitrogen', 'Potassium', 'Phosphorous']

def add_features(dff):
    # Add nutrient ratios
    dff['N_P_ratio'] = (dff['Nitrogen']+1) / (dff['Phosphorous'] + 1)  
    dff['N_K_ratio'] = (dff['Nitrogen']+1) / (dff['Potassium'] + 1)
    dff['P_K_ratio'] = (dff['Phosphorous']+1) / (dff['Potassium'] + 1)

    # Total nutrients
    dff['NPK_sum'] = dff['Nitrogen'] + dff['Phosphorous'] + dff['Potassium']
            
    # Normalized NPK values
    dff['N_proportion'] = dff['Nitrogen'] / dff['NPK_sum']
    dff['P_proportion'] = dff['Phosphorous'] / dff['NPK_sum'] 
    dff['K_proportion'] = dff['Potassium'] / dff['NPK_sum'] 
    
    # Moisture Temperature Ratio
    dff['MoistureTemp_Ratio'] = dff['Moisture'] / (dff['Temparature'] + 1e-5)
    
    # Composite Weather Index
    dff['WaterStressIndex'] = (dff['Moisture'] + dff['Humidity']) / (dff['Temparature'] + 1e-5)
    
    # Interaction of the categorical predictors
    dff['soil_x_crop'] = dff['Soil Type'].astype(str) + "_" + dff['Crop Type'].astype(str)
    
    # We add some second-degree interactions. We don't want to add e.g. an interaction between N_proportion and NPK_sum though.
    numeric_interactions = pd.DataFrame(index=dff.index)
  
    for var1, var2 in combinations(environment_vars, 2):
        colname = f"{var1}_x_{var2}"
        numeric_interactions[colname] = dff[var1] * dff[var2]
        
    for var1, var2 in combinations(nutrient_vars, 2):
        colname = f"{var1}_x_{var2}"
        numeric_interactions[colname] = dff[var1] * dff[var2]

    dff = pd.concat([dff, numeric_interactions], axis=1)
        
    return dff
    
train_df = add_features(train_df)
test_df = add_features(test_df)

new_features = [cname for cname in train_df.columns if train_df[cname].dtype in ['int64', 'float64'] and cname not in numeric_vars and cname != 'id']
print(f"We added {len(new_features)} new features, namely:", new_features)
numeric_vars = numeric_vars + new_features
print("The full set of numeric vars is", numeric_vars)


#Label encode the target
L_encoder = LabelEncoder()
y = L_encoder.fit_transform(train_df[target])

def prep_vars_verbose(dff):
    dff = dff.copy()
    before = dff.memory_usage(deep=True).sum() / 1024**2  # in MB

    for col in dff.columns:
        if dff[col].dtype == 'int64' and dff[col].min() >= -128 and dff[col].max() <= 127:
            dff[col] = dff[col].astype('int8')
        elif dff[col].dtype == 'float64' and np.allclose(dff[col], dff[col].astype('float16'),
                                                         rtol=1e-03, atol=1e-05, equal_nan=True):
            dff[col] = dff[col].astype('float16')
        elif dff[col].dtype == 'object':
            dff[col] = dff[col].astype('category')

    after = dff.memory_usage(deep=True).sum() / 1024**2
    print(f"Memory reduced from {before:.2f} MB to {after:.2f} MB ({(1 - after/before) * 100:.2f}% reduction)")
    return dff
        
train_df = prep_vars_verbose(train_df)
test_df = prep_vars_verbose(test_df)

X = train_df.drop(columns=[target, 'id'])


# MAP@3
def mapk(actual, preds, k=3):
    """
    actual: list of true labels
    preds:  list of lists (top-k predicted labels for each sample)
    """
    out = []
    for a, p in zip(actual, preds):
        p = p[:min(len(p), k)]
        
        if a in p[:k]:
            rank = p.index(a) + 1
            out.append(1.0 / rank)
        else:
            out.append(0.0)
    return np.mean(out)

gc.collect()


from sklearn.metrics import log_loss

# our full labeled data
X_full = X       
y_full = y      

M = 5                              # number of independent hold-outs
test_size = 0.2                    # fraction for each Váµ¢
random_state = 42

# small hyperparam space (3Â³Â·3Â² = 243 combos; we sample 15 each round)
param_dist = {
    'learning_rate':    [0.05, 0.1, 0.2],
    'max_depth':        [4, 6, 8],
    'min_child_weight': [1, 5, 10],
    'subsample':        [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
}

best_params_list = []
validation_scores = []

def run_holdout(i):
    print(f"\nâ�³ Starting fold {i+1}/{M}...", flush=True)
    t0 = time.time()

    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, stratify=y_full, test_size=test_size, random_state=random_state + i
    )
    print(f"ğŸ”¹ Data split done in {time.time() - t0:.2f}s", flush=True)

    model = XGBClassifier(
        objective='multi:softprob',
        num_class=7,
        tree_method='hist',
        enable_categorical=True,
        n_estimators=500,  
        use_label_encoder=False,
        eval_metric='mlogloss',
        early_stopping_rounds=20,
        random_state=random_state
    )

    # Wrap the search with timing
    def timed_fit(estimator, X, y):
        t_start = time.time()
        estimator.fit(
            X, y,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        return time.time() - t_start

    param_list = list(ParameterSampler(param_dist, n_iter=15, random_state=random_state + i))
    best_score = float('inf')
    best_model = None
    best_params = None
    for j, params in enumerate(param_list):
        model.set_params(**params)
        fit_time = timed_fit(model, X_train, y_train)
        score = log_loss(y_val, model.predict_proba(X_val))  # consistent scoring

        print(f"   â€¢ Config {j+1}/{len(param_list)} in {fit_time:.2f}s â†’ log_loss = {score:.4f}", flush=True)

        if score < best_score:
            best_score = score
            best_model = model
            best_params = params.copy()

    # MAP@3 evaluation
    probs = best_model.predict_proba(X_val)
    top3 = np.argsort(probs, axis=1)[:, -3:][:, ::-1]
    val_map3 = mapk(y_val.tolist(), top3.tolist(), k=3)

    print(f"âœ… Round {i+1}/{M} â†’ MAP@3 = {val_map3:.4f}, best params = {best_params}", flush=True)
    print(f"ğŸ•’ Total fold time: {time.time() - t0:.2f}s", flush=True)

    gc.collect()
    return best_params, val_map3

parallel_results = Parallel(n_jobs=3)(
    delayed(run_holdout)(i) for i in range(0, M)  
)

# parallel_results is a list of tuples: [(params1, val1), (params2, val2), ...]
records = []

for best_params, val_map3 in parallel_results:
    row = {**best_params, 'val_map3': val_map3}
    records.append(row)

df = pd.DataFrame(records)
print("Here are the best hyperparameter vectors for the 5 random train-validation splits:")
print(df)


# Compute the consensus hyperparameter vector and the MAP@3-weighted 
# hyperparameter vector.  Train the model on X_full, y_full using each.

df['fold'] = df.index + 1
df.rename(columns={'val_map3': 'MAP@3'}, inplace=True)
print(df)

#Helper: mode for categorical/discrete values
def mode(series):
    return Counter(series).most_common(1)[0][0]

# Helper: weighted average for continuous values
def weighted_avg(series, weights):
    return np.average(series, weights=weights)

# Extract weights
weights = df_sorted['MAP@3'].values

# Compute consensus (mode) and weighted average for each hyperparameter
consensus_params = {
    'subsample': mode(df_sorted['subsample']),
    'min_child_weight': mode(df_sorted['min_child_weight']),
    'max_depth': mode(df_sorted['max_depth']),
    'learning_rate': mode(df_sorted['learning_rate']),
    'colsample_bytree': mode(df_sorted['colsample_bytree']),
}

weighted_params = {
    'subsample': weighted_avg(df_sorted['subsample'], weights),
    'min_child_weight': weighted_avg(df_sorted['min_child_weight'], weights),
    'max_depth': int(round(weighted_avg(df_sorted['max_depth'], weights))),
    'learning_rate': weighted_avg(df_sorted['learning_rate'], weights),
    'colsample_bytree': weighted_avg(df_sorted['colsample_bytree'], weights),
}


def train_final_model(params, label):
    model = XGBClassifier(
        objective='multi:softprob',
        num_class=7,
        tree_method='hist',
        enable_categorical=True,
        n_estimators=500,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
        **params  # includes subsample, max_depth, etc.
    )
    model.fit(X_full, y_full, verbose=False)
    print(f"{label} model trained.")
    return model

# Train both models
model_consensus = train_final_model(consensus_params, "Consensus")
model_weighted = train_final_model(weighted_params, "Weighted")


def train_model_on_full_data(i):
    row = df[df['fold'] == i].iloc[0]
    
    model = XGBClassifier(
        objective='multi:softprob',
        num_class=7,
        tree_method='hist',
        enable_categorical=True,
        n_estimators=500,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42 + i,
        subsample=row['subsample'],
        min_child_weight=row['min_child_weight'],
        max_depth=int(row['max_depth']),
        learning_rate=row['learning_rate'],
        colsample_bytree=row['colsample_bytree']
    )
    
    model.fit(X_full, y_full, verbose=False)
    gc.collect()
    return model

# Train all 5 models in parallel
ensemble_models = Parallel(n_jobs=3)(
    delayed(train_model_on_full_data)(i) for i in range(1, 6)
)


def get_top3_predictions(model, X, label_encoder):
    proba = model.predict_proba(X)
    top3_indices = np.argsort(proba, axis=1)[:, -3:][:, ::-1]  # shape (n, 3)
    
    # Flatten â†’ decode â†’ reshape
    flat_decoded = label_encoder.inverse_transform(top3_indices.ravel())
    top3_labels = flat_decoded.reshape(top3_indices.shape)
    
    # Join top-3 labels into space-delimited strings
    return [' '.join(row) for row in top3_labels]

def ensemble_predict_top3(models, X, label_encoder):
    probas = [model.predict_proba(X) for model in models]
    avg_proba = np.mean(probas, axis=0)
    top3_indices = np.argsort(avg_proba, axis=1)[:, -3:][:, ::-1]
    
    flat_decoded = label_encoder.inverse_transform(top3_indices.ravel())
    top3_labels = flat_decoded.reshape(top3_indices.shape)
    
    return [' '.join(row) for row in top3_labels]


X_test_features = test_df.drop(columns=['id'])

# Consensus model
X_test_consensus = test_df.copy()
X_test_consensus['Fertilizer Name'] = get_top3_predictions(model_consensus, X_test_features, L_encoder)
X_test_consensus[['id', 'Fertilizer Name']].to_csv('consensus_model_predictions.csv', index=False)

# Weighted model
X_test_weighted = test_df.copy()
X_test_weighted['Fertilizer Name'] = get_top3_predictions(model_weighted, X_test_features, L_encoder)
X_test_weighted[['id', 'Fertilizer Name']].to_csv('weighted_model_predictions.csv', index=False)

# Ensemble model
X_test_ensemble = test_df.copy()
X_test_ensemble['Fertilizer Name'] = ensemble_predict_top3(ensemble_models, X_test_features, L_encoder)
X_test_ensemble[['id', 'Fertilizer Name']].to_csv('ensemble_model_predictions.csv', index=False)

