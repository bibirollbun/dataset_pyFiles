


!pip install -q xgboost==3.0.2 lightgbm==4.6.0 scikit-learn==1.7.1


%%time 

import pandas as pd, numpy as np, polars as pl
from gc import collect
from tqdm.notebook import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor as XGBR
from lightgbm import LGBMRegressor as LGBMR, log_evaluation, early_stopping
from catboost import CatBoostRegressor as CBR, Pool
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
# Gaussian Process Regressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# HistGradientBoosting Regressor
from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.base import clone
from sklearn.preprocessing import *
from sklearn.pipeline import make_pipeline, Pipeline

from warnings import filterwarnings 
filterwarnings("ignore")

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingRegressor



target = "CORRUCYSTIC_DENSITY"


%%time 

train  = pd.read_csv(f"/kaggle/input/new-files/preprocessed_train_data_2.csv", index_col = "LOCAL_IDENTIFIER")
test   = pd.read_csv(f"/kaggle/input/new-files/preprocessed_test_2.csv", index_col = "LOCAL_IDENTIFIER")
sub_fl = pd.read_csv(f"/kaggle/input/recruitment-task-for-gdsc-ml/SPECIMEN.csv", index_col = "LOCAL_IDENTIFIER")

print(f"\n\n---> Shapes = {train.shape}, {test.shape}")
strt_ftre = test.columns.tolist()

display(
    pd.concat(
        [
            train.describe().transpose(),
            train.nunique().to_frame().rename(columns = {0 : "n_unique"}),
            train.isna().sum().to_frame().rename(columns = {0 : "null_count"})
        ], axis=1
    ).
    style.
    set_caption(
        f"Basic description and analysis - train data"
    )
)

print("\n\n\n\n")
display(
    pd.concat(
        [
            test.describe().transpose(),
            test.nunique().to_frame().rename(columns = {0 : "n_unique"}),
            test.isna().sum().to_frame().rename(columns = {0 : "null_count"})
        ], axis=1
    ).
    style.
    set_caption(
        f"Basic description and analysis - test data"
    )
)

print("\n\n\n")

fig, ax = plt.subplots(1,1, figsize = (6, 4))
train[target].plot.kde(ax = ax)
ax.set_title(f"Target plot", fontweight = "bold", color = "maroon")
ax.set(xlabel = "", ylabel = "")
plt.tight_layout()
plt.show()


%%time 

train = train.dropna(subset = [target])

Xtrain = train.drop(columns = target)
ytrain = train[target]
Xtest  = test.copy()

#proxy_cols = [f"C{i}" for i in range(len(strt_ftre))]
#Xtrain.columns = proxy_cols
#Xtest.columns  = proxy_cols

cat_cols = list(Xtrain.select_dtypes(exclude = np.number).columns)

Xtrain[cat_cols] = Xtrain[cat_cols].astype("string").fillna("missing").astype("category")
Xtest[cat_cols]  = Xtest[cat_cols].astype("string").fillna("missing").astype("category")





import numpy as np
# Read feature importance CSV file
feature_importance_df = pd.read_csv('/kaggle/input/preprocessed-data/feature_importance_rankings.csv')
# Make copies of original data
Xtrain_weighted = Xtrain.copy()
Xtest_weighted = Xtest.copy()
# Keep only features present in the dataset
valid_features = feature_importance_df[feature_importance_df['feature'].isin(Xtrain.columns)]
if len(valid_features) == 0:
    print("WARNING: No matching features found. Using original data.")
else:
    # Get min/max importance for normalization
    max_importance = valid_features['importance_score'].max()
    min_importance = valid_features['importance_score'].min()
    
    feature_weights = {}
    for _, row in valid_features.iterrows():
        feature = row['feature']
        importance = row['importance_score']
        
        # Convert importance to multiplier between 0.5 and 2.0
        if max_importance == min_importance:
            weight = 1.0
        else:
            normalized = (importance - min_importance) / (max_importance - min_importance)
            weight = 0.5 + 1.5 * normalized
        
        feature_weights[feature] = weight
        Xtrain_weighted[feature] = Xtrain_weighted[feature] * weight
        Xtest_weighted[feature] = Xtest_weighted[feature] * weight
    print(f"Applied feature importance weighting to {len(feature_weights)} features.")
    print(f"Weight range: {min(feature_weights.values()):.3f} → {max(feature_weights.values()):.3f}")
# Replace original datasets with weighted versions
Xtrain = Xtrain_weighted
Xtest = Xtest_weighted
cv = KFold(1000, shuffle = True, random_state = 42)
Mdl_Master = {
    "XGB1R": XGBR(
        n_estimators     = 200,
        learning_rate    = 0.02,
        max_depth        = 5,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        reg_alpha        = 0.1,
        reg_lambda       = 0.1,
        random_state     = 42,
        verbosity        = 0,
    ),

    "LGBM1R": LGBMR(
        n_estimators  = 200,
        learning_rate = 0.02,
        max_depth     = 8,
        subsample     = 0.8,
        reg_alpha     = 0.1,
        reg_lambda    = 1.0,
        random_state  = 42,
        verbosity     = -1,
    ),

    "CB1R": CBR(
        iterations    = 200,
        learning_rate = 0.05,
        depth         = 6,
        l2_leaf_reg   = 1.0,
        random_state  = 42,
        verbose       = False,
    ),

    "BAGGING1R": BaggingRegressor(
        n_estimators       = 150,
        max_samples        = 0.85,
        max_features       = 0.7,
        bootstrap          = True,
        bootstrap_features = False,
        random_state       = 42,
        n_jobs             = 1,
    ),

    "LINEAR1R": Ridge(
        alpha      = 0.5,
        random_state = 42,
    ),

    "RF1R": RandomForestRegressor(
        n_estimators      = 200,
        max_depth         = 15,
        min_samples_split = 8,
        min_samples_leaf  = 2,
        max_features      = 0.8,
        bootstrap         = True,
        max_samples       = 0.9,
        random_state      = 42,
        n_jobs            = 1,
    ),

    "EXTRA1R": ExtraTreesRegressor(
        n_estimators      = 200,
        max_depth         = 12,
        min_samples_split = 5,
        min_samples_leaf  = 2,
        max_features      = 0.8,
        bootstrap         = False,
        random_state      = 42,
        n_jobs            = 1,
    ),
}


OOF_Preds, Mdl_Preds = [], []
# Meta-learning storage
meta_features_oof = []
meta_features_test = []
meta_target = []
fold_weights = []
# Define core models (get full weight) vs support models (get dampened weight)
core_models = ["XGB1R", "LGBM1R","CB1R"]
support_models = ["BAGGING1R","RF1R","LINEAR1R","EXTRA1R"]
for fold_nb, (train_idx, dev_idx) in tqdm(enumerate( cv.split(Xtrain, ytrain) ) ):
    print(f"---> Starting Fold {fold_nb + 1}")
    Xtr, ytr   = Xtrain.iloc[train_idx], ytrain.iloc[train_idx]
    Xdev, ydev = Xtrain.iloc[dev_idx],   ytrain.iloc[dev_idx]
    Xt         = Xtest.copy()
    oof_preds, test_preds = [], []
    fold_rmse = {}
    
    # Store individual model predictions for meta-learner
    fold_meta_features_oof = []
    fold_meta_features_test = []
    
    for method, mymodel in tqdm( Mdl_Master.items() ):
        model = make_pipeline(*[TargetEncoder(random_state = 42), mymodel])
        model.fit(Xtr, ytr)
        dev_preds = model.predict(Xdev)
        mdl_preds = model.predict(Xt)
        
        # Calculate RMSE for this model on this fold
        model_rmse = root_mean_squared_error(ydev, dev_preds)
        fold_rmse[method] = model_rmse
        
        dev_preds_df = pd.DataFrame( dev_preds, index = Xdev.index, columns = ["Preds"])
        mdl_preds_df = pd.DataFrame( mdl_preds, index = Xtest.index, columns = ["Preds"])
        oof_preds.append(dev_preds_df)
        test_preds.append(mdl_preds_df)
        
        # Store individual predictions for meta-learner
        fold_meta_features_oof.append(dev_preds)
        fold_meta_features_test.append(mdl_preds)
    
    # Stack predictions as features for meta-learner
    meta_features_fold_oof = np.column_stack(fold_meta_features_oof)
    meta_features_fold_test = np.column_stack(fold_meta_features_test)
    
    meta_features_oof.append(pd.DataFrame(
        meta_features_fold_oof, 
        index=Xdev.index, 
        columns=list(Mdl_Master.keys())
    ))
    
    meta_features_test.append(pd.DataFrame(
        meta_features_fold_test, 
        index=Xtest.index, 
        columns=list(Mdl_Master.keys())
    ))
    
    meta_target.append(ydev)
    
    # Calculate weights with dampening for support models
    weights = {}
    
    # Core models get full inverse RMSE weighting
    core_rmse = {k: v for k, v in fold_rmse.items() if k in core_models}
    core_inverse_rmse = [1/rmse for rmse in core_rmse.values()]
    core_total_inverse = sum(core_inverse_rmse)
    core_weights = {model: (1/fold_rmse[model])/core_total_inverse * 0.8 for model in core_models}  # 80% total weight
    
    # Support models get dampened weighting (max 20% total weight)
    support_rmse = {k: v for k, v in fold_rmse.items() if k in support_models}
    if support_rmse:
        support_inverse_rmse = [1/rmse for rmse in support_rmse.values()]
        support_total_inverse = sum(support_inverse_rmse)
        support_weights = {model: (1/fold_rmse[model])/support_total_inverse * 0.2 for model in support_models}  # 20% total weight
        weights.update(support_weights)
    
    weights.update(core_weights)
    fold_weights.append(weights)
    
    print(f"Fold {fold_nb + 1} RMSE: {fold_rmse}")
    print(f"Fold {fold_nb + 1} Weights: {weights}")
    
    # Apply weights to predictions
    weighted_oof = np.zeros(len(ydev))
    weighted_test = np.zeros(len(Xt))
    
    for i, (method, weight) in enumerate(zip(fold_rmse.keys(), weights.values())):
        weighted_oof += oof_preds[i]["Preds"].values * weight
        weighted_test += test_preds[i]["Preds"].values * weight
    
    weighted_oof_df = pd.DataFrame(weighted_oof, index=Xdev.index, columns=["Preds"])
    weighted_test_df = pd.DataFrame(weighted_test, index=Xtest.index, columns=["Preds"])
    
    OOF_Preds.append(weighted_oof_df)
    Mdl_Preds.append(weighted_test_df)
    
OOF_Preds = pd.concat(OOF_Preds, axis= 0).sort_index(ascending = True)
Mdl_Preds = (
    pd.concat(Mdl_Preds, axis= 0).
    sort_index(ascending = True).
    groupby(level = 0).
    mean()
)
score = root_mean_squared_error(ytrain, OOF_Preds.values.flatten())
print(f"\n---> Combined Score = {score:,.8f}\n\n")



# MULTI-LEVEL META-LEARNING SYSTEM
print(f"\n" + "="*70)
print("MULTI-LEVEL META-LEARNING SYSTEM")
print("Level 1: Ridge + Linear Regression + Neural Network")
print("Level 2: Ridge Meta-learner")
print("Level 3: Ridge Meta-learner")
print("="*70)

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

# Combine all folds for meta-learner training (assuming this comes from your existing code)
print("---> Combining base model features...")
meta_X_oof = pd.concat(meta_features_oof, axis=0).sort_index()
meta_y_oof = pd.concat(meta_target, axis=0).sort_index()
meta_X_test = pd.concat(meta_features_test, axis=0).groupby(level=0).mean().sort_index()

print(f"Base model OOF features shape: {meta_X_oof.shape}")
print(f"Base model test features shape: {meta_X_test.shape}")
print(f"Base models: {list(meta_X_oof.columns)}")

# ============================================================================
# LEVEL 1 META-LEARNERS: Ridge + Linear Regression + Neural Network
# ============================================================================

print(f"\n" + "="*50)
print("LEVEL 1 META-LEARNERS")
print("="*50)

# Prepare cross-validation for Level 1
cv_level1 = KFold(60, shuffle=True, random_state=42)

# Storage for Level 1 predictions
level1_oof_preds = pd.DataFrame(index=meta_X_oof.index)
level1_test_preds = pd.DataFrame(index=meta_X_test.index)

# ===== LEVEL 1 MODEL 1: RIDGE REGRESSION =====
print("\n---> Training Level 1 Ridge Regression...")

# Find best alpha for Ridge
alpha_values = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0]
best_ridge_alpha = None
best_ridge_score = float('inf')

for alpha in alpha_values:
    ridge_scores = []
    for train_idx, val_idx in cv_level1.split(meta_X_oof, meta_y_oof):
        X_tr, X_val = meta_X_oof.iloc[train_idx], meta_X_oof.iloc[val_idx]
        y_tr, y_val = meta_y_oof.iloc[train_idx], meta_y_oof.iloc[val_idx]
        
        ridge_model = Ridge(alpha=alpha, random_state=42)
        ridge_model.fit(X_tr, y_tr)
        val_preds = ridge_model.predict(X_val)
        ridge_scores.append(np.sqrt(mean_squared_error(y_val, val_preds)))
    
    avg_score = np.mean(ridge_scores)
    if avg_score < best_ridge_score:
        best_ridge_score = avg_score
        best_ridge_alpha = alpha

print(f"Best Ridge alpha: {best_ridge_alpha}, CV RMSE: {best_ridge_score:.6f}")

# Generate Ridge OOF predictions
ridge_oof = np.zeros(len(meta_X_oof))
for train_idx, val_idx in cv_level1.split(meta_X_oof, meta_y_oof):
    X_tr, X_val = meta_X_oof.iloc[train_idx], meta_X_oof.iloc[val_idx]
    y_tr, y_val = meta_y_oof.iloc[train_idx], meta_y_oof.iloc[val_idx]
    
    ridge_model = Ridge(alpha=best_ridge_alpha, random_state=42)
    ridge_model.fit(X_tr, y_tr)
    ridge_oof[val_idx] = ridge_model.predict(X_val)

# Generate Ridge test predictions
final_ridge = Ridge(alpha=best_ridge_alpha, random_state=42)
final_ridge.fit(meta_X_oof, meta_y_oof)
ridge_test = final_ridge.predict(meta_X_test)

level1_oof_preds['Ridge_L1'] = ridge_oof
level1_test_preds['Ridge_L1'] = ridge_test

# ===== LEVEL 1 MODEL 2: LINEAR REGRESSION =====
print("\n---> Training Level 1 Linear Regression...")

# Generate Linear Regression OOF predictions
lr_oof = np.zeros(len(meta_X_oof))
for train_idx, val_idx in cv_level1.split(meta_X_oof, meta_y_oof):
    X_tr, X_val = meta_X_oof.iloc[train_idx], meta_X_oof.iloc[val_idx]
    y_tr, y_val = meta_y_oof.iloc[train_idx], meta_y_oof.iloc[val_idx]
    
    lr_model = LinearRegression()
    lr_model.fit(X_tr, y_tr)
    lr_oof[val_idx] = lr_model.predict(X_val)

# Generate Linear Regression test predictions
final_lr = LinearRegression()
final_lr.fit(meta_X_oof, meta_y_oof)
lr_test = final_lr.predict(meta_X_test)

level1_oof_preds['LinearReg_L1'] = lr_oof
level1_test_preds['LinearReg_L1'] = lr_test

lr_rmse = np.sqrt(mean_squared_error(meta_y_oof, lr_oof))
print(f"Linear Regression RMSE: {lr_rmse:.6f}")

# ===== LEVEL 1 MODEL 3: NEURAL NETWORK =====
print("\n---> Training Level 1 Neural Network...")

# Scale features for Neural Network
scaler = StandardScaler()
meta_X_scaled = pd.DataFrame(
    scaler.fit_transform(meta_X_oof), 
    index=meta_X_oof.index, 
    columns=meta_X_oof.columns
)
meta_X_test_scaled = pd.DataFrame(
    scaler.transform(meta_X_test), 
    index=meta_X_test.index, 
    columns=meta_X_test.columns
)

# Try different Neural Network configurations
nn_configs = [
    {'hidden_layer_sizes': (50,), 'alpha': 0.001, 'learning_rate_init': 0.001},
    {'hidden_layer_sizes': (100,), 'alpha': 0.01, 'learning_rate_init': 0.001},
    {'hidden_layer_sizes': (50, 25), 'alpha': 0.001, 'learning_rate_init': 0.01},
    {'hidden_layer_sizes': (100, 50), 'alpha': 0.01, 'learning_rate_init': 0.001}
]

best_nn_config = None
best_nn_score = float('inf')

for config in nn_configs:
    nn_scores = []
    for train_idx, val_idx in cv_level1.split(meta_X_scaled, meta_y_oof):
        X_tr, X_val = meta_X_scaled.iloc[train_idx], meta_X_scaled.iloc[val_idx]
        y_tr, y_val = meta_y_oof.iloc[train_idx], meta_y_oof.iloc[val_idx]
        
        nn_model = MLPRegressor(
            hidden_layer_sizes=config['hidden_layer_sizes'],
            alpha=config['alpha'],
            learning_rate_init=config['learning_rate_init'],
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        nn_model.fit(X_tr, y_tr)
        val_preds = nn_model.predict(X_val)
        nn_scores.append(np.sqrt(mean_squared_error(y_val, val_preds)))
    
    avg_score = np.mean(nn_scores)
    if avg_score < best_nn_score:
        best_nn_score = avg_score
        best_nn_config = config

print(f"Best NN config: {best_nn_config}, CV RMSE: {best_nn_score:.6f}")

# Generate Neural Network OOF predictions
nn_oof = np.zeros(len(meta_X_scaled))
for train_idx, val_idx in cv_level1.split(meta_X_scaled, meta_y_oof):
    X_tr, X_val = meta_X_scaled.iloc[train_idx], meta_X_scaled.iloc[val_idx]
    y_tr, y_val = meta_y_oof.iloc[train_idx], meta_y_oof.iloc[val_idx]
    
    nn_model = MLPRegressor(
        hidden_layer_sizes=best_nn_config['hidden_layer_sizes'],
        alpha=best_nn_config['alpha'],
        learning_rate_init=best_nn_config['learning_rate_init'],
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    nn_model.fit(X_tr, y_tr)
    nn_oof[val_idx] = nn_model.predict(X_val)

# Generate Neural Network test predictions
final_nn = MLPRegressor(
    hidden_layer_sizes=best_nn_config['hidden_layer_sizes'],
    alpha=best_nn_config['alpha'],
    learning_rate_init=best_nn_config['learning_rate_init'],
    max_iter=1000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)
final_nn.fit(meta_X_scaled, meta_y_oof)
nn_test = final_nn.predict(meta_X_test_scaled)

level1_oof_preds['NeuralNet_L1'] = nn_oof
level1_test_preds['NeuralNet_L1'] = nn_test

print(f"\nLevel 1 OOF predictions shape: {level1_oof_preds.shape}")
print(f"Level 1 test predictions shape: {level1_test_preds.shape}")

# ============================================================================
# LEVEL 2 META-LEARNER: Ridge on Level 1 outputs
# ============================================================================

print(f"\n" + "="*50)
print("LEVEL 2 META-LEARNER (RIDGE)")
print("="*50)

# Find best alpha for Level 2 Ridge
cv_level2 = KFold(400, shuffle=True, random_state=43)
alpha_values_l2 = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0]
best_ridge_l2_alpha = None
best_ridge_l2_score = float('inf')

for alpha in alpha_values_l2:
    ridge_l2_scores = []
    for train_idx, val_idx in cv_level2.split(level1_oof_preds, meta_y_oof):
        X_tr, X_val = level1_oof_preds.iloc[train_idx], level1_oof_preds.iloc[val_idx]
        y_tr, y_val = meta_y_oof.iloc[train_idx], meta_y_oof.iloc[val_idx]
        
        ridge_l2_model = Ridge(alpha=alpha, random_state=42)
        ridge_l2_model.fit(X_tr, y_tr)
        val_preds = ridge_l2_model.predict(X_val)
        ridge_l2_scores.append(np.sqrt(mean_squared_error(y_val, val_preds)))
    
    avg_score = np.mean(ridge_l2_scores)
    if avg_score < best_ridge_l2_score:
        best_ridge_l2_score = avg_score
        best_ridge_l2_alpha = alpha

print(f"Best Level 2 Ridge alpha: {best_ridge_l2_alpha}, CV RMSE: {best_ridge_l2_score:.6f}")

# Generate Level 2 OOF predictions
level2_oof = np.zeros(len(level1_oof_preds))
for train_idx, val_idx in cv_level2.split(level1_oof_preds, meta_y_oof):
    X_tr, X_val = level1_oof_preds.iloc[train_idx], level1_oof_preds.iloc[val_idx]
    y_tr, y_val = meta_y_oof.iloc[train_idx], meta_y_oof.iloc[val_idx]
    
    ridge_l2_model = Ridge(alpha=best_ridge_l2_alpha, random_state=42)
    ridge_l2_model.fit(X_tr, y_tr)
    level2_oof[val_idx] = ridge_l2_model.predict(X_val)

# Generate Level 2 test predictions
final_ridge_l2 = Ridge(alpha=best_ridge_l2_alpha, random_state=42)
final_ridge_l2.fit(level1_oof_preds, meta_y_oof)
level2_test = final_ridge_l2.predict(level1_test_preds)

level2_oof_preds = pd.DataFrame({'Ridge_L2': level2_oof}, index=level1_oof_preds.index)
level2_test_preds = pd.DataFrame({'Ridge_L2': level2_test}, index=level1_test_preds.index)

level2_rmse = np.sqrt(mean_squared_error(meta_y_oof, level2_oof))
print(f"Level 2 RMSE: {level2_rmse:.6f}")

# ============================================================================
# LEVEL 3 META-LEARNER: Ridge on Level 2 + Level 1 combined
# ============================================================================

print(f"\n" + "="*50)
print("LEVEL 3 META-LEARNER (RIDGE)")
print("="*50)

# Combine Level 1 and Level 2 features
level3_oof_features = pd.concat([level1_oof_preds, level2_oof_preds], axis=1)
level3_test_features = pd.concat([level1_test_preds, level2_test_preds], axis=1)

print(f"Level 3 features: {list(level3_oof_features.columns)}")

# Find best alpha for Level 3 Ridge
cv_level3 = KFold(400, shuffle=True, random_state=44)
alpha_values_l3 = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0]
best_ridge_l3_alpha = None
best_ridge_l3_score = float('inf')

for alpha in alpha_values_l3:
    ridge_l3_scores = []
    for train_idx, val_idx in cv_level3.split(level3_oof_features, meta_y_oof):
        X_tr, X_val = level3_oof_features.iloc[train_idx], level3_oof_features.iloc[val_idx]
        y_tr, y_val = meta_y_oof.iloc[train_idx], meta_y_oof.iloc[val_idx]
        
        ridge_l3_model = Ridge(alpha=alpha, random_state=42)
        ridge_l3_model.fit(X_tr, y_tr)
        val_preds = ridge_l3_model.predict(X_val)
        ridge_l3_scores.append(np.sqrt(mean_squared_error(y_val, val_preds)))
    
    avg_score = np.mean(ridge_l3_scores)
    if avg_score < best_ridge_l3_score:
        best_ridge_l3_score = avg_score
        best_ridge_l3_alpha = alpha

print(f"Best Level 3 Ridge alpha: {best_ridge_l3_alpha}, CV RMSE: {best_ridge_l3_score:.6f}")

# Generate Level 3 final predictions
final_ridge_l3 = Ridge(alpha=best_ridge_l3_alpha, random_state=42)
final_ridge_l3.fit(level3_oof_features, meta_y_oof)
level3_oof = final_ridge_l3.predict(level3_oof_features)
level3_test = final_ridge_l3.predict(level3_test_features)

level3_rmse = np.sqrt(mean_squared_error(meta_y_oof, level3_oof))

# ============================================================================
# FINAL RESULTS AND COMPARISON
# ============================================================================

print(f"\n" + "="*70)
print("MULTI-LEVEL META-LEARNING RESULTS")
print("="*70)

# Calculate individual Level 1 model RMSEs
ridge_l1_rmse = np.sqrt(mean_squared_error(meta_y_oof, level1_oof_preds['Ridge_L1']))
lr_l1_rmse = np.sqrt(mean_squared_error(meta_y_oof, level1_oof_preds['LinearReg_L1']))
nn_l1_rmse = np.sqrt(mean_squared_error(meta_y_oof, level1_oof_preds['NeuralNet_L1']))

print(f"Level 1 Model Performance:")
print(f"  Ridge L1 RMSE:           {ridge_l1_rmse:.8f}")
print(f"  Linear Regression RMSE:  {lr_l1_rmse:.8f}")
print(f"  Neural Network RMSE:     {nn_l1_rmse:.8f}")
print(f"\nLevel 2 Ridge RMSE:        {level2_rmse:.8f}")
print(f"Level 3 Ridge RMSE:        {level3_rmse:.8f}")

# Compare with original results (assuming these exist from your code)
print(f"\nComparison with original:")
print(f"Original Ensemble RMSE:    {score:.8f}")
print(f"Level 3 Meta-learner:      {level3_rmse:.8f}")

improvement = score - level3_rmse
print(f"Improvement:               {improvement:+.8f} ({(improvement/score*100):+.4f}%)")

# Show Level 3 Ridge coefficients
print(f"\n" + "="*50)
print("LEVEL 3 META-LEARNER FEATURE WEIGHTS")
print("="*50)

ridge_l3_coeffs = pd.DataFrame({
    'Feature': level3_oof_features.columns,
    'Coefficient': final_ridge_l3.coef_,
    'Abs_Coefficient': np.abs(final_ridge_l3.coef_)
}).sort_values('Abs_Coefficient', ascending=False)

for _, row in ridge_l3_coeffs.iterrows():
    print(f"{row['Feature']:>15}: {row['Coefficient']:>8.4f} (|{row['Abs_Coefficient']:.4f}|)")

print(f"\nLevel 3 Ridge Intercept: {final_ridge_l3.intercept_:.6f}")

# Final predictions
print(f"\n" + "="*50)
print("FINAL PREDICTIONS")
print("="*50)

Level3_OOF_Preds = pd.DataFrame(level3_oof, index=meta_y_oof.index, columns=["Preds"])
Level3_Test_Preds = pd.DataFrame(level3_test, index=level3_test_features.index, columns=["Preds"])

print(f"Level 3 OOF predictions shape:  {Level3_OOF_Preds.shape}")
print(f"Level 3 Test predictions shape: {Level3_Test_Preds.shape}")

# Determine best model
all_scores = {
    "Original Ensemble": score,
    "Level 1 Ridge": ridge_l1_rmse,
    "Level 1 Linear Regression": lr_l1_rmse, 
    "Level 1 Neural Network": nn_l1_rmse,
    "Level 2 Meta-learner": level2_rmse,
    "Level 3 Meta-learner": level3_rmse
}

best_method = min(all_scores.keys(), key=lambda x: all_scores[x])
best_rmse = all_scores[best_method]

print(f"\nBest performing method: {best_method}")
print(f"Best RMSE: {best_rmse:.8f}")

if best_method == "Level 3 Meta-learner":
    print("✓ Use Level3_Test_Preds for final submission")
    Final_Predictions = Level3_Test_Preds
else:
    print(f"✓ Use predictions from {best_method}")

print(f"\nMulti-level meta-learning complete!")
print(f"Final prediction shape: {Final_Predictions.shape if 'Final_Predictions' in locals() else Level3_Test_Preds.shape}")


%%time 

sub_fl[target] = Final_Predictions
sub_fl.to_csv(f"submission.csv", index = True) 
!ls
print()
!head submission.csv

