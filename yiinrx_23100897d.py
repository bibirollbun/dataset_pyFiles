import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

print('Script Start ...')
# Read the data
public_aucs_csv = pd.read_csv("../input/300-probed-aucs-from-dont-overfit-ii/probed_aucs.csv")
print('public_aucs columns',public_aucs_csv.columns)
public_aucs = public_aucs_csv['public_auc'].values
public_aucs_names = public_aucs_csv['variable'].values
print('public_aucs values and names:',public_aucs[:10],public_aucs_names[:10])
public_aucs = pd.Series(public_aucs, index=public_aucs_names)
print('public_aucs series:\n',public_aucs.head())


# dat_train = pd.read_csv("../input/dont-overfit-ii/train.csv")
# dat_test = pd.read_csv("../input/dont-overfit-ii/test.csv")

# Using older dataset
dat_train = pd.read_csv("/kaggle/input/older-dataset-for-dont-overfit-ii-challenge/train.csv")
dat_test = pd.read_csv("/kaggle/input/older-dataset-for-dont-overfit-ii-challenge/test.csv")

print(f"Train shape: {dat_train.shape}")
print(f"Test shape:  {dat_test.shape}")
print(f"Public AUCs read: {len(public_aucs)} features")

# Clean the data a bit
if 'id' in dat_train.columns:
    dat_train = dat_train.drop(columns=['id'])
if 'id' in dat_test.columns:
    dat_test = dat_test.drop(columns=['id'])

# replace invalid chars with "_"
dat_train.columns = [c.replace(' ', '_').replace('-', '_') for c in dat_train.columns]
dat_test.columns = [c.replace(' ', '_').replace('-', '_') for c in dat_test.columns]

# Split into X matrices and a y vector
X_train = dat_train.drop(columns=['target']).values
X_sub = dat_test[dat_train.drop(columns=['target']).columns].values
y_train = dat_train['target'].values

print('After splitting into features and target ...')
print(f"X_train shape: {X_train.shape}")
print(f"y_train length: {len(y_train)}")
print('public_aucs series:\n',public_aucs.head())


def fastAUC(x, y):
    """Compute AUC quickly using rank method (same as the R version)."""
    x1 = x[y == 1]
    x2 = x[y == 0]
    n1 = len(x1)
    n2 = len(x2)
    r = pd.Series(np.concatenate([x1, x2])).rank().values
    auc = (np.sum(r[:n1]) - n1 * (n1 + 1) / 2) / (n1 * n2)
    return auc

def fastcolAUC(X, y):
    """Apply fastAUC to each column of X."""
    return np.apply_along_axis(lambda col: fastAUC(col, y), 0, X)

# Compute AUCs for all training columns
train_aucs = fastcolAUC(X_train, y_train)

print('dat_train columns:',dat_train.columns)
print('train_aucs columns:',train_aucs[:10])
print('public_aucs columns before:\n',public_aucs.head())

# Sort the 2 vectors of AUCs in the same order
public_aucs = public_aucs.rename(index=lambda x: x.replace('X', '')).reindex(
    dat_train.drop(columns=['target']).columns
)

print('train_aucs:',np.round(train_aucs[:10],4))
print("public_aucs aligned:", np.round(public_aucs.values[:10], 4))
print("shapes:", train_aucs.shape, public_aucs.shape)

# Plot
plt.figure(figsize=(6,6))
plt.scatter(
    train_aucs,
    public_aucs.values,
    facecolors='none',  
    edgecolors='black', 
    s=25                
)
plt.plot([0, 1], [0, 1], color='black', linewidth=1)
plt.title("Actual public vs train", fontsize=12)
plt.xlabel("train_aucs", fontsize=11)
plt.ylabel("public_aucs", fontsize=11)
plt.xlim(min(train_aucs) - 0.05, max(train_aucs) + 0.05)
plt.ylim(min(public_aucs.values) - 0.05, max(public_aucs.values) + 0.05)
plt.box(True)
plt.show()


train_rows = 250
test_rows = 19750
public_split = 0.10

public_rows = int(test_rows * public_split)
private_rows = int(test_rows * (1 - public_split))

total_rows = train_rows + public_rows
train_weight = train_rows / total_rows
public_weight = public_rows / total_rows

combined_aucs = train_weight * train_aucs + public_weight * public_aucs
coefficients = combined_aucs - 0.5

print(f'train weight: {round(train_weight,3)}, public_weight: {round(public_weight,3)}')


# Simulate training data
np.random.seed(1234)
ncols = 300
X_train = np.random.randn(train_rows, ncols)
X_public = np.random.randn(public_rows, ncols)
X_private = np.random.randn(private_rows, ncols)

# Apply known coefficients and add noise
coefficients_300 = np.array(coefficients)
y_train = (X_train @ coefficients_300) + np.random.rand(X_train.shape[0]) / 1.5
y_public = (X_public @ coefficients_300) + np.random.rand(X_public.shape[0]) / 1.5
y_private = (X_private @ coefficients_300) + np.random.rand(X_private.shape[0]) / 1.5

print(X_train.shape)
print(X_public.shape)
print(X_private.shape)

# Cut y into binary labels
CUT = np.sort(y_train)[90]
y_train = (y_train > CUT).astype(int)
y_public = (y_public > CUT).astype(int)
y_private = (y_private > CUT).astype(int)

# Calculate AUCs
auc_train_300 = fastcolAUC(X_train, y_train)
auc_public_300 = fastcolAUC(X_public, y_public)
auc_private_300 = fastcolAUC(X_private, y_private)

# Plot results
plt.figure(figsize=(6, 6))
plt.scatter(auc_train_300, auc_public_300,
            facecolors='none', edgecolors='black', s=40, linewidth=0.8)
plt.plot([0, 1], [0, 1], color='black', linewidth=0.8)
plt.title("Simulated public vs train - assuming 300 variables", fontsize=12, fontweight='bold')
plt.xlabel("auc_train_300", fontsize=10)
plt.ylabel("auc_public_300", fontsize=10)
plt.xlim(min(auc_train_300) - 0.01, max(auc_train_300) + 0.01)
plt.ylim(min(auc_public_300) - 0.01, max(auc_public_300) + 0.01)
plt.grid(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.show()


import sys, os
import pandas as pd
import numpy as np

from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import RobustScaler  
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, r2_score, make_scorer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# suppress warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# ------------------------------
# Settings
# ------------------------------
rfe_min_features = 12
rfe_step = 15
rfe_cv = 20
sss_n_splits = 12
sss_test_size = 0.2   # hold-out
grid_search_cv = 20
noise_std = 0.01
r2_threshold = 0.185
random_seed = 213

np.random.seed(random_seed)

dat_test = pd.read_csv("/kaggle/input/older-dataset-for-dont-overfit-ii-challenge/test.csv")
coefficients = np.array(coefficients)

# ------------------------------
# Load sim data
# ------------------------------
# Simulate training data
np.random.seed(1234)
ncols = 300
X_train = np.random.randn(2000, ncols)  ## change sim data size here

# Apply known coefficients and add noise
coefficients_300 = np.array(coefficients)  ## using different sim data (300,50,35)
y_train = (X_train @ coefficients_300) + np.random.rand(X_train.shape[0]) / 1.5
threshold_ratio = 90 / 250  # ~0.36
CUT = np.percentile(y_train, threshold_ratio * 100)
y_train = (y_train > CUT).astype(int)

sim_y = y_train
sim_X = X_train

# ------------------------------
# Split hold-out
# ------------------------------
X_train, X_holdout, y_train, y_holdout = train_test_split(
    sim_X, sim_y, test_size=sss_test_size, random_state=random_seed, stratify=sim_y
)

# ------------------------------
# Scale features
# ------------------------------
scaler = RobustScaler().fit(np.concatenate([X_train, X_holdout], axis=0))
X_train = scaler.transform(X_train)
X_holdout = scaler.transform(X_holdout)

# add a bit of noise to reduce overfitting
X_train += np.random.normal(0, noise_std, X_train.shape)

# ------------------------------
# Custom ROC-AUC scorer (robust to single-class predictions)
# ------------------------------
def scoring_roc_auc(y, y_pred):
    try:
        return roc_auc_score(y, y_pred)
    except:
        return 0.5

robust_roc_auc = make_scorer(scoring_roc_auc)

# ------------------------------
# Define model
# ------------------------------
model = Lasso(alpha=0.031, tol=0.01, random_state=random_seed, selection='random')
param_grid = {
    'alpha': [0.019,0.02,0.021,0.022,0.023,0.024,0.025,0.026,0.027,0.029,0.031],
    'tol': [0.001,0.0011,0.0012,0.0013,0.0014,0.0015,0.0016,0.0017]
}

feature_selector = RFECV(
    model, min_features_to_select=rfe_min_features,
    scoring=robust_roc_auc, step=rfe_step,
    verbose=0, cv=rfe_cv, n_jobs=-1
)

# ------------------------------
# Training + validation on hold-out
# ------------------------------

print("counter | val_mse  |  val_mae  |  val_roc  |  val_cos  |  val_dist  |  val_r2    | feature_count ")
print("-------------------------------------------------------------------------------------------------")

predictions = pd.DataFrame()
counter = 0

for train_index, val_index in StratifiedShuffleSplit(
        n_splits=sss_n_splits, test_size=sss_test_size, random_state=random_seed
    ).split(X_train, y_train):
    
    X_sub, val_X = X_train[train_index], X_train[val_index]
    y_sub, val_y = y_train[train_index], y_train[val_index]
    
    # Feature selection
    feature_selector.fit(X_sub, y_sub)
    X_sub_sel = feature_selector.transform(X_sub)
    val_X_sel = feature_selector.transform(val_X)
    X_holdout_sel = feature_selector.transform(X_holdout)
    
    # Grid search for best Lasso params
    grid_search = GridSearchCV(
        feature_selector.estimator_, param_grid=param_grid,
        verbose=0, n_jobs=-1, scoring=robust_roc_auc, cv=grid_search_cv
    )
    grid_search.fit(X_sub_sel, y_sub)
    
    # Validation
    val_y_pred = grid_search.best_estimator_.predict(val_X_sel)
    val_mse = mean_squared_error(val_y, val_y_pred)
    val_mae = mean_absolute_error(val_y, val_y_pred)
    val_roc = roc_auc_score(val_y, val_y_pred)
    val_cos = cosine_similarity(val_y.reshape(1, -1), val_y_pred.reshape(1, -1))[0][0]
    val_dst = euclidean_distances(val_y.reshape(1, -1), val_y_pred.reshape(1, -1))[0][0]
    val_r2 = r2_score(val_y, val_y_pred)
    
    # Only keep good models
    if val_r2 > r2_threshold:
        message = '<-- OK'
        prediction = grid_search.best_estimator_.predict(X_holdout_sel)
        predictions = pd.concat([predictions, pd.DataFrame(prediction)], axis=1)
    else:
        message = '<-- skipping'
    
    print("{0:2} | {1:.4f} | {2:.4f} | {3:.4f} | {4:.4f} | {5:.4f} | {6:.4f} | {7:3} {8}".format(
        counter, val_mse, val_mae, val_roc, val_cos, val_dst, val_r2, feature_selector.n_features_, message
    ))
    
    counter += 1

# ------------------------------
# Ensemble predictions on hold-out
# ------------------------------
mean_pred = predictions.mean(axis=1)
holdout_results = pd.DataFrame({
    'y_true': y_holdout,
    'y_pred': mean_pred
})
holdout_results.to_csv('holdout_predictions.csv', index=False)

print("-------------------------------------------------------------------------------------------------")
print("{}/{} models passed validation threshold and will be ensembled.".format(len(predictions.columns), sss_n_splits))


## submit
# ------------------------------
# Apply trained model to the X_sub
# ------------------------------
dat_test = pd.read_csv("/kaggle/input/older-dataset-for-dont-overfit-ii-challenge/test.csv")
X_sub = dat_test[dat_train.drop(columns=['target']).columns].values
X_sub_sel = feature_selector.transform(X_sub)
y_sub_pred = grid_search.best_estimator_.predict(X_sub_sel)

# ------------------------------
# Generate submission.csv
# ------------------------------
submission = pd.DataFrame({
    'id': dat_test['id'],
    'target': y_sub_pred
})
submission.to_csv('submission.csv', index=False)
print("submission.csv generated successfully!")


## Lasso for penalized sim data
# tried several times, but no significant improvement versus to the previous one
import sys, os
import pandas as pd
import numpy as np

from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, r2_score, make_scorer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# suppress warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# ------------------------------
# Settings
# ------------------------------
rfe_min_features = 12
rfe_step = 15
rfe_cv = 20
sss_n_splits = 12
sss_test_size = 0.2   # hold-out
grid_search_cv = 20
noise_std = 0.01
r2_threshold = 0.185
random_seed = 213

np.random.seed(random_seed)

dat_test = pd.read_csv("/kaggle/input/older-dataset-for-dont-overfit-ii-challenge/test.csv")
coefficients = np.array(coefficients)

## penalization equations
def fun_eur_50(cf):
    return 0.0320794465661013 + 1.38501844798023 * cf - 9.51464300036321 * cf**3 - 0.0654557703259389 * expit(85.9772069321308 * cf)
def fun_eur_300(cf):
    return 1.05780402210757 * cf * expit(0.0750590053462814 + 339.425848223982 * cf**2) - 0.000308402162785152

# ------------------------------
# Load sim data
# ------------------------------
# Simulate training data
np.random.seed(1234)
ncols = 300
X_train = np.random.randn(2000, ncols)

# Apply known coefficients and add noise
coefficients_300 = np.array(coefficients)
coefficients_300 = fun_eur_300(coefficients_300)
y_train = (X_train @ coefficients_300) + np.random.rand(X_train.shape[0]) / 1.5
threshold_ratio = 90 / 250  # ~0.36
CUT = np.percentile(y_train, threshold_ratio * 100)
y_train = (y_train > CUT).astype(int)

sim_y = y_train
sim_X = X_train

# ------------------------------
# Split hold-out
# ------------------------------
X_train, X_holdout, y_train, y_holdout = train_test_split(
    sim_X, sim_y, test_size=sss_test_size, random_state=random_seed, stratify=sim_y
)

# ------------------------------
# Scale features
# ------------------------------
scaler = RobustScaler().fit(np.concatenate([X_train, X_holdout], axis=0))
X_train = scaler.transform(X_train)
X_holdout = scaler.transform(X_holdout)

# add a bit of noise to reduce overfitting
X_train += np.random.normal(0, noise_std, X_train.shape)

# ------------------------------
# Custom ROC-AUC scorer (robust to single-class predictions)
# ------------------------------
def scoring_roc_auc(y, y_pred):
    try:
        return roc_auc_score(y, y_pred)
    except:
        return 0.5

robust_roc_auc = make_scorer(scoring_roc_auc)

# ------------------------------
# Define model
# ------------------------------
model = Lasso(alpha=0.031, tol=0.01, random_state=random_seed, selection='random')
param_grid = {
    'alpha': [0.019,0.02,0.021,0.022,0.023,0.024,0.025,0.026,0.027,0.029,0.031],
    'tol': [0.001,0.0011,0.0012,0.0013,0.0014,0.0015,0.0016,0.0017]
}

feature_selector = RFECV(
    model, min_features_to_select=rfe_min_features,
    scoring=robust_roc_auc, step=rfe_step,
    verbose=0, cv=rfe_cv, n_jobs=-1
)

# ------------------------------
# Training + validation on hold-out
# ------------------------------

print("counter | val_mse  |  val_mae  |  val_roc  |  val_cos  |  val_dist  |  val_r2    | feature_count ")
print("-------------------------------------------------------------------------------------------------")

predictions = pd.DataFrame()
counter = 0

for train_index, val_index in StratifiedShuffleSplit(
        n_splits=sss_n_splits, test_size=sss_test_size, random_state=random_seed
    ).split(X_train, y_train):
    
    X_sub, val_X = X_train[train_index], X_train[val_index]
    y_sub, val_y = y_train[train_index], y_train[val_index]
    
    # Feature selection
    feature_selector.fit(X_sub, y_sub)
    X_sub_sel = feature_selector.transform(X_sub)
    val_X_sel = feature_selector.transform(val_X)
    X_holdout_sel = feature_selector.transform(X_holdout)
    
    # Grid search for best Lasso params
    grid_search = GridSearchCV(
        feature_selector.estimator_, param_grid=param_grid,
        verbose=0, n_jobs=-1, scoring=robust_roc_auc, cv=grid_search_cv
    )
    grid_search.fit(X_sub_sel, y_sub)
    
    # Validation
    val_y_pred = grid_search.best_estimator_.predict(val_X_sel)
    val_mse = mean_squared_error(val_y, val_y_pred)
    val_mae = mean_absolute_error(val_y, val_y_pred)
    val_roc = roc_auc_score(val_y, val_y_pred)
    val_cos = cosine_similarity(val_y.reshape(1, -1), val_y_pred.reshape(1, -1))[0][0]
    val_dst = euclidean_distances(val_y.reshape(1, -1), val_y_pred.reshape(1, -1))[0][0]
    val_r2 = r2_score(val_y, val_y_pred)
    
    # Only keep good models
    if val_r2 > r2_threshold:
        message = '<-- OK'
        prediction = grid_search.best_estimator_.predict(X_holdout_sel)
        predictions = pd.concat([predictions, pd.DataFrame(prediction)], axis=1)
    else:
        message = '<-- skipping'
    
    print("{0:2} | {1:.4f} | {2:.4f} | {3:.4f} | {4:.4f} | {5:.4f} | {6:.4f} | {7:3} {8}".format(
        counter, val_mse, val_mae, val_roc, val_cos, val_dst, val_r2, feature_selector.n_features_, message
    ))
    
    counter += 1

# ------------------------------
# Ensemble predictions on hold-out
# ------------------------------
mean_pred = predictions.mean(axis=1)
holdout_results = pd.DataFrame({
    'y_true': y_holdout,
    'y_pred': mean_pred
})
holdout_results.to_csv('holdout_predictions.csv', index=False)

print("-------------------------------------------------------------------------------------------------")
print("{}/{} models passed validation threshold and will be ensembled.".format(len(predictions.columns), sss_n_splits))


## submit
# ------------------------------
# Apply trained model to the X_sub
# ------------------------------
dat_test = pd.read_csv("/kaggle/input/older-dataset-for-dont-overfit-ii-challenge/test.csv")
X_sub = dat_test[dat_train.drop(columns=['target']).columns].values
X_sub_sel = feature_selector.transform(X_sub)
y_sub_pred = grid_search.best_estimator_.predict(X_sub_sel)

# ------------------------------
# Generate submission.csv
# ------------------------------
submission = pd.DataFrame({
    'id': dat_test['id'],
    'target': y_sub_pred
})
submission.to_csv('submission.csv', index=False)
print("submission.csv generated successfully!")


from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LassoCV
model = LassoCV(alphas=np.logspace(-4,-1,20), cv=5, n_jobs=2, random_state=213, max_iter=5000)

train_sizes, train_scores, val_scores = learning_curve(
    model, X_train, y_train,
    cv=5, scoring='r2',
    train_sizes=np.linspace(0.1,1.0,8), n_jobs=2, shuffle=True, random_state=213
)

train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.figure(figsize=(6,4))
plt.fill_between(train_sizes, train_mean-train_std, train_mean+train_std, alpha=0.15)
plt.fill_between(train_sizes, val_mean-val_std, val_mean+val_std, alpha=0.15)
plt.plot(train_sizes, train_mean, 'o-', label='Train R2')
plt.plot(train_sizes, val_mean, 'o-', label='Validation R2')
plt.xlabel('Training set size')
plt.ylabel('R2 score')
plt.legend()
plt.grid(True)
plt.show()



def auc_train_pub_priv(x):
    """
    Compute AUCs on train, public, and private datasets
    for a given coefficient vector x.
    """
    x = x - 0.5
    
    pred_train = X_train @ x
    pred_public = X_public @ x
    pred_private = X_private @ x
    
    out = [
        fastAUC(pred_train, y_train),
        fastAUC(pred_public, y_public),
        fastAUC(pred_private, y_private)
    ]
    
    return np.round(out, 3)


# Model 1: mean of all the training variables
auc_train_pub_priv(np.ones(300))  # real life is (0.395, 0.433, ?????) (train, pub, priv)


# Model 2: AUCs based on training data only
auc_train_pub_priv(auc_train_300)    # real life is c(0.984, 0.739, ?????) (train, pub, priv)


# Model 3: AUCs based on public LB probing
auc_train_pub_priv(auc_public_300)   # real life is c(0.907, 0.926, ?????) (train, pub, priv)  - AUCs based only on the public LB data


coefficients_50 = np.array(coefficients)[np.argsort(np.abs(coefficients))]
coefficients_50[:250] = 0  

y_train = (X_train @ coefficients_50) + np.random.rand(X_train.shape[0]) / 1.5
y_public = (X_public @ coefficients_50) + np.random.rand(X_public.shape[0]) / 1.5
y_private = (X_private @ coefficients_50) + np.random.rand(X_private.shape[0]) / 1.5

CUT = np.sort(y_train)[90]
y_train = (y_train > CUT).astype(int)
y_public = (y_public > CUT).astype(int)
y_private = (y_private > CUT).astype(int)

auc_train_50 = fastcolAUC(X_train, y_train)
auc_public_50 = fastcolAUC(X_public, y_public)
auc_private_50 = fastcolAUC(X_private, y_private)

plt.figure(figsize=(6, 6))
plt.scatter(auc_train_50, auc_public_50,
            facecolors='none', edgecolors='black', s=40, linewidth=0.8)
plt.plot([0, 1], [0, 1], color='black', linewidth=0.8)
plt.title("Simulated public vs train - assuming 50 variables", fontsize=12, fontweight='bold')
plt.xlabel("auc_train_50", fontsize=10)
plt.ylabel("auc_public_50", fontsize=10)
plt.xlim(min(auc_train_50) - 0.01, max(auc_train_50) + 0.01)
plt.ylim(min(auc_public_50) - 0.01, max(auc_public_50) + 0.01)
plt.grid(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.show()


# Model 1: mean of all the training variables
auc_train_pub_priv(np.ones(300))  # real life is (0.395, 0.433, ?????) (train, pub, priv)


# Model 2: AUCs based on training data only
auc_train_pub_priv(auc_train_50)    # real life is c(0.984, 0.739, ?????) (train, pub, priv)


# Model 3: AUCs based on public LB probing
auc_train_pub_priv(auc_public_50)   # real life is c(0.907, 0.926, ?????) (train, pub, priv)  - AUCs based only on the public LB data


## coefficient 35
coefficients_35 = np.array(coefficients)[np.argsort(np.abs(coefficients))]
coefficients_35[:265] = 0  

y_train = (X_train @ coefficients_35) + np.random.rand(X_train.shape[0]) / 1.5
y_public = (X_public @ coefficients_35) + np.random.rand(X_public.shape[0]) / 1.5
y_private = (X_private @ coefficients_35) + np.random.rand(X_private.shape[0]) / 1.5

CUT = np.sort(y_train)[90]
y_train = (y_train > CUT).astype(int)
y_public = (y_public > CUT).astype(int)
y_private = (y_private > CUT).astype(int)

auc_train_35 = fastcolAUC(X_train, y_train)
auc_public_35 = fastcolAUC(X_public, y_public)
auc_private_35 = fastcolAUC(X_private, y_private)

plt.figure(figsize=(6, 6))
plt.scatter(auc_train_35, auc_public_35,
            facecolors='none', edgecolors='black', s=40, linewidth=0.8)
plt.plot([0, 1], [0, 1], color='black', linewidth=0.8)
plt.title("Simulated public vs train - assuming 35 variables", fontsize=12, fontweight='bold')
plt.xlabel("auc_train_35", fontsize=10)
plt.ylabel("auc_public_35", fontsize=10)
plt.xlim(min(auc_train_35) - 0.01, max(auc_train_35) + 0.01)
plt.ylim(min(auc_public_35) - 0.01, max(auc_public_35) + 0.01)
plt.grid(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.show()


## coefficient 35
# Model 1: mean of all the training variables
auc_train_pub_priv(np.ones(300))  # real life is (0.395, 0.433, ?????) (train, pub, priv)


## coefficient 35
# Model 2: AUCs based on training data only
auc_train_pub_priv(auc_train_35)    # real life is c(0.984, 0.739, ?????) (train, pub, priv)


## coefficient 35
# Model 3: AUCs based on public LB probing
auc_train_pub_priv(auc_public_35)   # real life is c(0.907, 0.926, ?????) (train, pub, priv)  - AUCs based only on the public LB data


# Weight the simulated AUCs the same as we did for the real data
auc_combined_300 = train_weight * auc_train_300 + public_weight * auc_public_300
auc_combined_50 = train_weight * auc_train_50 + public_weight * auc_public_50
auc_combined_35 = train_weight * auc_train_35 + public_weight * auc_public_35

# Subtract 0.5 to convert the AUCs to coefficients
cf_combined_300 = auc_combined_300 - 0.5
cf_combined_50 = auc_combined_50 - 0.5
cf_combined_35 = auc_combined_35 - 0.5

# I decided I wanted "symmetrical" equations, so I also saved the inverse of the AUC/CF pairs

# Save the Eureqa training data for 300 variables
out_1 = pd.DataFrame({
    "probed_cf": auc_combined_300 - 0.5,
    "real_cf": coefficients_300
})
out_2 = pd.DataFrame({
    "probed_cf": auc_combined_300 - 0.5,
    "real_cf": coefficients_300
})
out = pd.concat([out_1, out_2], ignore_index=True)
print(out.head())
out.to_csv("auc_training_300.csv", index=False)

# Save the Eureqa training data for 50 variables
out_1 = pd.DataFrame({
    "probed_cf": auc_combined_50 - 0.5,
    "real_cf": coefficients_50
})
out_2 = pd.DataFrame({
    "probed_cf": auc_combined_50 - 0.5,
    "real_cf": coefficients_50
})
out = pd.concat([out_1, out_2], ignore_index=True)
print(out.head())
out.to_csv("auc_training_50.csv", index=False)

# Save the Eureqa training data for 35 variables
out_1 = pd.DataFrame({
    "probed_cf": auc_combined_35 - 0.5,
    "real_cf": coefficients_35
})
out_2 = pd.DataFrame({
    "probed_cf": auc_combined_35 - 0.5,
    "real_cf": coefficients_35
})
out = pd.concat([out_1, out_2], ignore_index=True)
print(out.head())
out.to_csv("auc_training_35.csv", index=False)

# Fit eureqa models on auc_training_300.csv using MAE and auc_training_50.csv using MSE


import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from scipy.special import expit
import warnings
warnings.filterwarnings('ignore')

# ------------------- read data -------------------
dat_test = pd.read_csv("/kaggle/input/older-dataset-for-dont-overfit-ii-challenge/test.csv")
X_sub = np.array(X_sub) 
coefficients = np.array(coefficients)

# ------------------- non-linear expansion -------------------
def construct_extended_features(X, poly=None, fit_poly=True):
    """
    input: X (n_samples,1)
    ouput: expansioned feature matrix
    """
    if poly is None:
        poly = PolynomialFeatures(degree=3, include_bias=False)
        X_poly = poly.fit_transform(X)
    else:
        if fit_poly:
            X_poly = poly.fit_transform(X)
        else:
            X_poly = poly.transform(X)

    X_extra = np.column_stack([
        X,           # cf
        X**2,        # cf^2
        X**3,        # cf^3
        np.exp(X),   # exp(cf)
        np.exp(-X),  # exp(-cf)
        expit(X)     # sigmoid(cf)
    ])

    X_extended = np.hstack([X_poly, X_extra])
    return X_extended, poly

# ------------------- training functions -------------------
def train_ml_pipeline_on_simulation(sim_csv="auc_training_35.csv"): ##choose different sim data
    print("Loading simulation data...")
    sim_data = pd.read_csv(sim_csv)
    X_sim = sim_data['probed_cf'].values.reshape(-1, 1)
    y_sim = sim_data['real_cf'].values
    print(f"Simulation data: X.shape={X_sim.shape}, y.shape={y_sim.shape}")

    # parameter
    rfe_min_features = 1
    rfe_step = 1
    rfe_cv = 10
    sss_n_splits = 8
    sss_test_size = 0.2
    grid_search_cv = 10
    r2_threshold = 0.6
    random_seed = 213
    np.random.seed(random_seed)

    scaler = RobustScaler()
    X_sim_scaled = scaler.fit_transform(X_sim)

    r2_scorer = make_scorer(r2_score)
    counter = 0
    models_collection = []

    print("counter | val_mse  |  val_mae  | val_r2    | feature_count | message")
    print("---------------------------------------------------------------------")

    for train_index, val_index in ShuffleSplit(
        n_splits=sss_n_splits, 
        test_size=sss_test_size, 
        random_state=random_seed
    ).split(X_sim_scaled, y_sim):

        X_train_scaled = X_sim_scaled[train_index]
        X_val_scaled = X_sim_scaled[val_index]
        y_train = y_sim[train_index]
        y_val = y_sim[val_index]

        # feature expansion
        X_train_extended, poly_obj = construct_extended_features(X_train_scaled, fit_poly=True)
        X_val_extended, _ = construct_extended_features(X_val_scaled, poly=poly_obj, fit_poly=False)

        # LassoCV + RFECV
        lasso_cv = LassoCV(
            alphas=np.logspace(-4, 0, 50),
            cv=10,
            max_iter=10000,
            selection='random',
            random_state=random_seed
        )

        feature_selector = RFECV(
            estimator=lasso_cv,
            min_features_to_select=rfe_min_features,
            scoring=r2_scorer,
            step=rfe_step,
            cv=rfe_cv,
            n_jobs=-1,
            verbose=0
        )

        feature_selector.fit(X_train_extended, y_train)
        X_train_selected = feature_selector.transform(X_train_extended)
        X_val_selected = feature_selector.transform(X_val_extended)

        # grid search, able to use original parameters
        param_grid = {
            'alphas': [np.logspace(-4, 0, 50)]
        }
        grid_search = GridSearchCV(
            lasso_cv,
            param_grid={},  # blank: LassoCV
            scoring=r2_scorer,
            cv=grid_search_cv,
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train_selected, y_train)

        y_val_pred = grid_search.best_estimator_.predict(X_val_selected)
        val_mse = mean_squared_error(y_val, y_val_pred)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)

        if val_r2 > r2_threshold:
            message = '<-- OK'
            models_collection.append({
                'model': grid_search.best_estimator_,
                'feature_selector': feature_selector,
                'scaler': scaler,
                'poly': poly_obj,
                'val_r2': val_r2,
                'n_features': feature_selector.n_features_
            })
        else:
            message = '<-- skipping'

        print(f"{counter:2}      | {val_mse:.6f} | {val_mae:.6f} | {val_r2:.6f} | {feature_selector.n_features_:2}            | {message}")
        counter += 1

    print("---------------------------------------------------------------------")
    print(f"{len(models_collection)}/{sss_n_splits} models passed validation")
    return models_collection

# ------------------- apply to real data -------------------
def apply_trained_models_to_real_data(models_collection, original_coefficients, X_sub):
    all_predictions = []
    for i, model_info in enumerate(models_collection):
        model = model_info['model']
        feature_selector = model_info['feature_selector']
        scaler = model_info['scaler']
        poly = model_info['poly']
        n_features_expected = model_info['n_features']

        # feature expansion
        cf_2d = original_coefficients.reshape(-1, 1)
        cf_scaled = scaler.transform(cf_2d)
        cf_extended, _ = construct_extended_features(cf_scaled, poly=poly, fit_poly=False)
        cf_selected = feature_selector.transform(cf_extended)

        if cf_selected.shape[1] != n_features_expected:
            print(f"Warning: model {i} expected {n_features_expected} features, got {cf_selected.shape[1]}")
            continue

        penalized_cf = model.predict(cf_selected)
        test_pred = (X_sub @ penalized_cf).flatten()
        all_predictions.append(test_pred)

    # ------------------- submission -------------------
    if all_predictions:
        ensemble_pred = np.mean(all_predictions, axis=0)
        submission = pd.DataFrame({'id': dat_test['id'], 'target': ensemble_pred})
        submission.to_csv('sub_ml_pipeline_ensemble.csv', index=False)
        print(f"\nCreated sub_ml_pipeline_ensemble.csv with {len(all_predictions)} models")

        # 每个单模型
        for i, pred in enumerate(all_predictions):
            single_sub = pd.DataFrame({'id': dat_test['id'], 'target': pred})
            single_sub.to_csv(f'sub_ml_pipeline_model_{i}.csv', index=False)
            print('generate')
    else:
        print("No models were successfully applied!")

    return all_predictions

# ------------------- run pipeline-------------------
print("Starting migration of ML pipeline to simulation data...")
trained_models = train_ml_pipeline_on_simulation()

if trained_models:
    real_predictions = apply_trained_models_to_real_data(trained_models, coefficients, X_sub)
    if real_predictions:
        print(f"\nSuccessfully created {len(real_predictions)} predictions")
        print("Recommended submission: sub_ml_pipeline_ensemble.csv")
    else:
        print("\nFailed to apply any models to real data!")
else:
    print("No models passed validation threshold!")



## original version of Lasso
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, LassoCV
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
import warnings
warnings.filterwarnings('ignore')

dat_test = pd.read_csv("/kaggle/input/older-dataset-for-dont-overfit-ii-challenge/test.csv")
X_sub = np.array(X_sub)
coefficients = np.array(coefficients)


def train_ml_pipeline_on_simulation():
    print("Loading simulation data...")
    sim_data = pd.read_csv("auc_training_35.csv")

    X_sim = sim_data['probed_cf'].values.reshape(-1, 1)
    y_sim = sim_data['real_cf'].values
    
    print(f"Simulation data: X.shape={X_sim.shape}, y.shape={y_sim.shape}")

    # parameter
    rfe_min_features = 1
    rfe_step = 1
    rfe_cv = 10
    sss_n_splits = 8
    sss_test_size = 0.2
    grid_search_cv = 10
    r2_threshold = 0.6
    random_seed = 213
    
    np.random.seed(random_seed)

    # standardization
    scaler = RobustScaler()
    X_sim_scaled = scaler.fit_transform(X_sim)
    
    # capture non-linear relationship
    poly = PolynomialFeatures(degree=3, include_bias=False)
    X_sim_poly = poly.fit_transform(X_sim_scaled)
    
    print(f"After polynomial expansion: {X_sim_poly.shape}")
    
    # grid
    param_grid = {
        'alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
        'tol': [0.0001, 0.001, 0.01],
        'selection': ['cyclic', 'random']
    }
    
    r2_scorer = make_scorer(r2_score)
    
    print("counter | val_mse  |  val_mae  |  val_r2    | feature_count | message")
    print("---------------------------------------------------------------------")
    
    predictions_collection = []
    models_collection = []
    counter = 0
    
    # ShuffleSplit
    for train_index, val_index in ShuffleSplit(
        n_splits=sss_n_splits, 
        test_size=sss_test_size, 
        random_state=random_seed
    ).split(X_sim_poly, y_sim):
        
        X_train, X_val = X_sim_poly[train_index], X_sim_poly[val_index]
        y_train, y_val = y_sim[train_index], y_sim[val_index]
        
        # feature selector
        model = Lasso(alpha=0.01, tol=0.001, random_state=random_seed, 
                     selection='random', max_iter=10000)
        
        feature_selector = RFECV(
            model, 
            min_features_to_select=rfe_min_features,
            scoring=r2_scorer,
            step=rfe_step,
            cv=rfe_cv,
            n_jobs=-1,
            verbose=0
        )
        
        feature_selector.fit(X_train, y_train)
        X_train_selected = feature_selector.transform(X_train)
        X_val_selected = feature_selector.transform(X_val)
        
        grid_search = GridSearchCV(
            model, 
            param_grid=param_grid, 
            scoring=r2_scorer, 
            cv=grid_search_cv, 
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train_selected, y_train)
        
        # validation
        y_val_pred = grid_search.best_estimator_.predict(X_val_selected)
        val_mse = mean_squared_error(y_val, y_val_pred)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        
        
        if val_r2 > r2_threshold:
            message = '<-- OK'
            model_info = {
                'model': grid_search.best_estimator_,
                'feature_selector': feature_selector,
                'scaler': scaler,
                'poly': poly,
                'val_r2': val_r2,
                'n_features': feature_selector.n_features_
            }
            models_collection.append(model_info)
        else:
            message = '<-- skipping'
        
        print(f"{counter:2}      | {val_mse:.6f} | {val_mae:.6f} | {val_r2:.6f} | {feature_selector.n_features_:2}            | {message}")
        counter += 1
    
    print("---------------------------------------------------------------------")
    print(f"{len(models_collection)}/{sss_n_splits} models passed validation")
    
    return models_collection

def apply_trained_models_to_real_data(models_collection, original_coefficients, X_sub):
    
    print(f"\nApplying {len(models_collection)} trained models to real data...")
    
    all_predictions = []
    
    for i, model_info in enumerate(models_collection):
        model = model_info['model']
        feature_selector = model_info['feature_selector']
        scaler = model_info['scaler']
        poly = model_info['poly']
        n_features_expected = model_info['n_features']
        
        print(f"Model {i}: expecting {n_features_expected} features, R² = {model_info['val_r2']:.4f}")
        
        try:
            cf_2d = original_coefficients.reshape(-1, 1)
            cf_scaled = scaler.transform(cf_2d)
            cf_poly = poly.transform(cf_scaled)
            
            cf_selected = feature_selector.transform(cf_poly)
        
            if cf_selected.shape[1] != n_features_expected:
                print(f"  Warning: Expected {n_features_expected} features, got {cf_selected.shape[1]}")
                continue
                
            penalized_cf = model.predict(cf_selected)
            
            if len(penalized_cf) != len(original_coefficients):
                print(f"  Warning: Coefficient dimension mismatch: {len(penalized_cf)} vs {len(original_coefficients)}")
                continue
            
            test_pred = (X_sub @ penalized_cf).flatten()
            all_predictions.append(test_pred)
            print(f"  Successfully applied model {i}")
            
        except Exception as e:
            print(f"  Error applying model {i}: {str(e)}")
            continue
    
    if all_predictions:
        ensemble_pred = np.mean(all_predictions, axis=0)
        
        # submission
        submission = pd.DataFrame({
            'id': dat_test['id'],
            'target': ensemble_pred
        })
        submission.to_csv('sub_ml_pipeline_ensemble.csv', index=False)
        print(f"\nCreated sub_ml_pipeline_ensemble.csv with {len(all_predictions)} models")
    
        for i, pred in enumerate(all_predictions):
            single_sub = pd.DataFrame({
                'id': dat_test['id'],
                'target': pred
            })
            single_sub.to_csv(f'sub_ml_pipeline_model_{i}.csv', index=False)
    else:
        print("No models were successfully applied!")
    
    return all_predictions

# run all pipeline
print("Starting migration of ML pipeline to simulation data...")

trained_models = train_ml_pipeline_on_simulation()

if trained_models:
    real_predictions = apply_trained_models_to_real_data(
        trained_models, 
        coefficients,  
        X_sub          
    )
    
    if real_predictions:
        print(f"\nSuccessfully created {len(real_predictions)} predictions")
        print("Recommended submission: sub_ml_pipeline_ensemble.csv")
    else:
        print("\nFailed to apply any models to real data!")
else:
    print("No models passed validation threshold!")

