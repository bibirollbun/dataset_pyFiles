import pandas as pd 

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import mean_squared_error, make_scorer
import warnings

warnings.filterwarnings('ignore')


# Put theme of notebook 
from colorama import Fore, Style

# Colors
red = Fore.RED + Style.BRIGHT
mgta = Fore.MAGENTA + Style.BRIGHT
yllw = Fore.YELLOW + Style.BRIGHT
cyn = Fore.CYAN + Style.BRIGHT
blue = Fore.BLUE + Style.BRIGHT

# Reset
res = Style.RESET_ALL


import matplotlib.colors as mcolors

YELLOW = "#F7C53E"

CYAN_G = "#0CF7AF"
CYAB_DARK = "#11AB7C"

PURPLE = "#D826F8"
PURPLE_DARJ = "#9309AB"
PURPLE_L = "#b683d6"

BLUE = "#0C97FA"
RED = "#FA1D19"
ORANGE = "#FA9F19"
GREEN = "#0CFA58"
LIGTH_BLUE = "#01FADC"
S_BLUE = "#81c9e6"
DARK_BLUE = "#394be6"

PALETTE_7 = [PURPLE_DARJ, PURPLE_L, PURPLE, BLUE, LIGTH_BLUE, DARK_BLUE, S_BLUE]
PALETTE_7_C = [PURPLE_DARJ, BLUE, PURPLE, LIGTH_BLUE, PURPLE_L, S_BLUE, DARK_BLUE]

cmap_2 = mcolors.LinearSegmentedColormap.from_list("", [S_BLUE, PURPLE_DARJ])


import cupy as cp 

def map3_cupy(pred_proba, true_label):
    """
    Compute MAP@3 using CuPy for GPU acceleration.
    
    Args:
        pred_proba: CuPy array of shape (n_samples, 7) with predicted probabilities for 7 classes
        true_label: CuPy array of shape (n_samples,) with true class labels (integers 0-6)
    
    Returns:
        float: MAP@3 score
    """
    # Ensure inputs are CuPy arrays
    pred_proba = cp.asarray(pred_proba)
    true_label = cp.asarray(true_label, dtype=cp.int32)
    
    # Validate input shapes
    n_samples, n_classes = pred_proba.shape
    assert n_classes == 7, "pred_proba must have 7 classes"
    assert true_label.shape == (n_samples,), "true_label must have shape (n_samples,)"
    
    # Get top-3 predicted class indices (descending order)
    top3_indices = cp.argsort(pred_proba, axis=1)[:, -3:][:, ::-1]  # Shape: (n_samples, 3)
    
    # Compute AP@3 for each sample
    def compute_apk(true, preds):
        for i, pred in enumerate(preds):
            if pred == true:
                return 1.0 / (i + 1)
        return 0.0
    
    # Vectorize AP@3 computation
    ap_scores = cp.zeros(n_samples, dtype=cp.float32)
    for i in range(n_samples):
        ap_scores[i] = compute_apk(true_label[i], top3_indices[i])
    
    # Compute mean AP@3
    map3 = cp.mean(ap_scores)
    
    return float(map3.get())  # Convert to Python float for compatibility


def get_top_three(y_prob):
    # Sort the probability scores in descending order and get the index positions
    # of the top three probabilities for each sample.
    sorted_prob_ids = np.argsort(-y_prob)
    top3_diss_ids = sorted_prob_ids[:,:3]  # Take only the first three indices
    
    # Save the original shape of the top3_diss_ids array to reshape it later.
    original_shape = top3_diss_ids.shape
    
    # Use inverse_transform to get the disease labels associated with the top three indices.
    top3_diss = ord_encoder.inverse_transform(top3_diss_ids.reshape(-1,1))
    
    # Reshape the top3_diss array to match its original shape.
    top3_diss = top3_diss.reshape(original_shape)
    
    # Return the top three disease labels for each sample.
    return top3_diss


def show_corr_heatmap(df, title):
    
    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    plt.figure(figsize = (15, 10))
    plt.title(title)
    sns.heatmap(corr, annot = False, linewidths=.5, fmt=".2f", square=True, cmap=cmap_2, mask=mask, yticklabels=True, xticklabels=True)
    plt.show()


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
subm = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
orig = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")
train.head()


from sklearn.preprocessing import OrdinalEncoder
TARGET =  'Fertilizer Name'
ord_encoder = OrdinalEncoder()

ord_encoder.fit(train[[TARGET]])

orig[TARGET + "_lbl"] = ord_encoder.transform(orig[[TARGET]])
train[TARGET + "_lbl"] = ord_encoder.transform(train[[TARGET]])


NUMS = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium',
       'Phosphorous']
for idx, c in enumerate(NUMS[:-1]):
    for idx2, c2 in enumerate(NUMS[idx + 1 : ]):
        train[c + "___" + c2] = (train[c].astype(str) + "_" + train[c2].astype(str)).astype('category')
        train[c + "_+_" + c2] = train[c] + train[c2]
        train[c + "_*_" + c2] = train[c] * train[c2]
        train[c + "_/_" + c2] = train[c] / (train[c2] + 1)

        orig[c + "___" + c2] = orig[c].astype(str) + "_" + orig[c2].astype(str)
        orig[c + "_+_" + c2] = orig[c] + orig[c2]
        orig[c + "_*_" + c2] = orig[c] * orig[c2]
        orig[c + "_/_" + c2] = orig[c] / (orig[c2] + 1)


        test[c + "___" + c2] = (test[c].astype(str) + "_" + test[c2].astype(str)).astype('category')
        test[c + "_+_" + c2] = test[c] + test[c2]
        test[c + "_*_" + c2] = test[c] * test[c2]
        test[c + "_/_" + c2] = test[c] / (test[c2] + 1)

        
orig.shape


import matplotlib.pyplot as plt
import seaborn as sns

NUMS2 = list(train.select_dtypes(include='number').columns)[1:] 
NUMS2.remove('Fertilizer Name_lbl')
show_corr_heatmap(train[NUMS2+ [TARGET + "_lbl"]], "Training corr")
show_corr_heatmap(test[NUMS2], "Test corr")
show_corr_heatmap(orig[NUMS2] , "Original corr")


CA = ['Soil Type', 'Crop Type']

for c in CA:
    train[c] = train[c].astype('category')
    test[c] = test[c].astype('category')



fig, ax = plt.subplots(2, 2, figsize=[20, 12])

for i, c in enumerate(CA):
    train[c].value_counts().plot(kind='pie', colors=PALETTE_7,autopct='%1.1f%%',
                                startangle=90, ax=ax[i, 0])
    
    train[c].value_counts().plot(kind='barh', ax=ax[i, 1], color=PALETTE_7)


import matplotlib.pyplot as plt

"""" 
Differences of Training set and Original set base on Temparature

"""
plt.figure(figsize=[20, 8])
for idx, c in enumerate(NUMS[1:]):
    plt.subplot(2, 3, idx + 1)
    diff = orig.groupby('Temparature')[c].agg('mean')
    diff2 = train.groupby('Temparature')[c].agg('mean')
    plt.plot(diff, diff2, 'o')
    plt.plot([min(diff), max(diff)], [min(diff2), max(diff2)])
    plt.ylabel(f"Train {c}")
    plt.xlabel(f"Orig {c}")



COLS = list(test.iloc[:, 1:].columns)


CLASSES = train[TARGET].value_counts().shape[0]

len(COLS)


CATS = list(train.select_dtypes(include='category').columns)


catboost_params  = {
    'task_type': 'GPU',              # Use GPU
    'devices': '0',                  # Specify GPU device (e.g., '0' for first GPU)
    'iterations': 1000,              # Number of boosting iterations
    'learning_rate': 0.1,            # Step size for gradient descent
    'depth': 6,                      # Tree depth (6-10 is typical for GPU)
    'l2_leaf_reg': 3,                # L2 regularization
    'loss_function': 'MultiClass',   # Softmax for multiclass
    'eval_metric': 'MultiClass',     # Logloss for evaluation (default)
    'early_stopping_rounds': 50,     # Stop if no improvement in 50 rounds
    'random_seed': 42,               # For reproducibility
    'verbose': 0 ,
    'cat_features' : CATS 
}

xgboost_params = {
    'objective': 'multi:softprob',    # Softmax for multiclass
    'num_class': CLASSES,                  # Number of classes
    'tree_method': 'gpu_hist',       # Use GPU histogram-based algorithm
    'gpu_id': 0,                     # Specify GPU device
    'max_depth': 6,                  # Tree depth
    'eta': 0.1,                      # Learning rate
    'subsample': 0.8,                # Data subsampling
    'colsample_bytree': 0.8,         # Feature subsampling
    'lambda': 1.0,                   # L2 regularization
    'alpha': 0.1,                    # L1 regularization
    'seed': 42 ,
    'enable_categorical' : True# For reproducibility
}

lgbm_params = {
    'objective': 'multiclass',        # Softmax for multiclass
    'num_class': 7,                  # Number of classes
    'device_type': 'cpu',            # Use GPU
    'gpu_platform_id': 0,            # Specify GPU platform (if multiple GPUs)
    'gpu_device_id': 0,              # Specify GPU device
    'boosting_type': 'gbdt',         # Gradient Boosting Decision Trees
    'num_iterations': 10,          # Number of boosting iterations
    'learning_rate': 0.1,            # Step size
    'max_depth': 6,                  # Limit tree depth
    'num_leaves': 31,                # Number of leaves (2^max_depth - 1)
    'lambda_l1': 0.1,                # L1 regularization
    'lambda_l2': 0.1,                # L2 regularization
    'feature_fraction': 0.8,         # Feature subsampling
    'bagging_fraction': 0.8,         # Data subsampling
    'bagging_freq': 5,               # Bagging frequency
    'early_stopping_rounds': 50,     # Stop if no improvement
    'seed': 42,                      # For reproducibility
    'verbosity': -1  # Print progress
}



from sklearn.model_selection import KFold , StratifiedKFold
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, log_loss
from lightgbm import LGBMClassifier

SPLITS = 8
SEED = 12
test_probas_cat  = np.zeros((test.shape[0], CLASSES))
test_probas_xgb = np.zeros((test.shape[0], CLASSES))
test_probas_lgb = np.zeros((test.shape[0], CLASSES))

oof_probas_cat  = np.zeros((train.shape[0], CLASSES))
oof_probas_xgb  = np.zeros((train.shape[0], CLASSES))
oof_probas_lgb  = np.zeros((train.shape[0], CLASSES))

oof_dic = {"XGB" : oof_probas_xgb, "CAT" : oof_probas_cat, "LGBM" : oof_probas_lgb}
preds_dic = {"XGB" : test_probas_xgb, "CAT" : test_probas_cat, "LGBM" : test_probas_lgb}

X, y = train[COLS], train[[TARGET + "_lbl"]]


folds = StratifiedKFold(n_splits=SPLITS, shuffle=True, random_state=SEED)

xgboost_model = XGBClassifier(**xgboost_params)
catboost_model = CatBoostClassifier(**catboost_params)
lgbm_model = LGBMClassifier(**lgbm_params)

models = [xgboost_model, catboost_model, lgbm_model]

for idx, (train_idx, val_idx) in enumerate(folds.split(X, y)):
    x_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    x_valid, y_valid = X.iloc[val_idx], y.iloc[val_idx]

    lgbm_model.fit(
                    x_train, y_train,
                    eval_set=[(x_valid, y_valid)],
                    categorical_feature=CATS
                )

    xgboost_model.fit(
                    x_train, y_train,
                    eval_set=[(x_valid, y_valid)],
                    early_stopping_rounds=50,
                    verbose=0
                      )
    
    catboost_model.fit(
                    x_train, y_train,
                    eval_set=(x_valid, y_valid),
                    early_stopping_rounds=50,  
                    )
    
    
    for m, n in zip(models, ['XGB', 'CAT', 'LGBM']):
        
        val_preds = m.predict(x_valid)
        val_proba = m.predict_proba(x_valid)
        
        
        acc = accuracy_score(y_valid.values, val_preds)
        map3 = map3_cupy(val_proba, y_valid.values.reshape(-1))
        logloss = log_loss(y_valid, val_proba)
        
        
        print(f"{cyn}Model {res}:{n}    {blue}Fold {idx + 1}:{res}    {yllw}MAP@3 score{res} = {map3:.6f}\t {cyn}Log loss scores{res} = {logloss:.6f}\t {blue}Accuracy scores{res} = {acc:.6f}")
        oof_dic[n][val_idx] = val_proba

        tst = m.predict_proba(test[COLS])
        preds_dic[n] += tst
    print(40 * "--")



oof_dic['Ensemble'] = oof_dic["XGB"] + oof_dic["CAT"] + oof_dic["LGBM"]
preds_dic['Ensemble'] = preds_dic["XGB"] + preds_dic["CAT"] + preds_dic["LGBM"]


y_sub = get_top_three(preds_dic['Ensemble'])
prognosis =  np.apply_along_axis(lambda x: np.array(' '.join(x), dtype="object"), 1, y_sub)
xpt = np.hstack((np.reshape(test['id'].values, (-1,1)), np.reshape(prognosis, (-1,1))))
submission_df = pd.DataFrame(xpt, columns=['id', TARGET])
submission_df.to_csv("submission.csv", index=False)
submission_df.head()

