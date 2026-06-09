import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv("/kaggle/input/memcoin-graduation-basic-feature-extraction/train.csv")
train.head()


test = pd.read_csv("/kaggle/input/memcoin-graduation-basic-feature-extraction/test.csv")
test.head()


from sklearn.model_selection import StratifiedKFold, cross_val_score

def eval_model(model, X, y, cv=3, n_jobs=-1,print_results = True):
    stratified_kfold = StratifiedKFold(n_splits=cv)
    scores = cross_val_score(model, X, y, scoring='neg_log_loss',cv=stratified_kfold, n_jobs=n_jobs)
    if print_results:
        print("Mean Score:", np.mean(scores))
        print("Median Score:", np.median(scores))
        print("Std Score:", np.std(scores))
        print("Cross-validation Scores:", list(map(float, scores)))
    
    return scores



from sklearn.base import clone
from tqdm.auto import tqdm
import numpy as np

def lofo_drop_features(model, X, y, baseline_score=None, verbose=True, skip_cols=[]):
    """
    Run Leave-One-Feature-Out to identify non-contributing features.

    Parameters:
    - model: the sklearn pipeline/model to evaluate
    - X: input features as a pandas DataFrame
    - y: target array or series
    - eval_func: a callable like `eval_model` that returns a score (higher is better)
    - baseline_score: optional, if provided, used as the base score; otherwise it is computed
    - verbose: if True, prints details

    Returns:
    - List of feature names that can be dropped
    """
    if baseline_score is None:
        baseline_score = np.mean(eval_model(model, X, y, print_results=False))
        if verbose:
            print(f"Baseline score: {baseline_score:.5f}")

    droppable = []

    for col in tqdm(X.columns, desc="LOFO Evaluation"):
        if col in skip_cols:
            if verbose:
                print(f"⏩ Skipping {col} (used in transform)")
            continue
        X_lofo = X.drop(columns=[col])
        model_clone = clone(model)
        score = np.mean(eval_model(model_clone, X_lofo, y, print_results=False))
        delta = baseline_score - score 
        if score <= baseline_score:
            droppable.append((col, delta))
            if verbose:
                print(f"  ➤ {col} can be dropped (score: {score:.5f})")
        elif verbose:
            print(f"  ✖ {col} is important (score: {score:.5f})")
    droppable.sort(key=lambda x: -x[1])
    return droppable



from sklearn.base import clone
from tqdm.auto import tqdm
import numpy as np
import random

def greedy_feature_drop(model, X, y, random_state=42, verbose=True, skip_features=[]):
    """
    Random-order greedy feature elimination.

    Parameters:
    - model: sklearn pipeline or estimator
    - X: pandas DataFrame of input features
    - y: target
    - random_state: int seed for reproducibility
    - verbose: print progress
    - skip_features: list of feature names to exclude from testing

    Returns:
    - remaining_features: list of features that are kept
    - dropped_features: list of features that were dropped
    - final_score: the final score after dropping features
    """
    rng = random.Random(random_state)
    all_features = [col for col in X.columns if col not in skip_features]
    rng.shuffle(all_features)

    X_current = X.copy()
    model_clone = clone(model)
    baseline_score = np.mean(eval_model(model_clone, X_current, y, print_results=False))
    if verbose:
        print(f"Initial baseline score: {baseline_score:.5f}")

    remaining_features = list(X.columns)
    dropped_features = []

    for feature in tqdm(all_features, desc="Feature Selection"):
        candidate_features = [f for f in remaining_features if f != feature]
        X_candidate = X[candidate_features]

        model_clone = clone(model)
        score = np.mean(eval_model(model_clone, X_candidate, y, print_results=False))

        if score >= baseline_score:
            if verbose:
                print(f"✅ Dropped: {feature} (score: {score:.5f} ≥ {baseline_score:.5f})")
            remaining_features.remove(feature)
            dropped_features.append(feature)
            baseline_score = score
        elif verbose:
            print(f"❌ Kept:    {feature} (score: {score:.5f} < {baseline_score:.5f})")

    return remaining_features, dropped_features, baseline_score



columns_to_drop = ['mint','slot_min','first_15_sec_base_coin_amount_sum', 'first_15_sec_buy_percentage',
                   'token_sol_after_balance_ratio_mean', 'is_valid', 'creation_time', 
                   'provided_gas_limit_sum']
def transform(X:pd.DataFrame):
    X = X.copy()
    X.drop(columns=columns_to_drop,inplace=True)
    X = X.select_dtypes(['number'])
    X['sell_count'] = X['transaction_count'] - X['buy_count']
    X['sell_to_buy'] = X['sell_count'] / X['buy_count']
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = X[numeric_cols].replace([np.inf, -np.inf], 0)

    return X


X = train.drop(columns = ['has_graduated','slots_to_graduation'])
y = train['has_graduated']


from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
try:
    xgb_last
except NameError:
    xgb_last = float("-inf")
xgb = Pipeline([('transform',FunctionTransformer(transform)),
                ("xgb",XGBClassifier(n_jobs=-1,random_state = 0, missing=np.inf))])
print("Last: ", xgb_last)
mean = np.mean(eval_model(xgb,X,y,n_jobs=1))
print("improved" if mean > xgb_last  else "same" if mean == xgb_last else "worse")
xgb_last = mean


xgb.fit(X,y)


predictions = xgb.predict_proba(test)
probabilities = predictions[:, 1]
submission = pd.DataFrame({"mint" : test['mint'],'has_graduated':probabilities })
print(submission.head(10))
submission.to_csv("submission.csv",index = False)
print("Submission Saved")


pd.read_csv("/kaggle/input/solana-skill-sprint-memcoin-graduation/sample_submission.csv")

