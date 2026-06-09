import pandas as pd
import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings('ignore')


# Competition metric: driectly copied from demo-notebook: https://www.kaggle.com/code/michaelsemenoff/demo-extended
def winkler_score(y_true, lower, upper, alpha=0.1, return_coverage=False):
    """Compute the Winkler Interval Score for prediction intervals.

    Args:
        y_true (array-like): True observed values.
        lower (array-like): Lower bounds of prediction intervals.
        upper (array-like): Upper bounds of prediction intervals.
        alpha (float): Significance level (e.g., 0.1 for 90% intervals).
        return_coverage (bool): If True, also return empirical coverage.

    Returns:
        score (float): Mean Winkler Score.
        coverage (float, optional): Proportion of true values within intervals.
    """
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    width = upper - lower
    penalty_lower = 2 / alpha * (lower - y_true)
    penalty_upper = 2 / alpha * (y_true - upper)

    score = width.copy()
    score += np.where(y_true < lower, penalty_lower, 0)
    score += np.where(y_true > upper, penalty_upper, 0)

    if return_coverage:
        inside = (y_true >= lower) & (y_true <= upper)
        coverage = np.mean(inside)
        return np.mean(score), coverage

    return np.mean(score)


def preprocess_date(data):

    df = data.copy()

    df["sale_date"] = pd.to_datetime(df.sale_date)
    df["year"] = df["sale_date"].dt.year
    df["month"] = df["sale_date"].dt.month

    df.drop(["sale_date"], axis=1, inplace=True)
        
    return df


SEED = 69
FOLDS = 5
ALPHA = 0.1

train = pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv").set_index("id")
test = pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/test.csv").set_index("id")

train = preprocess_date(train)
test = preprocess_date(test)

cat_cols = ['sale_warning', 'join_status', 'city', 'zoning', 'subdivision', 'submarket']
num_cols = list(set(test.columns) ^ set(cat_cols)) 

encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
train[cat_cols] = encoder.fit_transform(train[cat_cols]).astype(int)
test[cat_cols] = encoder.transform(test[cat_cols]).astype(int)


# To get square foot price of neighbourhood without leaking in train
def retrieve_neighbours(model, X, y, k=5, exclude_0=False):
    # For leak-free retrival of distances and prices
    # exclude_0 = True excludes the closest neighbour (typically self when train)
    X = np.array(X)
    y = np.array(y)

    if exclude_0:
        distances, indices = model.kneighbors(X, n_neighbors=k+1)
    else:
        distances, indices = model.kneighbors(X, n_neighbors=k)

    preds = []
    dists = []
    
    for d, idxs in tqdm(zip(distances, indices), total=len(indices)):

        if exclude_0:
            d = d[1:]
            idxs = idxs[1:]
        pred = np.mean(y[idxs])
        dist = np.mean(d)
    
        preds.append(pred)
        dists.append(dist)
    
    return np.array(preds), np.array(dists)

def preprocess_knn_features(X_tr, X_va, y_tr, knn_features, knn_params):
    # Features based on direct neighbourhood
    scaler = StandardScaler()
    X_tr_knn = scaler.fit_transform(X_tr[knn_features])
    X_va_knn = scaler.transform(X_va[knn_features])
    knn = KNeighborsRegressor(**knn_params).fit(X_tr_knn, y_tr)

    k = knn_params["n_neighbors"]
    
    price_tr, d_tr = retrieve_neighbours(knn, X_tr_knn, y_tr, k=k, exclude_0=True)
    price_va, d_va = retrieve_neighbours(knn, X_va_knn, y_tr, k=k, exclude_0=False)

    X_tr = X_tr.copy()
    X_va = X_va.copy()
    X_tr["k_dist"], X_va["k_dist"] = d_tr, d_va
    X_tr["price_knn"], X_va["price_knn"] = price_tr, price_va

    return X_tr, X_va


class TwoStageUncertaintyModel:
    """
    A two-stage model for point predictions with uncertainty intervals.
    
    Stage 1 fits the main target.  
    Stage 2 models the absolute residuals using out-of-fold predictions.  
    Intervals are built around the point prediction, scaled by gamma0 and gamma1.
    """

    def __init__(
        self, 
        model0, 
        model1, 
        n_splits=5, 
        method="abs_error", 
        seed=None, 
        lower_bound=0, 
        alpha=0.1, 
        gamma0=1.0, 
        gamma1=1.0, 
        features1=None
    ):
        self.model0 = model0
        self.model1 = model1
        self.n_splits = n_splits
        self.method = method
        self.seed = seed
        self.gamma0 = gamma0
        self.gamma1 = gamma1
        self.lower_bound = lower_bound
        self.alpha = alpha 
        self.features1 = features1
        self.feature_importances0 = None
        self.feature_importances1 = None
        self.fitted_ = False

    def _prepare_features_for_model1(self, X, y_pred):
        if self.features1 == "same":
            if isinstance(X, pd.DataFrame):
                X_tmp = X.copy()
                X_tmp["y_pred"] = y_pred
            elif isinstance(X, np.ndarray):
                y_pred = y_pred.reshape(-1, 1)
                X_tmp = np.hstack([X, y_pred])
            else:
                raise ValueError("Unsupported data type for X with features1='same'")
        
        else:
            if not isinstance(X, pd.DataFrame):
                raise ValueError(f"features1 is specified as {self.features1}, which requires X to be a pandas DataFrame.")
            missing = set(self.features1) - set(X.columns)
            if missing:
                raise ValueError(f"Missing columns in X for features1: {missing}")
            
            X_tmp = X[self.features1].copy()
            X_tmp["y_pred"] = y_pred
        
        return X_tmp

    def _get_target(self, y, oof_preds):
        if self.method == "abs_error":
            return np.abs(y - oof_preds)
        elif self.method == "squared_error":
            return (y - oof_preds) ** 2 + 1e-6
        raise ValueError("method should be `abs_error`, `squared_error`")

    def _get_feature_importances(self, model):
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif hasattr(model, 'coef_'):
            return model.coef_
        return None

    def fit(self, X, y):
        y = np.asarray(y)
        oof_preds = np.zeros_like(y, dtype=float)
        
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        
        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr = y[train_idx]
            
            self.model0.fit(X_tr, y_tr)
            oof_preds[val_idx] = self.model0.predict(X_val)
    
        # Get target for Stage 2
        target = self._get_target(y, oof_preds)
    
        if self.lower_bound == "min":
            self.lower_bound = np.min(target)

        # Construct features for stage two
        if self.features1:
            X_resid_feat = self._prepare_features_for_model1(X, oof_preds)
        else:
            X_resid_feat = oof_preds.reshape(-1, 1)

        # Fit model1 (Stage2) 
        self.model1.fit(X_resid_feat, target)
        self.feature_importances1 = self._get_feature_importances(self.model1)
    
        # Refit on full dataset
        self.model0.fit(X, y)
        self.feature_importances0 = self._get_feature_importances(self.model0)
        self.fitted_ = True
        return self

    def build_interval(self, y_hat, err_hat):
        if self.method in {"squared_error"}:
            err_hat = np.sqrt(err_hat)
    
        if self.method in {"abs_error", "squared_error"}:
            lower = y_hat - self.gamma0 * err_hat
            upper = y_hat + self.gamma1 * err_hat
        else:
            raise ValueError(f"Unknown method: {self.method}")
            
        return lower, upper

    def predict(self, X, return_errors=False):
        if not self.fitted_:
            raise ValueError("Call fit() before predict()")
        # point prediction
        y_hat = self.model0.predict(X)

        # predict absolute error
        if self.features1:
            X_resid_feat = self._prepare_features_for_model1(X, y_hat)
        else:
            X_resid_feat = y_hat.reshape(-1, 1)

        err_hat = self.model1.predict(X_resid_feat)

        if self.lower_bound is not None:
            err_hat = np.maximum(err_hat, self.lower_bound)

        # Build interval based on method
        lower, upper = self.build_interval(y_hat, err_hat)

        if return_errors:
            return y_hat, lower, upper, err_hat
        return y_hat, lower, upper


# Number of neighbours for price/other features
knn_params = {
    'n_neighbors': 10
} 

# XGB params
xgb_params = {
    'n_estimators': 1500,
    'max_depth': 6,
    'learning_rate': 0.05,
    'random_state': SEED
}

# XGB params 1
xgb_params1 = {
    'objective': "reg:gamma", # <- important for squared error target
    'n_estimators': 1000,
    'max_depth': 4,
    'learning_rate': 0.1,
    'random_state': SEED
}

model0 = XGBRegressor(**xgb_params)
model1 = XGBRegressor(**xgb_params1) 

uncert_model = TwoStageUncertaintyModel(
    model0=model0,
    model1=model1,
    seed=SEED,
    method="squared_error",
    lower_bound=1000,
    n_splits=10,
    features1="same",
    gamma0=1.65,
    gamma1=1.75
)

features = cat_cols + num_cols + ["price_knn", "k_dist"]
knn_features = ["latitude", "longitude", "year"]
y = train["sale_price"]

scores = []
coverages = []
cv = KFold(shuffle=True, random_state=SEED, n_splits=FOLDS)

for i, (train_idx, val_idx) in enumerate(cv.split(train, y), 1):
    print(f"\n=== Fold: {i} ===")

    # split
    X_tr, X_vl = train.iloc[train_idx], train.iloc[val_idx]
    y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]

    # Add KNN features
    X_tr, X_vl = preprocess_knn_features(X_tr, X_vl, y_tr, knn_features, knn_params)

    # Fit model
    model = uncert_model.fit(X_tr[features], y_tr)

    # Predict
    _, pi_lower, pi_upper = model.predict(X_vl[features])
    # Clip Min&Max
    y_min, y_max = y_tr.min(), y_tr.max()
    pi_lower = np.clip(pi_lower, y_min, y_max)
    pi_upper = np.clip(pi_upper, y_min, y_max)

    # Calculate Winkler Score
    score, coverage = winkler_score(y_vl.values, pi_lower, pi_upper, alpha=ALPHA, return_coverage=True)
    print(f"Winkler score: {score:.4f} Coverage: {coverage:.4f}")
    scores.append(score)
    coverages.append(coverage)

print("\nAverage Winkler:", np.mean(scores))
print("Average Coverage:", np.mean(coverages))


# Add KNN features to train and test
X_train, X_test = preprocess_knn_features(train, test, y, knn_features, knn_params)

final_model = uncert_model.fit(X_train[features], y)
# Predict
_, pi_lower, pi_upper = final_model.predict(X_test[features])
# Clip Min&Max
y_min, y_max = y.min(), y.max()
pi_lower = np.clip(pi_lower, y_min, y_max)
pi_upper = np.clip(pi_upper, y_min, y_max)

# Prepare submission
submission = pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/sample_submission.csv")
submission["pi_lower"] = pi_lower
submission["pi_upper"] = pi_upper
intervals = submission["pi_upper"] - submission["pi_lower"]
display(pd.Series(intervals).describe())
display(submission)

submission.to_csv("submission.csv", index=False)




