import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from colorama import Fore, Style, init
from scipy.optimize import minimize
import warnings

init(autoreset=True)
warnings.filterwarnings('ignore')


# GPU setup
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Global var

BASE_PATH = "/kaggle/input/prediction-interval-competition-ii-house-price" 

SEED = 42
FOLDS = 5
ALPHA = 0.1



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


def retrieve_neighbours(model, X, y, k=5, exclude_0=False):
   X, y = np.array(X), np.array(y)
   distances, indices = model.kneighbors(X, n_neighbors=k + 1 if exclude_0 else k)
   preds, dists = [], []
   for d, idxs in zip(distances, indices):
       if exclude_0:
           d, idxs = d[1:], idxs[1:]
       preds.append(np.mean(y[idxs]))
       dists.append(np.mean(d))
   return np.array(preds), np.array(dists)


def preprocess_knn_features(X_tr, X_va, y_tr, knn_features, knn_params):
   scaler = StandardScaler()
   X_tr_knn, X_va_knn = scaler.fit_transform(X_tr[knn_features]), scaler.transform(X_va[knn_features])
   knn = KNeighborsRegressor(**knn_params).fit(X_tr_knn, y_tr)
   k = knn_params["n_neighbors"]
   price_tr, d_tr = retrieve_neighbours(knn, X_tr_knn, y_tr, k=k, exclude_0=True)
   price_va, d_va = retrieve_neighbours(knn, X_va_knn, y_tr, k=k, exclude_0=False)
   X_tr, X_va = X_tr.copy(), X_va.copy()
   X_tr["k_dist"], X_va["k_dist"] = d_tr, d_va
   X_tr["price_knn"], X_va["price_knn"] = price_tr, price_va
   return X_tr, X_va


# TWO-STAGE UNCERTAINTY MODEL

class TwoStageUncertaintyModel:
   def __init__(self, model0, model1, n_splits=5, method="squared_error", seed=None, lower_bound=1000, alpha=0.1, gamma0=1.65, gamma1=1.75, features1=None):
       self.model0, self.model1 = model0, model1
       self.n_splits, self.method, self.seed = n_splits, method, seed
       self.gamma0, self.gamma1 = gamma0, gamma1
       self.lower_bound, self.alpha, self.features1 = lower_bound, alpha, features1
       self.fitted_ = False

   def _prepare_features_for_model1(self, X, y_pred):
       X_tmp = X[self.features1].copy() if self.features1 != "same" else X.copy()
       X_tmp["y_pred"] = y_pred
       return X_tmp

   def _get_target(self, y, oof_preds):
       return (y - oof_preds) ** 2 + 1e-6 if self.method == "squared_error" else np.abs(y - oof_preds)

   def fit(self, X, y):
       y = np.asarray(y)
       oof_preds = np.zeros_like(y, dtype=float)
       kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
       for train_idx, val_idx in kf.split(X):
           X_tr, X_val, y_tr = X.iloc[train_idx], X.iloc[val_idx], y[train_idx]
           self.model0.fit(X_tr, y_tr)
           oof_preds[val_idx] = self.model0.predict(X_val)
       target = self._get_target(y, oof_preds)
       X_resid_feat = self._prepare_features_for_model1(X, oof_preds) if self.features1 else oof_preds.reshape(-1, 1)
       self.model1.fit(X_resid_feat, target)
       self.model0.fit(X, y)
       self.fitted_ = True
       return self

   def predict_components(self, X):
       if not self.fitted_: raise ValueError("Call fit() before predict()")
       y_hat = self.model0.predict(X)
       X_resid_feat = self._prepare_features_for_model1(X, y_hat) if self.features1 else y_hat.reshape(-1, 1)
       err_hat = self.model1.predict(X_resid_feat)
       err_hat = np.maximum(err_hat, self.lower_bound)
       return y_hat, err_hat

   def build_interval(self, y_hat, err_hat):
       err_hat_sqrt = np.sqrt(err_hat) if self.method == "squared_error" else err_hat
       lower = y_hat - self.gamma0 * err_hat_sqrt
       upper = y_hat + self.gamma1 * err_hat_sqrt
       return lower, upper

   def predict(self, X):
       y_hat, err_hat = self.predict_components(X)
       lower, upper = self.build_interval(y_hat, err_hat)
       return y_hat, lower, upper


train = pd.read_csv(BASE_PATH + '/dataset.csv').set_index("id")
test = pd.read_csv(BASE_PATH + '/test.csv').set_index("id")


train = preprocess_date(train)
test = preprocess_date(test)
cat_cols = [c for c in train.columns if train[c].dtype == 'object' and c != 'sale_price']
num_cols = list(set(test.columns) - set(cat_cols))
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
train[cat_cols] = encoder.fit_transform(train[cat_cols]).astype(int)
test[cat_cols] = encoder.transform(test[cat_cols]).astype(int)
print(f"Data ready: Train {train.shape}, Test {test.shape}")


# MODEL CONFIGURATION (TỐI ƯU HÓA)
knn_params = {'n_neighbors': 10}
xgb_params = {
    'n_estimators': 1000,
    'max_depth': 5,
    'learning_rate': 0.03,
    'subsample': 0.9,
    'colsample_bytree': 0.8,
    'reg_alpha': 1.0,
    'reg_lambda': 2.0,
    'random_state': SEED,
    'tree_method': 'hist',
    'device': device
}

xgb_params1 = {
    'objective': 'reg:gamma',
    'n_estimators': 500,
    'max_depth': 3,
    'learning_rate': 0.05,
    'subsample': 0.9,
    'colsample_bytree': 0.7,
    'reg_alpha': 2.0,
    'reg_lambda': 5.0,
    'random_state': SEED,
    'tree_method': 'hist',
    'device': device
}

model0 = XGBRegressor(**xgb_params)
model1 = XGBRegressor(**xgb_params1)

uncert_model = TwoStageUncertaintyModel(
    model0=model0,
    model1=model1,
    seed=SEED,
    method="squared_error",
    n_splits=10,
    features1="same",
    gamma0=1.65,
    gamma1=1.75
)

features = cat_cols + num_cols + ["price_knn", "k_dist"]
knn_features = ["latitude", "longitude", "year"]
y = train["sale_price"]



# CROSS-VALIDATION & OOF PREDICTION COLLECTION

print(f"\n Running Cross-Validation and Collecting OOF Predictions...")
oof_y_true, oof_y_hat, oof_err_hat, oof_indices = [], [], [], []
scores, coverages = [], []
cv = KFold(shuffle=True, random_state=SEED, n_splits=FOLDS)
for i, (train_idx, val_idx) in enumerate(cv.split(train, y), 1):
    print(f"\n=== Fold: {i} ===")
    X_tr, X_vl, y_tr, y_vl = train.iloc[train_idx], train.iloc[val_idx], y.iloc[train_idx], y.iloc[val_idx]
    X_tr, X_vl = preprocess_knn_features(X_tr, X_vl, y_tr, knn_features, knn_params)
    model = uncert_model.fit(X_tr[features], y_tr)
    y_hat_vl, err_hat_vl = model.predict_components(X_vl[features])
    pi_lower, pi_upper = model.build_interval(y_hat_vl, err_hat_vl)
    y_min, y_max = y_tr.min(), y_tr.max()
    pi_lower, pi_upper = np.clip(pi_lower, y_min, y_max), np.clip(pi_upper, y_min, y_max)
    score, coverage = winkler_score(y_vl.values, pi_lower, pi_upper, alpha=ALPHA, return_coverage=True)
    scores.append(score)
    coverages.append(coverage)
    oof_y_true.extend(y_vl.values), oof_y_hat.extend(y_hat_vl), oof_err_hat.extend(err_hat_vl), oof_indices.extend(val_idx)
    print(f"Winkler (fixed gamma): {score:,.0f} | Coverage: {coverage:.4f}")

oof_df = pd.DataFrame({'y_true': oof_y_true, 'y_hat': oof_y_hat, 'err_hat': oof_err_hat}, index=oof_indices).sort_index()
print(f"\n Initial CV Winkler (fixed gamma): {np.mean(scores):,.0f} ± {np.std(scores):,.0f}")


print("\n Optimizing Gamma Scaling Factors...")
def winkler_objective(gammas, y_true, y_hat, err_hat):
    gamma0, gamma1 = gammas
    err_hat_sqrt = np.sqrt(err_hat)
    lower = y_hat - gamma0 * err_hat_sqrt
    upper = y_hat + gamma1 * err_hat_sqrt
    return winkler_score(y_true, lower, upper)

initial_gammas = [uncert_model.gamma0, uncert_model.gamma1]
bounds = [(0.5, 4.0), (0.5, 4.0)]
opt_result = minimize(
    winkler_objective,
    initial_gammas,
    args=(oof_df['y_true'], oof_df['y_hat'], oof_df['err_hat']),
    method='Nelder-Mead',
    bounds=bounds
)

optimized_gamma0, optimized_gamma1 = opt_result.x
uncert_model.gamma0, uncert_model.gamma1 = optimized_gamma0, optimized_gamma1
optimized_score = opt_result.fun

print(f" Gamma optimization complete.")
print(f"   Initial Gammas: {initial_gammas[0]:.3f}, {initial_gammas[1]:.3f} -> Score: {np.mean(scores):,.0f}")
print(f"   {Fore.GREEN}Optimal Gammas: {optimized_gamma0:.3f}, {optimized_gamma1:.3f} -> Score: {optimized_score:,.0f}{Style.RESET_ALL}")
print(f"   Improvement: {np.mean(scores) - optimized_score:+,.0f} points")


# FINAL SUBMISSION

print(f"\n{Fore.GREEN} PROCEEDING WITH OPTIMIZED SUBMISSION{Style.RESET_ALL}")
print(" Training final model on full dataset with optimized gammas...")
X_train, X_test = preprocess_knn_features(train, test, y, knn_features, knn_params)
final_model = uncert_model.fit(X_train[features], y)

print(" Generating victory predictions...")
_, pi_lower, pi_upper = final_model.predict(X_test[features])
y_min, y_max = y.min(), y.max()
pi_lower, pi_upper = np.clip(pi_lower, y_min, y_max), np.clip(pi_upper, y_min, y_max)
submission_df = pd.DataFrame({'id': test.index, 'pi_lower': pi_lower, 'pi_upper': pi_upper})
submission_df.to_csv("submission_optimized_gamma.csv", index=False)

print(f"\n OPTIMIZED SUBMISSION CREATED!")
estimated_gap = 7304 # Using observed gap
predicted_lb = optimized_score + estimated_gap
print(f" New OOF CV Score: {optimized_score:,.0f}")
print(f" Predicted LB: {predicted_lb:,.0f} (using {estimated_gap:,.0f} gap)")

print(submission_df.head())
print(f"\n File saved: submission_optimized_gamma.csv")

