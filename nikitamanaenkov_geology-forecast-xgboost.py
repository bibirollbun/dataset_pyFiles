import pandas as pd
import numpy as np
import pywt
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import kstest
import warnings

warnings.filterwarnings("ignore")


test_df = pd.read_csv("/kaggle/input/geology-forecast-challenge-open/data/test.csv")
train_df = pd.read_csv("/kaggle/input/geology-forecast-challenge-open/data/train.csv")

input_cols = [str(i) for i in range(-299, 1)]
target_cols = [str(i) for i in range(1, 301)]

imputer = SimpleImputer(strategy="mean")
X_train_raw = imputer.fit_transform(train_df[input_cols])
X_test_raw = imputer.transform(test_df[input_cols])

# https://www.kaggle.com/code/nikitamanaenkov/geology-forecast-challenge-features 

def extract_features(X):
    df = pd.DataFrame(X)
    features = pd.DataFrame()
    X_filled = df.T.interpolate(limit_direction='both').T.fillna(0)
    X_np = X_filled.to_numpy()
    
    features["mean"] = df.mean(axis=1)
    features["std"] = df.std(axis=1)
    features["min"] = df.min(axis=1)
    features["max"] = df.max(axis=1)
    
    second_deriv = np.diff(X_np, n=2, axis=1)
    features["second_deriv_mean"] = second_deriv.mean(axis=1)
    features["second_deriv_std"] = second_deriv.std(axis=1)

    first_deriv = np.diff(X_np, axis=1)
    ratio = np.abs(first_deriv[:, 1:] / (first_deriv[:, :-1] + 1e-6))
    features['change_ratio'] = ratio.mean(axis=1)

    features['slope_mean'] = np.gradient(X_filled, axis=1).mean(axis=1)

    features['sign_changes'] = (np.diff(np.sign(X_filled.values), axis=1) != 0).sum(axis=1)
    
    features['zero_frac'] = (X_filled == 0).mean(axis=1)
    features['neg_frac'] = (X_filled < 0).mean(axis=1)
    
    weights = np.tile(np.arange(X_filled.shape[1]), (X_filled.shape[0], 1))
    features['center_of_mass'] = (X_filled.values * weights).sum(axis=1) / (X_filled.sum(axis=1) + 1e-6)
    
    ks_stats = []
    for row in X_np:
        m, s = row.mean(), row.std()
        stat = kstest((row - m) / (s + 1e-6), 'norm').statistic
        ks_stats.append(stat)
    features['ks_gauss'] = ks_stats
    
    features['auc'] = np.trapz(X_np, axis=1)
    
    return features


def wavelet_features(X):
    features = []
    for row in X:
        coeffs = pywt.wavedec(row, 'db1', level=3)
        flat = np.hstack(coeffs)
        features.append([
            np.mean(flat),
            np.std(flat),
            np.max(flat),
            np.min(flat)
        ])
    return np.array(features)


X_train_features = extract_features(X_train_raw)
X_test_features = extract_features(X_test_raw)

wavelet_feats = wavelet_features(X_train_raw)
X_train = np.hstack([X_train_raw, X_train_features, wavelet_feats])

wavelet_feats_test = wavelet_features(X_test_raw)
X_test = np.hstack([X_test_raw, X_test_features, wavelet_feats_test])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train = train_df[target_cols].values


best_model = XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.6,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

model = MultiOutputRegressor(best_model)

cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-validation MSE scores: {cv_scores}")
print(f"Average cross-validation MSE: {cv_scores.mean()}")

model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)

all_realizations = [predictions]
for i in range(9):
    noise = np.random.normal(loc=0, scale=0.5, size=predictions.shape)
    all_realizations.append(predictions + noise)



output_df = pd.DataFrame()
output_df["geology_id"] = test_df["geology_id"]

for i in range(1, 301):
    output_df[str(i)] = all_realizations[0][:, i - 1]

for r in range(1, 10):
    for i in range(1, 301):
        col_name = f"r_{r}_pos_{i}"
        output_df[col_name] = all_realizations[r][:, i - 1]

output_df.to_csv("submission.csv", index=False)
print("Submission saved as submission.csv")

output_df.head()

