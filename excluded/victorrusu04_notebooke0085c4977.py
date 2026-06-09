import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from scipy.stats import randint


train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")

# Drop rows without target
train = train.dropna(subset=["Price"])


X = train.drop(columns=["id", "Price"])
y = train["Price"]
X_test = test.drop(columns=["id"])


combined = pd.concat([X, X_test], axis=0)

# One-hot encode categoricals
combined = pd.get_dummies(combined, drop_first=True)

# Fill missing with 0
combined = combined.fillna(0)


scaler = StandardScaler()
combined_scaled = scaler.fit_transform(combined)


X = combined_scaled[:len(X)]
X_test = combined_scaled[len(X):]


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



param_dist = {
    'n_estimators': randint(100, 300),
    'max_depth': randint(5, 20),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['auto', 'sqrt', 'log2']
}

rfr = RandomForestRegressor(random_state=42, n_jobs=-1)
search = RandomizedSearchCV(rfr, param_distributions=param_dist,
                            n_iter=20, scoring='neg_root_mean_squared_error',
                            cv=3, verbose=1, random_state=42, n_jobs=-1)


search.fit(X_train, y_train)
best_model = search.best_estimator_


val_pred = best_model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"Validation RMSE (tuned RF): {rmse:.2f}")


test_pred = best_model.predict(X_test)


submission = sample_submission.copy()
submission["Price"] = test_pred
submission.to_csv("submission_rf_improved.csv", index=False)




