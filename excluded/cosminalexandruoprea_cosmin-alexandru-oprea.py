import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error

# 1. Încarcă datele
train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")

submission = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")

# 2. Feature Engineering
for df in [train, test]:
    df['Weight_per_Compartment'] = df['Weight Capacity (kg)'] / df['Compartments']
brand_median = train.groupby('Brand')['Price'].median()
train['Brand_mean_price'] = train['Brand'].map(brand_median)
test['Brand_mean_price'] = test['Brand'].map(brand_median)

# 3. Pregătește X, y, X_test
X = train.drop(columns=["id", "Price"])
y = train["Price"]
X_test = test.drop(columns=["id"])

# 4. Identifică coloane
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# 5. Preprocessor
num_imp = SimpleImputer(strategy='mean')
cat_imp = SimpleImputer(strategy='most_frequent')
one_hot = OneHotEncoder(handle_unknown='ignore', sparse=False)
preprocessor = ColumnTransformer([
    ('num', num_imp, num_cols),
    ('cat', make_pipeline(cat_imp, one_hot), cat_cols),
])

# 6. Split train/val
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Define și tunează RF
rf_pipe = make_pipeline(preprocessor, RandomForestRegressor(random_state=42))
param_dist = {
    'randomforestregressor__n_estimators': [100, 200],
    'randomforestregressor__max_depth': [None, 10],
    'randomforestregressor__min_samples_split': [2, 5]
}
rs = RandomizedSearchCV(rf_pipe, param_dist, n_iter=4, cv=3,
                        scoring='neg_mean_squared_error',
                        random_state=42, n_jobs=1)
rs.fit(X_train, y_train)
best_rf = rs.best_estimator_

# 8. Modele alternative
models = {
    'RF_tuned': best_rf,
    'HistGB': make_pipeline(preprocessor, HistGradientBoostingRegressor(random_state=42)),
    'XGBoost': make_pipeline(preprocessor, XGBRegressor(objective='reg:squarederror', random_state=42))
}

# 9. Compară RMSE
results = []
for name, mdl in models.items():
    start = time.time()
    mdl.fit(X_train, y_train)
    duration = round(time.time() - start, 2)
    preds = mdl.predict(X_val)
    rmse = mean_squared_error(y_val, preds, squared=False)
    results.append({'Model': name, 'RMSE': round(rmse, 4), 'Fit_s': duration})

# 10. Cross‑validation pentru modelul cel mai bun (RF_tuned)
best_model = best_rf
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = -cross_val_score(best_model, X, y, scoring='neg_mean_squared_error', cv=kf, n_jobs=1)
cv_rmse = round(np.sqrt(cv_scores).mean(), 4)
results.append({'Model': 'RF_tuned_CV5', 'RMSE': cv_rmse, 'Fit_s': None})

# 11. Afișează tabelul cu rezultate
df_results = pd.DataFrame(results).sort_values('RMSE').reset_index(drop=True)
print(df_results)

# 12. Retrain pe toate datele și generare submisie finală
best_model.fit(X, y)
submission["Price"] = best_model.predict(X_test)
submission.to_csv("submission.csv", index=False)
print("submission.csv created!")


