import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBRegressor
import lightgbm as lgb
# !pip install catboost
from catboost import CatBoostRegressor

pd.options.display.max_columns = None
pd.options.display.width = 180

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
sample = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")


train.head()


print(train.shape)


train.dtypes


train.describe()


num_cols = train.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols = train.select_dtypes(include=['object','bool', 'category']).columns.tolist()


for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(train[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.show()


unique_vals = pd.DataFrame({
    'Column': cat_cols,
    'Unique Values': [train[col].unique().tolist() for col in cat_cols]
})

unique_vals


target = 'accident_risk'
X = train.drop(columns=[target])
y = train[target]


corr = train.corr(numeric_only=True)[target].sort_values(ascending=False)
corr_matrix = train.corr(numeric_only=True)

plt.figure(figsize=(20, 15))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap — All Features (Including Target)')
plt.show()


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


if len(cat_cols) > 0:
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    oe.fit(X_train[cat_cols].astype(str).fillna("##nan##").values)

    def ord_transform(df):
        tmp = df.copy()
        tmp_cat = tmp[cat_cols].astype(str).fillna("##nan##").values
        tmp[cat_cols] = oe.transform(tmp_cat).astype(int)
        return tmp

    X_train_enc = ord_transform(X_train)
    X_test_enc  = ord_transform(X_test)
    X_pred_enc  = ord_transform(test)
else:
    X_train_enc = X_train.copy()
    X_test_enc  = X_test.copy()
    X_pred_enc  = test.copy()


if len(cat_cols) > 0:
    X_train_cb = X_train.copy()
    X_test_cb  = X_test.copy()
    X_pred_cb  = test.copy()
    for c in cat_cols:
        X_train_cb[c] = X_train_cb[c].astype('category')
        X_test_cb[c]  = X_test_cb[c].astype('category')
        X_pred_cb[c]  = X_pred_cb[c].astype('category')
    cat_feature_indices = [X_train_cb.columns.get_loc(c) for c in cat_cols]
else:
    X_train_cb = X_train.copy()
    X_test_cb  = X_test.copy()
    X_pred_cb  = test.copy()
    cat_feature_indices = []


rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
xgb = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1,
    verbosity=0
)
lgbm = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    n_jobs=-1
)
cat = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    random_seed=42,
    verbose=100
)


models = {
    "RandomForest": rf,
    "XGBoost": xgb,
    "LightGBM": lgbm,
    "CatBoost": cat
}


print("Models instantiated:", list(models.keys()))


def eval_regression(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return rmse, mae, r2

results = {}
for name, model in models.items():
    print(f"\n>>> Training {name} ...")
    if name == "CatBoost":
        model.fit(X_train_cb, y_train, cat_features=cat_feature_indices, use_best_model=False, verbose=False)
        preds = model.predict(X_test_cb)
    else:
        model.fit(X_train_enc, y_train)
        preds = model.predict(X_test_enc)
    rmse, mae, r2 = eval_regression(y_test, preds)
    results[name] = {"rmse": rmse, "mae": mae, "r2": r2}
    print(f"{name} -> RMSE: {rmse:.5f}, MAE: {mae:.5f}, R2: {r2:.5f}")


res_df = pd.DataFrame(results).T.sort_values('rmse')
print("Test Results (sorted by RMSE):")
display(res_df)


if len(cat_cols) > 0:
    oe_full = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    oe_full.fit(X[cat_cols].astype(str).fillna("##nan##").values)

    def ord_transform_full(df):
        tmp = df.copy()
        tmp_cat = tmp[cat_cols].astype(str).fillna("##nan##").values
        tmp[cat_cols] = oe_full.transform(tmp_cat).astype(int)
        return tmp

    X_full_enc = ord_transform_full(X)
    X_pred_enc_final = ord_transform_full(test)

    X_full_cb = X.copy()
    X_pred_cb_final = test.copy()
    for c in cat_cols:
        X_full_cb[c] = X_full_cb[c].astype('category')
        X_pred_cb_final[c] = X_pred_cb_final[c].astype('category')
    cat_feature_indices_full = [X_full_cb.columns.get_loc(c) for c in cat_cols]
else:
    X_full_enc = X.copy()
    X_pred_enc_final = test.copy()
    X_full_cb = X.copy()
    X_pred_cb_final = test.copy()
    cat_feature_indices_full = []

final_preds = pd.DataFrame(index=test.index)

# Train and predict each model
models['RandomForest'].fit(X_full_enc, y)
final_preds['RandomForest'] = models['RandomForest'].predict(X_pred_enc_final)

models['XGBoost'].fit(X_full_enc, y)
final_preds['XGBoost'] = models['XGBoost'].predict(X_pred_enc_final)

models['LightGBM'].fit(X_full_enc, y)
final_preds['LightGBM'] = models['LightGBM'].predict(X_pred_enc_final)

models['CatBoost'].fit(X_full_cb, y, cat_features=cat_feature_indices_full, verbose=0)
final_preds['CatBoost'] = models['CatBoost'].predict(X_pred_cb_final)

# Ensemble average
final_preds['Ensemble_mean'] = final_preds.mean(axis=1)

display(final_preds.head())


best_model = res_df.index[0]
print("Best model:", best_model)

model = models[best_model]
if hasattr(model, 'feature_importances_'):
    importances = pd.Series(model.feature_importances_, index=X_train_enc.columns).sort_values(ascending=False)[:20]
    plt.figure(figsize=(8,6))
    sns.barplot(x=importances.values, y=importances.index)
    plt.title(f"Top features — {best_model}")
    plt.show()
else:
    print("Selected model does not provide feature importances.")


num_cols = test.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols = test.select_dtypes(include=['object','bool']).columns.tolist()

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
encoder.fit(train[cat_cols])
test[cat_cols] = encoder.transform(test[cat_cols])


predictions = model.predict(test)

test['Prediction'] = predictions

print(test[['id', 'Prediction']].head())


df_to_save = test[['id', 'Prediction']]
df_to_save.to_csv('/kaggle/working/test_predictions.csv', index=False)

