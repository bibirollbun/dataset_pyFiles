import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Binomial
from statsmodels.stats.outliers_influence import variance_inflation_factor
from glob import glob
import lightgbm as lgb
from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings('ignore')

pd.set_option("display.max_colwidth", 1000)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def rmse_score(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test  = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sub   = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


train.shape


test.shape


info_train = pd.DataFrame({
    "DataType": train.dtypes,
    "MissingValues": train.isnull().sum(),
    "UniqueValues": train.nunique()
}).sort_values(by="MissingValues", ascending=False)

info_train['MissingValuesRatio'] = round(info_train['MissingValues'] / len(train),2)

print(info_train)


info_test = pd.DataFrame({
    "DataType": test.dtypes,
    "MissingValues": test.isnull().sum(),
    "UniqueValues": test.nunique()
}).sort_values(by="MissingValues", ascending=False)

info_test['MissingValuesRatio'] = round(info_test['MissingValues'] / len(test),2)

print(info_test)


def describe_columns(df):
    records = []
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.lower().str.strip()
            categories = list(df[c].unique())
            if set(categories) <= {"yes", "no"}:
                col_type = "binary"
            elif c == "education":
                col_type = "ordinal"
            else:
                col_type = "categorical"
            records.append({
                "Column": c,
                "Type": col_type,
                "Categories": categories
            })
        else: 
            min_val = df[c].min()
            max_val = df[c].max()
            if  set(df[c].dropna().unique()) <= {0, 1}:
                col_type = "binary"
                cats_str = "[0, 1]"
            else:
                col_type = "numeric"
                cats_str = f"min = {min_val}, max = {max_val}"
            records.append({
                "Column": c,
                "Type": col_type,
                "Categories": cats_str
            })
    return pd.DataFrame(records)


describe_columns(train)


describe_columns(test)


cat_cols = ['road_type', 'lighting', 'weather', 'time_of_day']
for c in cat_cols:
    if c in train.columns:
        train[c] = train[c].astype(str).str.lower().str.strip()
    if c in test.columns:
        test[c] = test[c].astype(str).str.lower().str.strip()

bin_cols = ['road_signs_present','public_road','holiday','school_season']
for c in bin_cols:
    if c in train.columns:
        train[c] = pd.to_numeric(train[c], errors='coerce').fillna(0).astype(float)
    if c in test.columns:
        test[c] = pd.to_numeric(test[c], errors='coerce').fillna(0).astype(float)

y = pd.to_numeric(train['accident_risk'], errors='coerce').astype(float)

print({c: train[c].unique()[:5] for c in cat_cols if c in train})


X_full = train.drop(columns=['id', 'accident_risk'], errors='ignore').copy()
X_full = pd.get_dummies(
    X_full,
    columns=['road_type', 'lighting', 'weather', 'time_of_day'],
    drop_first=True,
    dtype=np.float64
)

for b in ['road_signs_present','public_road','holiday','school_season']:
    if b in X_full.columns:
        X_full[b] = pd.to_numeric(X_full[b], errors='coerce').fillna(0.0).astype(float)

X_full = X_full.replace([np.inf, -np.inf], np.nan).fillna(0.0)
X_const = sm.add_constant(X_full, has_constant='add')

vif_data = pd.DataFrame({
    "feature": X_full.columns,
    "VIF": [variance_inflation_factor(X_full.values, i)
            for i in range(X_full.shape[1])]
}).sort_values('VIF', ascending=False)

display(vif_data.head(20))

high_vif = vif_data[vif_data['VIF'] > 10]
print(f"\nFeatures with VIF > 10:\n{high_vif}")



selected_features = [
    'curvature', 'speed_limit', 'num_lanes',
    'num_reported_accidents', 'road_type', 'time_of_day'
]

vif_base = train[selected_features].copy()

for c in ['road_type', 'time_of_day']:
    vif_base[c] = vif_base[c].astype(str).str.lower().str.strip()

X_vif = pd.get_dummies(
    vif_base,
    columns=['road_type', 'time_of_day'],
    drop_first=True,
    dtype=np.float64
)


vif_selected = pd.DataFrame({
    "feature": X_vif.columns,
    "VIF": [variance_inflation_factor(X_vif.values, i)
            for i in range(X_vif.shape[1])]
}).sort_values('VIF', ascending=False)

display(vif_selected)


formula_linear = """
accident_risk ~ curvature + speed_limit + num_reported_accidents + num_lanes
+ C(road_type) + C(time_of_day)
+ C(lighting) + C(weather)
"""

formula_poly = """
accident_risk ~ curvature + I(curvature**2)
+ speed_limit + I(speed_limit**2)
+ num_reported_accidents + num_lanes
+ C(road_type) + C(time_of_day)
+ C(lighting) + C(weather)
"""

tr, va = train_test_split(train, test_size=0.2, random_state=42)

m_lin = smf.ols(formula=formula_linear, data=tr).fit()
pred_lin = m_lin.predict(va).clip(0, 1)
rmse_lin = rmse_score(va['accident_risk'], pred_lin)

m_poly = smf.ols(formula=formula_poly, data=tr).fit()
pred_poly = m_poly.predict(va).clip(0, 1)
rmse_poly = rmse_score(va['accident_risk'], pred_poly)

family = sm.families.Binomial(link=sm.families.links.logit())
m_glm = smf.glm(formula=formula_poly, data=tr, family=family).fit()
pred_glm = m_glm.predict(va).clip(0, 1)
rmse_glm = rmse_score(va['accident_risk'], pred_glm)

results = pd.DataFrame({
    'Model': ['OLS linear (extended)', 'OLS poly (extended)', 'GLM poly (extended)'],
    'RMSE_holdout': [rmse_lin, rmse_poly, rmse_glm]
}).sort_values('RMSE_holdout')

display(results)


num_cols = ['curvature', 'speed_limit', 'num_lanes', 'num_reported_accidents']
cat_cols = ['road_type', 'time_of_day', 'lighting', 'weather']

def build_X(df):
    X = df[num_cols].copy()
    X['curvature_sq']   = X['curvature']**2
    X['speed_limit_sq'] = X['speed_limit']**2
    X_cat = pd.get_dummies(df[cat_cols], drop_first=True, dtype=np.float32)
    X = pd.concat([X, X_cat], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X.astype(np.float32)

X_tr = build_X(train)
X_te = build_X(test)
X_te = X_te.reindex(columns=X_tr.columns, fill_value=0.0).astype(np.float32)

y_tr = train['accident_risk'].astype(float)

print("Train/Test shapes:", X_tr.shape, X_te.shape)


ols_model = smf.ols(formula=formula_poly, data=train).fit()
pred_test_ols = np.clip(ols_model.predict(test), 0, 1).astype('float64')

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.03,
    'num_leaves': 64,
    'min_data_in_leaf': 50,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 1,
    'lambda_l2': 1.0,
    'verbose': -1,
    'seed': 42
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(X_tr), dtype=float)
pred_lgb = np.zeros(len(X_te), dtype=float)

for tr_idx, va_idx in kf.split(X_tr):
    X_trn, X_val = X_tr.iloc[tr_idx], X_tr.iloc[va_idx]
    y_trn, y_val = y_tr.iloc[tr_idx], y_tr.iloc[va_idx]

    dtrain = lgb.Dataset(X_trn, label=y_trn)
    dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=5000,
        valid_sets=[dtrain, dvalid],
        valid_names=['train','valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=200, verbose=False),
            lgb.log_evaluation(0)  # Ð±ÐµÐ· Ð»Ð¾Ð³Ð¾Ð²
        ]
    )

    oof[va_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    pred_lgb += model.predict(X_te, num_iteration=model.best_iteration) / kf.n_splits

rmse_lgb = rmse_score(y_tr, np.clip(oof, 0, 1))
print(f"LGB OOF RMSE: {rmse_lgb:.5f}")

w = 0.7
pred_blend = np.clip(w * pred_test_ols + (1 - w) * pred_lgb, 0, 1)

submission = pd.DataFrame({
    'id': pd.to_numeric(test['id'], errors='coerce').astype('Int64'),
    'accident_risk': pred_blend.round(3)
})
submission.to_csv('/kaggle/working/submission.csv', index=False, float_format="%.3f")
print("Saved: /kaggle/working/submission.csv")
print(submission.head())

