# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


pd.set_option('display.max_columns', None)

import warnings
warnings.filterwarnings("ignore")


# Import data
app = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')
print('Training data shape: ', app.shape)

app_test = pd.read_csv('/kaggle/input/home-credit-default-risk/application_test.csv')
print('Test data shape: ', app_test.shape)

prev = pd.read_csv('/kaggle/input/home-credit-default-risk/previous_application.csv')
print('Prev application data shape: ', prev.shape)

bureau_bal = pd.read_csv('/kaggle/input/home-credit-default-risk/bureau_balance.csv')
print('Bureau balance data shape: ', bureau_bal.shape)

bureau = pd.read_csv('/kaggle/input/home-credit-default-risk/bureau.csv')
print('Bureau data shape: ', bureau.shape)

pos = pd.read_csv('/kaggle/input/home-credit-default-risk/POS_CASH_balance.csv')
print('Pos_bal data shape: ', pos.shape)

cc = pd.read_csv('/kaggle/input/home-credit-default-risk/credit_card_balance.csv')
print('CC bal data shape: ', cc.shape)

inst = pd.read_csv('/kaggle/input/home-credit-default-risk/installments_payments.csv')
print('Installments_payments data shape: ', inst.shape)


import featuretools as ft

es = ft.EntitySet(id="home_credit")

es = es.add_dataframe(dataframe_name="app", dataframe=app, index="SK_ID_CURR")

es = es.add_dataframe(dataframe_name="bureau", dataframe=bureau, index="SK_ID_BUREAU")
es = es.add_dataframe(dataframe_name="bureau_bal", dataframe=bureau_bal, index="index")

es = es.add_dataframe(dataframe_name="prev", dataframe=prev, index="SK_ID_PREV")
es = es.add_dataframe(dataframe_name="pos", dataframe=pos, index="index")
es = es.add_dataframe(dataframe_name="cc", dataframe=cc, index="index")
es = es.add_dataframe(dataframe_name="inst", dataframe=inst, index="index")


es = es.add_relationship("app", "SK_ID_CURR", "bureau", "SK_ID_CURR")
es = es.add_relationship("bureau", "SK_ID_BUREAU", "bureau_bal", "SK_ID_BUREAU")

es = es.add_relationship("app", "SK_ID_CURR", "prev", "SK_ID_CURR")
es = es.add_relationship("app", "SK_ID_CURR", "pos", "SK_ID_CURR")
es = es.add_relationship("app", "SK_ID_CURR", "cc", "SK_ID_CURR")
es = es.add_relationship("app", "SK_ID_CURR", "inst", "SK_ID_CURR")


feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_dataframe_name="app",
    agg_primitives=["mean", "max", "std"],
    # trans_primitives=["divide_numeric"],
    max_depth=1
)


df = feature_matrix.copy()

df = df.replace([np.inf, -np.inf], np.nan)
missing_ratio = df.isnull().mean()

threshold = 0.7

cols_to_keep = missing_ratio[missing_ratio < threshold].index
df = df[cols_to_keep]

print("Remaining columns:", df.shape[1])
print("Dropped columns:", feature_matrix.shape[1] - df.shape[1])


df.head()


train = df.copy()

target = 'TARGET'
feature_cols = [col for col in train.columns if col != target]


def fit_woe_feature(x, y, min_pct=0.05, max_bins=10):
    
    df = pd.DataFrame({'x': x, 'y': y})
    
    # ðŸ”¥ Handle categorical safely
    if pd.api.types.is_categorical_dtype(df['x']):
        df['x'] = df['x'].cat.add_categories(['MISSING'])
        df['x'] = df['x'].fillna('MISSING')
    else:
        df['x'] = df['x'].astype(object).fillna('MISSING')
    
    # Detect numeric AFTER handling missing
    is_numeric = pd.api.types.is_numeric_dtype(x)
    
    if is_numeric:
        try:
            df['bin'] = pd.qcut(
                df['x'].astype(float),
                q=min(max_bins, df['x'].nunique()),
                duplicates='drop'
            )
        except:
            return None
    else:
        df['bin'] = df['x'].astype(str)
    
    # Aggregation
    stats = df.groupby('bin').agg(
        total=('y', 'count'),
        bad=('y', 'sum')
    ).reset_index()
    
    stats['good'] = stats['total'] - stats['bad']
    
    # Filter small bins
    stats['pct'] = stats['total'] / len(df)
    stats = stats[stats['pct'] >= min_pct]
    
    if len(stats) < 2:
        return None
    
    # Distribution
    stats['dist_good'] = stats['good'] / stats['good'].sum()
    stats['dist_bad'] = stats['bad'] / stats['bad'].sum()
    
    # Avoid log(0)
    stats['dist_good'] = stats['dist_good'].replace(0, 1e-6)
    stats['dist_bad'] = stats['dist_bad'].replace(0, 1e-6)
    
    stats['woe'] = np.log(stats['dist_good'] / stats['dist_bad'])
    stats['iv'] = (stats['dist_good'] - stats['dist_bad']) * stats['woe']
    
    mapping = dict(zip(stats['bin'].astype(str), stats['woe']))
    
    return {
        'mapping': mapping,
        'is_numeric': is_numeric,
        'bins': stats['bin'] if is_numeric else None,
        'iv': stats['iv'].sum()
    }


def transform_woe_feature(x, fit_result):
    
    if pd.api.types.is_categorical_dtype(x):
        x = x.cat.add_categories(['MISSING'])
        x = x.fillna('MISSING')
    else:
        x = x.astype(object).fillna('MISSING')
    
    if fit_result['is_numeric']:
        try:
            binned = pd.qcut(x.astype(float), q=len(fit_result['bins']), duplicates='drop')
            return binned.astype(str).map(fit_result['mapping']).fillna(0)
        except:
            return pd.Series(0, index=x.index)
    else:
        return x.astype(str).map(fit_result['mapping']).fillna(0)


results = {}
train_woe = pd.DataFrame({target: train[target]})

for col in feature_cols:
    
    fit_result = fit_woe_feature(train[col], train[target])
    
    if fit_result is None:
        continue
    
    results[col] = fit_result
    train_woe[col] = transform_woe_feature(train[col], fit_result)


train_woe





import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Ensure valid feature set
all_features = [col for col in train_woe.columns if col != target]

# IV map (ONLY valid features)
iv_map = {
    f: results[f]["iv"]
    for f in all_features
    if f in results and f in train_woe.columns
}

print(f"Initial features: {len(iv_map)}")

nunique = train_woe[all_features].nunique()
constant_cols = nunique[nunique <= 1].index.tolist()

train_woe = train_woe.drop(columns=constant_cols)

print(f"Removed constant features: {len(constant_cols)}")


# =========================
# Step 1: IV Filtering
# =========================

iv_threshold = 0.02

iv_table = (
    pd.DataFrame({
        "feature": list(iv_map.keys()),
        "iv": list(iv_map.values())
    })
    .sort_values("iv", ascending=False)
)

step1_features = iv_table.loc[
    iv_table["iv"] >= iv_threshold, "feature"
].tolist()

# ðŸ”¥ ALIGNMENT FIX (CRITICAL)
step1_features = [
    f for f in step1_features
    if f in train_woe.columns
]

print(f"After IV filter: {len(step1_features)}")


# # =========================
# # Step 2: Correlation Filtering
# # =========================

# corr_threshold = 0.7

# def woe_spread(feature):
#     m = results[feature]["mapping"]
#     return max(m.values()) - min(m.values())

# spread_map = {f: woe_spread(f) for f in step1_features}

# def choose_drop(a, b):
#     if iv_map[a] != iv_map[b]:
#         return a if iv_map[a] < iv_map[b] else b
#     if spread_map[a] != spread_map[b]:
#         return a if spread_map[a] < spread_map[b] else b
#     return max(a, b)

# step2_features = step1_features.copy()

# # sample for speed
# sample_df = train_woe.loc[:, step2_features].sample(
#     n=min(50000, len(train_woe)),
#     random_state=42
# )

# while True:
#     corr = sample_df.corr().abs()
#     upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
#     max_corr = upper.max().max()

#     if pd.isna(max_corr) or max_corr <= corr_threshold:
#         break

#     i, j = np.where(upper == max_corr)
#     f1 = upper.index[i[0]]
#     f2 = upper.columns[j[0]]

#     drop = choose_drop(f1, f2)
#     if drop in step2_features:
#         step2_features.remove(drop)

# print(f"After correlation filter: {len(step2_features)}")


step2_features= step1_features


# =========================
# Step 3: VIF Filtering
# =========================

vif_threshold = 10

def calc_vif(df):
    X = df.values.astype(float)
    return pd.Series(
        [variance_inflation_factor(X, i) for i in range(X.shape[1])],
        index=df.columns
    )

step3_features = step2_features.copy()

while True:
    vif = calc_vif(train_woe[step3_features])
    max_vif = vif.max()

    if max_vif <= vif_threshold:
        break

    drop = vif.idxmax()
    step3_features.remove(drop)

print(f"After VIF filter: {len(step3_features)}")


target_feature_count = 10

def fit_logit(X, y):
    X = sm.add_constant(X)
    model = sm.Logit(y, X).fit(disp=False)
    return pd.DataFrame({
        "feature": model.params.index,
        "coef": model.params.values,
        "p": model.pvalues.values
    })

def select_drop(df):
    df = df[df.feature != "const"]

    # drop positive coefficient first (credit logic)
    pos = df[df.coef > 0]
    if not pos.empty:
        return pos.sort_values(["p", "coef"], ascending=[False, True]).iloc[0]["feature"]

    return df.sort_values("p", ascending=False).iloc[0]["feature"]

final_features = step3_features.copy()

while len(final_features) > target_feature_count:
    model_df = fit_logit(train_woe[final_features], train_woe[target])
    drop = select_drop(model_df)
    final_features.remove(drop)

print(f"Final features: {len(final_features)}")





final_model = fit_logit(train_woe[final_features], train_woe[target])
final_model = final_model[final_model.feature != "const"]

final_model["iv"] = final_model["feature"].map(iv_map)

print("\nFinal Model Summary:")
display(final_model.sort_values("p"))





test = pd.read_csv("/kaggle/input/home-credit-default-risk/application_test.csv")


test_woe = pd.DataFrame()

for col in final_features:
    if col in test.columns:
        test_woe[col] = transform_woe_feature(test[col], results[col])
    else:
        # if feature came from featuretools / aggregation
        test_woe[col] = 0


test_woe = test_woe[final_features]


import statsmodels.api as sm

X_test = sm.add_constant(test_woe)

# Get coefficients from final model
coef_map = dict(zip(final_model["feature"], final_model["coef"]))

# Add intercept
intercept = coef_map.get("const", 0)

# Compute prediction
logit = intercept + np.dot(test_woe.values, final_model["coef"].values)
prob = 1 / (1 + np.exp(-logit))


submission = pd.DataFrame({
    "SK_ID_CURR": test["SK_ID_CURR"],
    "TARGET": prob
})

submission.to_csv("submission.csv", index=False)




