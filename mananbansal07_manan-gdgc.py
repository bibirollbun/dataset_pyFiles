import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


train = pd.read_csv('/kaggle/input/gdgc-ai-ml-inductions-batch-2025-26/train.csv')
test = pd.read_csv('/kaggle/input/gdgc-ai-ml-inductions-batch-2025-26/test.csv')
lookup = pd.read_csv('/kaggle/input/gdgc-ai-ml-inductions-batch-2025-26/feature_lookup.csv')

print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.head()


train.info()
print("\nMissing values:\n", train.isna().sum())

print("\nDescribe numerical:\n", train.describe())

# Separate target
TARGET = "relationship_probability"
ID = "ID"

# Feature list
features = [col for col in train.columns if col not in [TARGET, ID]]


# Identify numeric & categorical from lookup
cat_feats = lookup.loc[lookup['type'] == 'categorical', 'feature_code'].tolist()
num_feats = lookup.loc[lookup['type'] == 'numeric', 'feature_code'].tolist()

cat_feat_names = lookup.loc[lookup['type'] == 'categorical', 'relevance'].tolist()
num_feat_names = lookup.loc[lookup['type'] == 'numeric', 'relevance'].tolist()

print("Categorical features:", len(cat_feats))
print("Numeric features:", len(num_feats))


# 1. Target distribution
plt.figure(figsize=(6,4))
sns.histplot(train[TARGET], kde=True)
plt.title("Target Distribution")
plt.show()

# 2. Correlation heatmap (numeric)
plt.figure(figsize=(10,8))
sns.heatmap(train[num_feats + [TARGET]].corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 3. Compare feature distributions (Train vs Test)
for f in num_feats[:3]:  # top 3 numeric
    plt.figure(figsize=(6,4))
    sns.kdeplot(train[f], label="train")
    sns.kdeplot(test[f], label="test")
    plt.title(f"Train vs Test Distribution — {f}")
    plt.legend()
    plt.show()


print("Categorical : ")
for i in range(len(cat_feats)):
    print(cat_feats[i], cat_feat_names[i])

print()
print("Numeric : ")
for i in range(len(num_feats)):
    print(num_feats[i], num_feat_names[i])


# compute correlations
corrs = train[num_feats].corrwith(train['relationship_probability']).abs().sort_values(ascending=False)
top5 = corrs.head(5).index.tolist()
print("Top 5 numerical features : \n", top5)

for f in top5:
    plt.figure(figsize=(6,4))
    sns.scatterplot(data=train, x=f, y='relationship_probability', alpha=0.4)
    sns.regplot(data=train, x=f, y='relationship_probability', scatter=False, color='red')
    plt.title(f"{f} vs Relationship Probability")
    plt.show()



#box plot for cat features
for f in cat_feats:
    plt.figure(figsize=(8,4))
    sns.boxplot(data=train, x=f, y='relationship_probability')
    plt.xticks(rotation=45)
    plt.title(f"Relationship Probability Across Categories of {f}")
    plt.show()



for f in cat_feats:
    plt.figure(figsize=(8,4))
    temp = train.groupby(f)['relationship_probability'].mean().sort_values()
    sns.barplot(x=temp.index, y=temp.values)
    plt.xticks(rotation=45)
    plt.title(f"Average Relationship Probability by {f}")
    plt.ylabel("Mean Probability")
    plt.show()



sns.jointplot(
    data=train,
    x="F17",  # popularity
    y="F15",  # communication_skills
    kind="hex",
    cmap="Blues"
)
plt.suptitle("Popularity vs Communication Skills", y=1.02)
plt.show()



for f in cat_feats[:6]:
    plt.figure(figsize=(6,4))
    sns.countplot(data=train, x=f)
    plt.xticks(rotation=45)
    plt.title(f"Distribution of {f}")
    plt.show()



# since this dataset does not have any missing or null values 
# so i will skip that part

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_feats),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats)
    ]
)

preprocessor.fit(train[cat_feats + num_feats])

X_train = preprocessor.transform(train[cat_feats + num_feats])
X_test  = preprocessor.transform(test[cat_feats + num_feats])

y_train = train["relationship_probability"]



from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def final_competition_score(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)

    # normalized forms
    rmse_n = max(0, 1 - rmse / 100)
    mae_n  = max(0, 1 - mae / 100)
    r2_n   = max(0, min(1, r2))

    weighted = 0.40 * rmse_n + 0.30 * mae_n + 0.30 * r2_n
    return weighted * 100



from sklearn.model_selection import train_test_split

X = train[cat_feats + num_feats]
y = train["relationship_probability"]

# Apply preprocessing
X_trans = preprocessor.transform(X)

X_train_s, X_val_s, y_train_s, y_val_s = train_test_split(
    X_trans, y, test_size=0.2, random_state=42
)



from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train_s, y_train_s)

pred_lr = lr.predict(X_val_s)
print("Linear Regression Score:", final_competition_score(y_val_s, pred_lr))



from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0, solver='lsqr')
ridge.fit(X_train_s, y_train_s)
pred_ridge = ridge.predict(X_val_s)
print("Ridge Score:", final_competition_score(y_val_s, pred_ridge))



from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.0005)
lasso.fit(X_train_s, y_train_s)

pred_lasso = lasso.predict(X_val_s)
print("Lasso Score:", final_competition_score(y_val_s, pred_lasso))



from sklearn.linear_model import ElasticNet

en = ElasticNet(alpha=0.0005, l1_ratio=0.5)
en.fit(X_train_s, y_train_s)

pred_en = en.predict(X_val_s)
print("ElasticNet Score:", final_competition_score(y_val_s, pred_en))



from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(
    max_depth=6,
    random_state=42
)
dt.fit(X_train_s, y_train_s)

pred_dt = dt.predict(X_val_s)
print("Decision Tree Score:", final_competition_score(y_val_s, pred_dt))



from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train_s, y_train_s)

pred_knn = knn.predict(X_val_s)
print("KNN Score:", final_competition_score(y_val_s, pred_knn))



import pandas as pd

results = {
    "Model": ["Linear Regression", "Ridge", "Lasso", "ElasticNet", "Decision Tree", "KNN"],
    "Score": [
        final_competition_score(y_val_s, pred_lr),
        final_competition_score(y_val_s, pred_ridge),
        final_competition_score(y_val_s, pred_lasso),
        final_competition_score(y_val_s, pred_en),
        final_competition_score(y_val_s, pred_dt),
        final_competition_score(y_val_s, pred_knn)
    ]
}

results_df = pd.DataFrame(results)
results_df



# From this basic comparison we can see that Elastic net regression
# Combination of (ridge + lasso) is giving good results.
# Whereas as Desicion tree and KNN did not perform good which is surprising.


# import lightgbm as lgb

# lgbm = lgb.LGBMRegressor(
#     n_estimators=2000,
#     learning_rate=0.02,
#     max_depth=-1,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     reg_alpha=0.1,
#     reg_lambda=0.1,
#     random_state=42
# )

# lgbm.fit(
#     X_train_s,
#     y_train_s,
#     eval_set=[(X_val_s, y_val_s)],
#     eval_metric="rmse",
#     callbacks=[lgb.early_stopping(100)]
# )

# pred_lgbm = lgbm.predict(X_val_s)
# print("LightGBM Score:", final_competition_score(y_val_s, pred_lgbm))



# from xgboost import XGBRegressor

# xgb = XGBRegressor(
#     n_estimators=2000,
#     learning_rate=0.03,
#     max_depth=7,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     objective="reg:squarederror",
#     random_state=42
# )

# xgb.fit(
#     X_train_s,
#     y_train_s,
#     eval_set=[(X_val_s, y_val_s)],
#     early_stopping_rounds=100,
#     verbose=False
# )

# pred_xgb = xgb.predict(X_val_s)
# print("XGBoost Score:", final_competition_score(y_val_s, pred_xgb))



# from catboost import CatBoostRegressor

# cat = CatBoostRegressor(
#     depth=8,
#     learning_rate=0.03,
#     iterations=1500,
#     loss_function="RMSE",
#     random_seed=42,
#     verbose=0
# )
# cat.fit(X_train_s, y_train_s)

# pred_cat = cat.predict(X_val_s)
# print("CatBoost Score:", final_competition_score(y_val_s, pred_cat))



# from sklearn.ensemble import RandomForestRegressor

# rf = RandomForestRegressor(
#     n_estimators=500,
#     max_depth=15,
#     random_state=42
# )

# rf.fit(X_train_s, y_train_s)
# pred_rf = rf.predict(X_val_s)
# print("Random Forest Score:", final_competition_score(y_val_s, pred_rf))



# from sklearn.ensemble import GradientBoostingRegressor

# gbr = GradientBoostingRegressor(
#     n_estimators=1200,
#     learning_rate=0.03,
#     max_depth=5,
#     random_state=42
# )

# gbr.fit(X_train_s, y_train_s)
# pred_gbr = gbr.predict(X_val_s)
# print("GradientBoosting Score:", final_competition_score(y_val_s, pred_gbr))



# results_adv = pd.DataFrame({
#     "Model": [
#         "LightGBM", "XGBoost", "CatBoost",
#         "RandomForest", "GradientBoosting"
#     ],
#     "Score": [
#         final_competition_score(y_val_s, pred_lgbm),
#         final_competition_score(y_val_s, pred_xgb),
#         final_competition_score(y_val_s, pred_cat),
#         final_competition_score(y_val_s, pred_rf),
#         final_competition_score(y_val_s, pred_gbr),
#     ]
# })

# results_adv



# Testing out advanced models tells that the data is mostly linear
# which means if i work with linear regression ahead i can improve
# the accuray more.


corrs = train[num_feats].corrwith(train['relationship_probability']).abs()
top10 = corrs.sort_values(ascending=False).head(10).index.tolist()

print(top10)


from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(
    degree=2,
    interaction_only=False,
    include_bias=False
)

poly_train = poly.fit_transform(train[top10])
poly_test = poly.transform(test[top10])

poly_cols = poly.get_feature_names_out(top10)

poly_train_df = pd.DataFrame(poly_train, columns=poly_cols)
poly_test_df = pd.DataFrame(poly_test, columns=poly_cols)



# Remove original feature names from polynomial columns
poly_cols_clean = [c for c in poly_cols if c not in top10]


print("Before:", len(poly_cols))
print("After removing originals:", len(poly_cols_clean))



poly_train_df = pd.DataFrame(poly_train, columns=poly_cols)
poly_train_df = poly_train_df[poly_cols_clean]

poly_test_df = pd.DataFrame(poly_test, columns=poly_cols)
poly_test_df = poly_test_df[poly_cols_clean]

# Check duplicates in polynomial features
print("Duplicates in poly cols:", poly_train_df.columns.duplicated().sum())



train_poly = pd.concat([train, poly_train_df], axis=1)
test_poly  = pd.concat([test,  poly_test_df], axis=1)



# Updated numeric feature list
num_feats_poly = num_feats.copy()

for c in poly_cols_clean:
    if c not in num_feats_poly:
        num_feats_poly.append(c)

preprocessor_poly = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_feats_poly),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats)
    ]
)

preprocessor_poly.fit(train_poly[cat_feats + num_feats_poly])



from sklearn.model_selection import train_test_split

X = train_poly[cat_feats + num_feats_poly]  # updated feature set
y = train_poly["relationship_probability"]

X_train_s, X_val_s, y_train_s, y_val_s = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
X_train_trans = preprocessor_poly.transform(X_train_s)
X_val_trans   = preprocessor_poly.transform(X_val_s)



from sklearn.linear_model import ElasticNet

en_poly = ElasticNet(
    alpha=0.0003,      # small regularization
    l1_ratio=0.6,      # mix of L1/L2
    max_iter=5000,
    random_state=42
)

en_poly.fit(X_train_trans, y_train_s)

pred_poly = en_poly.predict(X_val_trans)

print("ElasticNet + Polynomial Features Score:",
      final_competition_score(y_val_s, pred_poly))

print("RMSE:", mean_squared_error(y_val_s, pred_poly, squared=False))
print("MAE:", mean_absolute_error(y_val_s, pred_poly))
print("R2:", r2_score(y_val_s, pred_poly))



from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNet

# Prepare
X_full = train_poly[cat_feats + num_feats_poly]
y_full = train_poly["relationship_probability"]

# Transform EVERYTHING using the preprocessor
preprocessor_poly.fit(X_full)

X_full_trans = preprocessor_poly.transform(X_full)

# 5-Fold setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train_poly))
models = []

fold_num = 1
for train_idx, val_idx in kf.split(X_full_trans):

    print(f"FOLD {fold_num} ---------------------------")
    fold_num += 1
    
    X_tr, X_val = X_full_trans[train_idx], X_full_trans[val_idx]
    y_tr, y_val = y_full.iloc[train_idx], y_full.iloc[val_idx]

    # ElasticNet (best params so far)
    model = ElasticNet(
        alpha=0.0003,
        l1_ratio=0.6,
        max_iter=5000,
        random_state=42
    )

    model.fit(X_tr, y_tr)
    pred = model.predict(X_val)

    # Save predictions
    oof_preds[val_idx] = pred

    # Save model
    models.append(model)

# Final OOF evaluation
print("\n==================== OOF SCORE ====================")
print("Final OOF Competition Score:", final_competition_score(y_full, oof_preds))
print("OOF RMSE:", mean_squared_error(y_full, oof_preds, squared=False))
print("OOF MAE:", mean_absolute_error(y_full, oof_preds))
print("OOF R2:", r2_score(y_full, oof_preds))



X_test_full = test_poly[cat_feats + num_feats_poly]

# Transform using the fitted preprocessor
X_test_trans = preprocessor_poly.transform(X_test_full)


elastic_preds = []

for model in models:
    elastic_preds.append(model.predict(X_test_trans))

# Average predictions
elastic_final_preds = np.mean(elastic_preds, axis=0)

# Clip to valid range
elastic_final_preds = np.clip(elastic_final_preds, 0, 100)




print(elastic_final_preds)
print(len(elastic_final_preds))


submission = pd.DataFrame({
    "ID": test["ID"],
    "relationship_probability": elastic_final_preds
})

submission.to_csv("submission.csv", index=False)
submission.head()


