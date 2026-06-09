import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score,roc_auc_score,make_scorer
from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import plot_confusion_matrix


df = pd.read_csv('/kaggle/input/mercedes-benz-greener-manufacturing/train.csv.zip')


df


from sklearn.preprocessing import LabelEncoder

# Create a copy to avoid modifying the original DataFrame
df_encoded = df.copy()

string_cols = df.select_dtypes(include='object').columns

for col in string_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])

# Remove columns with zero variance
zero_variance_cols = df_encoded.columns[df_encoded.var() == 0]
df_encoded = df_encoded.drop(columns=zero_variance_cols)


X=df_encoded.drop('y',axis=1).copy()
Y=df_encoded['y'].copy()


X.head()


X_fe = X.copy()
# X_test_fe = add_row_features(X_test)


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5, random_state=42)
X_fe["cluster"] = kmeans.fit_predict(X)
# X_test_fe["cluster"] = kmeans.predict(X_test)


from sklearn.feature_selection import VarianceThreshold

vt = VarianceThreshold(threshold=0.0)
X_fe = pd.DataFrame(vt.fit_transform(X_fe))
# X_test_fe = pd.DataFrame(vt.transform(X_test_fe))


# top_feats = X.corrwith(Y).abs().sort_values(ascending=False).head(6).index.tolist()
# print(top_feats)


# X_fe["int_1"] = X[top_feats[0]] * X[top_feats[1]]
# X_fe["int_2"] = X[top_feats[2]] / (X[top_feats[3]] + 1e-6)

# X_test_fe["int_1"] = X_test[top_feats[0]] * X_test[top_feats[1]]
# X_test_fe["int_2"] = X_test[top_feats[2]] / (X_test[top_feats[3]] + 1e-6)


from sklearn.feature_selection import VarianceThreshold

vt = VarianceThreshold(threshold=0.0)
X_fe = pd.DataFrame(vt.fit_transform(X_fe))


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_fe, Y, test_size=0.2, random_state = 42)


# Finding D Matrix
dtrain = xgb.DMatrix(X, label=Y)


# Setting Parameters
params = {
    "objective": "reg:squarederror",
    "learning_rate": 0.01,
    "max_depth": 3,
    "subsample": 0.6,
    "colsample_bytree": 0.6,
    "tree_method": "hist",
    "reg_lambda": 5,
    "reg_alpha": 1000,
    "eval_metric": "rmse"
}


# Finding best possible value of n_estimators
cv = xgb.cv(
    params,
    dtrain,
    num_boost_round=5000,
    nfold=5,
    early_stopping_rounds=100
)


best_rounds = len(cv)
print(best_rounds)


clf_xgb = xgb.XGBRegressor(
    objective="reg:squarederror",
    tree_method="hist",
    random_state=42,
    n_estimators=5000,
    learning_rate=0.01,
    max_depth=2,
    reg_lambda=5,
    reg_alpha=1000,
    subsample=0.6,
    colsample_bytree=0.6
)
clf_xgb.fit(X_train,Y_train,verbose=True,early_stopping_rounds=50,eval_metric='rmse',eval_set=[(X_test, Y_test)] )


Y_pred = clf_xgb.predict(X_test)


ss_res = ((Y_test - Y_pred) ** 2).sum()
ss_tot = ((Y_test - Y_test.mean()) ** 2).sum()
r2_manual = 1 - ss_res / ss_tot
print("Manual R²:", r2_manual)


mae = (abs(Y_test - Y_pred)).mean()
print("MAE:", mae)


import numpy as np
rmse = np.sqrt(((Y_test - Y_pred) ** 2).mean())
print("RMSE:", rmse)


# from sklearn.model_selection import RepeatedKFold, cross_val_score

# rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

# scores = cross_val_score(
#     clf_xgb,
#     X_fe, Y,
#     cv=rkf,
#     scoring="neg_root_mean_squared_error"
# )

# print("CV RMSE per fold:", -scores)
# print("Mean CV RMSE:", -scores.mean())
# print("Std RMSE:", scores.std())


test_df = pd.read_csv('/kaggle/input/mercedes-benz-greener-manufacturing/test.csv.zip')


# Select object type columns (strings)
string_cols = test_df.select_dtypes(include='object').columns

# Create a copy to avoid modifying the original DataFrame
test_df_encoded = test_df.copy()

# Perform label encoding
for col in string_cols:
    le = LabelEncoder()
    test_df_encoded[col] = le.fit_transform(test_df_encoded[col])

test_df_encoded = test_df_encoded.drop(columns=zero_variance_cols)


test_df_encoded["cluster"] = kmeans.predict(test_df_encoded)
test_df_encoded = pd.DataFrame(vt.transform(test_df_encoded))


Y_test_pred = clf_xgb.predict(test_df_encoded)


submission = pd.DataFrame({
    "ID": test_df['ID'],
    "y": Y_test_pred
})

submission.to_csv('submission.csv', index=False)

print("Submission file successfully created!")

