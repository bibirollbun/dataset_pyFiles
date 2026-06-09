import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from catboost import CatBoostClassifier, Pool
import lightgbm as lgb

from sklearn.ensemble import StackingClassifier


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv',index_col="id")
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv',index_col="id")


train.head()


test.head()


print(train.shape, test.shape)
print(train.dtypes)
print(train['y'].value_counts(normalize=True))


train_features = train.columns
test_features = test.columns


print("Train Columns Unique Value")
for i in train_features:
    print("\n",i)
    print(train[i].unique())


print("Test Columns Unique Value")
for i in test_features:
    print("\n",i)
    print(test[i].unique())


def preprocess_with_label_encoding(train_df, test_df):
    train = train_df.copy()
    test = test_df.copy()
    
    for df in [train, test]:
        df['was_contacted'] = (df['pdays'] != -1).astype(int)
        df['pdays'] = df['pdays'].replace(-1, 0)

    cat_cols = train.select_dtypes(include=['object']).columns.tolist()

    for col in cat_cols:
        le = LabelEncoder()
        le.fit(pd.concat([train[col], test[col]], axis=0).astype(str))

        train[col] = le.transform(train[col].astype(str))
        test[col]  = le.transform(test[col].astype(str))

    return train, test

train_pp, test_pp = preprocess_with_label_encoding(train, test)


print(train_pp.isna().sum())
print(test_pp.isna().sum())


print(train['y'].value_counts(normalize=True))


SEED = 42
NFOLDS = 5

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X = train_pp.drop(columns=["y"])
y = train_pp["y"].values


# 1) Logistic Regression with class weights
log_model = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)

# 2) CatBoost with class_weights
neg, pos = (y==0).sum(), (y==1).sum()
pos_weight = neg/pos
cat_model = CatBoostClassifier(
    iterations=2000, learning_rate=0.05, depth=6,
    loss_function="Logloss", eval_metric="AUC",
    random_seed=SEED, class_weights=[1.0, pos_weight],
    verbose=False
)

# 3) LightGBM with scale_pos_weight
lgb_model = lgb.LGBMClassifier(
    objective="binary", metric="auc",
    boosting_type="gbdt", learning_rate=0.05,
    num_leaves=31, scale_pos_weight=pos_weight,
    random_state=SEED
)



def evaluate_model(model, model_name, X, y):
    oof_preds = np.zeros(len(X))
    aucs = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model.fit(X_tr, y_tr)
        preds = model.predict_proba(X_val)[:,1]
        oof_preds[val_idx] = preds

        auc = roc_auc_score(y_val, preds)
        aucs.append(auc)
        print(f"[{model_name}] Fold {fold} AUC: {auc:.5f}")

    overall = roc_auc_score(y, oof_preds)
    print(f"[{model_name}] OOF AUC: {overall:.5f} | mean {np.mean(aucs):.5f} ± {np.std(aucs):.5f}")
    return overall, oof_preds

auc_log, oof_log = evaluate_model(log_model, "Logistic Regression", X, y)
auc_cat, oof_cat = evaluate_model(cat_model, "CatBoost", X, y)
auc_lgb, oof_lgb = evaluate_model(lgb_model, "LightGBM", X, y)



# stack = StackingClassifier(
#     estimators=[
#         ("lr", log_model),
#         ("cat", cat_model),
#         ("lgb", lgb_model)
#     ],
#     final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
#     passthrough=True,   # include original features in meta-learner
#     n_jobs=-1
# )

# auc_stack, oof_stack = evaluate_model(stack, "Stacked Model", X, y)



X_full = train_pp.drop(columns=["y"])
y_full = train_pp["y"].values
X_test = test_pp

# ----- 1) Pick the best single model by OOF AUC
model_scores = {
    "Logistic": auc_log,
    "CatBoost": auc_cat,
    "LightGBM": auc_lgb,
}
best_name = max(model_scores, key=model_scores.get)
print("Best single model by OOF AUC:", best_name, "->", model_scores[best_name])

# Map to the actual model instance you created earlier
name_to_model = {
    "Logistic": log_model,
    "CatBoost": cat_model,
    "LightGBM": lgb_model,
}
best_model = name_to_model[best_name]

# **Refit** on all data and predict test
best_model.fit(X_full, y_full)
test_pred_best = best_model.predict_proba(X_test)[:, 1]

sub_best = pd.DataFrame({
    "id": test_pp.index,
    "y": test_pred_best
})
sub_best.to_csv("submission.csv", index=False)
print("Saved:", "submission_best_single.csv")




