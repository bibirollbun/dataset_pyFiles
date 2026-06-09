import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd
import numpy as np

train_df = pd.read_csv("/kaggle/input/terrain-prices-reggression/train.csv")
test_df = pd.read_csv("/kaggle/input/terrain-prices-reggression/test.csv")


# print(train_df.isnull().sum())
# print(test_df.isnull().sum())


X = train_df.drop(["target", "id"], axis=1)
y = train_df["target"]
X_test = test_df.drop(["id"], axis=1)


def add_features(df):
    df["total_price"] = df["land_area_m2"] * df["price_per_m2"]
    return df

X = add_features(X)
X_test = add_features(X_test)


cat_cols = X.select_dtypes(include="object").columns.tolist()
print(cat_cols)


for col in cat_cols:
    print(f"\"{col}\" colum:\n>>> {sorted(X[col].unique())}\n>>> {sorted(X_test[col].unique())}\n")


from sklearn.preprocessing import LabelEncoder

for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    X_test[col] = le.transform(X_test[col])


from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

n_splits = 3
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

models = []
val_scores = []
val_preds = np.zeros(len(y))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"Fold {fold+1}/{n_splits} >>>")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = LGBMRegressor(
        boosting_type='goss',
        n_estimators=20000,
        learning_rate=0.001,
        num_leaves=10,
        # max_depth=-1,
        min_child_samples=1,
        # reg_alpha=0,
        # reg_lambda=0,
        max_bin=900,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='r2',
        callbacks=[
            early_stopping(stopping_rounds=100),
            log_evaluation(period=1000)
        ]
    )

    # test
    test_preds += model.predict(X_test) / n_splits

    # val
    val_pred = model.predict(X_val)
    val_preds[val_idx] = val_pred
    val_score = r2_score(y_val, val_pred)
    print(f"Best R2: {val_score:.5f}")
    
    val_scores.append(val_score)
    models.append(model)

print(f"Mean CV R2: {np.mean(val_scores):.5f}")


all_importances = []

for model in models:
    if hasattr(model, "feature_importances_"):
        all_importances.append(model.feature_importances_)

if all_importances:
    avg_importances = pd.Series(np.mean(all_importances, axis=0), index=X.columns)
    avg_importances.sort_values(ascending=False).head(10).plot(kind='barh')


submission = pd.DataFrame({"id": test_df["id"], "target": test_preds})
submission.to_csv("submission.csv", index=False)
submission.head()

