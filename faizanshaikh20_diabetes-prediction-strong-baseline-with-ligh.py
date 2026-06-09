import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

import lightgbm as lgb



train_df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

print(train_df.shape, test_df.shape)
train_df.head()



train_df.describe().style.background_gradient(cmap='Reds')


TARGET = "diagnosed_diabetes"
ID_COL = "id"

y = train_df[TARGET]
X = train_df.drop(columns=[TARGET, ID_COL])
X_test = test_df.drop(columns=[ID_COL])



import matplotlib.pyplot as plt
y.value_counts().plot(kind='bar')
plt.title('Targer Distribution')
plt.show()


cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

print("Numerical:", len(num_cols))
print("Categorical:", len(cat_cols))



num_imputer = SimpleImputer(strategy="median")

X_num = num_imputer.fit_transform(X[num_cols])
X_test_num = num_imputer.transform(X_test[num_cols])



cat_encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

X_cat = cat_encoder.fit_transform(X[cat_cols])
X_test_cat = cat_encoder.transform(X_test[cat_cols])



X_final = np.hstack([X_num, X_cat])
X_test_final = np.hstack([X_test_num, X_test_cat])



skf = StratifiedKFold(
    n_splits=7,
    shuffle=True,
    random_state=42
)

model = lgb.LGBMClassifier(
    n_estimators=800,
    learning_rate=0.04,
    num_leaves=31,
    subsample=0.85,
    colsample_bytree=0.85,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)



auc_scores = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_final, y)):
    X_tr, X_val = X_final[tr_idx], X_final[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    model.fit(X_tr, y_tr)

    val_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_pred)
    auc_scores.append(auc)

    print(f"Fold {fold+1} AUC: {auc:.5f}")



print("="*40)
print(f"Mean CV AUC: {np.mean(auc_scores):.5f}")
print(f"Std CV AUC : {np.std(auc_scores):.5f}")



model.fit(X_final, y)
test_pred = model.predict_proba(X_test_final)[:, 1]

submission = pd.DataFrame({
    "id": test_df["id"],
    "diagnosed_diabetes": test_pred
})

submission.to_csv("/kaggle/working/submission.csv", index=False)
submission.head()





