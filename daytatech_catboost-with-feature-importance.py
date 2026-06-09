from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool, metrics, cv
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv", index_col = "id")

for col in df.columns:
    if df[col].dtype == "O":
        df[col] = df[col].astype("category")

le = LabelEncoder()
le.fit(["unknown", "primary", "secondary", "tertiary"])
df['education'] = le.fit_transform(df['education'])
le = LabelEncoder()
le.fit(["jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
df['month'] = le.fit_transform(df['month'])

X = df.drop(columns=["y"])
y = df.y
#(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size = .3, random_state = 0)
#cat_model = CatBoostClassifier(cat_features=list(X.select_dtypes("category")), random_seed=42, eval_metric=metrics.AUC())
#cat_model.fit(X_train, y_train)
#cat_pred = cat_model.predict_proba(X_test)[:,1]
#roc_auc_score(y_test,cat_pred)

cat_model = CatBoostClassifier(n_estimators=200, silent=True, cat_features=list(X.select_dtypes("category")), random_seed=42, eval_metric=metrics.AUC())
cat_model.fit(X, y)

feature_imp = cat_model.get_feature_importance()
feature_names = X.columns
feature_imp = pd.DataFrame(feature_imp, index=feature_names, columns=["imp"])
# feature_imp.sort_values("imp").plot.bar(); plt.show()
print(feature_imp.sort_values("imp"))

# Submission
df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv", index_col = "id")
for col in df.columns:
    if df[col].dtype == "O":
        df[col] = df[col].astype("category")

le = LabelEncoder()
le.fit(["unknown", "primary", "secondary", "tertiary"])
df['education'] = le.fit_transform(df['education'])
le = LabelEncoder()
le.fit(["jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
df['month'] = le.fit_transform(df['month'])

cat_pred = cat_model.predict_proba(df)[:,1]
results = pd.DataFrame(cat_pred, index=df.index, columns=["y"])
results.to_csv("submission.csv")
results.head(5)



