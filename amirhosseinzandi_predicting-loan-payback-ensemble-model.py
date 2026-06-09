! pip install -q ydata-profiling


! pip install -q catboost


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier


from IPython.display import IFrame
import warnings
warnings.filterwarnings("ignore")


df_train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
test_ids = df_test['id']

# Remove IDs
df_train = df_train.drop("id", axis=1)
df_test = df_test.drop("id", axis=1)


df_train


target = "loan_paid_back"
x = df_train.drop(target, axis=1)
y = df_train[target]

num_cols = x.select_dtypes(include=np.number).columns.tolist()
cat_cols = x.select_dtypes(exclude=np.number).columns.tolist()


from ydata_profiling import ProfileReport

profile = ProfileReport(df_train , title="Profiling Report", explorative=True)
profile.to_file("profiling_report.html")


IFrame(src='profiling_report.html', width=1000, height=1500)


# Label distribution

# Counting
value_counts = df_train[target].value_counts()

# Plotting
value_counts.plot(kind="bar")
plt.xlabel("Value")
plt.xticks(rotation=0)
plt.ylabel("Count")
plt.title("Value Counts")
plt.show()


# Preprocessing
for col in cat_cols:
    x[col] = x[col].astype("category")
    df_test[col] = df_test[col].astype("category")
    

scaler = MinMaxScaler()
x[num_cols] = scaler.fit_transform(x[num_cols])
df_test[num_cols] = scaler.transform(df_test[num_cols])

# Models
xgb = XGBClassifier(enable_categorical=True)
cat = CatBoostClassifier(cat_features=cat_cols , verbose=0)
lgb = LGBMClassifier(categorical_feature=cat_cols ,verbosity=-1)

# Final model
final_model = StackingClassifier(
    estimators=[('xgb', xgb), ('cat', cat), ('lgb', lgb)],
    final_estimator=LogisticRegression(),
    cv=5,
    n_jobs=1
)

final_model.fit(x, y)


# Prediction
final_preds = final_model.predict_proba(df_test)[:, 1]


submission = pd.DataFrame({
    "id": test_ids,
    "loan_paid_back": final_preds
})

submission.to_csv("submission.csv", index=False)


submission.sample(3)


from sklearn.inspection import permutation_importance


result = permutation_importance(final_model, x, y, n_repeats=5, random_state=42)
importance = result.importances_mean

df_importance = pd.DataFrame({
    "feature": x.columns,
    "importance": importance
})

df_importance = df_importance.sort_values(by="importance", ascending=False).reset_index(drop=True)
df_importance


