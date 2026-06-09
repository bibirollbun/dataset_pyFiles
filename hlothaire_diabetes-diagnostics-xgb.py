import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import math
import warnings

warnings.filterwarnings("ignore")
sns.set(style="darkgrid")

pd.set_option("display.max_columns", None)


train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')

print('Train Shape:', train.shape)
print('Test Shape:', test.shape)
print('Submission:', submission.shape)


train.head()


train.info()


train.isnull().sum()


target_col = 'diagnosed_diabetes'
feature_cols = [col for col in train.columns if col not in ['id', target_col]]
cat_cols = train.select_dtypes('object').columns.to_list()
num_cols = [col for col in feature_cols if col not in cat_cols]

print("Features :", feature_cols)
print("\nCat cols :", cat_cols)


train[feature_cols + [target_col]].isnull().sum().sort_values(ascending=False)



train[num_cols].describe().T


plt.figure()
sns.countplot(data=train, x="diagnosed_diabetes")
plt.title("Target Distribution")
plt.xlabel("Diabetes (False/True)")
plt.ylabel("Count")
plt.show()


n_cols = 3
n_rows = math.ceil(len(num_cols) / n_cols)

plt.figure(figsize=(18, 5 * n_rows))

for i, col in enumerate(num_cols, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.histplot(train[col], kde=True, bins=40, color='blue')
    plt.title(f"Distribution de {col}")
    plt.xlabel(col)
    plt.ylabel("Count")

plt.suptitle("Numeric Feature Distributions", y=1.02, fontsize=16)
plt.tight_layout()
plt.show()


n_cols = 3
n_rows = math.ceil(len(cat_cols) / n_cols)

plt.figure(figsize=(18, 5 * n_rows))

for i, col in enumerate(cat_cols, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.countplot(data=train, x=col, hue=target_col, palette='viridis')
    plt.title(f"Distribution {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=30)

plt.suptitle("Categorical Feature Distributions", y=1.02, fontsize=16)
plt.tight_layout()
plt.show()


plt.figure(figsize=(18, 14))
corr = train[num_cols + [target_col]].corr()

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    linewidths=0.5
)

plt.title("Correlation Matrix (Numeric Features + Target)", fontsize=16)
plt.show()



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier


skewed_cols = [
    'physical_activity_minutes_per_week',
    'screen_time_hours_per_day',
    'triglycerides'
]

train_fe = train.copy()

for col in skewed_cols:
    train_fe[col] = np.log1p(train_fe[col])


X = train_fe[feature_cols]
y = train_fe[target_col]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)



preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ]
)



xgb_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    random_state=42
)

model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("xgb", xgb_model)
])



model.fit(X_train, y_train)


y_pred = model.predict(X_val)
y_proba = model.predict_proba(X_val)[:, 1]

print(classification_report(y_val, y_pred))
print("ROC-AUC :", roc_auc_score(y_val, y_proba))


test_fe = test.copy()

for col in skewed_cols:
    test_fe[col] = np.log1p(test_fe[col])

X_test = test_fe[feature_cols]

test_pred = model.predict_proba(X_test)[:, 1]



submission = pd.DataFrame({
    "id": test["id"],
    "diabetes_probability": test_pred
})

submission.to_csv("submission.csv", index=False)

submission.head()

