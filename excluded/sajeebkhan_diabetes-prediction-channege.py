# 1. Libraries ğŸ“š

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

plt.style.use("seaborn-v0_8")
sns.set_palette("coolwarm")



# 2. Dataset ğŸ“Š

train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

train.head()



target = "diagnosed_diabetes"

X = train.drop(columns=["id", target])
y = train[target]

test_data = test.drop(columns=["id"])



# Distribution of target
plt.figure(figsize=(6,4))
sns.countplot(x=y)
plt.title("Target Distribution")
plt.show()



train.shape,test.shape


train.isnull().sum().sort_values(ascending=False).head()



num_cols= train.select_dtypes(include=['int64', 'float64']).columns
train[num_cols].hist(figsize=(15,12), bins=30)
plt.show()


numeric_cols = X.select_dtypes(include=["float64", "int64"]).columns

plt.figure(figsize=(15,10))
sns.heatmap(train[numeric_cols].corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap (Numeric Features)", fontsize=16)
plt.show()



# 4. Preprocessing ğŸ› ï¸�  (fixed version)

# 1) Separate features again (just to be safe)
X = train.drop(columns=["id", target])
y = train[target]
test_data = test.drop(columns=["id"])

# 2) Find categorical columns
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
print("Categorical columns:", cat_cols)

# 3) One-hot encode categoricals
X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=True)
test_enc = pd.get_dummies(test_data, columns=cat_cols, drop_first=True)

# 4) Align train & test columns
X_enc, test_enc = X_enc.align(test_enc, join="left", axis=1, fill_value=0)

# 5) Scale (now everything is numeric)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_enc)
test_scaled = scaler.transform(test_enc)

# Save feature names for later (feature importance)
feature_names = X_enc.columns

X_scaled[:5]



results_df = (
    pd.DataFrame(results.items(), columns=["Model", "Accuracy"])
    .sort_values("Accuracy", ascending=False)
    .reset_index(drop=True)
)

results_df



plt.figure(figsize=(8,4))
sns.barplot(data=results_df, x="Accuracy", y="Model")
plt.title("Model Comparison")
plt.xlim(0, 1)
plt.show()



# 5. ML Modeling âš™ï¸�

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

models = {
    "Logistic Regression": LogisticRegression(max_iter=300),
    "Random Forest (100 trees)": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

models



!pip install catboost lightgbm xgboost --quiet



from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



# 6. Training & Evaluation ğŸ”¥

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split data for validation
X_train, X_valid, y_train, y_valid = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

results = {}

print("ğŸ”� Evaluating Models...\n")

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)

    preds = model.predict(X_valid)
    acc = accuracy_score(y_valid, preds)

    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}\n")

results



X_train, X_valid, y_train, y_valid = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)



# CatBoost
cat_model = CatBoostClassifier(
    iterations=400,
    learning_rate=0.05,
    depth=8,
    verbose=0,
    random_state=42
)
cat_model.fit(X_train, y_train)

# LightGBM
lgbm_model = LGBMClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
lgbm_model.fit(X_train, y_train)

# XGBoost
xgb_model = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42
)
xgb_model.fit(X_train, y_train)



models = {
    "CatBoost": cat_model,
    "LightGBM": lgbm_model,
    "XGBoost": xgb_model
}

for name, model in models.items():
    preds = model.predict(X_valid)
    acc = accuracy_score(y_valid, preds)
    print(f"{name} Accuracy: {acc:.4f}")



estimators = [
    ("cat", cat_model),
    ("lgbm", lgbm_model),
    ("xgb", xgb_model)
]

stack_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=200),
    passthrough=True
)

stack_model.fit(X_train, y_train)



stack_preds = stack_model.predict(X_valid)
stack_acc = accuracy_score(y_valid, stack_preds)
stack_acc



# 7. Best Model Selection â­�

best_model_name = max(results, key=results.get)
best_model_name



best_model = models[best_model_name]
best_model.fit(X_scaled, y)

best_model



!pip install lightgbm --quiet
from lightgbm import LGBMClassifier

# Train LightGBM (best for tabular competitions)
best_model = LGBMClassifier(
    n_estimators=900,
    learning_rate=0.03,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight='balanced',  # VERY IMPORTANT
    random_state=42
)

best_model.fit(X_scaled, y)
best_model



final_model = stack_model
final_model.fit(X_scaled, y)

test_predictions = final_model.predict(test_scaled)

submission = pd.DataFrame({
    "id": test["id"],
    "diagnosed_diabetes": test_predictions
})

submission.to_csv("submission.csv", index=False)
submission.head()



pd.Series(test_predictions).value_counts()


