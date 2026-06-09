# Import Libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
import xgboost as xgb


# Load and Explore Data

train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
submission_df = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Train columns: {train_df.columns.tolist()}")


print("Missing values in train:\n", train_df.isna().sum())
print("\nMissing values in test:\n", test_df.isna().sum())


# Target Distribution
plt.figure(figsize=(5, 3))
sns.countplot(x="Personality", data=train_df, palette="Set2")
plt.title("Target Distribution")
plt.show()


# Numerical Correlation Heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(train_df.select_dtypes(include='number').corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()


# Boxplots of numerical features by Personality
num_cols = train_df.select_dtypes(include='number').drop(columns=["id"]).columns
for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="Personality", y=col, data=train_df, palette="Set3")
    plt.title(f"{col} by Personality")
    plt.show()


# Categorical feature distribution
cat_cols = ["Stage_fear", "Drained_after_socializing"]
for col in cat_cols:
    plt.figure(figsize=(5, 3))
    sns.countplot(data=train_df, x=col, hue="Personality", palette="pastel")
    plt.title(f"{col} vs Personality")
    plt.show()


# ğŸ’¡ Drop ID column
train_df.drop(columns=["id"], inplace=True)
test_ids = test_df["id"]
test_df.drop(columns=["id"], inplace=True)


train_df["is_train"] = 1
test_df["is_train"] = 0
test_df["Personality"] = np.nan
combined = pd.concat([train_df, test_df], ignore_index=True)


# === Basic Preprocessing ===
categorical_cols = ["Stage_fear", "Drained_after_socializing"]
numerical_cols = [col for col in train_df.columns if col not in categorical_cols + ["Personality", "is_train"]]



# Fill missing values
for col in numerical_cols:
    combined[col] = combined[col].fillna(combined[col].median())
for col in categorical_cols:
    combined[col] = combined[col].fillna("missing")


def cv_target_encode(train_df, test_df, target, cols, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_te = train_df.copy()
    test_te = test_df.copy()
    global_means = train_df[target].value_counts(normalize=True).to_dict()

    for col in cols:
        oof = np.zeros(len(train_df))
        test_vals = np.zeros(len(test_df))
        for train_idx, val_idx in skf.split(train_df, train_df[target]):
            X_tr, X_val = train_df.iloc[train_idx], train_df.iloc[val_idx]
            means = X_tr.groupby(col)[target].value_counts(normalize=True).unstack().fillna(0)
            for i in val_idx:
                val = train_df.iloc[i][col]
                oof[i] = means.loc[val]["Extrovert"] if val in means.index else global_means.get("Extrovert", 0.5)
        train_te[col + "_te"] = oof

        # Test encoding
        full_means = train_df.groupby(col)[target].value_counts(normalize=True).unstack().fillna(0)
        for i in range(len(test_df)):
            val = test_df.iloc[i][col]
            test_vals[i] = full_means.loc[val]["Extrovert"] if val in full_means.index else global_means.get("Extrovert", 0.5)
        test_te[col + "_te"] = test_vals
    return train_te, test_te



train_data = combined[combined["is_train"] == 1].copy()
test_data = combined[combined["is_train"] == 0].copy()
train_te, test_te = cv_target_encode(train_data, test_data, "Personality", categorical_cols)



features = numerical_cols + [col + "_te" for col in categorical_cols]
X = train_te[features].values
X_test = test_te[features].values
y = train_te["Personality"]
le = LabelEncoder()
y_encoded = le.fit_transform(y)



# === Optuna-tuned base models ===
cat_params = {
    "iterations": 1000, "depth": 8, "learning_rate": 0.05,
    "l2_leaf_reg": 3.5, "border_count": 128, "random_strength": 1,
    "bagging_temperature": 0.3, "od_type": "Iter", "od_wait": 30,
    "verbose": 0, "random_seed": 42
}

xgb_params = {
    "n_estimators": 1000, "max_depth": 6, "learning_rate": 0.05,
    "subsample": 0.9, "colsample_bytree": 0.8, "gamma": 0.5,
    "reg_alpha": 1.0, "reg_lambda": 1.0, "use_label_encoder": False,
    "eval_metric": "logloss", "random_state": 42
}


# === Reuse tuned params ===
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_cat, oof_xgb = np.zeros((len(X), 2)), np.zeros((len(X), 2))
test_cat_preds, test_xgb_preds = [], []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

    cat_model = CatBoostClassifier(**cat_params)
    cat_model.fit(X_train, y_train)
    oof_cat[val_idx] = cat_model.predict_proba(X_val)
    test_cat_preds.append(cat_model.predict_proba(X_test))

    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_train, y_train)
    oof_xgb[val_idx] = xgb_model.predict_proba(X_val)
    test_xgb_preds.append(xgb_model.predict_proba(X_test))



# === Train Models ===
X_meta = np.hstack([oof_cat, oof_xgb])
X_test_meta = np.hstack([
    np.mean(test_cat_preds, axis=0),
    np.mean(test_xgb_preds, axis=0)
])




# === Weighted Soft Voting Ensemble ===
meta_model = LogisticRegression()
meta_model.fit(X_meta, y_encoded)
final_preds = meta_model.predict(X_test_meta)
decoded_preds = le.inverse_transform(final_preds)



submission_df = pd.DataFrame({
    "id": test_ids,
    "Personality": decoded_preds
})
submission_df.to_csv("submission_stacked.csv", index=False)
print("ğŸ“� Saved: submission_stacked.csv")




