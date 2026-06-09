import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
%matplotlib inline
sns.set_style("whitegrid")

warnings.filterwarnings('ignore')

train_df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
print('Train Shape:', train_df.shape)
print('Test Shape:', test_df.shape)

train_df.head(3)


train_df.info()


train_df['diagnosed_diabetes'].value_counts(normalize=True)


categorical_cols = train_df.select_dtypes(include='object').columns
numeric_cols = train_df.select_dtypes(exclude='object').columns
print('Categorical Columns:', categorical_cols)
print('Numeric Columns:', numeric_cols)


INPUTS = [col for col in numeric_cols if col not in ['id', 'diagnosed_diabetes']] + categorical_cols.tolist()
TARGET = 'diagnosed_diabetes'

X_train = train_df[INPUTS]
y_train = train_df[TARGET]


X_train.nunique()


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from lightgbm import LGBMClassifier, early_stopping


N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)


for col in categorical_cols:
    X_train[col] = X_train[col].astype("category")
    test_df[col] = test_df[col].astype("category")


oof_lgb = np.zeros(len(X_train))
test_preds_lgb = np.zeros(len(test_df))


params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 50,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "random_state": 42,
}

# -----------------------------
# Training loop
# -----------------------------
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), 1):
    X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
    y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

    # ---- LightGBM ----
    lgb = LGBMClassifier(
        **params,
        n_estimators=5000
    )

    lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        categorical_feature=categorical_cols.tolist(),
        callbacks=[early_stopping(200, verbose=False)]
    )

    va_pred = lgb.predict_proba(X_va)[:, 1]
    oof_lgb[va_idx] = va_pred

    fold_auc = roc_auc_score(y_va, va_pred)
    print(f"Fold {fold} AUC: {fold_auc:.6f} | Best iter: {lgb.best_iteration_}")

    test_preds_lgb = lgb.predict_proba(test_df[INPUTS])[:, 1] / N_SPLITS

# -----------------------------
# Overall CV score
# -----------------------------
oof_auc_lgb = roc_auc_score(y_train, oof_lgb)
print(f"\nOOF ROC-AUC: {oof_auc_lgb:.6f}")


submission_df = pd.DataFrame({
    "id": test_df['id'],
    "diagnosed_diabetes": test_preds_lgb
})

submission_df.to_csv("/kaggle/working/submission_lgb.csv", index=False)
submission_df.head()


fpr, tpr, _ = roc_curve(y_train, oof_lgb)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {oof_auc_lgb:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()


!pip install catboost -q


from catboost import CatBoostClassifier


oof_cat = np.zeros(len(X_train))
test_preds_cat = np.zeros(len(test_df))


# -----------------------------
# Training loop
# -----------------------------
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), 1):
    X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
    y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

    # ---- CatBoost ----
    cat = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=5000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=False,
        early_stopping_rounds=200
    )

    cat.fit(
        X_tr,
        y_tr,
        eval_set=(X_va, y_va),
        cat_features=[X_train.columns.get_loc(col) for col in categorical_cols]
    )

    va_pred = cat.predict_proba(X_va)[:, 1]
    oof_cat[va_idx] = va_pred

    fold_auc = roc_auc_score(y_va, va_pred)
    print(f"Fold {fold} CatBoost AUC: {fold_auc:.6f}")

    test_preds_cat += cat.predict_proba(test_df[INPUTS])[:, 1] / N_SPLITS

# -----------------------------
# Overall CV score
# -----------------------------
print(f"\nCatBoost OOF ROC-AUC: {roc_auc_score(y_train, oof_cat):.6f}")


submission = pd.DataFrame({
    "id": test_df["id"],
    "diagnosed_diabetes": test_preds_cat
})

submission.to_csv("/kaggle/working/submission_cat.csv", index=False)
submission.head()


oof_ensemble = 0.6 * oof_lgb + 0.4 * oof_cat
test_preds_ensemble = 0.6 * test_preds_lgb + 0.4 * test_preds_cat


submission = pd.DataFrame({
    "id": test_df["id"],
    "diagnosed_diabetes": test_preds_ensemble
})

submission.to_csv("/kaggle/working/submission_lgb_cat_ensemble.csv", index=False)
submission.head()


print(f"\nBlended OOF ROC-AUC: {roc_auc_score(y_train, oof_ensemble):.6f}")


def plot_target_distributions(df, feature, target):
    plt.figure(figsize=(6,4))
    sns.kdeplot(df[df[target]==0][feature], label="No Diabetes", fill=True)
    sns.kdeplot(df[df[target]==1][feature], label="Diabetes", fill=True)
    plt.title(f"{feature} vs {target}")
    plt.legend()
    plt.show()
    print("\n")

# Example usage
for f in ['bmi', 'family_history_diabetes', 'age', 'waist_to_hip_ratio', 'hypertension_history']:
      plot_target_distributions(train_df, f, "diagnosed_diabetes")


plt.figure(figsize=(6,5))
sns.scatterplot(
    data=train_df.sample(10000, random_state=42),
    x="hdl_cholesterol",
    y="cholesterol_total",
    hue="diagnosed_diabetes",
    alpha=0.3
)
plt.title("Total Cholesterol vs HDL")
plt.show()


plt.figure(figsize=(6,5))
sns.scatterplot(
    data=train_df.sample(10000, random_state=42),
    x="hdl_cholesterol",
    y="ldl_cholesterol",
    hue="diagnosed_diabetes",
    alpha=0.3
)
plt.title("LDL vs HDL")
plt.show()


plt.figure(figsize=(6,5))
sns.scatterplot(
    data=train_df.sample(10000, random_state=42),
    x="bmi",
    y="triglycerides",
    hue="diagnosed_diabetes",
    alpha=0.3
)
plt.title("Triglycerides vs BMI")
plt.show()


# === Feature Importance (LightGBM) ===
plt.figure(figsize=(10,6))
importance = lgb.feature_importances_
idx = np.argsort(importance)[::-1]

sns.barplot(x=importance[idx][:15], y=X_train.columns[idx][:15])
plt.title("Top 15 Features — LightGBM Importance")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()


def add_features(df):
    """
    Add engineered features for diabetes prediction.

    Features added:
    - age_bin: Age categories
    - cholesterol_ratio: LDL/HDL ratio
    - triglycerides_hdl_ratio: Triglycerides/HDL ratio
    - pulse_pressure: Difference between systolic and diastolic BP
    - mean_arterial_pressure: Average arterial pressure
    """
    # Age bins
    df["age_bin"] = pd.cut(
        df["age"],
        bins=[18, 40, 60, 89],
        labels=["young", "mid", "senior"]
    ).astype("object")

    # Cholesterol ratio (LDL/HDL)
    df["cholesterol_ratio"] = df["ldl_cholesterol"] / df["hdl_cholesterol"]

    # Triglycerides/HDL ratio
    df["triglycerides_hdl_ratio"] = df["triglycerides"] / df["hdl_cholesterol"]

    # Blood pressure features
    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
    df["mean_arterial_pressure"] = df["diastolic_bp"] + (df["pulse_pressure"] / 3.0)

    return df


train_df = add_features(train_df)
test_df = add_features(test_df)


categorical_cols = train_df.select_dtypes(include='object').columns
numeric_cols = train_df.select_dtypes(exclude='object').columns
print('Categorical Columns:', categorical_cols)
print('Numeric Columns:', numeric_cols)


INPUTS = [col for col in numeric_cols if col not in ['id', 'diagnosed_diabetes']] + categorical_cols.tolist()
TARGET = 'diagnosed_diabetes'

X_train = train_df[INPUTS]
y_train = train_df[TARGET]


from sklearn.linear_model import LogisticRegression


N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_lgb_final = np.zeros(len(X_train))
oof_cat_final = np.zeros(len(X_train))
test_preds_lgb_final = np.zeros(len(test_df))
test_preds_cat_final = np.zeros(len(test_df))


for col in categorical_cols:
    X_train[col] = X_train[col].astype("category")
    test_df[col] = test_df[col].astype("category")


params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 50,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "random_state": 42,
}

# -----------------------------
# Training loop
# -----------------------------
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), 1):
    X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
    y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

    lgb = LGBMClassifier(
        **params,
        n_estimators=5000
    )

    lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        categorical_feature=categorical_cols.tolist(),
        callbacks=[early_stopping(200, verbose=False)]
    )

    va_pred = lgb.predict_proba(X_va)[:, 1]
    oof_lgb_final[va_idx] = va_pred
    test_preds_lgb_final += lgb.predict_proba(test_df[INPUTS])[:, 1] / N_SPLITS

# -----------------------------
# Overall CV score
# -----------------------------
print(f"\nLightGBM OOF ROC-AUC: {roc_auc_score(y_train, oof_lgb_final):.6f}")


for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), 1):
    X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
    y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
    cat = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=5000,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=3.0,
            random_seed=42,
            verbose=False,
            early_stopping_rounds=200
        )

    cat.fit(
            X_tr,
            y_tr,
            eval_set=(X_va, y_va),
            cat_features=[X_train.columns.get_loc(col) for col in categorical_cols]
        )

    va_pred = cat.predict_proba(X_va)[:, 1]
    oof_cat_final[va_idx] = va_pred
    test_preds_cat_final += cat.predict_proba(test_df[INPUTS])[:, 1] / N_SPLITS

# -----------------------------
# Overall CV score
# -----------------------------
print(f"\nCatBoost OOF ROC-AUC: {roc_auc_score(y_train, oof_cat_final):.6f}")


stack_train = np.vstack([oof_lgb_final, oof_cat_final]).T
stack_test = np.vstack([test_preds_lgb, test_preds_cat]).T

meta = LogisticRegression(max_iter=2000)
meta.fit(stack_train, y_train)


test_preds_stack = meta.predict_proba(stack_test)[:, 1]
print("\nFinal Stacked ROC:", roc_auc_score(y_train, meta.predict_proba(stack_train)[:,1]))


submission_df = pd.DataFrame({
    "id": test_df['id'],
    "diagnosed_diabetes": test_preds_stack
})

submission_df.to_csv("/kaggle/working/submission_stack.csv", index=False)
submission_df.head()


oof_df = pd.DataFrame({"pred": oof_lgb_final})
plt.figure(figsize=(10, 5))

sns.kdeplot(
    oof_df["pred"],
    label="OOF Predictions (Train)",
    fill=True,
    alpha=0.3
)
sns.kdeplot(
    submission_df["diagnosed_diabetes"],
    label="Test Predictions",
    fill=True,
    alpha=0.3
)
plt.title("Prediction Distribution: OOF vs Test")
plt.xlabel("Predicted Probability")
plt.ylabel("Density")
plt.legend()
plt.show()

