# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
import xgboost as xgb

# --- Configuration ---
SEED = 42
TARGET = "diagnosed_diabetes"


# Robust ordinal mapping
def map_ordinals(df):
    df = df.copy()

    smoke_map = {
        "Never": 0, "No": 0,
        "Former": 1,
        "Current": 2, "Smoker": 2, "Yes": 2
    }

    if "smoking_status" in df.columns:
        df["smoking_status_risk"] = (
            df["smoking_status"].map(smoke_map).fillna(0).astype("int8")
        )

    return df


# Medical feature engineering
def engineer_medical_features(df):
    df = df.copy()

    # BMI categories
    df["BMI_Cat"] = pd.cut(
        df["bmi"],
        bins=[-1, 18.5, 25, 30, 100],
        labels=[0, 1, 2, 3]
    ).astype(int)

    # Blood pressure risk (systolic)
    df["BP_Risk_Level"] = pd.cut(
        df["systolic_bp"],
        bins=[-1, 120, 130, 140, 300],
        labels=[0, 1, 2, 3]
    ).astype(int)

    # Interaction features
    df["Visceral_Fat"] = df["bmi"] * df["waist_to_hip_ratio"]
    df["AIP"] = np.log((df["triglycerides"] + 1) / (df["hdl_cholesterol"] + 1))
    df["MAP"] = (df["systolic_bp"] + 2 * df["diastolic_bp"]) / 3

    return df


# Main function
def main():
    # Load data
    train_df = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
    test_df = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

    test_ids = test_df["id"]
    train_df = train_df.drop(columns=["id"])
    test_df = test_df.drop(columns=["id"])

    # Apply feature engineering
    train_df = map_ordinals(train_df)
    test_df = map_ordinals(test_df)

    train_df = engineer_medical_features(train_df)
    test_df = engineer_medical_features(test_df)

    # --- Clustering / Patient profiling ---
    cluster_cols = [
        "age", "bmi", "systolic_bp",
        "cholesterol_total",
        "diet_score",
        "physical_activity_minutes_per_week"
    ]

    scaler = StandardScaler()
    combined = pd.concat([train_df[cluster_cols], test_df[cluster_cols]], axis=0)
    combined_scaled = scaler.fit_transform(combined)

    kmeans = KMeans(n_clusters=7, random_state=SEED, n_init=10)
    kmeans.fit(combined_scaled)

    train_df["Cluster_ID"] = kmeans.predict(scaler.transform(train_df[cluster_cols]))
    test_df["Cluster_ID"] = kmeans.predict(scaler.transform(test_df[cluster_cols]))

    train_df["Cluster_ID"] = train_df["Cluster_ID"].astype("category")
    test_df["Cluster_ID"] = test_df["Cluster_ID"].astype("category")

    # Split features / target
    X = train_df.drop(columns=[TARGET])
    y = train_df[TARGET]

    X_test = test_df.copy()

    # Identify categorical columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Ensure categorical dtype
    for col in cat_cols:
        X[col] = X[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    # Train / validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.1,
        random_state=SEED,
        stratify=y
    )

    # XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=3000,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.7,
        colsample_bytree=0.6,
        reg_lambda=2.0,
        scale_pos_weight=1.2,
        enable_categorical=True,
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
        early_stopping_rounds=150
    )

    # Train
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )

    # Validation AUC
    val_preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_preds)
    print(f"Validation AUC: {auc:.5f}")

    # Predict test
    test_preds = model.predict_proba(X_test)[:, 1]

    # Submission
    submission = pd.DataFrame({
        "id": test_ids,
        "diagnosed_diabetes": test_preds
    })

    submission.to_csv("submission.csv", index=False)
    print("Saved submission.csv")


# Execute main
if __name__ == "__main__":
    main()

