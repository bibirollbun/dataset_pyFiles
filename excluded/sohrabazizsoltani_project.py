%%writefile main.py
import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ----------------------------
# تنظیمات کلی
# ----------------------------
SEED = 42
TARGET = "Listening_Time_minutes"

DATA_DIR = Path("/kaggle/input/playground-series-s5e4")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"


# --------------------------
# 1) Load Data
# --------------------------
def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test


# --------------------------
# 2) Preprocess
# --------------------------
def preprocess(train, test):
    y = train[TARGET]
    train = train.drop(columns=[TARGET])

    combined = pd.concat([train, test], ignore_index=True)

    numeric_cols = combined.select_dtypes(include=[np.number]).columns
    categorical_cols = [c for c in combined.columns if c not in numeric_cols]

    for col in numeric_cols:
        combined[col] = combined[col].fillna(combined[col].median())

    for col in categorical_cols:
        combined[col] = combined[col].fillna("Missing")
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col].astype(str))

    X = combined.iloc[:len(train), :].reset_index(drop=True)
    X_test = combined.iloc[len(train):, :].reset_index(drop=True)

    return X, X_test, y


# --------------------------
# 3) Optuna Objective
# --------------------------
def objective(trial, X_train, X_val, y_train, y_val):

    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "seed": SEED,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "num_leaves": trial.suggest_int("num_leaves", 20, 200),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
    }

    train_set = lgb.Dataset(X_train, y_train)
    val_set = lgb.Dataset(X_val, y_val)

    model = lgb.train(
        params,
        train_set,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, preds) ** 0.5
    return rmse


# --------------------------
# 4) Train Final Model
# --------------------------
def train_full_model(X, y, best_params, best_iter):

    params = best_params.copy()
    params.update({
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "seed": SEED,
    })

    train_set = lgb.Dataset(X, y)

    model = lgb.train(
        params,
        train_set,
        num_boost_round=best_iter,
    )

    return model

# ---------------------------------------------------
# 5) Main Pipeline
# ---------------------------------------------------
def main():
    print("== Stage 2 | Load & Preprocess ==")
    train, test = load_data()
    X, X_test, y = preprocess(train, test)

    print("== Stage 3 | Optuna Hyperparameter Tuning ==")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_train, X_val, y_train, y_val),
                   n_trials=5)  # برای سرعت بیشتر

    best_params = study.best_params
    best_iter = 500

    print("Best Params:", best_params)

    print("\n== Stage 5 | Training Final Model ==")
    final_model = train_full_model(X, y, best_params, best_iter)

    # ذخیره مدل
    final_model.save_model("final_lgbm_model.txt")
    print("Final model saved → final_lgbm_model.txt")

    # ذخیره Feature Importance
    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance_gain": final_model.feature_importance(importance_type="gain"),
        "importance_split": final_model.feature_importance(importance_type="split")
    })
    feature_importance.sort_values("importance_gain", ascending=False, inplace=True)
    feature_importance.to_csv("feature_importance_lgbm.csv", index=False)
    print("feature_importance_lgbm.csv saved ✓")

    # پیش‌بینی داده تست
    test_preds = final_model.predict(X_test)

    submission = pd.DataFrame({
        "id": test["id"],
        TARGET: test_preds
    })

    submission.to_csv("submission.csv", index=False)
    print("submission.csv saved ✓")

    print("\nPipeline Completed ✓")


if __name__ == "__main__":
    main()


!python main.py


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from pathlib import Path
import os

DATA_DIR = Path("/kaggle/input/playground-series-s5e4")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
TARGET_COL = "Listening_Time_minutes"

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAY_MAPPING = {day: idx for idx, day in enumerate(DAY_ORDER, start=1)}

TIME_BUCKETS = {
    "Morning": 9 * 60,
    "Afternoon": 15 * 60,
    "Evening": 20 * 60,
    "Night": 23 * 60,
    "Late Night": 1 * 60,
    "Early Morning": 5 * 60
}

def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test

def parse_publication_time(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip()
    if ":" in value:
        try:
            h, m, s = map(int, value.split(":"))
            return h * 60 + m
        except ValueError:
            pass
    return TIME_BUCKETS.get(value, np.nan)

def fill_with_group_median(source_df, target_df, col, group_col):
    medians = source_df.groupby(group_col)[col].median()
    target_df[col] = target_df[col].fillna(target_df[group_col].map(medians))
    target_df[col] = target_df[col].fillna(source_df[col].median())
    return target_df

def clean_data(train_df, test_df):
    train = train_df.copy()
    test = test_df.copy()

    # تبدیل روز به عدد
    for df in (train, test):
        df["Publication_Day"] = df["Publication_Day"].map(DAY_MAPPING).fillna(0)

    # تبدیل زمان انتشار به دقیقه
    for df in (train, test):
        df["Publication_Time_minutes"] = df["Publication_Time"].apply(parse_publication_time)
        df["Publication_Time_minutes"] = df["Publication_Time_minutes"].fillna(
            df["Publication_Time_minutes"].median()
        )
        df.drop(columns=["Publication_Time"], inplace=True)

    for df in (train, test):
        df["Episode_Length_minutes"] = df["Episode_Length_minutes"].astype(float)
        df["Guest_Popularity_percentage"] = df["Guest_Popularity_percentage"].astype(float)

    # پر کردن با median گروه
    train = fill_with_group_median(train, train, "Episode_Length_minutes", "Podcast_Name")
    test = fill_with_group_median(train, test, "Episode_Length_minutes", "Podcast_Name")

    train = fill_with_group_median(train, train, "Guest_Popularity_percentage", "Podcast_Name")
    test = fill_with_group_median(train, test, "Guest_Popularity_percentage", "Podcast_Name")

    # تعداد تبلیغ‌ها
    median_ads = train["Number_of_Ads"].median()
    train["Number_of_Ads"] = train["Number_of_Ads"].fillna(median_ads)
    test["Number_of_Ads"] = test["Number_of_Ads"].fillna(median_ads)

    # احساس اپیزود
    sentiment_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
    for df in (train, test):
        df["Episode_Sentiment"] = df["Episode_Sentiment"].map(sentiment_map).fillna(1)

    # لیبل انکود کردن
    text_cols = ["Podcast_Name", "Episode_Title", "Genre"]
    for col in text_cols:
        le = LabelEncoder()
        combined = pd.concat([train[col], test[col]], axis=0).astype(str)
        le.fit(combined)
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))

    # ویژگی‌های جدید
    for df in (train, test):
        df["Episode_Ratio"] = df["Episode_Length_minutes"] / (df["Guest_Popularity_percentage"] + 1)
        df["Day_Time_Combo"] = df["Publication_Day"] * 1000 + df["Publication_Time_minutes"]
        df["Has_Ads"] = (df["Number_of_Ads"] > 0).astype(int)

    return train, test

def train_xgb_model(train_clean):
    features = [col for col in train_clean.columns if col != TARGET_COL]
    X = train_clean[features]
    y = train_clean[TARGET_COL]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    params = {
        "n_estimators": 800,        # کمتر از 1500 برای سرعت بیشتر
        "learning_rate": 0.05,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "random_state": 42,
        "tree_method": "hist"
    }

    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=200
    )

    preds = model.predict(X_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, preds))
    print(f"\nXGBoost Validation RMSE: {rmse:.4f}")

    return model, features

def create_submission(model, features, train_clean, test_clean, filename="submission_xgb.csv"):
    model.fit(train_clean[features], train_clean[TARGET_COL])
    preds = model.predict(test_clean[features])

    sample_path = DATA_DIR / "sample_submission.csv"
    if sample_path.exists():
        submission = pd.read_csv(sample_path)
    else:
        submission = pd.DataFrame({"id": test_clean["id"]})
    submission[TARGET_COL] = preds
    submission.to_csv(filename, index=False)
    print(f"\n{filename} saved!")

if __name__ == "__main__":
    train, test = load_data()
    train_clean, test_clean = clean_data(train, test)

    print("\nTrain clean shape:", train_clean.shape)
    print("Test clean shape:", test_clean.shape)

    xgb_model, feature_cols = train_xgb_model(train_clean)
    create_submission(xgb_model, feature_cols, train_clean, test_clean)


import pandas as pd

# Load both submissions
lgb = pd.read_csv("submission.csv")
xgb = pd.read_csv("submission_xgb.csv")

# Check that IDs match
if not (lgb["id"].equals(xgb["id"])):
    raise ValueError("ID mismatch between files!")

# Ensemble weights
w_lgb = 0.60    # weight for LightGBM
w_xgb = 0.40    # weight for XGBoost

# Weighted average
ensemble_pred = (w_lgb * lgb["Listening_Time_minutes"] +
                 w_xgb * xgb["Listening_Time_minutes"])

# Save final ensemble submission
final_df = pd.DataFrame({
    "id": lgb["id"],
    "Listening_Time_minutes": ensemble_pred
})

final_df.to_csv("submission_ensemble.csv", index=False)

print("Ensemble submission saved as submission_ensemble.csv")


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# مسیرها
DATA_DIR = Path("/kaggle/input/playground-series-s5e4")
WORK_DIR = Path("/kaggle/working")

FEATURE_IMPORTANCE_FILE = WORK_DIR / "feature_importance_lgbm.csv"
TRAIN_PATH = DATA_DIR / "train.csv"
SUBMISSION_PATH = WORK_DIR / "submission.csv"

PLOTS_DIR = WORK_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

TARGET = "Listening_Time_minutes"


# -----------------------------
# 1) Feature Importance Plot
# -----------------------------
def plot_feature_importance():
    if not FEATURE_IMPORTANCE_FILE.exists():
        print("feature_importance_lgbm.csv وجود ندارد.")
        return

    df = pd.read_csv(FEATURE_IMPORTANCE_FILE)
    df = df.sort_values("importance_gain", ascending=False).head(25)

    plt.figure(figsize=(10, 8))
    sns.barplot(x="importance_gain", y="feature", data=df, palette="viridis")
    plt.title("Top 25 Feature Importance (Gain)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_importance_gain.png")
    plt.show()
    plt.close()

    print("✓ feature_importance_gain.png ذخیره شد.")


# -----------------------------
# 2) Correlation Heatmap
# -----------------------------
def plot_correlation():
    if not TRAIN_PATH.exists():
        print("train.csv پیدا نشد.")
        return

    df = pd.read_csv(TRAIN_PATH)
    numeric = df.select_dtypes(include=[np.number])

    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric.corr(), cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "correlation_heatmap.png")
    plt.show()
    plt.close()

    print("✓ correlation_heatmap.png ذخیره شد.")


# -----------------------------
# 3) Distribution of Target
# -----------------------------
def plot_target_distribution():
    if not TRAIN_PATH.exists():
        print("train.csv پیدا نشد.")
        return

    df = pd.read_csv(TRAIN_PATH)

    plt.figure(figsize=(8, 6))
    sns.histplot(df[TARGET], bins=40, kde=True, color="blue")
    plt.title(f"Distribution of {TARGET}")
    plt.xlabel(TARGET)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "target_distribution.png")
    plt.show()
    plt.close()

    print("✓ target_distribution.png ذخیره شد.")


# -----------------------------
# 4) Prediction vs Ground Truth
# -----------------------------
def plot_pred_vs_true():
    if not SUBMISSION_PATH.exists():
        print("submission.csv وجود ندارد.")
        return

    sub = pd.read_csv(SUBMISSION_PATH)
    train = pd.read_csv(TRAIN_PATH)

    plt.figure(figsize=(8, 6))
    plt.scatter(train.index, train[TARGET], s=10, label="Actual (Train)", alpha=0.6)
    plt.scatter(sub.index, sub[TARGET], s=10, label="Predicted (Test)", alpha=0.6)
    plt.legend()
    plt.title("Actual vs Predicted Comparison (Train/Test)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "actual_vs_predicted.png")
    plt.show()
    plt.close()

    print("✓ actual_vs_predicted.png ذخیره شد.")


# -----------------------------
# 5) Residual Plot (Training Approx)
# -----------------------------
def plot_residuals():
    if not TRAIN_PATH.exists():
        print("train.csv پیدا نشد.")
        return

    df = pd.read_csv(TRAIN_PATH)
    y = df[TARGET]

    pred_mean = y.mean()
    residuals = y - pred_mean

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=np.arange(len(residuals)), y=residuals, alpha=0.5)
    plt.title("Residual Plot (Approx)")
    plt.axhline(0, color="red")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "residual_plot.png")
    plt.show()
    plt.close()

    print("✓ residual_plot.png ذخیره شد.")


def main():
    print("== Stage 9 | Generating Plots ==")

    plot_feature_importance()
    plot_correlation()
    plot_target_distribution()
    plot_pred_vs_true()
    plot_residuals()

    print("\nتمام نمودارها ساخته شدند و در پوشه plots ذخیره شدند. ✓")


if __name__ == "__main__":
    main()


%%writefile stage10_shap.py
import argparse
import os
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

import matplotlib


def _running_inside_notebook() -> bool:
    try:
        from IPython import get_ipython  # pylint: disable=import-error
        return get_ipython() is not None
    except Exception:
        return False


FORCE_SHOW = os.environ.get("SHAP_INLINE_SHOW", "").strip() == "1"
FORCE_HIDE = os.environ.get("SHAP_INLINE_HIDE", "").strip() == "1"

if FORCE_SHOW and FORCE_HIDE:
    raise ValueError("SHAP_INLINE_SHOW و SHAP_INLINE_HIDE را همزمان تنظیم نکن.")

if FORCE_SHOW:
    DISPLAY_INLINE = True
elif FORCE_HIDE:
    DISPLAY_INLINE = False
else:
    DISPLAY_INLINE = _running_inside_notebook()

if not DISPLAY_INLINE:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

SEED = 42
DATA_DIR = Path("/kaggle/input/playground-series-s5e4")
WORK_DIR = Path("/kaggle/working")

MODEL_PATH = WORK_DIR / "final_lgbm_model.txt"
TRAIN_PATH = DATA_DIR / "train.csv"
TARGET = "Listening_Time_minutes"

OUTPUT_DIR = WORK_DIR / "shap_plots"
OUTPUT_DIR.mkdir(exist_ok=True)


def preprocess_for_shap(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in df.columns if c not in numeric + [TARGET]]

    for col in numeric:
        df[col] = df[col].fillna(df[col].median())

    for col in categorical:
        df[col] = df[col].astype(str).fillna("Missing")
        df[col] = df[col].astype("category").cat.codes

    return df


def compute_shap_in_chunks(explainer: shap.TreeExplainer,
                           X: pd.DataFrame,
                           chunk: int) -> np.ndarray:
    sv_list = []
    for i in range(0, len(X), chunk):
        X_chunk = X.iloc[i: i + chunk]
        print(f"  computing chunk {i}..{i+len(X_chunk)-1} ({len(X_chunk)} rows)")
        sv = explainer.shap_values(X_chunk)
        sv_list.append(sv)
    return np.vstack(sv_list)


def _finalize_plot(path: Path, dpi: int = 150):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    if DISPLAY_INLINE:
        plt.show()
    plt.close()


def main(mode: str = "sample", sample_n: int = 2000, chunk: int = 5000):
    print("== Stage 10 | SHAP Plots ==")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"فایل مدل پیدا نشد: {MODEL_PATH}\n"
            "ابتدا Stage 5 (main.py) را اجرا کن تا final_lgbm_model.txt ساخته شود."
        )

    model = lgb.Booster(model_file=str(MODEL_PATH))

    df = pd.read_csv(TRAIN_PATH)
    df_processed = preprocess_for_shap(df)

    X = df_processed.drop(columns=[TARGET])

    if mode == "sample":
        n = min(sample_n, len(X))
        print(f"Mode=sample → نمونه‌گیری {n} ردیف از {len(X)} کل")
        X_use = X.sample(n=n, random_state=SEED).reset_index(drop=True)
    elif mode == "full":
        print(f"Mode=full → استفاده از کل دیتاست ({len(X)} rows) با chunk={chunk}")
        X_use = X
    else:
        raise ValueError("mode باید 'sample' یا 'full' باشد")

    print("Creating SHAP explainer...")
    explainer = shap.TreeExplainer(model)

    print("Computing SHAP values...")
    if mode == "sample":
        shap_values = explainer.shap_values(X_use)
    else:
        shap_values = compute_shap_in_chunks(explainer, X_use, chunk)

    print("Saving shap_summary_plot.png ...")
    shap.summary_plot(shap_values, X_use, show=False)
    _finalize_plot(OUTPUT_DIR / "shap_summary_plot.png")

    print("Saving shap_bar_plot.png ...")
    shap.summary_plot(shap_values, X_use, plot_type="bar", show=False)
    _finalize_plot(OUTPUT_DIR / "shap_bar_plot.png")

    print("Saving shap_dependence_plot.png ...")
    top_feature = X_use.columns[np.abs(shap_values).mean(axis=0).argmax()]
    shap.dependence_plot(top_feature, shap_values, X_use, show=False)
    _finalize_plot(OUTPUT_DIR / "shap_dependence_plot.png")

    print("\nStage 10 Completed ✓")
    print(f"Files saved in {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sample", "full"], default="sample",
                        help="sample (پیش‌فرض) یا full برای کل دیتا")
    parser.add_argument("--sample_n", type=int, default=2000,
                        help="تعداد نمونه در حالت sample")
    parser.add_argument("--chunk", type=int, default=5000,
                        help="اندازهٔ chunk در حالت full")
    args = parser.parse_args()
    main(mode=args.mode, sample_n=args.sample_n, chunk=args.chunk)


%run stage10_shap.py --mode sample


%%writefile stage11_submission.py
import pandas as pd
import lightgbm as lgb
from pathlib import Path

# توابع و ثابت‌ها را مستقیماً از main ایمپورت می‌کنیم
from main import load_data, preprocess, TARGET

MODEL_PATH = Path("final_lgbm_model.txt")
OUTPUT_PATH = Path("submission_final.csv")


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"فایل مدل {MODEL_PATH} پیدا نشد. مطمئن شو مرحله‌ی قبلی اجرا شده و مدل ذخیره شده است."
        )

    print("== Stage 11 | Build Final Submission ==")

    train_df, test_df = load_data()
    X, X_test, _ = preprocess(train_df, test_df)

    booster = lgb.Booster(model_file=str(MODEL_PATH))
    print("Final model loaded ✓")

    test_preds = booster.predict(X_test)
    submission = pd.DataFrame({
        "id": test_df["id"],
        TARGET: test_preds
    })

    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"submission_final.csv saved → {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()


!python stage11_submission.py


import pandas as pd
pd.read_csv("submission_final.csv").head()


print("Notebook execution completed successfully.")

