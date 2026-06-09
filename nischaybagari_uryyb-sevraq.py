# you know this part.
import pandas as pd
import numpy as np

TRAIN_PATH = r"/kaggle/input/recruitment-task-for-gdsc-ml/MiNDAT.csv"
TEST_PATH  = r"/kaggle/input/recruitment-task-for-gdsc-ml/MiNDAT_UNK.csv"

df = pd.read_csv(TRAIN_PATH)


# you know this part.
!pip install --upgrade scikit-learn


# you know this part.
import sklearn
print(sklearn.__version__)


import matplotlib.pyplot as plt
import seaborn as sns

# correlogram
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", annot=False)
plt.title("correlogram)", fontsize=14)
plt.show()

# histograms
df.hist(figsize=(15, 12), bins=30, edgecolor='black')
plt.suptitle("histograms", fontsize=16)
plt.show()


# we dont want the model to know something it shouldn't, do we?
df = df.dropna(subset=["CORRUCYSTIC_DENSITY"])



import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_dataframe(df, special_cols, label_encoders=None, is_train=True):
    """
    you know the drill.
      - rename columns
      - mindless label encoding
      - imputation
      - splitting target `y` from the observable `X`
    """

    # ---------------------------
    # rename columns.
    # ---------------------------
    renaming = {}
    renaming.update(special_cols)

    # generic names are all the others will get.
    generic_cols = [col for col in df.columns if col not in special_cols]
    renaming.update({generic_cols[i]: f"n_{i+1}" for i in range(len(generic_cols))})

    df = df.rename(columns=renaming)

    # ---------------------------
    # nums and cats separation. meow
    # ---------------------------
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = df.select_dtypes(exclude=["object"]).columns.tolist()

    # ---------------------------
    # imputation.
    # ---------------------------
    df[categorical_cols] = df[categorical_cols].fillna("UNK")
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # ---------------------------
    # mindless label encoding.
    # ---------------------------
    if is_train:
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])
            # Add UNK to classes
            le.classes_ = np.append(le.classes_, "UNK")
            label_encoders[col] = le
    else:
        for col in categorical_cols:
            le = label_encoders[col]
            df[col] = df[col].astype(str)
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else "UNK")
            df[col] = le.transform(df[col])

    # ---------------------------
    # split the target from the observables.
    # ---------------------------
    if is_train and "target" in df.columns:
        X = df.drop(columns=["target"])
        y = df["target"]
        return X, y, label_encoders
    else:
        return df, label_encoders



special_cols = {
    "LOCAL_IDENTIFIER": "id",
    "CORRUCYSTIC_DENSITY": "target",

    "v0rt3X": "vortex1",
    "v1rt3X": "vortex2",
    "r1Ng": "ring1",
    "r2Ng": "ring2",
    "b1oRb13": "blobie1",
    "b2oRb13": "blobie2",

    "MINDSPIKE_VERSION": "mndspke",
    "maT_r": "matter",
}

X, y, label_encoders = preprocess_dataframe(df, special_cols, is_train=True)


import optuna
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb


from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,
    test_size=0.2,      # 20% validation split
    random_state=42,    # for reproducibility
)


from sklearn.model_selection import KFold

def objective(trial):
    # suggest hyperparameters.
    params = {
        "objective": "reg:squarederror",
        "verbosity": 0,
        "tree_method": "hist",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "early_stopping_rounds": 10
    }

    # k-fold cv.
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    rmse_list = []

    for train_idx, val_idx in kf.split(X):
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train_cv, y_train_cv,
            eval_set=[(X_val_cv, y_val_cv)],
            
            verbose=False
        )

        y_pred = model.predict(X_val_cv)
        rmse = np.sqrt(mean_squared_error(y_val_cv, y_pred))
        rmse_list.append(rmse)

    return np.mean(rmse_list)  # average rmse across folds.


# --- run study ---
study = optuna.create_study(direction="minimize")  # minimize rmse.
study.optimize(objective, n_trials=10)  # feel free to increase trials to (over)fit.

print("best trial:")
print(study.best_trial.params)


best_params = study.best_trial.params

# fit the model.
import xgboost as xgb
model = xgb.XGBRegressor(
   **best_params
)
model.fit(X_train, y_train)


# test the model.
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
y_pred = model.predict(X_valid)
print(f"RMSE: {root_mean_squared_error(y_valid, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_valid, y_pred):.4f}")


# you know the drill.

# fit.
model.fit(X, y)

# read and preprocess test data.
df_unk = pd.read_csv(TEST_PATH)
X_unk, _ = preprocess_dataframe(
    df_unk.copy(),
    special_cols,
    label_encoders=label_encoders,
    is_train=False
)

# ensure same column order.
X_unk = X_unk.reindex(columns=X.columns, fill_value=0)

# make predictions.
y_unk_pred = model.predict(X_unk)

# build submission file.
submission = pd.DataFrame({
    "LOCAL_IDENTIFIER": df_unk["LOCAL_IDENTIFIER"].astype(int),
    "CORRUCYSTIC_DENSITY": y_unk_pred.astype(float)
})

submission.to_csv("submission.csv", index=False)
print(submission.head())


