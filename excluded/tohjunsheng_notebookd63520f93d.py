# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ============================================================
# ğŸ�† Linking Writing Processes to Writing Quality (Enhanced)
# Includes Feature Importance Correlation + Hyperparameter Tuning
# ============================================================

import pandas as pd
import numpy as np
import re, warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import nltk

# ============================================================
# 1ï¸�âƒ£ Essay Reconstruction
# ============================================================

def getEssays(df):
    textInputDf = df[['id', 'activity', 'cursor_position', 'text_change']].copy()
    textInputDf = textInputDf[textInputDf.activity != 'Nonproduction']
    valCountsArr = textInputDf['id'].value_counts(sort=False).values
    lastIndex = 0
    essaySeries = pd.Series(dtype=str)

    for index, valCount in enumerate(valCountsArr):
        currTextInput = textInputDf[['activity', 'cursor_position', 'text_change']].iloc[lastIndex:lastIndex + valCount]
        lastIndex += valCount
        essayText = ""

        for Input in currTextInput.values:
            if Input[0] == 'Replace':
                replaceTxt = Input[2].split(' => ')
                essayText = essayText[:Input[1] - len(replaceTxt[1])] + replaceTxt[1] + essayText[Input[1] - len(replaceTxt[1]) + len(replaceTxt[0]):]
                continue

            if Input[0] == 'Paste':
                essayText = essayText[:Input[1] - len(Input[2])] + Input[2] + essayText[Input[1] - len(Input[2]):]
                continue

            if Input[0] == 'Remove/Cut':
                essayText = essayText[:Input[1]] + essayText[Input[1] + len(Input[2]):]
                continue

            if "M" in Input[0]:
                croppedTxt = Input[0][10:]
                splitTxt = croppedTxt.split(' To ')
                valueArr = [item.split(', ') for item in splitTxt]
                moveData = (int(valueArr[0][0][1:]), int(valueArr[0][1][:-1]),
                            int(valueArr[1][0][1:]), int(valueArr[1][1][:-1]))

                if moveData[0] != moveData[2]:
                    if moveData[0] < moveData[2]:
                        essayText = essayText[:moveData[0]] + essayText[moveData[1]:moveData[3]] + essayText[moveData[0]:moveData[1]] + essayText[moveData[3]:]
                    else:
                        essayText = essayText[:moveData[2]] + essayText[moveData[0]:moveData[1]] + essayText[moveData[2]:moveData[0]] + essayText[moveData[1]:]
                continue

            essayText = essayText[:Input[1] - len(Input[2])] + Input[2] + essayText[Input[1] - len(Input[2]):]

        essaySeries[index] = essayText

    essaySeries.index = textInputDf['id'].unique()
    return essaySeries


# ============================================================
# 2ï¸�âƒ£ Text-based features
# ============================================================

def essay_structure_features(df):
    feats = []
    for _, row in df.iterrows():
        text = row["essay"]
        sents = re.split(r"[.!?]", text)
        sents = [s.strip() for s in sents if len(s.strip()) > 0]
        words = re.findall(r"\w+", text)
        word_lens = [len(w) for w in words]
        char_len = len(text)
        f = {
            "id": row["id"],
            "char_len": char_len,
            "n_words": len(words),
            "n_sent": len(sents),
            "avg_word_len": np.mean(word_lens) if word_lens else 0,
            "std_word_len": np.std(word_lens) if word_lens else 0,
            "avg_sent_len": np.mean([len(re.findall(r'\w+', s)) for s in sents]) if sents else 0,
            "std_sent_len": np.std([len(re.findall(r'\w+', s)) for s in sents]) if sents else 0,
            "commas": text.count(','),
            "periods": text.count('.'),
            "excls": text.count('!'),
            "ques": text.count('?'),
            "punct_density": sum([text.count(p) for p in [',', '.', '!', '?', ';']]) / (char_len + 1e-6)
        }
        feats.append(f)
    return pd.DataFrame(feats)


# ============================================================
# 3ï¸�âƒ£ Process-based features
# ============================================================

def make_feats(df):
    df = df.copy()
    df["activity"] = df["activity"].apply(lambda x: "Move" if isinstance(x, str) and x.startswith("Move From") else x)
    out = pd.DataFrame({"id": df["id"].unique()})

    def get_time_features(sub):
        total_time = sub["up_time"].max() - sub["down_time"].min()
        avg_action = sub["action_time"].mean()
        median_action = sub["action_time"].median()
        return pd.Series({"total_time": total_time, "avg_action_time": avg_action, "median_action_time": median_action})

    def get_activity_features(sub):
        counts = sub["activity"].value_counts().to_dict()
        return pd.Series({
            "num_input": counts.get("Input", 0),
            "num_remove": counts.get("Remove/Cut", 0),
            "num_paste": counts.get("Paste", 0),
            "num_replace": counts.get("Replace", 0),
            "num_nonprod": counts.get("Nonproduction", 0),
            "num_move": counts.get("Move", 0)
        })

    def get_text_features(sub):
        final_wc = sub["word_count"].iloc[-1]
        cursor_var = sub["cursor_position"].std()
        delete_ratio = (sub["activity"] == "Remove/Cut").sum() / len(sub)
        return pd.Series({"final_word_count": final_wc, "cursor_var": cursor_var, "delete_ratio": delete_ratio})

    def get_pause_features(sub):
        sub = sub.sort_values("down_time")
        sub["time_diff"] = sub["down_time"].diff()
        pauses = sub["time_diff"][sub["time_diff"] > 2000]
        return pd.Series({"num_pauses": len(pauses), "avg_pause": pauses.mean() if len(pauses) else 0})

    feats = []
    for essay_id, sub in df.groupby("id"):
        f = {"id": essay_id}
        f.update(get_time_features(sub))
        f.update(get_activity_features(sub))
        f.update(get_text_features(sub))
        f.update(get_pause_features(sub))
        feats.append(f)
    out = out.merge(pd.DataFrame(feats), on="id", how="left")

    essay_time = df.groupby("id").apply(lambda x: (x["up_time"].max() - x["down_time"].min()) / 1000).reset_index(name="duration_s")
    out = out.merge(essay_time, on="id", how="left")

    key_counts = df.groupby("id")["event_id"].count().reset_index(name="n_events")
    out = out.merge(key_counts, on="id", how="left")
    out["keys_per_sec"] = out["n_events"] / out["duration_s"].replace(0, np.nan)

    df["up_time_lagged"] = df.groupby("id")["up_time"].shift(1)
    df["pause_s"] = (df["down_time"] - df["up_time_lagged"]).fillna(0) / 1000
    pauses = df.groupby("id")["pause_s"].agg(["mean", "median", "max", "std"]).reset_index()
    pauses.columns = ["id", "pause_mean", "pause_median", "pause_max", "pause_std"]
    out = out.merge(pauses, on="id", how="left")

    action_stats = df.groupby("id")["action_time"].agg(["mean", "std", "max", "sum"]).reset_index()
    action_stats.columns = ["id", "action_mean", "action_std", "action_max", "action_sum"]
    out = out.merge(action_stats, on="id", how="left")

    cursor_stats = df.groupby("id")["cursor_position"].agg(["max", "nunique"]).reset_index()
    cursor_stats.columns = ["id", "cursor_max", "cursor_unique"]
    out = out.merge(cursor_stats, on="id", how="left")

    act_counts = df.groupby(["id", "activity"]).size().unstack(fill_value=0).reset_index()
    act_counts.columns = [f"activity_{c}" if c != "id" else "id" for c in act_counts.columns]
    out = out.merge(act_counts, on="id", how="left")

    out["cursor_per_sec"] = out["cursor_max"] / out["duration_s"]
    out["action_density"] = out["action_sum"] / out["duration_s"]
    out["delete_to_input_ratio"] = out["num_remove"] / (out["num_input"] + 1e-6)
    return out


# ============================================================
# 4ï¸�âƒ£ Load Data & Reconstruct Essays
# ============================================================

train_logs = pd.read_csv("/kaggle/input/linking-writing-processes-to-writing-quality/train_logs.csv")
test_logs = pd.read_csv("/kaggle/input/linking-writing-processes-to-writing-quality/test_logs.csv")
train_scores = pd.read_csv("/kaggle/input/linking-writing-processes-to-writing-quality/train_scores.csv")

print("Train essays:", train_logs["id"].nunique(), " | Test essays:", test_logs["id"].nunique())

train_essays = getEssays(train_logs).reset_index().rename(columns={"index": "id", 0: "essay"})
test_essays = getEssays(test_logs).reset_index().rename(columns={"index": "id", 0: "essay"})

missing_ids = set(train_logs["id"].unique()) - set(train_essays["id"].unique())
train_logs = train_logs[~train_logs["id"].isin(missing_ids)].reset_index(drop=True)
train_scores = train_scores[~train_scores["id"].isin(missing_ids)].reset_index(drop=True)

print(f"Dropped {len(missing_ids)} bad ID(s): {missing_ids}")


# ============================================================
# 5ï¸�âƒ£ Feature Engineering
# ============================================================

train_text_feats = essay_structure_features(train_essays)
test_text_feats = essay_structure_features(test_essays)
train_proc_feats = make_feats(train_logs)
test_proc_feats = make_feats(test_logs)

train_feats = train_proc_feats.merge(train_text_feats, on="id", how="left")
test_feats = test_proc_feats.merge(test_text_feats, on="id", how="left")
train_feats = train_feats.merge(train_scores, on="id", how="left")

na_cols = train_feats.columns[train_feats.isna().any()].tolist()
train_feats = train_feats.drop(columns=na_cols)
test_feats = test_feats.drop(columns=na_cols, errors="ignore")

missing_in_test = set(train_feats.columns) - set(test_feats.columns)
for col in missing_in_test:
    if col != "score":
        test_feats[col] = 0
test_feats = test_feats[[c for c in train_feats.columns if c != "score"]]

train_cols = [c for c in train_feats.columns if c not in ["id", "score"]]
X, y = train_feats[train_cols], train_feats["score"]
X_test = test_feats[train_cols]

X.columns = X.columns.str.replace(r"[\[\]<>]", "", regex=True)
X_test.columns = X_test.columns.str.replace(r"[\[\]<>]", "", regex=True)

# ============================================================
# 6ï¸�âƒ£ Model Training with CV + Tuning
# ============================================================

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# --- Hyperparameter tuning (optional) ---
rf_param_grid = {
    "n_estimators": [200, 300, 400, 500],
    "max_depth": [8, 10, 12, 14, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

rf_random = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_distributions=rf_param_grid,
    n_iter=10,
    cv=3,
    scoring="neg_root_mean_squared_error",
    random_state=42,
    verbose=1,
    n_jobs=-1
)
rf_random.fit(X, y)
best_rf = rf_random.best_estimator_
print("Best RF Params:", rf_random.best_params_)

# --- XGBoost tuning ---
xgb_param_grid = {
    "n_estimators": [300, 400, 600],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [4, 6, 8],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0]
}
xgb_random = RandomizedSearchCV(
    xgb.XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
    param_distributions=xgb_param_grid,
    n_iter=10,
    cv=3,
    scoring="neg_root_mean_squared_error",
    random_state=42,
    verbose=1,
    n_jobs=-1
)
xgb_random.fit(X, y)
best_xgb = xgb_random.best_estimator_
print("Best XGB Params:", xgb_random.best_params_)

# --- LightGBM tuning ---
lgb_param_grid = {
    "n_estimators": [300, 400, 600],
    "learning_rate": [0.01, 0.05, 0.1],
    "num_leaves": [15, 31, 63],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "max_depth": [-1, 6, 10]
}
lgb_random = RandomizedSearchCV(
    lgb.LGBMRegressor(random_state=42, n_jobs=-1),
    param_distributions=lgb_param_grid,
    n_iter=10,
    cv=3,
    scoring="neg_root_mean_squared_error",
    random_state=42,
    verbose=1,
    n_jobs=-1
)
lgb_random.fit(X, y)
best_lgb = lgb_random.best_estimator_
print("Best LGB Params:", lgb_random.best_params_)

# --- Use tuned models ---
models = {"rf": best_rf, "xgb": best_xgb, "lgb": best_lgb}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = pd.DataFrame({"id": train_feats["id"], "rf": 0, "xgb": 0, "lgb": 0})
test_preds = pd.DataFrame({"id": test_feats["id"], "rf": 0, "xgb": 0, "lgb": 0})

for name, model in models.items():
    print(f"\nTraining {name.upper()}...")
    oof = np.zeros(len(X))
    preds = np.zeros(len(X_test))
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        model.fit(X_tr, y_tr)
        oof[val_idx] = model.predict(X_val)
        preds += model.predict(X_test) / kf.n_splits
    oof_preds[name] = oof
    test_preds[name] = preds
    print(f"{name.upper()} CV RMSE: {rmse(y, oof):.4f}")

# ============================================================
# 7ï¸�âƒ£ Feature Importance Correlation
# ============================================================

def plot_feature_importances(models, X):
    importances = {}
    for name, model in models.items():
        if hasattr(model, "feature_importances_"):
            importances[name] = model.feature_importances_
    imp_df = pd.DataFrame(importances, index=X.columns)
    corr = imp_df.corr()
    print("\nğŸ“Š Feature Importance Correlation:\n", corr)
    plt.figure(figsize=(5, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Importance Correlation Between Models")
    plt.show()

plot_feature_importances(models, X)

# ============================================================
# 8ï¸�âƒ£ Ridge Stacking (Level-2)
# ============================================================

meta_train = oof_preds[["rf", "xgb", "lgb"]]
meta_test = test_preds[["rf", "xgb", "lgb"]]

meta_model = RidgeCV(alphas=np.logspace(-3, 3, 20))
meta_model.fit(meta_train, y)
final_oof = meta_model.predict(meta_train)
final_test = meta_model.predict(meta_test)

print("\nFinal Stacked RMSE:", rmse(y, final_oof))
print("Meta Weights:", meta_model.coef_)

# ============================================================
# 9ï¸�âƒ£ Save Submission
# ============================================================

submission = pd.DataFrame({"id": test_feats["id"], "score": np.ravel(final_test)})
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("\nâœ… submission.csv saved successfully!")
print(submission.head())


