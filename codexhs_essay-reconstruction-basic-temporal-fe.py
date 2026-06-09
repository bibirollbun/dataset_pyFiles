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


import pandas as pd
import numpy as np
import re, warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import nltk


import numpy as np
import pandas as pd
import re 

def getEssays(df):
    # Copy required columns
    textInputDf = df[['id', 'activity', 'cursor_position', 'text_change']].copy()
    
    # Get rid of text inputs that make no change
    # Note: Shift was unpreditcable so ignored
    textInputDf = textInputDf[textInputDf.activity != 'Nonproduction']

    # Get how much each Id there is
    valCountsArr = textInputDf['id'].value_counts(sort=False).values

    # Holds the final index of the previous Id
    lastIndex = 0

    # Holds all the essays
    essaySeries = pd.Series()

    # Fills essay series with essays
    for index, valCount in enumerate(valCountsArr):

        # Indexes down_time at current Id
        currTextInput = textInputDf[['activity', 'cursor_position', 'text_change']].iloc[lastIndex : lastIndex + valCount]

        # Update the last index
        lastIndex += valCount

        # Where the essay content will be stored
        essayText = ""

        
        # Produces the essay
        for Input in currTextInput.values:
            
            # Input[0] = activity
            # Input[2] = cursor_position
            # Input[3] = text_change
            
            # If activity = Replace
            if Input[0] == 'Replace':
                # splits text_change at ' => '
                replaceTxt = Input[2].split(' => ')
                
                # DONT TOUCH
                essayText = essayText[:Input[1] - len(replaceTxt[1])] + replaceTxt[1] + essayText[Input[1] - len(replaceTxt[1]) + len(replaceTxt[0]):]
                continue

                
            # If activity = Paste    
            if Input[0] == 'Paste':
                # DONT TOUCH
                essayText = essayText[:Input[1] - len(Input[2])] + Input[2] + essayText[Input[1] - len(Input[2]):]
                continue

                
            # If activity = Remove/Cut
            if Input[0] == 'Remove/Cut':
                # DONT TOUCH
                essayText = essayText[:Input[1]] + essayText[Input[1] + len(Input[2]):]
                continue

                
            # If activity = Move...
            if "M" in Input[0]:
                # Gets rid of the "Move from to" text
                croppedTxt = Input[0][10:]
                
                # Splits cropped text by ' To '
                splitTxt = croppedTxt.split(' To ')
                
                # Splits split text again by ', ' for each item
                valueArr = [item.split(', ') for item in splitTxt]
                
                # Move from [2, 4] To [5, 7] = (2, 4, 5, 7)
                moveData = (int(valueArr[0][0][1:]), int(valueArr[0][1][:-1]), int(valueArr[1][0][1:]), int(valueArr[1][1][:-1]))

                # Skip if someone manages to activiate this by moving to same place
                if moveData[0] != moveData[2]:
                    # Check if they move text forward in essay (they are different)
                    if moveData[0] < moveData[2]:
                        # DONT TOUCH
                        essayText = essayText[:moveData[0]] + essayText[moveData[1]:moveData[3]] + essayText[moveData[0]:moveData[1]] + essayText[moveData[3]:]
                    else:
                        # DONT TOUCH
                        essayText = essayText[:moveData[2]] + essayText[moveData[0]:moveData[1]] + essayText[moveData[2]:moveData[0]] + essayText[moveData[1]:]
                continue
                
                
            # If just input
            # DONT TOUCH
            essayText = essayText[:Input[1] - len(Input[2])] + Input[2] + essayText[Input[1] - len(Input[2]):]

            
        # Sets essay at index  
        essaySeries[index] = essayText
     
    
    # Sets essay series index to the ids
    essaySeries.index =  textInputDf['id'].unique()
    
    
    # Returns the essay series
    return essaySeries

# Cap abnormal essays > 35 minutes (time diff > 35*60*1000 ms, allow some leeway)
def cap_long_essays(df, max_minutes=35):
    """
    For each essay ID:
    - Compute elapsed time since the essay started.
    - Keep only keystrokes within the first `max_minutes`.
    - Does NOT drop entire essays — only late keystrokes.
    """
    cutoff = max_minutes * 60 * 1000  # minutes → milliseconds
    
    # Align each essay's timeline to start at zero
    df["elapsed"] = df.groupby("id")["down_time"].transform(lambda x: x - x.min())
    
    # Keep only events within allowed window
    df = df[df["elapsed"] <= cutoff].copy()
    
    # Drop helper column to keep dataset clean
    df = df.drop(columns="elapsed")
    
    return df

# Fix backward down_time issues
def fix_backward(df):
    df["down_time"] = df.groupby("id")["down_time"].transform(lambda x: np.maximum.accumulate(x))
    df["up_time"]   = df.groupby("id")["up_time"].transform(lambda x: np.maximum.accumulate(x))
    return df

def essay_structure_features(df):
    feats = []
    for _, row in df.iterrows():
        text = row["essay"]
        sents = re.split(r"[.!?]", text)
        sents = [s.strip() for s in sents if len(s.strip())>0]
        words = re.findall(r"q+", text)
        word_lens = [len(w) for w in words]
        char_len = len(text)

        f = {
            "id": row["id"],
            "char_len": char_len,
            "n_words": len(words),
            "n_sent": len(sents),
            "avg_word_len": np.mean(word_lens) if word_lens else 0,
            "std_word_len": np.std(word_lens) if word_lens else 0,
            "avg_sent_len": np.mean([len(re.findall(r'q+', s)) for s in sents]) if sents else 0,
            "std_sent_len": np.std([len(re.findall(r'q+', s)) for s in sents]) if sents else 0,
            "commas": text.count(','),
            "periods": text.count('.'),
            "excls": text.count('!'),
            "ques": text.count('?'),
            "punct_density": sum([text.count(p) for p in [',','.','!','?',';']])/(char_len+1e-6)
        }
        feats.append(f)
    return pd.DataFrame(feats)

def make_feats(df):
    # --- 1️⃣ Simplify 'Move From ... To ...' activities early ---
    df = df.copy()
    df["activity"] = df["activity"].apply(
        lambda x: "Move" if isinstance(x, str) and x.startswith("Move From") else x
    )

    out = pd.DataFrame({"id": df["id"].unique()})

    # =============================================================
    # 2️⃣ Original "simple" high-performing features (per essay)
    # =============================================================

    def get_time_features(sub):
        total_time = sub["up_time"].max() - sub["down_time"].min()
        avg_action = sub["action_time"].mean()
        median_action = sub["action_time"].median()
        return pd.Series({
            "total_time": total_time,
            "avg_action_time": avg_action,
            "median_action_time": median_action,
        })

    def get_activity_features(sub):
        counts = sub["activity"].value_counts().to_dict()
        return pd.Series({
            "num_input": counts.get("Input", 0),
            "num_remove": counts.get("Remove/Cut", 0),
            "num_paste": counts.get("Paste", 0),
            "num_replace": counts.get("Replace", 0),
            "num_nonprod": counts.get("Nonproduction", 0),
            "num_move": counts.get("Move", 0),
        })

    def get_text_features(sub):
        final_wc = sub["word_count"].iloc[-1]
        cursor_var = sub["cursor_position"].std()
        delete_ratio = (sub["activity"] == "Remove/Cut").sum() / len(sub)
        return pd.Series({
            "final_word_count": final_wc,
            "cursor_var": cursor_var,
            "delete_ratio": delete_ratio
        })

    def get_pause_features(sub):
        sub = sub.sort_values("down_time")
        sub["time_diff"] = sub["down_time"].diff()
        pauses = sub["time_diff"][sub["time_diff"] > 2000]
        return pd.Series({
            "num_pauses": len(pauses),
            "avg_pause": pauses.mean() if len(pauses) else 0
        })

    # =============================================================
    # 3️⃣ Compute those simple features per essay id
    # =============================================================
    simple_feats = []
    for essay_id, sub in df.groupby("id"):
        f = {"id": essay_id}
        f.update(get_time_features(sub))
        f.update(get_activity_features(sub))
        f.update(get_text_features(sub))
        f.update(get_pause_features(sub))
        simple_feats.append(f)
    simple_feats = pd.DataFrame(simple_feats)
    out = out.merge(simple_feats, on="id", how="left")

    # =============================================================
    # 4️⃣ Add richer aggregate / timing features
    # =============================================================

    # ---- Basic durations (seconds) ----
    essay_time = df.groupby("id").apply(
        lambda x: (x["up_time"].max() - x["down_time"].min()) / 1000
    ).reset_index(name="duration_s")
    out = out.merge(essay_time, on="id", how="left")

    # ---- Key speed metrics ----
    key_counts = df.groupby("id")["event_id"].count().reset_index(name="n_events")
    out = out.merge(key_counts, on="id", how="left")
    out["keys_per_sec"] = out["n_events"] / out["duration_s"].replace(0, np.nan)

    # ---- Pause / latency features ----
    df["up_time_lagged"] = df.groupby("id")["up_time"].shift(1)
    df["pause_s"] = (df["down_time"] - df["up_time_lagged"]).fillna(0) / 1000
    pauses = df.groupby("id")["pause_s"].agg(["mean", "median", "max", "std"]).reset_index()
    pauses.columns = ["id", "pause_mean", "pause_median", "pause_max", "pause_std"]
    out = out.merge(pauses, on="id", how="left")

    # ---- Action time stats ----
    action_stats = df.groupby("id")["action_time"].agg(["mean", "std", "max", "sum"]).reset_index()
    action_stats.columns = ["id", "action_mean", "action_std", "action_max", "action_sum"]
    out = out.merge(action_stats, on="id", how="left")

    # ---- Cursor dynamics ----
    cursor_stats = df.groupby("id")["cursor_position"].agg(["max", "nunique"]).reset_index()
    cursor_stats.columns = ["id", "cursor_max", "cursor_unique"]
    out = out.merge(cursor_stats, on="id", how="left")

    # ---- Simplified activity counts (no explosion) ----
    act_counts = (
        df.groupby(["id", "activity"])
          .size()
          .unstack(fill_value=0)
          .reset_index()
    )
    act_counts.columns = [f"activity_{c}" if c != "id" else "id" for c in act_counts.columns]
    out = out.merge(act_counts, on="id", how="left")

    # ---- Ratios ----
    out["cursor_per_sec"] = out["cursor_max"] / out["duration_s"]
    out["action_density"] = out["action_sum"] / out["duration_s"]
    out["delete_to_input_ratio"] = (
        out["num_remove"] / (out["num_input"] + 1e-6)
        if "num_remove" in out.columns and "num_input" in out.columns
        else 0
    )

    return out



train_logs = pd.read_csv("/kaggle/input/linking-writing-processes-to-writing-quality/train_logs.csv")
test_logs  = pd.read_csv("/kaggle/input/linking-writing-processes-to-writing-quality/test_logs.csv")
train_scores = pd.read_csv("/kaggle/input/linking-writing-processes-to-writing-quality/train_scores.csv")


print("Train essays:", train_logs["id"].nunique(), " | Test essays:", test_logs["id"].nunique())



train_essays = getEssays(train_logs).reset_index()
train_essays.columns = ["id", "essay"]
test_essays = getEssays(test_logs).reset_index()
test_essays.columns = ["id", "essay"]


print(train_essays["id"].nunique())
print(test_essays["id"].nunique())


# original full set of IDs from train_logs
orig_ids = set(train_logs["id"].unique())

# IDs from reconstructed essays (after loading npz)
recon_ids = set(train_essays["id"].unique())

# which ID(s) are missing in reconstructed
missing_ids = orig_ids - recon_ids
extra_ids   = recon_ids - orig_ids   # should be empty, but good to check

print("Missing IDs:", missing_ids)
print("Extra IDs (if any):", extra_ids)
print("Count check → original:", len(orig_ids), " | reconstructed:", len(recon_ids))


# ID to drop
bad_ids = list(missing_ids)

# Drop from train_logs and train_scores
train_logs = train_logs[~train_logs["id"].isin(bad_ids)].reset_index(drop=True)
train_scores = train_scores[~train_scores["id"].isin(bad_ids)].reset_index(drop=True)

# Confirm
print(f"Dropped ID: {bad_ids}")
print("Remaining IDs → train_logs:", train_logs['id'].nunique(),
      "| train_scores:", train_scores['id'].nunique())


nltk.download('punkt')


train_text_feats = essay_structure_features(train_essays)
test_text_feats  = essay_structure_features(test_essays)


## Visualize train text feats

print("Train text feats: \n", train_text_feats.head)


train_proc_feats = make_feats(train_logs)
test_proc_feats  = make_feats(test_logs)


## Visualize train proc feats

print("Train proc feats:\n", train_proc_feats)


train_feats = train_proc_feats.merge(train_text_feats, on="id", how="left")
test_feats  = test_proc_feats.merge(test_text_feats, on="id", how="left")


train_feats = train_feats.merge(train_scores, on="id", how="left")

# Drop any feature with NaN in train
na_cols = train_feats.columns[train_feats.isna().any()].tolist()
print("Number of na cols:", len(na_cols))
print("The columns are:", na_cols)

train_feats = train_feats.drop(columns=na_cols)
test_feats  = test_feats.drop(columns=na_cols)


# Make sure both dataframes have the same set of columns
missing_in_test = set(train_feats.columns) - set(test_feats.columns)
missing_in_train = set(test_feats.columns) - set(train_feats.columns)

# Add missing columns filled with 0
for col in missing_in_test:
    if col not in ["score"]:  # don't add target to test
        test_feats[col] = 0

for col in missing_in_train:
    train_feats[col] = 0

# Reorder columns to match
common_cols = [c for c in train_feats.columns if c not in ["score"]]  # 'id' already included here
test_feats = test_feats[common_cols]


target_col = ['score']
drop_cols = ['id']
train_cols = [c for c in train_feats.columns if c not in target_col + drop_cols]


## Visualize train cols, train_feats, and test_feats

print("Train cols: \n", train_cols)
print("Train feats: \n", train_feats)
print("Test feats: \n", test_feats)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


X, y = train_feats[train_cols], train_feats[target_col]
X_test = test_feats[train_cols]

# Clean up feature names for all models
X.columns = X.columns.str.replace(r"[\[\]<>]", "", regex=True)
X_test.columns = X_test.columns.str.replace(r"[\[\]<>]", "", regex=True)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = pd.DataFrame({"id": train_feats["id"], "rf":0, "xgb":0, "lgb":0})
test_preds = pd.DataFrame(
    {
        "rf": 0.0,
        "xgb": 0.0,
        "lgb": 0.0,
    },
    index=test_feats["id"]
).reset_index(names="id")



print("test feats id:\n", test_feats)


# --- Random Forest ---
rf_params = {"n_estimators":300, "max_depth":12, "random_state":42, "n_jobs":-1}
rf_model = RandomForestRegressor(**rf_params)

# --- XGBoost ---
xgb_params = {"n_estimators":400, "learning_rate":0.05, "max_depth":6, "subsample":0.8, "colsample_bytree":0.8, "random_state":42}
xgb_model = xgb.XGBRegressor(**xgb_params)

# --- LightGBM ---
lgb_params = {"n_estimators":400, "learning_rate":0.05, "num_leaves":31, "subsample":0.8, "colsample_bytree":0.8, "random_state":42}
lgb_model = lgb.LGBMRegressor(**lgb_params)


models = {"rf":rf_model, "xgb":xgb_model, "lgb":lgb_model}

for name, model in models.items():
    print(f"\nTraining {name.upper()}...")
    oof = np.zeros(len(X))
    preds = np.zeros(len(X_test))
    for fold,(tr_idx,val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        model.fit(X_tr, y_tr)
        oof[val_idx] = model.predict(X_val)
        preds += model.predict(X_test)/kf.n_splits
    oof_preds[name] = oof
    test_preds[name] = preds
    print(f"{name} RMSE:", rmse(y, oof))


print(test_preds)


meta_train = oof_preds[["rf","xgb","lgb"]]
meta_test  = test_preds[["rf","xgb","lgb"]]

meta_model = RidgeCV(alphas=np.logspace(-3,3,20))
meta_model.fit(meta_train, y)
final_oof = meta_model.predict(meta_train)
final_test = meta_model.predict(meta_test)

print("\nFinal Stacked RMSE:", rmse(y, final_oof))


print("✅ Shapes check:")
print("test_feats['id']:", np.shape(test_feats["id"].values))
print("final_test:", np.shape(final_test))


# Check shape before flattening
print("Before flattening:", np.shape(final_test))

# Flatten if needed
final_test = np.ravel(final_test)

# Confirm
print("After flattening:", np.shape(final_test))


# Save predictions
submission = pd.DataFrame({"id": test_feats["id"], "score": final_test})
submission.to_csv('/kaggle/working/submission.csv', index=False)
print(submission.head())

