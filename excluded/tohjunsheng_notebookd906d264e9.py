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
# 1. IMPORTS & GLOBAL SETTINGS
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import re
import string
import pickle
import numpy as np
import pandas as pd

from scipy.stats import skew
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

import polars as pl

from lightgbm import LGBMRegressor, log_evaluation
from catboost import CatBoostRegressor
import xgboost as xgb


# ============================================================
# 2. PURE-PYTHON READABILITY METRICS (OFFLINE)
# ============================================================

def count_syllables(word):
    word = word.lower()
    vowels = "aeiouy"
    count = 0
    prev = False
    for c in word:
        if c in vowels:
            if not prev:
                count += 1
            prev = True
        else:
            prev = False
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)

def automated_readability_index(text):
    words = text.split()
    if len(words) == 0:
        return 0
    chars = sum(len(w) for w in words)
    sents = max(1, len(re.split(r"[.!?]", text)) - 1)
    return 4.71*(chars/len(words)) + 0.5*(len(words)/sents) - 21.43

def coleman_liau_index(text):
    words = text.split()
    if len(words) == 0:
        return 0
    letters = sum(len(re.sub(r'[^A-Za-z]', '', w)) for w in words)
    sents = max(1, len(re.split(r"[.!?]", text)) - 1)
    L = letters/len(words)*100
    S = sents/len(words)*100
    return 0.0588*L - 0.296*S - 15.8

def mcalpine_eflaw(text):
    words = text.split()
    if not words:
        return 0
    polys = sum(1 for w in words if count_syllables(w) >= 3)
    return polys/len(words)*100


# ============================================================
# 3. TEXT RECONSTRUCTION
# ============================================================

def revealing_text(df):
    USER_ID = df["id"].iloc[0]
    textInputDf = df[df.activity != "Nonproduction"][["activity","cursor_position","text_change"]]
    essay = ""

    for act, cur, chg in textInputDf.values:

        if act == "Replace":
            old, new = chg.split(" => ")
            essay = essay[:cur-len(new)] + new + essay[cur-len(new)+len(old):]
            continue

        if act == "Paste":
            essay = essay[:cur-len(chg)] + chg + essay[cur-len(chg):]
            continue

        if act == "Remove/Cut":
            essay = essay[:cur] + essay[cur+len(chg):]
            continue

        if "M" in act:
            cropped = act[10:]
            pairs = [p.split(", ") for p in cropped.split(" To ")]
            s1,e1 = int(pairs[0][0][1:]), int(pairs[0][1][:-1])
            s2,e2 = int(pairs[1][0][1:]), int(pairs[1][1][:-1])
            if s1 < s2:
                essay = essay[:s1] + essay[e1:e2] + essay[s1:e1] + essay[e2:]
            else:
                essay = essay[:s2] + essay[s1:e1] + essay[s2:s1] + essay[e1:]
            continue

        essay = essay[:cur-len(chg)] + chg + essay[cur-len(chg):]

    return USER_ID, essay


# ============================================================
# 4. TEXT FEATURE ENGINEERING
# ============================================================

def standardize_text(txt):
    txt = re.sub(r"\t","",txt)
    txt = re.sub(r"\n +","\n",txt)
    txt = re.sub(r" +\n","\n",txt)
    txt = re.sub(r"\n{2,}","\n",txt)
    txt = re.sub(r" {2,}"," ",txt)
    return txt.strip()

def get_text_chunk_features(df):
    df["text_length"] = df["revealed_text"].apply(len)
    df["num_newlines"] = df["revealed_text"].apply(lambda x: x.count("\n"))

    df["automated_readability_index"] = df["revealed_text"].apply(automated_readability_index)
    df["mcalpine_eflaw"] = df["revealed_text"].apply(mcalpine_eflaw)
    df["coleman_liau"] = df["revealed_text"].apply(coleman_liau_index)

    df["repetitiveness"] = df["revealed_text"].apply(lambda x: x.count("q")/max(1,len(x)))
    df["word_count"] = df["revealed_text"].apply(lambda x: len(x.split()))

    df["avg_word_length"] = df["revealed_text"].apply(
        lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0
    )
    df["word_lexical_diversity"] = df["revealed_text"].apply(
        lambda x: len(set(x.split()))/len(x.split()) if x.split() else 0
    )

    return df

def word_feats(df):
    df2 = df.copy()
    df2["word"] = df2["revealed_text"].apply(lambda x: re.split(r"[ \n.!?]",x))
    df2 = df2.explode("word")
    df2["word_len"] = df2["word"].apply(len)
    df2 = df2[df2["word_len"]>0]

    agg = df2.groupby("id")["word_len"].agg(
        ["count","min","max","first","last","median","sum","std"]
    )
    agg.columns = [f"word_len_{c}" for c in agg.columns]
    return agg.reset_index()

def sent_feats(df):
    df2 = df.copy()
    df2["sent"] = df2["revealed_text"].apply(lambda x: re.split(r"[.!?]",x))
    df2 = df2.explode("sent")
    df2["sent"] = df2["sent"].str.strip()
    df2["sent_len"] = df2["sent"].apply(len)
    df2["sent_word_count"] = df2["sent"].apply(lambda x: len(x.split()))

    df2 = df2[df2["sent_len"]>0]

    a1 = df2.groupby("id")["sent_len"].agg(
        ["count","min","max","first","last","median","sum","std"]
    )
    a2 = df2.groupby("id")["sent_word_count"].agg(
        ["count","min","max","first","last","median","sum","std"]
    )

    out = pd.concat([a1,a2],axis=1)
    out.columns = (
        [f"sent_len_{c}" for c in a1.columns] +
        [f"sent_word_count_{c}" for c in a2.columns]
    )
    out = out.reset_index()
    out.rename(columns={"sent_len_count":"sent_count"}, inplace=True)
    out.drop(columns=["sent_word_count_count"], inplace=True)
    return out

def parag_feats(df):
    df2 = df.copy()
    df2["paragraph"] = df2["revealed_text"].apply(lambda x: x.split("\n"))
    df2 = df2.explode("paragraph")

    df2["paragraph_len"] = df2["paragraph"].apply(len)
    df2["paragraph_word_count"] = df2["paragraph"].apply(lambda x: len(x.split()))
    df2["paragraph_sent_count"] = df2["paragraph"].apply(lambda x: len(re.split(r"[.!?]",x)))

    df2 = df2[df2["paragraph_len"]>2]

    a1 = df2.groupby("id")["paragraph_len"].agg(
        ["count","min","max","first","last","median","sum","std"]
    )
    a2 = df2.groupby("id")["paragraph_word_count"].agg(
        ["count","min","max","first","last","median","sum","std"]
    )
    a3 = df2.groupby("id")["paragraph_sent_count"].agg(
        ["count","min","max","first","last","median","sum","std"]
    )

    out = pd.concat([a1,a2,a3],axis=1)
    cols = []
    for i,c in enumerate(out.columns):
        if i<8: cols.append("paragraph_len_"+c)
        elif i<16: cols.append("paragraph_word_count_"+c)
        else: cols.append("paragraph_sent_count_"+c)
    out.columns = cols

    out = out.reset_index()
    out.rename(columns={"paragraph_len_count":"paragraph_count"}, inplace=True)
    out.drop(columns=["paragraph_word_count_count","paragraph_sent_count_count"], inplace=True)
    return out

def TextProcessor(df):
    df = df.copy()
    df.loc[df["revealed_text"].str.replace(" ","")=="","revealed_text"] = "q"
    df["revealed_text"] = df["revealed_text"].apply(standardize_text)

    df = get_text_chunk_features(df)
    df = df.merge(word_feats(df), on="id", how="left")
    df = df.merge(sent_feats(df), on="id", how="left")
    df = df.merge(parag_feats(df), on="id", how="left")
    return df


# ============================================================
# 5. RAW KEYSTROKE FEATURE ENGINEERING
# ============================================================

def count_by_values(df, colname, values):
    fts = df.select(pl.col("id").unique(maintain_order=True))
    for i,v in enumerate(values):
        tmp = df.group_by("id").agg(
            pl.col(colname).is_in([v]).sum().alias(f"{colname}_{i}_cnt")
        )
        fts = fts.join(tmp, on="id")
    return fts

def event_count_feats(df):
    acts = ['Input','Remove/Cut','Nonproduction','Replace','Paste']
    events = ['q','Space','Backspace','Shift','ArrowRight','Leftclick','ArrowLeft',
        '.',',','ArrowDown','ArrowUp','Enter','CapsLock',"'",'Delete','Unidentified']
    tchanges = ['q',' ','.',',','\n',"'",'"','-','?',';','=','/','\\',':']

    out = count_by_values(df,"activity",acts)
    out = out.join(count_by_values(df,"text_change",tchanges), on="id")
    out = out.join(count_by_values(df,"down_event",events), on="id")
    return out.to_pandas()

# === FIXED FOR POLARS (no .suffix) ===
def num_colstat_feats(df):
    nums = ['down_time','up_time','action_time','cursor_position','word_count','event_id']

    agg_exprs = [
        pl.col("action_time").sum().alias("action_time_sum")
    ]

    for col in nums:
        agg_exprs.append(pl.col(col).std().alias(f"{col}_std"))
        agg_exprs.append(pl.col(col).median().alias(f"{col}_median"))
        agg_exprs.append(pl.col(col).min().alias(f"{col}_min"))
        agg_exprs.append(pl.col(col).max().alias(f"{col}_max"))

    agg = df.group_by("id").agg(agg_exprs)
    return agg.to_pandas()

def cat_colstat_feats(df):
    agg = df.group_by("id").agg(
        pl.n_unique(["activity","down_event","up_event","text_change"])
    )
    return agg.to_pandas()

def pause_stat_aggregator(df, prefix):
    return df.group_by("id").agg([
        pl.col("time_diff").max().alias(f"{prefix}_max_pause_time"),
        pl.col("time_diff").median().alias(f"{prefix}_median_pause_time"),
        pl.col("time_diff").mean().alias(f"{prefix}_mean_pause_time"),
        pl.col("time_diff").min().alias(f"{prefix}_min_pause_time"),
        pl.col("time_diff").std().alias(f"{prefix}_std_pause_time"),
        pl.col("time_diff").sum().alias(f"{prefix}_total_pause_time"),

        pl.col("time_diff").filter((pl.col("time_diff")>0.5)&(pl.col("time_diff")<=1)).count().alias(f"{prefix}_pauses_half_sec"),
        pl.col("time_diff").filter((pl.col("time_diff")>1)&(pl.col("time_diff")<=2)).count().alias(f"{prefix}_pauses_1_sec"),
        pl.col("time_diff").filter((pl.col("time_diff")>2)&(pl.col("time_diff")<=3)).count().alias(f"{prefix}_pauses_2_sec"),
        pl.col("time_diff").filter(pl.col("time_diff")>3).count().alias(f"{prefix}_pauses_3_sec"),
    ])

# === FIXED FOR POLARS (no .cumsum)
def idle_time_feats(df):
    temp = df.with_columns(
        pl.col("up_time").shift().over("id").alias("up_time_lagged")
    )
    temp = temp.with_columns(
        ((pl.col("down_time") - pl.col("up_time_lagged")).abs()/1000)
        .fill_null(0)
        .alias("time_diff")
    )

    temp = temp.with_columns(
        (pl.col("up_event")=="Space").alias("is_space"),
        (pl.col("up_event")==".").alias("is_dot"),
        (pl.col("up_event")=="Enter").alias("is_enter")
    )

    temp = temp.with_columns(
        pl.col("is_space").cum_sum().over("id").alias("word_id"),
        pl.col("is_dot").cum_sum().over("id").alias("sentence_id"),
        pl.col("is_enter").cum_sum().over("id").alias("paragraph_id")
    )

    temp2 = temp.filter(pl.col("activity").is_in(["Input","Remove/Cut"]))

    iw = pause_stat_aggregator(temp2,"iw")

    bww = temp2.group_by(["id","word_id"]).agg(pl.col("time_diff").first())
    bww = pause_stat_aggregator(bww,"bww")

    bws = temp2.group_by(["id","sentence_id"]).agg(pl.col("time_diff").first())
    bws = pause_stat_aggregator(bws,"bws")

    bwp = temp2.group_by(["id","paragraph_id"]).agg(pl.col("time_diff").first())
    bwp = pause_stat_aggregator(bwp,"bwp")

    return (
        iw.join(bww,on="id")
          .join(bws,on="id")
          .join(bwp,on="id")
          .to_pandas()
    )

# === FIXED FOR POLARS (no .suffix)
def burst_features(df, burst_type):
    temp = df.with_columns(
        pl.col("up_time").shift().over("id").alias("up_time_lagged")
    )
    temp = temp.with_columns(
        ((pl.col("down_time") - pl.col("up_time_lagged")).abs()/1000)
        .fill_null(0)
        .alias("time_diff")
    )

    if burst_type=="p":
        temp = temp.with_columns(pl.col("activity").is_in(["Input"]))
    else:
        temp = temp.with_columns(pl.col("activity").is_in(["Remove/Cut"]))

    temp = temp.with_columns(
        (pl.col("action_time")/1000).alias("action_time_s"),
        (pl.col("up_time")/1000).alias("up_time_s"),
        pl.when(pl.col("activity")).then(pl.col("activity").rle_id()).alias(f"{burst_type}_burst_group")
    ).drop_nulls()

    g = temp.group_by(["id",f"{burst_type}_burst_group"]).agg([
        pl.col("activity").count().alias(f"{burst_type}_burst_group_keypress_count"),
        pl.col("action_time_s").sum().alias(f"{burst_type}_burst_group_timespent"),
        pl.col("action_time_s").mean().alias(f"{burst_type}_burst_keypress_timespent_mean"),
        pl.col("action_time_s").std().alias(f"{burst_type}_burst_keypress_timespent_std"),
        pl.col("up_time_s").min().alias(f"{burst_type}_burst_keypress_timestamp_first"),
        pl.col("up_time_s").max().alias(f"{burst_type}_burst_keypress_timestamp_last"),
    ])

    out = g.group_by("id").agg([
        pl.col(f"{burst_type}_burst_group_keypress_count").sum().alias(f"{burst_type}_burst_keypress_count_sum"),
        pl.col(f"{burst_type}_burst_group_keypress_count").mean().alias(f"{burst_type}_burst_keypress_count_mean"),
        pl.col(f"{burst_type}_burst_group_keypress_count").std().alias(f"{burst_type}_burst_keypress_count_std"),
        pl.col(f"{burst_type}_burst_group_keypress_count").max().alias(f"{burst_type}_burst_keypress_count_max"),

        pl.col(f"{burst_type}_burst_group_timespent").sum().alias(f"{burst_type}_burst_timespent_sum"),
        pl.col(f"{burst_type}_burst_group_timespent").mean().alias(f"{burst_type}_burst_timespent_mean"),
        pl.col(f"{burst_type}_burst_group_timespent").std().alias(f"{burst_type}_burst_timespent_std"),
        pl.col(f"{burst_type}_burst_group_timespent").max().alias(f"{burst_type}_burst_timespent_max"),

        pl.col(f"{burst_type}_burst_keypress_timespent_mean").mean().alias(f"{burst_type}_burst_keypress_timespent_mean"),
        pl.col(f"{burst_type}_burst_keypress_timespent_std").mean().alias(f"{burst_type}_burst_keypress_timespent_std"),

        pl.col(f"{burst_type}_burst_keypress_timestamp_first").min().alias(f"{burst_type}_burst_keypress_timestamp_first"),
        pl.col(f"{burst_type}_burst_keypress_timestamp_last").max().alias(f"{burst_type}_burst_keypress_timestamp_last"),
    ])

    return out.to_pandas()

def get_keys_pressed_per_second(raw_df):
    a = raw_df[raw_df["activity"].isin(["Input","Remove/Cut"])].groupby("id").agg(
        keys_pressed=("event_id","count")
    ).reset_index()

    b = raw_df.groupby("id").agg(
        min_down_time=("down_time","min"),
        max_up_time=("up_time","max")
    ).reset_index()

    out = a.merge(b,on="id")
    out["keys_per_second"] = out["keys_pressed"]/((out["max_up_time"]-out["min_down_time"])/1000)
    return out[["id","keys_per_second"]]

def RawProcessor(raw_df):
    raw_pl = pl.from_pandas(raw_df)

    print("Creating kpps features...")
    feat = get_keys_pressed_per_second(raw_df)

    print("Event counts...")
    feat = feat.merge(event_count_feats(raw_pl), on="id", how="left")

    print("Numeric/categorical stats...")
    feat = feat.merge(num_colstat_feats(raw_pl), on="id", how="left")
    feat = feat.merge(cat_colstat_feats(raw_pl), on="id", how="left")

    print("Pause features...")
    feat = feat.merge(idle_time_feats(raw_pl), on="id", how="left")

    print("Burst features...")
    feat = feat.merge(burst_features(raw_pl,"p"), on="id", how="left")
    feat = feat.merge(burst_features(raw_pl,"r"), on="id", how="left")

    feat["p_bursts_timeratio"] = feat["p_burst_timespent_sum"]/(feat["up_time_max"]/1000)
    feat["r_bursts_timeratio"] = feat["r_burst_timespent_sum"]/(feat["up_time_max"]/1000)
    feat["action_timeratio"] = feat["action_time_sum"]/feat["up_time_max"]
    feat["pause_timeratio"] = feat["iw_total_pause_time"]/(feat["up_time_max"]/1000)
    feat["pausecount_timeratio"] = feat["iw_pauses_2_sec"]/(feat["up_time_max"]/1000)

    feat["word_time_ratio"] = feat["word_count_max"]/(feat["up_time_max"]/1000)
    feat["word_event_ratio"] = feat["word_count_max"]/(feat["up_time_max"]/1000)
    feat["event_time_ratio"] = feat["event_id_max"]/(feat["up_time_max"]/1000)

    return feat


# ============================================================
# 6. BUILD FULL TRAINING DATA (KAGGLE PATHS)
# ============================================================

print("Loading data...")

train_scores = pd.read_csv('/kaggle/input/linking-writing-processes-to-writing-quality/train_scores.csv')
raw_df = pd.read_csv('/kaggle/input/linking-writing-processes-to-writing-quality/train_logs.csv')

print("Extracting raw keystroke features...")
raw_feats = RawProcessor(raw_df)

print("Reconstructing essays...")
reveal = raw_df.groupby("id").apply(revealing_text)
df = pd.DataFrame(reveal.tolist(), columns=["id","revealed_text"])
df = df.merge(train_scores, on="id")

print("Extracting text features...")
df = TextProcessor(df)

df = df.merge(raw_feats, on="id")
df["text_length_timeratio"] = df["text_length"]/(df["up_time_max"]/1000)

feature_cols = df.drop(["id","revealed_text","score"], axis=1).columns
label = "score"

# ============================================================
# 6.5 ADD TFIDF + SVD EMBEDDINGS  (VERY IMPORTANT BOOST)
# ============================================================

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

print("Building TF-IDF matrix...")

tfidf = TfidfVectorizer(
    max_features=20000,         # safe for Kaggle
    ngram_range=(1, 2),
    min_df=2
)

X_tfidf = tfidf.fit_transform(df["revealed_text"])

# SVD reduces 20k → 120 features
svd_dim = 120
print(f"Performing SVD → {svd_dim} components...")

svd = TruncatedSVD(n_components=svd_dim, random_state=42)
X_svd = svd.fit_transform(X_tfidf)

# Attach SVD features to df
for i in range(svd_dim):
    df[f"svd_{i}"] = X_svd[:, i]

print("TF-IDF + SVD added:", svd_dim, "features")

# Update feature columns
feature_cols = df.drop(["id","revealed_text","score"], axis=1).columns



# ============================================================
# 7. MODEL TRAINING (LGBM / XGB / CATBOOST)
# ============================================================

rmse = lambda y,p: mean_squared_error(y,p,squared=False)

models_to_ensemble = ["lgbm","xgboost","catboost"]
models = {m: [] for m in models_to_ensemble}
oof_df = pd.DataFrame()

params = {
    "lgbm": {
        "reg_alpha":1.0894,"reg_lambda":6.2909,"colsample_bytree":0.6218,
        "subsample":0.9579,"learning_rate":0.0027,"max_depth":8,
        "num_leaves":947,"min_child_samples":57,"n_estimators":2500,
        "metric":"rmse","random_state":42,"verbosity":-1,"force_col_wise":True
    },
    "xgboost":{
        "max_depth":2,"learning_rate":0.00998,"n_estimators":1000,
        "min_child_weight":17,"gamma":0.1288,"subsample":0.5078,
        "colsample_bytree":0.735,"reg_alpha":0.6709,"reg_lambda":0.0681,
        "random_state":1,"tree_method":"hist"
    },
    "catboost":{
        "learning_rate":0.0249,"depth":5,"l2_leaf_reg":3.71,
        "subsample":0.185,"colsample_bylevel":0.655,
        "min_data_in_leaf":93,"iterations":1000,
        "random_state":1,"silent":True,"use_best_model":False
    }
}

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
splits = list(skf.split(df, df["score"].astype(str)))

for model_name in models_to_ensemble:
    print(f"\n===== Training {model_name} =====")
    oof_folds = pd.DataFrame()

    for fold,(tr,va) in enumerate(splits):
        xtr = df.loc[tr, feature_cols]
        ytr = df.loc[tr, label]
        xva = df.loc[va, feature_cols]
        yva = df.loc[va, label]
        idva = df.loc[va, "id"]

        if model_name == "lgbm":
            model = LGBMRegressor(**params["lgbm"])
            model.fit(xtr, ytr, callbacks=[log_evaluation(period=0)])
        elif model_name == "xgboost":
            model = xgb.XGBRegressor(**params["xgboost"])
            model.fit(xtr, ytr)
        else:
            model = CatBoostRegressor(**params["catboost"])
            model.fit(xtr, ytr)

        preds = model.predict(xva)

        fold_df = pd.DataFrame({
            "id":idva,"score":yva,
            f"{model_name}_preds":preds
        })
        oof_folds = pd.concat([oof_folds, fold_df])
        models[model_name].append(model)

        print(f"Fold {fold}: RMSE = {rmse(yva, preds):.5f}")

    if oof_df.empty:
        oof_df = oof_folds
    else:
        oof_df[f"{model_name}_preds"] = oof_folds[f"{model_name}_preds"]

    print(f"{model_name} CV:", rmse(oof_df["score"], oof_df[f"{model_name}_preds"]))


# ============================================================
# 8. ENSEMBLE OPTIMIZATION
# ============================================================

pred_cols = [f"{m}_preds" for m in models_to_ensemble]

def objective(w):
    blend = (oof_df[pred_cols] * w).sum(axis=1)
    return rmse(oof_df["score"], blend)

def find_weights():
    init = np.ones(len(models_to_ensemble)) / len(models_to_ensemble)
    bounds = [(0, 1)] * len(models_to_ensemble)
    res = minimize(objective, init, bounds=bounds, method="SLSQP")
    w = res.x
    return w / w.sum()

optimized_weights = find_weights()
print("\nOptimized Weights:", optimized_weights)

oof_df["ensemble_preds"] = (oof_df[pred_cols] * optimized_weights).sum(axis=1)
print("Ensemble CV Score:", rmse(oof_df["score"], oof_df["ensemble_preds"]))


# ============================================================
# 9. SAVE MODELS
# ============================================================

with open("allmodels.mdls", "wb") as f:
    pickle.dump(models, f)

print("\nAll models saved successfully!")


# ============================================================
# 10. Generate Submission (Test Inference)
# ============================================================

print("Loading test logs...")
test_logs = pd.read_csv('/kaggle/input/linking-writing-processes-to-writing-quality/test_logs.csv')

print("Extracting raw keystroke features...")
test_raw_feats = RawProcessor(test_logs)

print("Reconstructing test essays...")
test_reveal = (
    test_logs.groupby("id")
    .apply(revealing_text)
    .reset_index(drop=True)
)

test_df = pd.DataFrame(test_reveal.tolist(), columns=["id", "revealed_text"])

print("Processing text features...")
test_df = TextProcessor(test_df)

print("Merging raw + text features...")
test_df = test_df.merge(test_raw_feats, on="id", how="left")
test_df["text_length_timeratio"] = test_df["text_length"] / (test_df["up_time_max"] / 1000)

# ============================================================
# Add TF-IDF + SVD for Test Set
# ============================================================

print("Applying TF-IDF + SVD to test set...")

X_tfidf_test = tfidf.transform(test_df["revealed_text"])
X_svd_test = svd.transform(X_tfidf_test)

for i in range(svd_dim):
    test_df[f"svd_{i}"] = X_svd_test[:, i]

# Final matrix
X_test = test_df[feature_cols].fillna(0)


# ============================================================
# 11. Predict using ensemble
# ============================================================

print("Predicting with ensemble...")

test_preds = np.zeros(len(X_test))

for idx, model_name in enumerate(models_to_ensemble):
    model_list = models[model_name]
    fold_preds = np.zeros(len(X_test))
    
    for model in model_list:
        fold_preds += model.predict(X_test) / len(model_list)
    
    test_preds += optimized_weights[idx] * fold_preds


# ============================================================
# 12. Create Submission File
# ============================================================

submission = pd.DataFrame({
    "id": test_df["id"],
    "score": test_preds
})

submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)

print("Submission saved to:", submission_path)
submission.head()


