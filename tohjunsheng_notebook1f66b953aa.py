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
# 2. PURE-PYTHON READABILITY METRICS
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
    """
    Reconstruct the final essay text for each participant ID.
    """
    user_id = df["id"].iloc[0]
    text_df = df[df.activity != "Nonproduction"][["activity", "cursor_position", "text_change"]]

    essay = ""

    for act, cur, chg in text_df.values:

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

        # Movement-based acts
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

        # Normal insert
        essay = essay[:cur-len(chg)] + chg + essay[cur-len(chg):]

    return user_id, essay


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

    df["repetitiveness"] = df["revealed_text"].apply(lambda x: x.count("q")/max(1, len(x)))
    df["word_count"] = df["revealed_text"].apply(lambda x: len(x.split()))

    df["avg_word_length"] = df["revealed_text"].apply(
        lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0
    )
    df["word_lexical_diversity"] = df["revealed_text"].apply(
        lambda x: len(set(x.split())) / len(x.split()) if x.split() else 0
    )

    return df

def word_feats(df):
    df2 = df.copy()
    df2["word"] = df2["revealed_text"].apply(lambda x: re.split(r"[ \n.!?]", x))
    df2 = df2.explode("word")
    df2["word_len"] = df2["word"].apply(len)
    df2 = df2[df2["word_len"] > 0]

    agg = df2.groupby("id")["word_len"].agg(
        ["count", "min", "max", "first", "last", "median", "sum", "std"]
    )
    agg.columns = [f"word_len_{c}" for c in agg.columns]
    return agg.reset_index()

def sent_feats(df):
    df2 = df.copy()
    df2["sent"] = df2["revealed_text"].apply(lambda x: re.split(r"[.!?]", x))
    df2 = df2.explode("sent")
    df2["sent"] = df2["sent"].str.strip()
    df2["sent_len"] = df2["sent"].apply(len)
    df2["sent_word_count"] = df2["sent"].apply(lambda x: len(x.split()))
    df2 = df2[df2["sent_len"] > 0]

    a1 = df2.groupby("id")["sent_len"].agg(
        ["count","min","max","first","last","median","sum","std"]
    )
    a2 = df2.groupby("id")["sent_word_count"].agg(
        ["count","min","max","first","last","median","sum","std"]
    )

    out = pd.concat([a1, a2], axis=1)
    out.columns = (
        [f"sent_len_{c}" for c in a1.columns] +
        [f"sent_word_count_{c}" for c in a2.columns]
    )

    out = out.reset_index()
    out.rename(columns={"sent_len_count": "sent_count"}, inplace=True)
    out.drop(columns=["sent_word_count_count"], inplace=True)

    return out

def parag_feats(df):
    df2 = df.copy()
    df2["paragraph"] = df2["revealed_text"].apply(lambda x: x.split("\n"))
    df2 = df2.explode("paragraph")

    df2["paragraph_len"] = df2["paragraph"].apply(len)
    df2["paragraph_word_count"] = df2["paragraph"].apply(lambda x: len(x.split()))
    df2["paragraph_sent_count"] = df2["paragraph"].apply(lambda x: len(re.split(r"[.!?]", x)))

    df2 = df2[df2["paragraph_len"] > 2]

    a1 = df2.groupby("id")["paragraph_len"].agg(
        ["count","min","max","first","last","median","sum","std"]
    )
    a2 = df2.groupby("id")["paragraph_word_count"].agg(
        ["count","min","max","first","last","median","sum","std"]
    )
    a3 = df2.groupby("id")["paragraph_sent_count"].agg(
        ["count","min","max","first","last","median","sum","std"]
    )

    out = pd.concat([a1, a2, a3], axis=1)

    cols = []
    for i, c in enumerate(out.columns):
        if i < 8:
            cols.append("paragraph_len_" + c)
        elif i < 16:
            cols.append("paragraph_word_count_" + c)
        else:
            cols.append("paragraph_sent_count_" + c)

    out.columns = cols
    out = out.reset_index()

    out.rename(columns={"paragraph_len_count": "paragraph_count"}, inplace=True)
    out.drop(columns=["paragraph_word_count_count", "paragraph_sent_count_count"], inplace=True)

    return out


def count_punctuation_errors(text):
    if not isinstance(text, str) or len(text) == 0:
        return {
            "punct_error_space_before_comma": 0,
            "punct_error_space_before_period": 0,
            "punct_error_missing_space_after_comma": 0,
            "punct_error_missing_space_after_period": 0,
            "punct_error_double_punct": 0,
            "punct_error_total": 0
        }

    errors = {}

    # Space before comma / period
    errors["punct_error_space_before_comma"] = len(re.findall(r"\s+,", text))
    errors["punct_error_space_before_period"] = len(re.findall(r"\s+\.", text))

    # Missing space after punctuation (comma or period)
    errors["punct_error_missing_space_after_comma"] = len(re.findall(r",[A-Za-z]", text))
    errors["punct_error_missing_space_after_period"] = len(re.findall(r"\.[A-Za-z]", text))

    # Double punctuation !! ?? ..., ??!
    errors["punct_error_double_punct"] = len(re.findall(r"([!?.,])\1{1,}", text))

    # Total
    errors["punct_error_total"] = sum(errors.values())
    
    return errors


def TextProcessor(df):
    df = df.copy()

    # Replace empty essays with a placeholder
    df.loc[df["revealed_text"].str.replace(" ", "", regex=False) == "", "revealed_text"] = "q"

    # Standardize whitespace and formatting
    df["revealed_text"] = df["revealed_text"].apply(standardize_text)

    # Basic text-level features
    df = get_text_chunk_features(df)

    # Advanced text structure features
    df = df.merge(word_feats(df), on="id", how="left")
    df = df.merge(sent_feats(df), on="id", how="left")
    df = df.merge(parag_feats(df), on="id", how="left")

    # =============================
    # NEW: Punctuation error features
    # =============================
    print("Computing punctuation error features...")
    punct = df["revealed_text"].apply(count_punctuation_errors)

    df["punct_error_space_before_comma"]   = punct.apply(lambda x: x.get("punct_error_space_before_comma", 0))
    df["punct_error_space_before_period"]  = punct.apply(lambda x: x.get("punct_error_space_before_period", 0))
    df["punct_error_missing_space_after_comma"]  = punct.apply(lambda x: x.get("punct_error_missing_space_after_comma", 0))
    df["punct_error_missing_space_after_period"] = punct.apply(lambda x: x.get("punct_error_missing_space_after_period", 0))
    df["punct_error_double_punct"]         = punct.apply(lambda x: x.get("punct_error_double_punct", 0))
    df["punct_error_total"]                = punct.apply(lambda x: x.get("punct_error_total", 0))

    return df


# ============================================================
# 5. RAW KEYSTROKE FEATURE ENGINEERING
# ============================================================

def count_by_values(df, colname, values):
    """
    Count occurrences of specific values within a given column for each ID.
    """
    fts = df.select(pl.col("id").unique(maintain_order=True))

    for i, v in enumerate(values):
        tmp = df.group_by("id").agg(
            pl.col(colname).is_in([v]).sum().alias(f"{colname}_{i}_cnt")
        )
        fts = fts.join(tmp, on="id")

    return fts


def event_count_feats(df):
    acts = ['Input', 'Remove/Cut', 'Nonproduction', 'Replace', 'Paste']
    events = [
        'q','Space','Backspace','Shift','ArrowRight','Leftclick','ArrowLeft',
        '.',',','ArrowDown','ArrowUp','Enter','CapsLock',"'",'Delete','Unidentified'
    ]
    tchanges = ['q',' ','.',',','\n',"'",'"','-','?',';','=','/','\\',':']

    out = count_by_values(df, "activity", acts)
    out = out.join(count_by_values(df, "text_change", tchanges), on="id")
    out = out.join(count_by_values(df, "down_event", events), on="id")

    return out.to_pandas()


# ============================================================
#  Numeric & Categorical Stats
# ============================================================

def num_colstat_feats(df):
    """
    Compute statistical aggregations for numeric columns.
    """
    nums = ['down_time','up_time','action_time','cursor_position','word_count','event_id']

    agg_exprs = [
        pl.col("action_time").sum().alias("action_time_sum")
    ]

    for col in nums:
        agg_exprs += [
            pl.col(col).std().alias(f"{col}_std"),
            pl.col(col).median().alias(f"{col}_median"),
            pl.col(col).min().alias(f"{col}_min"),
            pl.col(col).max().alias(f"{col}_max")
        ]

    agg = df.group_by("id").agg(agg_exprs)
    return agg.to_pandas()


def cat_colstat_feats(df):
    """
    Count unique categorical values per ID.
    """
    agg = df.group_by("id").agg(
        pl.n_unique(["activity", "down_event", "up_event", "text_change"])
    )
    return agg.to_pandas()


# ============================================================
#  Pause Aggregations
# ============================================================

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


# ============================================================
#  Idle Time & Pause Features
# ============================================================

def idle_time_feats(df):
    temp = df.with_columns(
        pl.col("up_time").shift().over("id").alias("up_time_lagged")
    )

    temp = temp.with_columns(
        ((pl.col("down_time") - pl.col("up_time_lagged")).abs()/1000)
        .fill_null(0)
        .alias("time_diff")
    )

    # Identify space/sentence/paragraph delimiters
    temp = temp.with_columns(
        (pl.col("up_event")=="Space").alias("is_space"),
        (pl.col("up_event")==".").alias("is_dot"),
        (pl.col("up_event")=="Enter").alias("is_enter")
    )

    # Running counters
    temp = temp.with_columns(
        pl.col("is_space").cum_sum().over("id").alias("word_id"),
        pl.col("is_dot").cum_sum().over("id").alias("sentence_id"),
        pl.col("is_enter").cum_sum().over("id").alias("paragraph_id")
    )

    temp2 = temp.filter(pl.col("activity").is_in(["Input","Remove/Cut"]))

    iw = pause_stat_aggregator(temp2, "iw")

    # Pause before words
    bww = temp2.group_by(["id", "word_id"]).agg(pl.col("time_diff").first())
    bww = pause_stat_aggregator(bww, "bww")

    # Pause before sentences
    bws = temp2.group_by(["id","sentence_id"]).agg(pl.col("time_diff").first())
    bws = pause_stat_aggregator(bws, "bws")

    # Pause before paragraphs
    bwp = temp2.group_by(["id","paragraph_id"]).agg(pl.col("time_diff").first())
    bwp = pause_stat_aggregator(bwp, "bwp")

    return (
        iw.join(bww, on="id")
          .join(bws, on="id")
          .join(bwp, on="id")
          .to_pandas()
    )


# ============================================================
#  Burst Features
# ============================================================

def burst_features(df, burst_type):
    temp = df.with_columns(
        pl.col("up_time").shift().over("id").alias("up_time_lagged")
    )

    temp = temp.with_columns(
        ((pl.col("down_time") - pl.col("up_time_lagged")).abs()/1000)
        .fill_null(0)
        .alias("time_diff")
    )

    # Define what counts as a burst (Input vs Remove)
    if burst_type == "p":
        temp = temp.with_columns(pl.col("activity").is_in(["Input"]))
    else:
        temp = temp.with_columns(pl.col("activity").is_in(["Remove/Cut"]))

    temp = temp.with_columns(
        (pl.col("activity")).alias("burst_flag"),
        (pl.col("action_time")/1000).alias("action_time_s"),
        (pl.col("up_time")/1000).alias("up_time_s"),
        pl.when(pl.col("activity")).then(
            pl.col("activity").rle_id()
        ).alias(f"{burst_type}_burst_group")
    ).drop_nulls()

    # Burst-level aggregates
    g = temp.group_by(["id", f"{burst_type}_burst_group"]).agg([
        pl.col("activity").count().alias(f"{burst_type}_burst_group_keypress_count"),
        pl.col("action_time_s").sum().alias(f"{burst_type}_burst_group_timespent"),
        pl.col("action_time_s").mean().alias(f"{burst_type}_burst_keypress_timespent_mean"),
        pl.col("action_time_s").std().alias(f"{burst_type}_burst_keypress_timespent_std"),
        pl.col("up_time_s").min().alias(f"{burst_type}_burst_keypress_timestamp_first"),
        pl.col("up_time_s").max().alias(f"{burst_type}_burst_keypress_timestamp_last"),
    ])

    # ID-level aggregates
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


# ============================================================
#  Keys-per-second
# ============================================================

def get_keys_pressed_per_second(raw_df):
    a = raw_df[raw_df["activity"].isin(["Input","Remove/Cut"])].groupby("id").agg(
        keys_pressed=("event_id", "count")
    ).reset_index()

    b = raw_df.groupby("id").agg(
        min_down_time=("down_time", "min"),
        max_up_time=("up_time", "max")
    ).reset_index()

    out = a.merge(b, on="id")
    out["keys_per_second"] = out["keys_pressed"] / (
        (out["max_up_time"] - out["min_down_time"]) / 1000
    )

    return out[["id", "keys_per_second"]]


def compute_time_to_word_targets(raw_df, word_targets=[200, 300, 400, 500]):
    """
    Computes the timestamp (in seconds) when each writer reaches 200, 300, 400, 500 words.
    Uses raw keystroke logs chronologically.
    """
    results = []

    for uid, user_df in raw_df.groupby("id"):
        user_df = user_df.sort_values("down_time")

        word_count = 0
        text = ""
        hit_times = {t: np.nan for t in word_targets}

        for _, row in user_df.iterrows():
            act = row["activity"]
            change = row["text_change"]
            cur = row["cursor_position"]
            t = row["down_time"] / 1000.0   # convert to seconds

            # Apply text-change operations
            if act == "Input":
                text = text[:cur] + change + text[cur:]
            elif act == "Remove/Cut":
                text = text[:cur] + text[cur+len(change):]
            elif act == "Paste":
                text = text[:cur] + change + text[cur:]

            # Count words
            word_count = len(text.split())

            # Check targets
            for target in word_targets:
                if np.isnan(hit_times[target]) and word_count >= target:
                    hit_times[target] = t

        results.append({
            "id": uid,
            "time_to_200_words": hit_times[200],
            "time_to_300_words": hit_times[300],
            "time_to_400_words": hit_times[400],
            "time_to_500_words": hit_times[500],
        })

    return pd.DataFrame(results)


# ============================================================
#  Main RAW Processor for Keystrokes
# ============================================================

def RawProcessor(raw_df):
    raw_pl = pl.from_pandas(raw_df)

    print("Creating kpps features...")
    feat = get_keys_pressed_per_second(raw_df)

    print("Computing time-to-word targets...")
    tt = compute_time_to_word_targets(raw_df)
    feat = feat.merge(tt, on="id", how="left")

    print("Event counts...")
    feat = feat.merge(event_count_feats(raw_pl), on="id", how="left")

    print("Numeric/categorical stats...")
    feat = feat.merge(num_colstat_feats(raw_pl), on="id", how="left")
    feat = feat.merge(cat_colstat_feats(raw_pl), on="id", how="left")

    print("Pause features...")
    feat = feat.merge(idle_time_feats(raw_pl), on="id", how="left")

    print("Burst features...")
    feat = feat.merge(burst_features(raw_pl, "p"), on="id", how="left")
    feat = feat.merge(burst_features(raw_pl, "r"), on="id", how="left")

    # Ratios
    feat["p_bursts_timeratio"] = feat["p_burst_timespent_sum"] / (feat["up_time_max"]/1000)
    feat["r_bursts_timeratio"] = feat["r_burst_timespent_sum"] / (feat["up_time_max"]/1000)
    feat["action_timeratio"] = feat["action_time_sum"] / feat["up_time_max"]
    feat["pause_timeratio"] = feat["iw_total_pause_time"] / (feat["up_time_max"]/1000)
    feat["pausecount_timeratio"] = feat["iw_pauses_2_sec"] / (feat["up_time_max"]/1000)

    feat["word_time_ratio"] = feat["word_count_max"] / (feat["up_time_max"]/1000)
    feat["word_event_ratio"] = feat["word_count_max"] / (feat["up_time_max"]/1000)
    feat["event_time_ratio"] = feat["event_id_max"] / (feat["up_time_max"]/1000)

    return feat


# ============================================================
# 6. BUILD FULL TRAINING DATASET
# ============================================================

print("Loading data...")

train_scores = pd.read_csv('/kaggle/input/linking-writing-processes-to-writing-quality/train_scores.csv')
raw_df = pd.read_csv('/kaggle/input/linking-writing-processes-to-writing-quality/train_logs.csv')

print("Extracting raw keystroke features...")
raw_feats = RawProcessor(raw_df)

print("Reconstructing essays...")
reveal = raw_df.groupby("id").apply(revealing_text)
df = pd.DataFrame(reveal.tolist(), columns=["id", "revealed_text"])
df = df.merge(train_scores, on="id")

print("Extracting text features...")
df = TextProcessor(df)

df = df.merge(raw_feats, on="id")
df["text_length_timeratio"] = df["text_length"] / (df["up_time_max"] / 1000)

# Columns for training
feature_cols = df.drop(["id", "revealed_text", "score"], axis=1).columns
label = "score"

rmse = lambda y, p: mean_squared_error(y, p, squared=False)


# ============================================================
# 7. MODEL TRAINING (LGBM / XGB / CATBOOST)
# ============================================================

models_to_ensemble = ["lgbm", "xgboost", "catboost"]
models = {m: [] for m in models_to_ensemble}
oof_df = pd.DataFrame()

# Predefined tuned hyperparameters
params = {
    "lgbm": {
        "reg_alpha":1.0894, "reg_lambda":6.2909, "colsample_bytree":0.6218,
        "subsample":0.9579, "learning_rate":0.0027, "max_depth":8,
        "num_leaves":947, "min_child_samples":57, "n_estimators":2500,
        "metric":"rmse", "random_state":42, "verbosity":-1, "force_col_wise":True
    },
    "xgboost":{
        "max_depth":2, "learning_rate":0.00998, "n_estimators":1000,
        "min_child_weight":17, "gamma":0.1288, "subsample":0.5078,
        "colsample_bytree":0.735, "reg_alpha":0.6709, "reg_lambda":0.0681,
        "random_state":1, "tree_method":"hist"
    },
    "catboost":{
        "learning_rate":0.0249, "depth":5, "l2_leaf_reg":3.71,
        "subsample":0.185, "colsample_bylevel":0.655,
        "min_data_in_leaf":93, "iterations":1000,
        "random_state":1, "silent":True, "use_best_model":False
    }
}

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
splits = list(skf.split(df, df["score"].astype(str)))

# Loop through each model
for model_name in models_to_ensemble:
    print(f"\n===== Training {model_name} =====")
    oof_folds = pd.DataFrame()

    for fold, (tr, va) in enumerate(splits):
        xtr = df.loc[tr, feature_cols]
        ytr = df.loc[tr, label]
        xva = df.loc[va, feature_cols]
        yva = df.loc[va, label]
        idva = df.loc[va, "id"]

        # ----- Model selection -----
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
            "id": idva,
            "score": yva,
            f"{model_name}_preds": preds
        })
        oof_folds = pd.concat([oof_folds, fold_df])
        models[model_name].append(model)

        print(f"Fold {fold}: RMSE = {rmse(yva, preds):.5f}")

    # Add to OO F dataframe
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

# Final out-of-fold ensemble predictions
oof_df["ensemble_preds"] = (oof_df[pred_cols] * optimized_weights).sum(axis=1)
print("Ensemble CV Score:", rmse(oof_df["score"], oof_df["ensemble_preds"]))


# ============================================================
# 9. SAVE MODELS
# ============================================================

with open("allmodels.mdls", "wb") as f:
    pickle.dump(models, f)

print("\nAll models saved successfully!")


# ============================================================
# 10. TEST INFERENCE (Build Test Data)
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

# Final test matrix
X_test = test_df[feature_cols].fillna(0)


# ============================================================
# 11. PREDICT USING ENSEMBLE
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
# 12. CREATE SUBMISSION FILE
# ============================================================

submission = pd.DataFrame({
    "id": test_df["id"],
    "score": test_preds
})

submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)

print("Submission saved to:", submission_path)
submission.head()

