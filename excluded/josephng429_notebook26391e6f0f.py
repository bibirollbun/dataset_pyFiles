# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

# List input files (optional)
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# HF / transformers offline safeguards
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import optuna
OPTUNA_AVAILABLE = True

USE_OPTUNA = False
OPTUNA_TRIALS = 30
OPTUNA_FOLDS = 5

# ============================================================
# ğŸ�† Linking Writing Processes to Writing Quality (Enhanced)
# Includes Feature Importance Correlation + Hyperparameter Tuning
# ============================================================

import re, warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV, Ridge, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone

from catboost import CatBoostRegressor, Pool
import xgboost as xgb
import lightgbm as lgb

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import nltk
import torch
import gc
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset
import shutil
import atexit

# ============================================================
# 1. Essay Reconstruction
# ============================================================

def getEssays(df):
    df = df.sort_values(['id', 'event_id'], kind='mergesort').reset_index(drop=True)
    textInputDf = df[['id', 'activity', 'cursor_position', 'text_change']].copy()

    # Remove nonproduction (no actual text change)
    textInputDf = textInputDf[textInputDf.activity != 'Nonproduction']

    valCountsArr = textInputDf['id'].value_counts(sort=False).values
    lastIndex = 0
    essaySeries = pd.Series(dtype=object)

    for index, valCount in enumerate(valCountsArr):
        currTextInput = textInputDf[['activity', 'cursor_position', 'text_change']].iloc[lastIndex:lastIndex + valCount]
        lastIndex += valCount
        essayText = ""

        for Input in currTextInput.values:
            # Input[0] = activity
            # Input[1] = cursor_position
            # Input[2] = text_change

            if Input[0] == 'Replace':
                replaceTxt = Input[2].split(' => ')
                essayText = (
                    essayText[:Input[1] - len(replaceTxt[1])]
                    + replaceTxt[1]
                    + essayText[Input[1] - len(replaceTxt[1]) + len(replaceTxt[0]):]
                )
                continue

            if Input[0] == 'Paste':
                essayText = (
                    essayText[:Input[1] - len(Input[2])]
                    + Input[2]
                    + essayText[Input[1] - len(Input[2]):]
                )
                continue

            if Input[0] == 'Remove/Cut':
                essayText = essayText[:Input[1]] + essayText[Input[1] + len(Input[2]):]
                continue

            if "M" in Input[0]:
                croppedTxt = Input[0][10:]
                splitTxt = croppedTxt.split(' To ')
                valueArr = [item.split(', ') for item in splitTxt]
                moveData = (
                    int(valueArr[0][0][1:]),
                    int(valueArr[0][1][:-1]),
                    int(valueArr[1][0][1:]),
                    int(valueArr[1][1][:-1]),
                )

                if moveData[0] != moveData[2]:
                    if moveData[0] < moveData[2]:
                        essayText = (
                            essayText[:moveData[0]]
                            + essayText[moveData[1]:moveData[3]]
                            + essayText[moveData[0]:moveData[1]]
                            + essayText[moveData[3]:]
                        )
                    else:
                        essayText = (
                            essayText[:moveData[2]]
                            + essayText[moveData[0]:moveData[1]]
                            + essayText[moveData[2]:moveData[0]]
                            + essayText[moveData[1]:]
                        )
                continue

            # Plain input
            essayText = (
                essayText[:Input[1] - len(Input[2])]
                + Input[2]
                + essayText[Input[1] - len(Input[2]):]
            )

        essaySeries[index] = essayText

    essaySeries.index = textInputDf['id'].unique()
    return essaySeries


# ============================================================
# 2. Text-based features (extended with lexical richness)
# ============================================================

def essay_structure_features(df):
    feats = []
    for _, row in df.iterrows():
        text = str(row["essay"])

        sents = re.split(r"[.!?]", text)
        sents = [s.strip() for s in sents if len(s.strip()) > 0]
        words = re.findall(r"\w+", text)
        word_lens = [len(w) for w in words]
        char_len = len(text)

        sent_word_counts = [len(re.findall(r"\w+", s)) for s in sents] if sents else []

        unique_words = set(w.lower() for w in words)
        type_token_ratio = len(unique_words) / (len(words) + 1e-6)
        long_words = [w for w in words if len(w) >= 7]
        long_word_ratio = len(long_words) / (len(words) + 1e-6)
        upper_chars = sum(1 for c in text if c.isupper())
        upper_char_ratio = upper_chars / (char_len + 1e-6)

        denom = (char_len + 1e-6) / 1000.0
        semicolons_per_k = text.count(';') / denom
        colons_per_k = text.count(':') / denom
        dashes_per_k = text.count('-') / denom

        try:
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
            stopwords = ENGLISH_STOP_WORDS
        except Exception:
            stopwords = set()
        stopword_rate = (sum(1 for w in words if w.lower() in stopwords) / (len(words) + 1e-6))

        f = {
            "id": row["id"],
            "char_len": char_len,
            "n_words": len(words),
            "n_sent": len(sents),
            "avg_word_len": np.mean(word_lens) if word_lens else 0,
            "std_word_len": np.std(word_lens) if word_lens else 0,
            "avg_sent_len": np.mean(sent_word_counts) if sent_word_counts else 0,
            "std_sent_len": np.std(sent_word_counts) if sent_word_counts else 0,
            "commas": text.count(','),
            "periods": text.count('.'),
            "excls": text.count('!'),
            "ques": text.count('?'),
            "punct_density": sum(text.count(p) for p in [',', '.', '!', '?', ';']) / (char_len + 1e-6),
            "type_token_ratio": type_token_ratio,
            "long_word_ratio": long_word_ratio,
            "upper_char_ratio": upper_char_ratio,
            "semicolons_per_k": semicolons_per_k,
            "colons_per_k": colons_per_k,
            "dashes_per_k": dashes_per_k,
            "stopword_rate": stopword_rate,
        }
        feats.append(f)
    return pd.DataFrame(feats)


# ============================================================
# 3. Process-based features (extended with time segments + pause bins)
# ============================================================

def make_feats(df):
    df = df.copy()
    df["activity"] = df["activity"].apply(
        lambda x: "Move" if isinstance(x, str) and x.startswith("Move From") else x
    )
    out = pd.DataFrame({"id": df["id"].unique()})

    def get_time_features(sub):
        total_time = sub["up_time"].max() - sub["down_time"].min()
        avg_action = sub["action_time"].mean()
        median_action = sub["action_time"].median()
        return pd.Series({
            "total_time": total_time,
            "avg_action_time": avg_action,
            "median_action_time": median_action
        })

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

    feats = []
    for essay_id, sub in df.groupby("id"):
        f = {"id": essay_id}
        f.update(get_time_features(sub))
        f.update(get_activity_features(sub))
        f.update(get_text_features(sub))
        f.update(get_pause_features(sub))
        feats.append(f)
    out = out.merge(pd.DataFrame(feats), on="id", how="left")

    # duration and event counts
    essay_time = df.groupby("id").apply(
        lambda x: (x["up_time"].max() - x["down_time"].min()) / 1000
    ).reset_index(name="duration_s")
    out = out.merge(essay_time, on="id", how="left")

    key_counts = df.groupby("id")["event_id"].count().reset_index(name="n_events")
    out = out.merge(key_counts, on="id", how="left")
    out["keys_per_sec"] = out["n_events"] / out["duration_s"].replace(0, np.nan)

    # pause stats
    df["up_time_lagged"] = df.groupby("id")["up_time"].shift(1)
    df["pause_s"] = (df["down_time"] - df["up_time_lagged"]).fillna(0) / 1000
    pauses = df.groupby("id")["pause_s"].agg(["mean", "median", "max", "std"]).reset_index()
    pauses.columns = ["id", "pause_mean", "pause_median", "pause_max", "pause_std"]
    out = out.merge(pauses, on="id", how="left")

    # action stats
    action_stats = df.groupby("id")["action_time"].agg(["mean", "std", "max", "sum"]).reset_index()
    action_stats.columns = ["id", "action_mean", "action_std", "action_max", "action_sum"]
    out = out.merge(action_stats, on="id", how="left")

    # cursor stats
    cursor_stats = df.groupby("id")["cursor_position"].agg(["max", "nunique"]).reset_index()
    cursor_stats.columns = ["id", "cursor_max", "cursor_unique"]
    out = out.merge(cursor_stats, on="id", how="left")

    # activity counts
    act_counts = df.groupby(["id", "activity"]).size().unstack(fill_value=0).reset_index()
    act_counts.columns = [f"activity_{c}" if c != "id" else "id" for c in act_counts.columns]
    out = out.merge(act_counts, on="id", how="left")

    # ratios
    out["cursor_per_sec"] = out["cursor_max"] / out["duration_s"]
    out["action_density"] = out["action_sum"] / out["duration_s"]
    out["delete_to_input_ratio"] = out["num_remove"] / (out["num_input"] + 1e-6)

    # ========= EXTRA BEHAVIOURAL FEATURES =========
    df_sorted = df.sort_values(["id", "down_time"]).copy()

    # Normalised time
    df_sorted["elapsed_ms"] = df_sorted.groupby("id")["down_time"].transform(lambda x: x - x.min())
    total_ms = df_sorted.groupby("id")["elapsed_ms"].transform("max").replace(0, 1)
    df_sorted["time_frac"] = df_sorted["elapsed_ms"] / total_ms

    df_sorted["segment"] = pd.cut(
        df_sorted["time_frac"],
        bins=[0.0, 1/3, 2/3, 1.0000001],
        labels=["early", "mid", "late"],
        include_lowest=True
    )

    seg_counts = df_sorted.groupby(["id", "segment"])["event_id"].count().unstack(fill_value=0)
    for s in ["early", "mid", "late"]:
        if s not in seg_counts.columns:
            seg_counts[s] = 0
    seg_counts = seg_counts[["early", "mid", "late"]]
    seg_counts.columns = ["n_events_early", "n_events_mid", "n_events_late"]
    seg_counts = seg_counts.reset_index()

    total_events = (
        seg_counts[["n_events_early", "n_events_mid", "n_events_late"]]
        .sum(axis=1)
        .replace(0, 1)
    )
    for s in ["early", "mid", "late"]:
        seg_counts[f"event_frac_{s}"] = seg_counts[f"n_events_{s}"] / total_events

    out = out.merge(seg_counts, on="id", how="left")

    # Pause bins
    df_sorted["pause_bin"] = pd.cut(
        df_sorted["pause_s"],
        bins=[0.0, 0.3, 1.0, 3.0, np.inf],
        labels=["short", "medium", "long", "very_long"],
        include_lowest=True,
        right=False
    )

    pause_counts = df_sorted.groupby(["id", "pause_bin"])["pause_s"].count().unstack(fill_value=0)
    for b in ["short", "medium", "long", "very_long"]:
        if b not in pause_counts.columns:
            pause_counts[b] = 0
    pause_counts = pause_counts[["short", "medium", "long", "very_long"]]
    pause_counts.columns = [
        "n_pause_short",
        "n_pause_medium",
        "n_pause_long",
        "n_pause_very_long",
    ]
    pause_counts = pause_counts.reset_index()

    total_pauses = (
        pause_counts[
            ["n_pause_short", "n_pause_medium", "n_pause_long", "n_pause_very_long"]
        ].sum(axis=1)
        .replace(0, 1)
    )
    for b in ["short", "medium", "long", "very_long"]:
        pause_counts[f"pause_frac_{b}"] = pause_counts[f"n_pause_{b}"] / total_pauses

    out = out.merge(pause_counts, on="id", how="left")
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out


# ============================================================
# 4. Extra feature utilities (currently unused but available)
# ============================================================

def input_word_stats_from_logs(df, all_ids):
    sub = df.loc[~df['text_change'].astype(str).str.contains('=>', regex=False)].copy()
    sub = sub.loc[sub['text_change'].astype(str) != 'NoChange']
    sub = sub.loc[sub['activity'].isin(['Input', 'Paste', 'Replace'])]

    s = (
        sub.groupby('id')['text_change']
        .apply(lambda s: ''.join(map(str, s)))
        .reindex(all_ids, fill_value="")
    )
    concat = s.reset_index(name='all_text')

    def _word_len_stats(text):
        words = re.findall(r"[A-Za-z]+", text)
        if not words:
            return pd.Series({
                'input_word_count': 0,
                'input_word_len_mean': 0, 'input_word_len_max': 0,
                'input_word_len_std': 0, 'input_word_len_median': 0,
                'input_word_len_skew': 0,
            })
        lens = np.fromiter((len(w) for w in words), dtype=float)
        return pd.Series({
            'input_word_count': len(words),
            'input_word_len_mean': float(lens.mean()),
            'input_word_len_max': float(lens.max()),
            'input_word_len_std': float(lens.std(ddof=0)),
            'input_word_len_median': float(np.median(lens)),
            'input_word_len_skew': float(pd.Series(lens).skew()),
        })

    stats = concat['all_text'].apply(_word_len_stats).fillna(0)
    out = pd.concat([concat[['id']], stats], axis=1)
    return out


def text_change_stats(df):
    feats = []
    sub = df[df['activity'].isin(['Input', 'Remove/Cut', 'Paste'])].copy()
    sub['tc_len'] = sub['text_change'].astype(str).str.len()

    for eid, g in sub.groupby('id'):
        f = {'id': eid}

        g_input = g[g['activity'] == 'Input']['tc_len']
        f['input_tc_len_mean'] = g_input.mean()
        f['input_tc_len_std'] = g_input.std()
        f['input_tc_len_max'] = g_input.max()

        g_remove = g[g['activity'] == 'Remove/Cut']['tc_len']
        f['remove_tc_len_mean'] = g_remove.mean()
        f['remove_tc_len_max'] = g_remove.max()

        g_paste = g[g['activity'] == 'Paste']['tc_len']
        f['paste_tc_len_mean'] = g_paste.mean()
        f['paste_tc_len_max'] = g_paste.max()

        feats.append(f)

    out = pd.DataFrame(feats)
    return out.fillna(0)


def numeric_aggs(df):
    num_cols = [c for c in ['down_time', 'up_time', 'action_time', 'cursor_position', 'word_count'] if c in df.columns]
    agg = df.groupby('id')[num_cols].agg(['mean', 'std', 'median', 'min', 'max', 'sum'])
    agg.columns = ['_'.join(c) for c in agg.columns]
    agg = agg.reset_index()
    return agg


def _run_lengths(mask):
    if len(mask) == 0:
        return []
    runs = []
    cnt = 1 if mask[0] else 0
    for i in range(1, len(mask)):
        if mask[i] and mask[i - 1]:
            cnt += 1
        else:
            if mask[i - 1]:
                runs.append(cnt)
            cnt = 1 if mask[i] else 0
    if len(mask) and mask[-1]:
        runs.append(cnt)
    return runs


def burst_features(df):
    feats = []
    for eid, g in df.sort_values(['id', 'down_time']).groupby('id'):
        g2 = g[g['activity'].isin(['Input', 'Remove/Cut'])].copy()
        g2['prev_up'] = g2['up_time'].shift(1)
        td = (g2['down_time'] - g2['prev_up']).fillna(0) / 1000.0
        mask_p = td.values < 2.0
        pruns = _run_lengths(mask_p.astype(bool))

        mask_r = (g2['activity'] == 'Remove/Cut').values
        rruns = _run_lengths(mask_r.astype(bool))

        def agg_runs(runs, prefix):
            if len(runs) == 0:
                return {
                    f'{prefix}_mean': 0.0, f'{prefix}_std': 0.0, f'{prefix}_median': 0.0,
                    f'{prefix}_max': 0.0, f'{prefix}_count': 0, f'{prefix}_first': 0, f'{prefix}_last': 0
                }
            r = np.array(runs, dtype=float)
            return {
                f'{prefix}_mean': float(r.mean()), f'{prefix}_std': float(r.std(ddof=0)),
                f'{prefix}_median': float(np.median(r)), f'{prefix}_max': float(r.max()),
                f'{prefix}_count': int(len(r)), f'{prefix}_first': int(r[0]), f'{prefix}_last': int(r[-1])
            }

        f = {'id': eid}
        f.update(agg_runs(pruns, 'P_bursts'))
        f.update(agg_runs(rruns, 'R_bursts'))
        feats.append(f)
    return pd.DataFrame(feats)


def advanced_revision_features(df):
    out_rows = []
    df = df.sort_values(['id', 'event_id']).reset_index(drop=True)

    for essay_id, g in df.groupby('id'):
        g = g.copy()

        g['EOT'] = g['cursor_position'].cummax()
        g['dist_from_EOT'] = g['EOT'] - g['cursor_position']

        rm_events = g[g['activity'] == 'Remove/Cut']
        immediate_deletes = (rm_events['dist_from_EOT'] <= 1).sum()
        distant_deletes = (rm_events['dist_from_EOT'] > 1).sum()

        g['prev_up_time'] = g['up_time'].shift(1)
        g['pause_ms'] = (g['down_time'] - g['prev_up_time']).fillna(0)

        g['is_distant_delete'] = (g['activity'] == 'Remove/Cut') & (g['dist_from_EOT'] > 1)
        g['pause_before_distant_delete'] = g['pause_ms'] * g['is_distant_delete'].shift(-1).fillna(0)

        pauses_before_revision = g[g['pause_before_distant_delete'] > 0]['pause_before_distant_delete']

        f = {
            'id': essay_id,
            'distant_delete_count': distant_deletes,
            'immediate_delete_count': immediate_deletes,
            'distant_delete_ratio': distant_deletes / (distant_deletes + immediate_deletes + 1e-6),
            'mean_pause_before_revision': pauses_before_revision.mean(),
            'std_pause_before_revision': pauses_before_revision.std(),
            'total_pause_before_revision': pauses_before_revision.sum(),
            'mean_dist_from_EOT_all': g['dist_from_EOT'].mean(),
            'mean_dist_from_EOT_revising': g[g['dist_from_EOT'] > 1]['dist_from_EOT'].mean(),
        }
        out_rows.append(f)

    return pd.DataFrame(out_rows).fillna(0)


# Grammar / paragraph / replace features are defined but not called in this pipeline
global_lt_tool = None
global_lt_server_process = None

def get_grammar_features(essays_df, lt_path="/kaggle/input/language-tool-python-5-7/LanguageTool-5.7"):
    import language_tool_python

    global global_lt_tool, global_lt_server_process

    if global_lt_tool is None:
        print(f"Initializing LanguageTool from: {lt_path}")
        os.environ['JAVA_HOME'] = '/opt/conda/bin/java'
        writeable_path = "/kaggle/working/LanguageTool-5.7"

        if os.path.exists(writeable_path):
            shutil.rmtree(writeable_path)

        try:
            shutil.copytree(lt_path, writeable_path)
        except Exception as e:
            print(f"Failed to copy LanguageTool: {e}")
            feats = essays_df[['id']].copy()
            feats['grammar_errors'] = 0
            feats['spelling_errors'] = 0
            feats['style_errors'] = 0
            feats['total_errors'] = 0
            return feats

        config = {'languageToolJavaPath': os.path.join(writeable_path, 'languagetool-server.jar')}

        try:
            global_lt_tool = language_tool_python.LanguageTool('en-US', config=config)
            global_lt_server_process = global_lt_tool.server_process

            def cleanup_lt():
                print("Shutting down LanguageTool server...")
                if global_lt_tool:
                    global_lt_tool.close()
                if global_lt_server_process:
                    global_lt_server_process.terminate()
                    global_lt_server_process.wait()
                if os.path.exists(writeable_path):
                    shutil.rmtree(writeable_path)
                print("LanguageTool cleanup complete.")

            atexit.register(cleanup_lt)

        except Exception as e:
            print(f"Failed to initialize LanguageTool: {e}")
            feats = essays_df[['id']].copy()
            feats['grammar_errors'] = 0
            feats['spelling_errors'] = 0
            feats['style_errors'] = 0
            feats['total_errors'] = 0
            return feats

    feats = []
    print("Checking essays for grammar errors...")
    for _, row in tqdm(essays_df.iterrows(), total=len(essays_df)):
        text = str(row['essay'])
        if len(text.strip()) < 50:
            f = {'id': row['id'], 'grammar_errors': 0, 'spelling_errors': 0, 'style_errors': 0, 'total_errors': 0}
            feats.append(f)
            continue

        try:
            matches = global_lt_tool.check(text)
            f = {
                'id': row['id'],
                'grammar_errors': sum(1 for m in matches if m.category == 'GRAMMAR'),
                'spelling_errors': sum(1 for m in matches if m.category == 'SPELLING' or 'TYPO' in m.category),
                'style_errors': sum(1 for m in matches if m.category == 'STYLE' or m.category == 'TYPOGRAPHY'),
                'total_errors': len(matches)
            }
            feats.append(f)

        except Exception as e:
            print(f"Error checking essay {row['id']}: {e}")
            f = {'id': row['id'], 'grammar_errors': 0, 'spelling_errors': 0, 'style_errors': 0, 'total_errors': 0}
            feats.append(f)

    return pd.DataFrame(feats).fillna(0)


def get_replace_features(df):
    df_replace = df[df['activity'] == 'Replace'].copy()
    feats = []

    for essay_id, g in df_replace.groupby('id'):
        g['text_change_split'] = g['text_change'].str.split(' => ')
        g['old_text'] = g['text_change_split'].str[0].str.strip()
        g['new_text'] = g['text_change_split'].str[1].str.strip()

        g['old_len'] = g['old_text'].str.len()
        g['new_len'] = g['new_text'].str.len()
        g['len_diff'] = g['new_len'] - g['old_len']

        typo_fixes = (g['len_diff'] == 0).sum()
        elaborations = (g['len_diff'] > 0).sum()
        concisions = (g['len_diff'] < 0).sum()

        feats.append({
            'id': essay_id,
            'n_replace': len(g),
            'mean_replace_len_diff': g['len_diff'].mean(),
            'std_replace_len_diff': g['len_diff'].std(),
            'total_replace_len_diff': g['len_diff'].sum(),
            'typo_fix_ratio': typo_fixes / (len(g) + 1e-6),
            'elaboration_ratio': elaborations / (len(g) + 1e-6),
            'concision_ratio': concisions / (len(g) + 1e-6),
        })

    all_ids_df = pd.DataFrame({'id': df['id'].unique()})
    feat_cols = [
        'id', 'n_replace', 'mean_replace_len_diff', 'std_replace_len_diff',
        'total_replace_len_diff', 'typo_fix_ratio', 'elaboration_ratio', 'concision_ratio'
    ]

    if not feats:
        out_df = pd.DataFrame(columns=feat_cols)
    else:
        out_df = pd.DataFrame(feats)

    final_df = all_ids_df.merge(out_df, on='id', how='left').fillna(0)
    return final_df


def get_paragraph_features(essays_df):
    feats = []
    for _, row in essays_df.iterrows():
        text = str(row['essay'])
        paragraphs = re.split(r'\n+', text)
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 10]

        if not paragraphs:
            f = {
                'id': row['id'],
                'n_paragraphs': 0,
                'mean_paragraph_len': 0,
                'std_paragraph_len': 0,
                'mean_paragraph_words': 0,
                'std_paragraph_words': 0,
            }
            feats.append(f)
            continue

        para_lens_char = [len(p) for p in paragraphs]
        para_lens_words = [len(re.findall(r'\w+', p)) for p in paragraphs]

        f = {
            'id': row['id'],
            'n_paragraphs': len(paragraphs),
            'mean_paragraph_len': np.mean(para_lens_char),
            'std_paragraph_len': np.std(para_lens_char),
            'mean_paragraph_words': np.mean(para_lens_words),
            'std_paragraph_words': np.std(para_lens_words),
        }
        feats.append(f)

    return pd.DataFrame(feats).fillna(0)


# ============================================================
# 5. Load Data & Reconstruct Essays
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
all_ids = pd.Index(sorted(set(train_essays['id']).union(set(test_essays['id']))), name='id')


# ============================================================
# 6. Classic word TF-IDF bigrams + SVD
# ============================================================

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    lowercase=True
)

train_texts = train_essays["essay"].fillna("")
test_texts = test_essays["essay"].fillna("")

print("Fitting TF-IDF...")
X_tfidf = tfidf.fit_transform(train_texts)
X_test_tfidf = tfidf.transform(test_texts)

svd = TruncatedSVD(n_components=64, random_state=42)
print("Fitting SVD...")
train_tfidf_svd = svd.fit_transform(X_tfidf)
test_tfidf_svd = svd.transform(X_test_tfidf)

tfidf_cols = [f"tfidf_svd_{i}" for i in range(train_tfidf_svd.shape[1])]

train_tfidf_df = pd.DataFrame(train_tfidf_svd, columns=tfidf_cols)
train_tfidf_df.insert(0, "id", train_essays["id"].values)

test_tfidf_df = pd.DataFrame(test_tfidf_svd, columns=tfidf_cols)
test_tfidf_df.insert(0, "id", test_essays["id"].values)

print("TF-IDF+SVD features shape:", train_tfidf_df.shape)


# ============================================================
# 7. Diversified TF-IDF views
# ============================================================

def build_tfidf_views(train_texts, test_texts, random_state=42):
    views = []
    configs = [
        dict(name="w12_std", analyzer="word", ngram_range=(1, 2),
             lowercase=True, max_features=5000, sublinear_tf=False, binary=False, stop_words=None),
        dict(name="w12_sublinear", analyzer="word", ngram_range=(1, 2),
             lowercase=True, max_features=5000, sublinear_tf=True, binary=False, stop_words=None),
        dict(name="w13_binary", analyzer="word", ngram_range=(1, 3),
             lowercase=True, max_features=7000, sublinear_tf=False, binary=True, stop_words=None),
        dict(name="w12_stop", analyzer="word", ngram_range=(1, 2),
             lowercase=True, max_features=6000, sublinear_tf=False, binary=False, stop_words="english"),
        dict(name="c35_wb", analyzer="char_wb", ngram_range=(3, 5),
             lowercase=True, max_features=8000, sublinear_tf=False, binary=False, stop_words=None),
        dict(name="c47", analyzer="char", ngram_range=(4, 7),
             lowercase=True, max_features=8000, sublinear_tf=True, binary=False, stop_words=None),
    ]

    for cfg in configs:
        vec = TfidfVectorizer(
            analyzer=cfg["analyzer"], ngram_range=cfg["ngram_range"],
            lowercase=cfg["lowercase"], max_features=cfg["max_features"],
            sublinear_tf=cfg["sublinear_tf"], binary=cfg["binary"], stop_words=cfg["stop_words"]
        )
        Xtr = vec.fit_transform(train_texts)
        Xte = vec.transform(test_texts)

        base_rank = 64 if cfg["analyzer"].startswith("word") else 48
        max_rank = max(1, min(Xtr.shape[0] - 1, Xtr.shape[1] - 1))
        rank = min(base_rank, max_rank)

        if rank < 2:
            print(f"[TFIDF VIEW SKIPPED] {cfg['name']} has too few features (n_features={Xtr.shape[1]}).")
            continue

        svd = TruncatedSVD(n_components=rank, random_state=random_state)
        Ztr = svd.fit_transform(Xtr)
        Zte = svd.transform(Xte)

        cols = [f"{cfg['name']}_svd_{i}" for i in range(rank)]
        df_tr = pd.DataFrame(Ztr, columns=cols)
        df_te = pd.DataFrame(Zte, columns=cols)
        views.append((cfg["name"], df_tr, df_te))

    return views


train_texts = train_essays["essay"].fillna("")
test_texts = test_essays["essay"].fillna("")
tfidf_views = build_tfidf_views(train_texts, test_texts)


# ============================================================
# 7b. Merge features & prepare matrices
# ============================================================

train_feats = train_tfidf_df.merge(train_scores, on="id", how="left")
test_feats = test_tfidf_df.copy()

base_cols = [c for c in train_feats.columns if c not in ["id", "score"]]

X = train_feats[base_cols].copy()
y = train_feats["score"].astype(float).copy()
X_test = test_feats[base_cols].copy()

feature_views = {}
feature_views["tfidf_svd"] = (X, X_test)

for name, df_tr, df_te in tfidf_views:
    feature_views[name] = (df_tr, df_te)

print("train_feats shape:", train_feats.shape)
print("test_feats  shape:", test_feats.shape)
print("Base feature columns:", len(base_cols))
print("feature_views:", list(feature_views.keys()))
print("X shape:", X.shape, "| X_test shape:", X_test.shape)


# ============================================================
# SBERT (defined but not used in this run)
# ============================================================

SBERT_DIR = "/kaggle/input/sentencetransformersallminilml6v2"
SBERT_DIM = 384

def build_sbert_features_offline(train_texts, test_texts, ids_train, ids_test,
                                 model_dir=SBERT_DIR, dim=SBERT_DIM,
                                 batch_size=256, max_len=256, device="cpu"):

    required = ["config.json", "pytorch_model.bin", "tokenizer.json"]
    try:
        missing = [f for f in required if not os.path.exists(os.path.join(model_dir, f))]
        if missing:
            raise FileNotFoundError(f"SBERT model files missing: {missing} in {model_dir}")
    except Exception as e:
        print(f"[SBERT WARNING] {e}")
        sbert_cols = [f"sbert_{i}" for i in range(dim)]
        ztr = pd.DataFrame(np.zeros((len(ids_train), dim), np.float32), columns=sbert_cols)
        zte = pd.DataFrame(np.zeros((len(ids_test), dim), np.float32), columns=sbert_cols)
        ztr.insert(0, "id", ids_train.values); zte.insert(0, "id", ids_test.values)
        return ztr, zte

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True)
    model.to(device).eval()
    torch.set_num_threads(2)

    @torch.no_grad()
    def _encode(texts):
        chunks = []
        for i in range(0, len(texts), batch_size):
            batch = list(map(str, texts[i:i + batch_size]))
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=max_len, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            hs = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            emb = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            chunks.append(emb.cpu().numpy().astype(np.float32))
        return np.vstack(chunks)

    tr_series = pd.Series(train_texts).astype(str)
    te_series = pd.Series(test_texts).astype(str)
    tr_unique = tr_series.drop_duplicates()
    te_unique = te_series.drop_duplicates()

    tr_map = dict(zip(tr_unique.index, range(len(tr_unique))))
    te_map = dict(zip(te_unique.index, range(len(te_unique))))

    emb_tr_u = _encode(tr_unique.tolist())
    emb_te_u = _encode(te_unique.tolist())

    emb_tr = np.zeros((len(tr_series), emb_tr_u.shape[1]), np.float32)
    emb_te = np.zeros((len(te_series), emb_te_u.shape[1]), np.float32)
    for idx, pos in tr_map.items(): emb_tr[idx] = emb_tr_u[pos]
    for idx, pos in te_map.items(): emb_te[idx] = emb_te_u[pos]

    if emb_tr.shape[1] != dim:
        if emb_tr.shape[1] > dim:
            emb_tr, emb_te = emb_tr[:, :dim], emb_te[:, :dim]
        else:
            pad_tr = np.zeros((emb_tr.shape[0], dim - emb_tr.shape[1]), np.float32)
            pad_te = np.zeros((emb_te.shape[0], dim - emb_te.shape[1]), np.float32)
            emb_tr, emb_te = np.hstack([emb_tr, pad_tr]), np.hstack([emb_te, pad_te])

    cols = [f"sbert_{i}" for i in range(dim)]
    sbert_tr = pd.DataFrame(emb_tr, columns=cols); sbert_tr.insert(0, "id", ids_train.values)
    sbert_te = pd.DataFrame(emb_te, columns=cols); sbert_te.insert(0, "id", ids_test.values)
    print(f"Added SBERT features offline: {dim} dims | train={len(sbert_tr)} test={len(sbert_te)}")
    return sbert_tr, sbert_te


# ============================================================
# 8. Model Training with CV (tree models + linear text models)
# ============================================================

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Best params (from offline search)
rf_best_params = {
    'n_estimators': 652,
    'max_depth': 16,
    'min_samples_split': 9,
    'min_samples_leaf': 2,
    'max_features': 0.571259374981976,
    'random_state': 42,
    'n_jobs': -1,
}
xgb_best_params = {
    'n_estimators': 917,
    'max_depth': 3,
    'learning_rate': 0.010165732265520047,
    'subsample': 0.7383490273976494,
    'colsample_bytree': 0.7633333718927601,
    'reg_alpha': 1.452792067012294,
    'reg_lambda': 1.785498731400768,
    'min_child_weight': 6.6938953287759135,
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1,
}
lgb_best_params = {
    'n_estimators': 478,
    'num_leaves': 25,
    'max_depth': 3,
    'learning_rate': 0.020081552088704914,
    'subsample': 0.6962805629998192,
    'colsample_bytree': 0.6375828970481828,
    'reg_alpha': 0.4066458727082568,
    'reg_lambda': 1.414132007206554,
    'min_child_samples': 75,
    'objective': 'regression',
    'random_state': 42,
    'n_jobs': -1,
}
cat_best_params = {
    'iterations': 1380,
    'depth': 5,
    'learning_rate': 0.03195332803299333,
    'l2_leaf_reg': 5.701336410383776,
    'subsample': 0.8682663553348593,
    'rsm': 0.7760617223116183,
    'random_strength': 0.9510903829255853,
    'loss_function': 'RMSE',
    'random_seed': 42,
    'verbose': False,
    'allow_writing_files': False,
}

best_rf = RandomForestRegressor(**rf_best_params)
best_xgb = xgb.XGBRegressor(**xgb_best_params)
best_lgb = lgb.LGBMRegressor(**lgb_best_params)
best_cat = CatBoostRegressor(**cat_best_params)

models = {"rf": best_rf, "xgb": best_xgb, "lgb": best_lgb, "cat": best_cat}

y_strat = (y * 2).round().astype(int)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_cols = {"id": train_feats["id"].values}
test_cols = {"id": test_feats["id"].values}


def train_one_view(Xv, Xv_test, view_tag):
    view_oof = {}
    view_test = {}

    for mname, base_model in models.items():
        model = base_model.__class__(**base_model.get_params())
        print(f"\n[View={view_tag}] Training {mname.upper()}...")

        oof = np.zeros(len(Xv))
        preds = np.zeros(len(Xv_test))

        for fold, (tr_idx, val_idx) in enumerate(kf.split(Xv, y_strat)):
            X_tr, X_val = Xv.iloc[tr_idx], Xv.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

            if isinstance(model, xgb.XGBRegressor):
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                    early_stopping_rounds=50,
                )
            elif isinstance(model, lgb.LGBMRegressor):
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_val, y_val)],
                    eval_metric="rmse",
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )
            elif isinstance(model, CatBoostRegressor):
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=(X_val, y_val),
                    use_best_model=True,
                    early_stopping_rounds=50,
                    verbose=False,
                )
            else:
                model.fit(X_tr, y_tr)

            oof[val_idx] = model.predict(X_val)
            preds += model.predict(Xv_test) / kf.n_splits

        tag = f"{mname}@{view_tag}"
        view_oof[tag] = oof
        view_test[tag] = preds
        print(f"[View={view_tag}] {mname.upper()} CV RMSE: {rmse(y, oof):.5f}")

    return view_oof, view_test


# Train tree models on all feature views
for view_tag, (Xv, Xv_test) in feature_views.items():
    voof, vtest = train_one_view(Xv, Xv_test, view_tag)
    oof_cols.update(voof)
    test_cols.update(vtest)


# ============================================================
# 9. High-dimensional linear text models (TF-IDF + Ridge)
# ============================================================

def train_ridge_tfidf(X_sp, X_test_sp, y, kf, alpha, tag):
    """
    Train Ridge on sparse TF-IDF with explicit solver to avoid scipy.cg 'tol' issue.
    """
    oof = np.zeros(X_sp.shape[0])
    preds = np.zeros(X_test_sp.shape[0])

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_sp, y_strat)):
        X_tr, X_val = X_sp[tr_idx], X_sp[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        # ğŸ”§ FIX: use solver='lsqr' so sklearn doesn't call sparse_cg/cg with 'tol'
        model = Ridge(alpha=alpha, solver="lsqr")
        model.fit(X_tr, y_tr)
        oof[val_idx] = model.predict(X_val)
        preds += model.predict(X_test_sp) / kf.n_splits

    print(f"[Linear] {tag} CV RMSE: {rmse(y, oof):.5f}")
    return oof, preds


text_train = train_essays["essay"].fillna("").astype(str).tolist()
text_test = test_essays["essay"].fillna("").astype(str).tolist()

linear_text_configs = [
    (
        "ridge_char_3_5",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            max_features=150000,
            sublinear_tf=True,
            lowercase=True,
        ),
        8.0,
    ),
    (
        "ridge_char_2_6",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 6),
            min_df=2,
            max_features=150000,
            sublinear_tf=True,
            lowercase=True,
        ),
        10.0,
    ),
    (
        "ridge_word_1_2",
        TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_features=100000,
            sublinear_tf=True,
            lowercase=True,
            stop_words="english",
        ),
        5.0,
    ),
]

for tag, vec, alpha in linear_text_configs:
    print(f"\nFitting TF-IDF for {tag} ...")
    X_tr_lin = vec.fit_transform(text_train)
    X_te_lin = vec.transform(text_test)
    print(f"{tag} shapes: train={X_tr_lin.shape}, test={X_te_lin.shape}")

    oof_lin, test_lin = train_ridge_tfidf(X_tr_lin, X_te_lin, y, kf, alpha=alpha, tag=tag)
    oof_cols[tag] = oof_lin
    test_cols[tag] = test_lin


# Materialize OOF/Test prediction matrices
oof_preds = pd.DataFrame(oof_cols)
test_preds = pd.DataFrame(test_cols)

print("oof_preds columns (sample):", [c for c in oof_preds.columns if c != "id"][:10], "...")
print("test_preds columns (sample):", [c for c in test_preds.columns if c != "id"][:10], "...")


# ============================================================
# 10. Feature Importance Correlation (tree models only)
# ============================================================

def plot_feature_importances(models, X, y):
    importances = {}
    for name, model in models.items():
        if hasattr(model, "feature_importances_"):
            m = clone(model)
            m.fit(X, y)
            importances[name] = m.feature_importances_
        elif isinstance(model, CatBoostRegressor):
            m = CatBoostRegressor(**model.get_params())
            m.set_params(verbose=False, allow_writing_files=False, loss_function="RMSE", random_seed=42)
            m.fit(X, y, verbose=False)
            importances[name] = m.get_feature_importance(Pool(X, y))

    imp_df = pd.DataFrame(importances, index=X.columns)
    if not imp_df.empty:
        corr = imp_df.corr()
        print("\nğŸ“Š Feature Importance Correlation:\n", corr)
        plt.figure(figsize=(5, 4))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Feature Importance Correlation Between Tree Models")
        plt.show()
    else:
        print("No feature importances available for the provided models.")
    return imp_df


imp_df = plot_feature_importances(models, X, y)


# ============================================================
# 11. Correlation-aware stacking meta-learner
# ============================================================

def greedy_select(oof_df, target, corr_thresh=0.95):
    cols = [c for c in oof_df.columns if c != "id"]
    rmse_single = {c: np.sqrt(mean_squared_error(target, oof_df[c])) for c in cols}
    ordered = sorted(cols, key=rmse_single.get)
    corr = oof_df[cols].corr().abs()

    selected = []
    best_rmse = float("inf")
    for cand in ordered:
        if not selected:
            selected = [cand]
            best_rmse = rmse_single[cand]
            continue

        if any(corr.loc[cand, s] > corr_thresh for s in selected):
            continue

        trial = oof_df[selected + [cand]].values
        test_meta = RidgeCV(alphas=np.logspace(-3, 3, 20))
        test_meta.fit(trial, target)
        oof_hat = test_meta.predict(trial)
        cand_rmse = np.sqrt(mean_squared_error(target, oof_hat))

        if cand_rmse + 1e-7 < best_rmse:
            selected.append(cand)
            best_rmse = cand_rmse

    return selected


def fit_meta_and_score(X_meta, y, meta_kind="ridge"):
    if meta_kind == "ridge":
        meta = RidgeCV(alphas=np.logspace(-3, 3, 40))
    elif meta_kind == "lasso":
        meta = LassoCV(alphas=np.logspace(-4, 1, 50), cv=5, n_jobs=-1, max_iter=20000)
    elif meta_kind == "elastic":
        meta = ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
            alphas=np.logspace(-4, 1, 40),
            cv=5,
            n_jobs=-1,
            max_iter=20000,
        )
    else:
        raise ValueError(meta_kind)

    meta.fit(X_meta, y)
    oof_hat = meta.predict(X_meta)
    rmse_val = np.sqrt(mean_squared_error(y, oof_hat))
    return meta, rmse_val, oof_hat


def meta_sweep(oof_preds_df, y, test_preds_df, corr_thresholds=(0.95, 0.90, 0.99)):
    results = []
    for thr in corr_thresholds:
        chosen = greedy_select(oof_preds_df, y, corr_thresh=thr)
        Xm = oof_preds_df[chosen].values
        Xt = test_preds_df[chosen].values

        for kind in ["ridge", "lasso", "elastic"]:
            meta, rmse_val, oof_hat = fit_meta_and_score(Xm, y, meta_kind=kind)
            results.append(
                {
                    "corr_gate": thr,
                    "meta": kind,
                    "cols": chosen,
                    "rmse": rmse_val,
                    "model": meta,
                    "oof_hat": oof_hat,
                    "Xt": Xt,
                }
            )

    best = min(results, key=lambda r: r["rmse"])
    print(
        f"[META] Best: gate={best['corr_gate']} | {best['meta']} | "
        f"k={len(best['cols'])} | OOF {best['rmse']:.6f}"
    )
    coefs = getattr(best["model"], "coef_", None)
    if coefs is not None:
        print(
            "Meta weights (top 12):",
            sorted(list(zip(best["cols"], coefs)), key=lambda x: -abs(x[1]))[:12],
        )
    return best


best_meta = meta_sweep(oof_preds, y, test_preds, corr_thresholds=(0.95, 0.90, 0.99))

final_oof = best_meta["oof_hat"]
final_test = best_meta["model"].predict(best_meta["Xt"])
print(f"Final Stacked RMSE: {np.sqrt(mean_squared_error(y, final_oof)):.6f}")


# ============================================================
# 12. Save Submission
# ============================================================

submission = pd.DataFrame({"id": test_feats["id"], "score": np.ravel(final_test)})
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("\nâœ… submission.csv saved successfully!")
print(submission.head())


