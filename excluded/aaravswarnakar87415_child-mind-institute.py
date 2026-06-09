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


import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm



# ---------- 1) Load files ----------
DATA_DIR = Path('.')  # adjust if needed
train_events = pd.read_csv('/kaggle/input/child-mind-institute-detect-sleep-states/train_events.csv')
test_series = pd.read_parquet('/kaggle/input/child-mind-institute-detect-sleep-states/test_series.parquet')
sample_sub = pd.read_csv('/kaggle/input/child-mind-institute-detect-sleep-states/sample_submission.csv')



print("train_events rows:", len(train_events))
print("test_series rows:", len(test_series))
print("sample_sub rows:", len(sample_sub))
display(train_events.head())
display(test_series.head())



print("\nTest series columns:", test_series.columns.tolist())
print("Series count:", test_series['series_id'].nunique())



def detect_sleep_window(df_series,
                        enmo_col='enmo',
                        step_col='step',
                        roll_win=30,
                        low_percentile=20,
                        min_length_steps=30):
    """
    Input: df_series - DataFrame of single series sorted by step.
    Returns: (onset_step, wakeup_step) or None if not found.
    """
    s = df_series.copy()
    if enmo_col not in s.columns:
        return None

    # Ensure sorted
    s = s.sort_values(step_col).reset_index(drop=True)

    # Rolling median to smooth short spikes
    s['enmo_roll'] = s[enmo_col].rolling(roll_win, center=True, min_periods=1).median()

    # threshold: percentile of rolling enmo (per-series adaptive)
    threshold = np.percentile(s['enmo_roll'].dropna(), low_percentile)
    # create boolean mask low activity
    s['low'] = (s['enmo_roll'] <= threshold).astype(int)

    # find contiguous runs of low==1
    # create group id for every change in mask
    s['grp'] = (s['low'] != s['low'].shift(1)).cumsum()
    runs = s.groupby(['grp', 'low']).agg(
        start_idx=('index', 'first') if 'index' in s.columns else (step_col, 'first'),
        end_idx=(step_col, 'last'),
        count=(step_col, 'count')
    ).reset_index()
    # keep only low runs
    low_runs = runs[runs['low'] == 1].copy()
    if low_runs.empty:
        return None

    # choose the longest low run (by count)
    best = low_runs.sort_values('count', ascending=False).iloc[0]
    if best['count'] < min_length_steps:
        # run too short -> no valid sleep window found
        return None

    # map start and end steps
    # start step is first step where low==1 in that group
    grp_id = best['grp']
    grp_rows = s[s['grp'] == grp_id]
    onset_step = int(grp_rows[step_col].iloc[0])
    wakeup_step = int(grp_rows[step_col].iloc[-1])
    return onset_step, wakeup_step



# Better implementation for contiguous detection (works without 'index' column)
def detect_sleep_window_v2(df_series,
                           enmo_col='enmo',
                           step_col='step',
                           roll_win=30,
                           low_percentile=20,
                           min_length_steps=30):
    s = df_series.copy().sort_values(step_col).reset_index(drop=True)
    if enmo_col not in s.columns:
        return None
    s['enmo_roll'] = s[enmo_col].rolling(roll_win, center=True, min_periods=1).median()
    thresh = np.percentile(s['enmo_roll'].dropna(), low_percentile)
    mask = (s['enmo_roll'] <= thresh).astype(int).values
    if mask.sum() == 0:
        return None

    # find contiguous True segments
    segments = []
    start = None
    for i, v in enumerate(mask):
        if v == 1 and start is None:
            start = i
        elif v == 0 and start is not None:
            segments.append((start, i-1))
            start = None
    if start is not None:
        segments.append((start, len(mask)-1))

    if not segments:
        return None

    # choose longest segment
    seg_lengths = [(s_e - s_s + 1, s_s, s_e) for (s_s, s_e) in segments]
    seg_lengths.sort(reverse=True)
    longest = seg_lengths[0]
    length, s_idx, e_idx = longest
    if length < min_length_steps:
        return None

    onset_step = int(s[step_col].iloc[s_idx])
    wakeup_step = int(s[step_col].iloc[e_idx])
    return onset_step, wakeup_step


# ---------- 4) Run detection across all series in test set ----------
# We'll run series-by-series. If test set is large, this may take time.
# Use tqdm for progress bar.
out_rows = []
series_ids = test_series['series_id'].unique()
print("Total series to process (test):", len(series_ids))

# Parameters you may tune:
ROLL_WIN = 60           # smoothing window (samples)
LOW_PERCENTILE = 20     # percentile to define low activity
MIN_LENGTH_STEPS = 30   # minimum contiguous low region length (samples) to be considered sleep

for sid in tqdm(series_ids):
    s_df = test_series[test_series['series_id'] == sid][['series_id', 'step', 'enmo']].copy()
    # if enmo missing, skip
    if s_df['enmo'].isna().all():
        continue

    res = detect_sleep_window_v2(s_df,
                                 enmo_col='enmo',
                                 step_col='step',
                                 roll_win=ROLL_WIN,
                                 low_percentile=LOW_PERCENTILE,
                                 min_length_steps=MIN_LENGTH_STEPS)
    if res is None:
        # No predicted sleep window -> skip or optionally predict nothing
        continue
    onset, wakeup = res

    # Add two predictions: onset and wakeup, with a heuristic score (1.0)
    # Row ordering / row_id will be generated later to match sample submission format
    out_rows.append({'series_id': sid, 'step': onset, 'event': 'onset', 'score': 1.0})
    out_rows.append({'series_id': sid, 'step': wakeup, 'event': 'wakeup', 'score': 1.0})



# ---------- 5) Build submission DataFrame ----------
pred_df = pd.DataFrame(out_rows)
print("Predictions count:", len(pred_df))
display(pred_df.head())

# If sample_submission contains a specific ordering or enumerated 'row_id', we must follow it.
# The sample submission format in this competition uses a 'row_id' which is a simple enumeration.
# We'll create a submission dataframe with columns: row_id, series_id, step, event, score
if 'row_id' in sample_sub.columns:
    # create row_id in the order we generated predictions
    pred_df = pred_df.reset_index(drop=True)
    pred_df.insert(0, 'row_id', pred_df.index.astype(int))
    sub_df = pred_df[['row_id', 'series_id', 'step', 'event', 'score']]
else:
    # If sample submission doesn't have row_id, just use our columns
    sub_df = pred_df[['series_id', 'step', 'event', 'score']]


# ---------- 6) Save submission ----------
out_path = DATA_DIR / 'submission.csv'
sub_df.to_csv(out_path, index=False)
print("Saved submission to:", out_path)

# ---------- 7) Quick evaluation on train (optional) ----------
# We can apply the same heuristic on training series and compare to train_events (VERY rough).
# This helps check if the heuristic is doing something sensible.
do_quick_train_check = True
if do_quick_train_check:
    # load train series if available (train_series.parquet). Many public kernels have train_series.parquet.
    try:
        train_series = pd.read_parquet(DATA_DIR / 'train_series.parquet')
        # run detection on each train series and compare to train_events
        # We'll compute how many labeled events we found (onset/wakeup)
        found = 0
        total_events = len(train_events)
        # build predicted events for train
        preds = []
        for sid in train_series['series_id'].unique():
            s_df = train_series[train_series['series_id'] == sid][['series_id','step','enmo']].copy()
            res = detect_sleep_window_v2(s_df,
                                         enmo_col='enmo',
                                         step_col='step',
                                         roll_win=ROLL_WIN,
                                         low_percentile=LOW_PERCENTILE,
                                         min_length_steps=MIN_LENGTH_STEPS)
            if res is None:
                continue
            onset, wakeup = res
            preds.append((sid, onset, 'onset'))
            preds.append((sid, wakeup, 'wakeup'))
        preds_df = pd.DataFrame(preds, columns=['series_id','step','event'])
        # naive matching: exact match on (series_id, step, event)
        merged = train_events.merge(preds_df, on=['series_id','step','event'], how='inner')
        found = len(merged)
        print(f"Quick train check: found {found} of {total_events} labelled events by exact match (very strict).")
    except FileNotFoundError:
        print("train_series.parquet not found — skipping train check (optional).")
    except Exception as e:
        print("Train check error (skipping).", e)



