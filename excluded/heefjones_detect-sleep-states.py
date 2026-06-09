# data science
import numpy as np
import polars as pl
import pandas as pd
import datetime as dt

# machine learning
from xgboost import XGBClassifier
from tqdm import tqdm
import gc


# read in the data
root = '/kaggle/input/child-mind-institute-detect-sleep-states/'
data = pl.read_parquet(root + 'train_series.parquet')
events = pl.read_csv(root + 'train_events.csv')
test = pl.read_parquet(root + 'test_series.parquet')

# cast step
events = events.with_columns([pl.col("step").cast(pl.UInt32).alias("step")])


# create (onsent, wakeup) pairs
pairs = (events.pivot(values="step", index=["series_id", "night"], on="event"))

# filter the pairs for rows where exactly one of the two is null
faulty_pairs = pairs.filter((pl.col("onset").is_null() & pl.col("wakeup").is_not_null()) | (pl.col("onset").is_not_null() & pl.col("wakeup").is_null()))

# view
faulty_pairs


# iterate through faulty_pairs['series_id', 'night'] and index into events
for series_id, night in zip(faulty_pairs['series_id'], faulty_pairs['night']):
    # fill all "step" and "timestamp" values with null
    events = events.with_columns(
        pl.when((pl.col('series_id') == series_id) & (pl.col('night') == night)).then(None).otherwise(pl.col('step')).alias('step'),
        pl.when((pl.col('series_id') == series_id) & (pl.col('night') == night)).then(None).otherwise(pl.col('timestamp')).alias('timestamp'))

# check again for mismatches
pairs = (events.pivot(values="step", index=["series_id", "night"], on="event"))
faulty_pairs = pairs.filter((pl.col("onset").is_null() & pl.col("wakeup").is_not_null()) | (pl.col("onset").is_not_null() & pl.col("wakeup").is_null()))
faulty_pairs


def add_date_cols(df):
    """
    Add date columns to a DataFrame.

    Args:
    - df (pl.DataFrame): The DataFrame to modify.

    Returns:
    - df (pl.DataFrame): The modified DataFrame.
    """

    df = df.with_columns([
        # remove the timezone offset (get first 19 characters), convert to datetime
        pl.col('timestamp').str.slice(0, 19).str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S").alias('timestamp_datetime')]).with_columns([

            # create date and hour columns
            pl.col('timestamp_datetime').dt.date().alias('date'),
            pl.col('timestamp_datetime').dt.hour().alias('hour')])
    
    # drop the unnecessary columns
    df = df.drop(['timestamp', 'timestamp_datetime'])

    return df


# add date cols
data = add_date_cols(data)
events = add_date_cols(events)
test = add_date_cols(test)

# check range of dates
events.filter(pl.col('date').is_not_null())['date'].min(), events.filter(pl.col('date').is_not_null())['date'].max()


def label_data(df):
    """
    Create an "asleep" label column (1 for asleep, 0 for awake) based on the "event" column.

    Args:
    - df (pl.DataFrame): DataFrame containing sleep data.

    Returns:
    - (pl.DataFrame): DataFrame with the "asleep" label column added.
    """

    # create a 'sleep_indicator' column: +1 if event == "onset", -1 if event == "wakeup", else 0
    df = df.sort(["series_id", "step"]).with_columns(
        pl.when(pl.col("event") == "onset")
        .then(1)
        .when(pl.col("event") == "wakeup")
        .then(-1)
        .otherwise(0)
        .alias("sleep_indicator"))

    # do a cumulative sum of 'sleep_indicator' so that rows after an "onset" have cumsum > 0 until we hit a "wakeup"
    df = df.with_columns(pl.col("sleep_indicator").cum_sum().over("series_id").alias("sleep_cumsum"))

    # create a boolean column 'asleep' that is True if sleep_cumsum > 0
    df = df.with_columns((pl.col("sleep_cumsum") > 0).alias("asleep"))

    # drop old columns
    return df.drop(['event', 'sleep_indicator', 'sleep_cumsum'])


# label data with events
data = data.join(events[['series_id', 'step', 'event']], on=['series_id', 'step'], how='left').sort(by=['series_id', 'step'])

# add "asleep" column
data_labeled = label_data(data)

# check
data_labeled.sample()


def create_features(df):
    """
    Create features for each series.

    Args:
    - df: contains the series data with columns 'enmo', 'anglez'

    Returns:
    - pl.dataframe: dataframe with new features added.
    """

    # sort by series_id and step
    df = df.sort(["series_id", "step"])

    # convert to lazy frame for better performance
    lazy_df = df.lazy()

    # add diff cols
    lazy_df = lazy_df.with_columns([(pl.col('anglez').diff().abs().fill_null(0)).alias('anglez_diff'), 
                                 (pl.col('enmo').diff().abs().fill_null(0)).alias('enmo_diff')])


    # define the rolling aggregation functions
    agg_funcs = {"min":  lambda col, w: pl.col(col).rolling_min(w, center=True).fill_nan(0), 
                 "max":  lambda col, w: pl.col(col).rolling_max(w, center=True).fill_nan(0), 
                 "mean": lambda col, w: pl.col(col).rolling_mean(w, center=True).fill_nan(0), 
                 "std":  lambda col, w: pl.col(col).rolling_std(w, center=True).fill_nan(0)}

    # list of windows in minutes
    windows = [1, 3, 5, 7.5, 10, 12.5, 15, 20, 25, 30, 60, 120, 180, 240, 480]

    # loop over each window duration, creating rolling features
    for m in windows:
        # multiply by 12 to convert 5-second steps to minutes
        window_size = int(m * 12)

        # create a list to hold the expressions for this window size
        exprs = []

        # create rolling features for 'anglez' and 'enmo'
        for col in ['anglez', 'enmo']:
            for stat, func in agg_funcs.items():
                alias_name = f'{col}_{m}m_{stat}'
                exprs.append(func(col, window_size).alias(alias_name))
            
            # difference features
            diff_col = f'{col}_diff'
            for stat, func in agg_funcs.items():
                alias_name = f'{diff_col}_{m}m_{stat}'
                exprs.append(func(diff_col, window_size).alias(alias_name))
        
        # add the rolling features
        lazy_df = lazy_df.with_columns(exprs)

    # collect the results back into a standard df
    return lazy_df.collect()


def batch_data(data, batch_size=1_000_000):
    """
    Create batches of data for training.

    Args:
    - data (pl.DataFrame): Data to be batched.
    - batch_size (int): Size of each batch. Default is 1 million.

    Returns:
    - (generator): Yields batches of data.
    """

    # iterate through batches
    for i in range(0, len(data), batch_size):
        # get batch
        batch = data[i:i + batch_size]

        # generate features from the batch
        yield create_features(batch)


# create xgb
xgb = XGBClassifier(random_state=9, n_jobs=-1, tree_method='hist')

# define non-feature cols
non_feat_cols = ['series_id', 'step', 'date', 'asleep']

# iterate through batches
for i, features in tqdm(enumerate(batch_data(data_labeled)), desc='Iterating through batches'):
    # clear memory
    gc.collect()

    # define x and y
    X_batch = features.drop(non_feat_cols)
    y_batch = features.select("asleep").to_numpy().ravel()

    if i == 0:
        # for the first batch, train normally
        xgb.fit(X_batch, y_batch)
    else:
        # continue training using the current booster
        xgb.fit(X_batch, y_batch, xgb_model=xgb.get_booster())


def predict(data, classifier):
    """
    Takes a time series of (containing features and labels) and a classifier and returns a formatted submission dataframe.

    Args:
    - data (pl.DataFrame): Contains series data and sleep events.
    - classifier (sklearn.classifier): Trained classifier for prediction.

    Returns:
    - event_preds_df (pd.DataFrame): Contains predicted sleep events.
    """
    
    # non-feature columns
    non_feat_cols = ['series_id', 'step', 'date']

    # get unique series_ids as a list
    series_ids = data.select("series_id").unique().to_series().to_list()

    # list to accumulate event predictions
    event_preds = []

    # iterate through series_ids
    for sid in tqdm(series_ids, desc="Processing users"):
        # filter the data for the current series id
        user_data = data.filter(pl.col("series_id") == sid).sort("step")

        # create features
        features = create_features(user_data)

        # define X
        X = features.drop(non_feat_cols)

        # get preds
        preds = classifier.predict(X)
        probs = classifier.predict_proba(X)[:, 1]
        
        # append step, date, and predictions to X
        X = X.with_columns([pl.Series("step", features["step"]), pl.Series("date", features["date"]), pl.Series("pred", preds), pl.Series("prob", probs)])
        
        # calculate the difference in predictions to find changes
        X = X.with_columns(pl.col("pred").diff().alias("pred_diff"))
        
        # extract the 'step' values where the change indicates an onset (0 -> 1) or wakeup (1 -> 0)
        pred_onsets = X.filter(pl.col("pred_diff") > 0)["step"].to_list()
        pred_wakeups = X.filter(pl.col("pred_diff") < 0)["step"].to_list()
        
        # process events if we have at least one onset and wakeup
        if len(pred_onsets) > 0 and len(pred_wakeups) > 0:
            # if first wakeup occurs before the first onset, drop it
            if pred_wakeups[0] < pred_onsets[0]:
                pred_wakeups = pred_wakeups[1:]
            # if last onset occurs after the last wakeup, drop it
            if pred_onsets and pred_wakeups and pred_onsets[-1] > pred_wakeups[-1]:
                pred_onsets = pred_onsets[:-1]
            
            # create sleep segments only if the duration is at least 30 minutes
            segments = [(onset, wakeup) for onset, wakeup in zip(pred_onsets, pred_wakeups) if (wakeup - onset) >= (30 * 12)]

            # merge segments that are close together
            merged_segments = []
            current_start, current_end = segments[0]
            for onset, wakeup in segments[1:]:
                # merge segments that are separated by less than 2 consecutive hours
                if onset - current_end < (120 * 12):
                    current_end = wakeup
                else:
                    merged_segments.append((current_start, current_end))
                    current_start, current_end = onset, wakeup
            merged_segments.append((current_start, current_end))

            # keep only the longest window per night
            segments_by_night = {}
            for onset, wakeup in merged_segments:
                # get the date of the onset step
                night_key = X.filter(pl.col("step") == onset).select(pl.col("date")).to_series()[0]

                # get duration of the segment
                duration = wakeup - onset

                # check if current segment is longer than the existing one
                if night_key not in segments_by_night or duration > segments_by_night[night_key]["duration"]:
                    # update the segment for this night
                    segments_by_night[night_key] = {"onset": onset, "wakeup": wakeup, "duration": duration}

            # iterate through segments and get scores
            for night_key, seg in segments_by_night.items():
                # get onset and wakeup times
                onset_step, wakeup_step = seg["onset"], seg["wakeup"]

                # record only the longest sleep window per night
                sleep_segment = X.filter((pl.col("step") >= onset_step) & (pl.col("step") < wakeup_step))
                score = sleep_segment.select(pl.col("prob")).mean().item()
                
                # get onset and wakeup dates
                onset_date = X.filter(pl.col("step") == onset_step).select(pl.col("date")).to_series()[0]
                wakeup_date = X.filter(pl.col("step") == wakeup_step).select(pl.col("date")).to_series()[0]
                
                # append events to the list
                event_preds.append({"series_id": sid, "step": onset_step, "event": "onset", "score": score, "date": onset_date})
                event_preds.append({"series_id": sid, "step": wakeup_step, "event": "wakeup", "score": score, "date": wakeup_date})

    # create a pandas df for the preds
    if event_preds:
        event_preds_df = pd.DataFrame(event_preds)
    else:
        # create an empty DataFrame with the six required columns if no events were detected
        event_preds_df = pd.DataFrame(columns=['series_id', 'step', 'event', 'score', 'date'])

    # add 'row_id' col
    event_preds_df['row_id'] = range(len(event_preds_df))

    # reorder cols
    return event_preds_df[['row_id', 'series_id', 'step', 'event', 'score', 'date']]


# get predictions
preds = predict(test, xgb).drop('date', axis=1)

# check
preds


# save
preds.to_csv('submission.csv', index=False)




