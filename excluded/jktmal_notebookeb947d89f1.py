import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

from sklearn.ensemble import VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def generate_features(train_logs):
    
    chars_per_essay = train_logs[train_logs['activity'] == 'Input'].groupby('id').agg(
        chars_per_essay=('event_id', 'nunique')
    )



    grouped_train_logs = train_logs.groupby('id').agg(
        words_per_essay=('word_count', 'last'),
        max_up_time=('up_time', 'max'),
        min_down_time=('down_time', 'min'),
        events_per_essay=('event_id', 'nunique')
    )

 

    grouped_train_logs['writing_duration'] = grouped_train_logs['max_up_time'] - grouped_train_logs['min_down_time']

    grouped_train_logs = grouped_train_logs.merge(chars_per_essay, left_index=True, right_index=True)

    grouped_train_logs['chars_per_minute'] = 1000 * 60 * grouped_train_logs['chars_per_essay'] / grouped_train_logs['writing_duration']

    ### Pauzy

    train_logs.sort_values(by=['id', 'down_time'], inplace=True)

    train_logs['pause_time'] = (train_logs.groupby('id')['down_time'].shift(0) - train_logs.groupby('id')['up_time'].shift(1)).fillna(0)

  

    train_logs['is_pause'] = train_logs['pause_time'] > 2000

    pause_features = train_logs.groupby('id').agg(
        total_pauses=('is_pause', 'sum'),
        total_pause_time=('pause_time', 'sum')
    )


    grouped_train_logs = grouped_train_logs.merge(pause_features, left_index=True, right_index=True)

  

    grouped_train_logs['perc_pause_time'] = grouped_train_logs['total_pause_time'] / grouped_train_logs['writing_duration']

    grouped_train_logs['pauses_per_words'] = grouped_train_logs['total_pauses'] / grouped_train_logs['words_per_essay']

    ### Aktywności

   

    train_logs['text_change_len'] = train_logs['text_change'].str.len()

  
    train_logs.loc[train_logs['text_change'] == 'NoChange', 'text_change_len'] = 0

  

    activity_per_essay = {}
    activites = ['Remove/Cut', 'Nonproduction', 'Replace', 'Paste']

    for act in activites:
        activity_per_essay[act] = train_logs[train_logs['activity'] == act].groupby('id').agg(
            activity_per_essay=('event_id', 'nunique'),
            activity_len_per_essay=('text_change_len', 'sum')

        )

    activity_per_essay['Remove/Cut']

    for act in activites:
        grouped_train_logs = grouped_train_logs.merge(activity_per_essay[act],
                                                      left_index=True,
                                                      right_index=True,
                                                      suffixes=('', f"_{act}"), how='left')



    grouped_train_logs.rename(columns={'activity_per_essay': 'removes_per_essay',
                                       'activity_len_per_essay': 'removes_len_per_essay'}, inplace=True)

    grouped_train_logs['removes_per_words'] = grouped_train_logs['removes_per_essay'] / grouped_train_logs['words_per_essay']

    grouped_train_logs['replaces_per_words'] = grouped_train_logs['activity_per_essay_Replace'] / grouped_train_logs['words_per_essay']

    grouped_train_logs.fillna(0, inplace=True)

   


    ## Zaawansowane cechy

    grouped_train_logs['chars_per_words'] = grouped_train_logs['chars_per_essay'] / grouped_train_logs['words_per_essay']

    grouped_train_logs.reset_index(inplace=True)



    train_logs = train_logs.merge(grouped_train_logs[['id', 'min_down_time', 'writing_duration', 'words_per_essay']], on='id')

    train_logs['perc_time'] = (train_logs['up_time'] - train_logs['min_down_time']) / train_logs['writing_duration']
    train_logs['perc_words'] = train_logs['word_count'] / train_logs['words_per_essay']


    thres_res = {}

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for t in thresholds:
        subset = train_logs[train_logs['perc_words'] >= t]
        subset = pd.DataFrame(subset.groupby('id')['perc_time'].min())
        thres_res[t] = subset

    thres_tab = thres_res[thresholds[0]][['perc_time']]

    for t in thresholds[1:]:
        thres_tab = thres_tab.merge(thres_res[t][['perc_time']],
                                    left_index=True, right_index=True,
                                    suffixes=('', f"_{t}"))

    thres_cols = [f"perc_time_{t}" for t in thresholds]
    thres_tab.columns = thres_cols

 

    grouped_train_logs = grouped_train_logs.merge(thres_tab.reset_index(), on='id')


    ## Burst


    train_logs['group'] = (train_logs['is_pause'].cumsum().shift(fill_value=0))


    grouped = train_logs[~train_logs['is_pause'] & (train_logs['activity'] == 'Input')].groupby(['id', 'group'])

    writing_time_no_pause = grouped['up_time'].last() - grouped['down_time'].first()



    words_no_pause = grouped['word_count'].last() - grouped['word_count'].first()

  

    combined = pd.DataFrame({'writing_time_no_pause': writing_time_no_pause,
                             'words_no_pause': words_no_pause}).reset_index()

    train_logs = train_logs.merge(combined, on=['id', 'group'], how='left')

   

    burst_tab = train_logs.groupby('id')[['words_no_pause', 'writing_time_no_pause']].max().reset_index()

    n_bursts = train_logs.groupby('id')['group'].nunique()

    burst_tab = burst_tab.merge(n_bursts.reset_index())

  

    grouped_train_logs = grouped_train_logs.merge(burst_tab)

   

    grouped_train_logs['bursts_per_words'] = grouped_train_logs['group'] / grouped_train_logs['words_per_essay']
    grouped_train_logs['words_no_pause_per_words'] = grouped_train_logs['words_no_pause'] / grouped_train_logs['words_per_essay']
    grouped_train_logs['writin_time_no_pause_per_time'] = grouped_train_logs['writing_time_no_pause'] / grouped_train_logs['writing_duration']

    
    # R-burst

    train_logs['is_change'] = (train_logs['activity'].shift(1) != train_logs['activity'].shift(0)) & (train_logs['activity'] != 'Input')
    train_logs.loc[0, 'is_change'] = False
    train_logs['group'] = (train_logs['is_change'].cumsum().shift(fill_value=0))
    grouped = train_logs.groupby(['id', 'group'])

    writing_time_no_pause_r_burst = grouped['up_time'].last() - grouped['down_time'].first()
    words_no_pause_r_burst = grouped['word_count'].last() - grouped['word_count'].first()

    combined = pd.DataFrame({'writing_time_no_pause_r_burst': writing_time_no_pause_r_burst,
                             'words_no_pause_r_burst': words_no_pause_r_burst}).reset_index()

    train_logs = train_logs.merge(combined, on=['id', 'group'], how='left')


    burst_tab_r = train_logs.groupby('id')[['words_no_pause_r_burst', 'writing_time_no_pause_r_burst']].max().reset_index()

    n_bursts_r = train_logs.groupby('id')['group'].nunique()

    burst_tab_r = burst_tab_r.merge(n_bursts_r.reset_index())


    burst_tab_r.rename(columns={"group":"group_r_burst"}, inplace=True)

    grouped_train_logs = grouped_train_logs.merge(burst_tab_r)

    ## Cursor change rate

    train_logs['cursor_change'] = (train_logs['cursor_position'].shift(1) - train_logs['cursor_position'].shift(0)).fillna(1)

    mean_cursor_change = train_logs.groupby('id')['cursor_change'].mean()

    sd_cursor_change = train_logs.groupby('id')['cursor_change'].std()

    grouped_train_logs = grouped_train_logs.merge(mean_cursor_change.reset_index(), on='id')
    grouped_train_logs = grouped_train_logs.merge(sd_cursor_change.reset_index(), on='id')

    ## Revision

    revisions = ['Remove/Cut', 'Replace', 'Paste']
    revisions_per_essay = train_logs[train_logs['activity'].isin(revisions) & train_logs['cursor_change'] == 0].groupby('id').agg(
        immediate_revisions_per_essay=('event_id', 'nunique'))
    grouped_train_logs = grouped_train_logs.merge(revisions_per_essay.reset_index(), on='id', how='left')
    revisions_per_essay = train_logs[train_logs['activity'].isin(revisions) & train_logs['cursor_change'] != 0].groupby('id').agg(
        distant_revisions_per_essay=('event_id', 'nunique'))
    grouped_train_logs = grouped_train_logs.merge(revisions_per_essay.reset_index(), on='id', how='left')

    ## Chars per minute in process

    active = train_logs[train_logs['activity'] == 'Input']

    train_logs['process_duration'] = (active['up_time'] - active['down_time'])

    writing_process_duration = train_logs.groupby('id')['process_duration'].sum()

    production_duration = train_logs.groupby('id')['process_duration'].sum()

    grouped_train_logs = grouped_train_logs.merge(production_duration.reset_index(), on='id')

    grouped_train_logs['chars_per_minute_production'] = 1000 * 60 * grouped_train_logs['chars_per_essay'] / grouped_train_logs['process_duration']

    
    grouped_train_logs.rename(columns={"cursor_change_x":"mean_cursor_change", "cursor_change_y":"sd_cursor_change"}, inplace=True)
    grouped_train_logs.fillna(0, inplace=True)

    
    
    
    return grouped_train_logs


df = pd.read_csv("/kaggle/input/linking-writing-processes-to-writing-quality/train_logs.csv")
test_df = pd.read_csv("/kaggle/input/linking-writing-processes-to-writing-quality/test_logs.csv")
df_scores = pd.read_csv("/kaggle/input/linking-writing-processes-to-writing-quality/train_scores.csv")
sample_submission = pd.read_csv("/kaggle/input/linking-writing-processes-to-writing-quality/sample_submission.csv")


df_features = generate_features(df)


df_features


df_scores


X_train_regression = df_features.drop(['id'], axis=1)
y_train_regression = df_scores['score']


scaler = StandardScaler()

X_train_normalized = scaler.fit_transform(X_train_regression)


linear_reg = LinearRegression()
nn_reg = MLPRegressor(max_iter=1000, early_stopping=True, random_state=1)
rf_reg = RandomForestRegressor(max_depth=4, random_state=1)


estimators = [('linear', linear_reg),
              ('neural_network', nn_reg),
              ('rf', rf_reg)]


ereg = VotingRegressor(estimators)


ereg.fit(X_train_normalized, y_train_regression)


df_test_features = generate_features(test_df)


X_test_regression = df_test_features.drop(['id'], axis=1)


X_test_normalized = scaler.transform(X_test_regression)


df_test_features['score'] = ereg.predict(X_test_normalized)


submission_path = 'submission.csv'

df_test_features[['id', 'score']].to_csv(submission_path, index=False)




