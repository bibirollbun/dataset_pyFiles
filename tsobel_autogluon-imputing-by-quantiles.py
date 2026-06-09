!pip install autogluon pandas


from google.colab import drive
import pandas as pd
from autogluon.tabular import TabularPredictor


drive.mount('/content/drive')


df_train = pd.read_csv("/content/drive/MyDrive/datasets/introvert_extrovert/train.csv")
df_test = pd.read_csv("/content/drive/MyDrive/datasets/introvert_extrovert/test.csv")
submission = pd.read_csv("/content/drive/MyDrive/datasets/introvert_extrovert/sample_submission.csv")


df_datasert = (
    pd
    .read_csv('/content/drive/MyDrive/datasets/introvert_extrovert/personality_datasert.csv')
    .rename(columns={
        "Personality": "match_p"
    })
    .drop_duplicates(['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
       'Going_outside', 'Drained_after_socializing', 'Friends_circle_size',
       'Post_frequency'])
)

merge_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
              'Going_outside', 'Drained_after_socializing',
              'Friends_circle_size', 'Post_frequency']


df_test = df_test.merge(df_datasert, how='left', on=merge_cols)
df_train = df_train.merge(df_datasert, how='left', on=merge_cols)

train_ID = df_train['id']
test_ID = df_test['id']

df_train.drop("id", axis = 1, inplace=True)
df_test.drop("id", axis = 1, inplace=True)

ntrain = df_train.shape[0]
ntest = df_test.shape[0]
y_train = df_train['Personality'].values

df_train_features = df_train.drop(['Personality'], axis=1)
all_data = pd.concat((df_train_features, df_test)).reset_index(drop=True)


all_data['social_attend_bin'] = pd.qcut(
    all_data['Social_event_attendance'],
    q=[0, 0.25, 0.5, 0.75, 1.0],
    labels=['Q1', 'Q2', 'Q3', 'Q4']
)

def fill_by_group_median(df, group_col, target_col):
    return df[target_col].fillna(df.groupby(group_col)[target_col].transform('median'))

all_data['Time_spent_Alone'] = fill_by_group_median(
    all_data, group_col='social_attend_bin', target_col='Time_spent_Alone'
)

all_data.drop(columns=['social_attend_bin'], inplace=True)


all_data['Going_outside_bin'] = pd.qcut(
    all_data['Going_outside'],
    q=[0, 0.25, 0.5, 0.75, 1.0],
    labels=['Q1', 'Q2', 'Q3', 'Q4']
)

def fill_by_group_median(df, group_col, target_col):
    return df[target_col].fillna(df.groupby(group_col)[target_col].transform('median'))

all_data['Time_spent_Alone'] = fill_by_group_median(
    all_data, group_col='Going_outside_bin', target_col='Time_spent_Alone'
)

all_data.drop(columns=['Going_outside_bin'], inplace=True)


all_data['Going_outside_bin'] = pd.qcut(
    all_data['Going_outside'],
    q=[0, 0.25, 0.5, 0.75, 1.0],
    labels=['Q1', 'Q2', 'Q3', 'Q4']
)

def fill_by_group_median(df, group_col, target_col):
    return df[target_col].fillna(df.groupby(group_col)[target_col].transform('median'))

all_data['Social_event_attendance'] = fill_by_group_median(
    all_data, group_col='Going_outside_bin', target_col='Social_event_attendance'
)

all_data.drop(columns=['Going_outside_bin'], inplace=True)


all_data['Friends_circle_bin'] = pd.qcut(
    all_data['Friends_circle_size'],
    q=[0, 0.25, 0.5, 0.75, 1.0],
    labels=['Q1', 'Q2', 'Q3', 'Q4']
)

def fill_by_group_median(df, group_col, target_col):
    return df[target_col].fillna(df.groupby(group_col)[target_col].transform('median'))

all_data['Social_event_attendance'] = fill_by_group_median(
    all_data, group_col='Friends_circle_bin', target_col='Social_event_attendance'
)


all_data.drop(columns=['Friends_circle_bin'], inplace=True)


all_data['Post_frequency_bin'] = pd.qcut(
    all_data['Post_frequency'],
    q=[0, 0.25, 0.5, 0.75, 1.0],
    labels=['Q1', 'Q2', 'Q3', 'Q4']
)

def fill_by_group_median(df, group_col, target_col):
    return df[target_col].fillna(df.groupby(group_col)[target_col].transform('median'))

all_data['Social_event_attendance'] = fill_by_group_median(
    all_data, group_col='Post_frequency_bin', target_col='Social_event_attendance'
)

all_data.drop(columns=['Post_frequency_bin'], inplace=True)


def fill_missing_by_quantile_group(df, group_source_col, target_col, quantiles=[0, 0.25, 0.5, 0.75, 1.0], labels=None):
    if labels is None:
        labels = [f'Q{i+1}' for i in range(len(quantiles)-1)]

    temp_bin_col = f'{group_source_col}_bin'

    df[temp_bin_col] = pd.qcut(df[group_source_col], q=quantiles, labels=labels)

    df[target_col] = df[target_col].fillna(df.groupby(temp_bin_col)[target_col].transform('median'))

    df.drop(columns=[temp_bin_col], inplace=True)

    return df

all_data = fill_missing_by_quantile_group(
    df=all_data,
    group_source_col='Social_event_attendance',
    target_col='Going_outside'
)

all_data = fill_missing_by_quantile_group(
    df=all_data,
    group_source_col='Post_frequency',
    target_col='Friends_circle_size'
)
all_data = fill_missing_by_quantile_group(
    df=all_data,
    group_source_col='Going_outside',
    target_col='Friends_circle_size'
)
all_data = fill_missing_by_quantile_group(
    df=all_data,
    group_source_col='Friends_circle_size',
    target_col='Post_frequency'
)


all_data.fillna({
    'Stage_fear': 'unknown',
    'Drained_after_socializing': 'unknown'
}, inplace=True)
all_data.info()


all_data = pd.get_dummies(all_data, columns=['Stage_fear', 'Drained_after_socializing','match_p'], prefix=['Stage', 'Drained','match'])


X_train = all_data[:ntrain].copy()
X_train['Personality'] = y_train
X_test = all_data[ntrain:]


predictor = TabularPredictor(label="Personality")

predictor.fit(train_data=X_train, presets='best_quality', time_limit=60*40)


y_preds = predictor.predict(X_test)

y_preds.index.name = 'id'

y_preds.to_csv("submission.csv")

