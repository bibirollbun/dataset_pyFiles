!python -m pip install --upgrade -q pip uv
!python -m uv pip install --system -q autogluon


from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.impute import KNNImputer
import pandas as pd
import os


from google.colab import userdata
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
os.environ['KAGGLE_KEY'] = userdata.get('KAGGLE_KEY')
os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME')


!kaggle competitions download -c playground-series-s5e7


!unzip playground-series-s5e7.zip


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
sub_df = pd.read_csv('sample_submission.csv')


train_df.info()


from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import numpy as np

def preprocess_data(df, is_train=True, scaler=None, imputer=None):
    df = df.copy()

    if 'Personality' not in df.columns:
        df['Personality'] = np.nan

    binary_map = {'Yes': 1, 'No': 0}
    df['Stage_fear'] = df['Stage_fear'].map(binary_map)
    df['Drained_after_socializing'] = df['Drained_after_socializing'].map(binary_map)

    if is_train:
        df['Personality'] = df['Personality'].map({'Extrovert': 0, 'Introvert': 1})

    num_cols = df.select_dtypes(include='number').columns
    if 'Personality' in num_cols:
        num_cols = num_cols.drop('Personality')

    if is_train:
        imputer = KNNImputer(n_neighbors=5)
        df[num_cols] = imputer.fit_transform(df[num_cols])
    else:
        df[num_cols] = imputer.transform(df[num_cols])

    df['social_energy'] = (
        df['Social_event_attendance'] +
        df['Going_outside'] +
        df['Friends_circle_size'] +
        df['Post_frequency']
    ) - df['Time_spent_Alone']

    df['fear_vs_attend'] = df['Stage_fear'] * df['Social_event_attendance']
    df['drain_ratio'] = df['Drained_after_socializing'] / (df['Friends_circle_size'] + 1)

    scale_cols = ['Time_spent_Alone', 'Friends_circle_size', 'Post_frequency', 'social_energy', 'drain_ratio']

    if is_train:
        scaler = StandardScaler()
        df[scale_cols] = scaler.fit_transform(df[scale_cols])
    else:
        df[scale_cols] = scaler.transform(df[scale_cols])

    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    return df, scaler, imputer



train_df, scaler, imputer = preprocess_data(train_df, is_train=True)

test_df, _, _ = preprocess_data(test_df, is_train=False, scaler=scaler, imputer=imputer)


train_df.info()


train_df


test_df.info()


sub_df


from autogluon.tabular import TabularPredictor

label = "Personality"

# tabpfnmix_default = {
#     "model_path_classifier": "autogluon/tabpfn-mix-1.0-classifier",
#     "n_ensembles": 3,
#     "max_epochs": 5,
# }

# hyperparameters = {
#     "TABPFNMIX": [tabpfnmix_default],
# }

problem_type = "multiclass"

predictor = TabularPredictor(
    label=label,
    problem_type=problem_type,
    eval_metric="accuracy"
)

predictor = predictor.fit(
    train_data=train_df,
    presets="high",
    # hyperparameters=hyperparameters,
    time_limit=3600,
    verbosity=3,
    auto_stack=True
)


predictor.leaderboard(train_df)


test_df = test_df.drop(columns=['Personality'])


test_df


predictions = predictor.predict(test_df)


test_df['Personality'] = predictions.map({0: 'Extrovert', 1: 'Introvert'})


test_df


sub_df['Personality'] = test_df['Personality']
sub_df


sub_df


sub_df.to_csv("gluon_high_3600.csv",index=False)


!kaggle competitions submit -c playground-series-s5e7 -f gluon_high_3600.csv -m let_go

