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


!pip install /kaggle/input/offline-sktime/scikit_base-0.12.5-py3-none-any.whl
!pip install /kaggle/input/offline-sktime/sktime-0.38.5-py3-none-any.whl


from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import f1_score
import kaggle_evaluation.cmi_inference_server
from datetime import datetime
import polars as pl
pd.set_option('display.max_columns', 500)
import warnings
warnings.filterwarnings("ignore")


train_df = pd.read_csv(r'/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
test_df = pd.read_csv(r'/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv')


train_df.shape, test_df.shape


test_pl = pl.read_csv(r'/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv')
test_demo_pl = pl.read_csv(r'/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv')


train_df.head(2)


train_df[['sequence_type','phase','gesture']].drop_duplicates()


train_df.head(2)


train_df.shape


for col in [col for col in train_df.columns if col.startswith('tof_')]:
    train_df[col] = train_df[col].fillna(-2)

for col in [col for col in train_df.columns if col.startswith('thm_')]:
    train_df[col] = train_df[col].fillna(-2)

for col in [col for col in train_df.columns if col.startswith('rot_')]:
    train_df[col] = train_df[col].fillna(-2)


train_df.head(2)


tof1 = [col for col in test_df.columns if col.startswith('tof_1')]
tof2 = [col for col in test_df.columns if col.startswith('tof_2')]
tof3 = [col for col in test_df.columns if col.startswith('tof_3')]
tof4 = [col for col in test_df.columns if col.startswith('tof_4')]
tof5 = [col for col in test_df.columns if col.startswith('tof_5')]


d = {'tof1': tof1, 'tof2':tof2, 'tof3':tof3, 'tof4':tof4, 'tof5':tof5}


for k, col in d.items():
    test_df[f'Mean_{k}'] = test_df[col].mean(axis = 1)
    test_df[f'Min_{k}'] = test_df[col].mean(axis = 1)
    test_df[f'Max_{k}'] = test_df[col].mean(axis = 1)

    train_df[f'Mean_{k}'] = train_df[col].mean(axis = 1)
    train_df[f'Min_{k}'] = train_df[col].mean(axis = 1)
    train_df[f'Max_{k}'] = train_df[col].mean(axis = 1)


train_df.head(2)


train_df.drop(columns = [col for col in train_df.columns if col.startswith('tof')], axis = 1 , inplace = True)
test_df.drop(columns = [col for col in test_df.columns if col.startswith('tof')], axis = 1 , inplace = True)


train_df.head(2)


test_df.head(2)


train_df.shape, test_df.shape


total_columns = ['acc_x','acc_y','acc_z',
                 'rot_x', 'rot_y', 'rot_z', 'rot_w',
                'thm_1','thm_2','thm_3','thm_4','thm_5',
                'Mean_tof1','Min_tof1','Max_tof1',
                 'Mean_tof2','Min_tof2','Max_tof2',
                 'Mean_tof3','Min_tof3', 'Max_tof3',
                 'Mean_tof4','Min_tof4', 'Max_tof4',
                 'Mean_tof5', 'Min_tof5', 	'Max_tof5'] 


from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()


train_scaled = pd.DataFrame(scaler.fit_transform(train_df[total_columns]), 
             columns = scaler.get_feature_names_out())


test_scaled = pd.DataFrame(scaler.transform(test_df[total_columns]), 
             columns = scaler.get_feature_names_out())


train_df.drop(total_columns, axis = 1 , inplace = True)
test_df.drop(total_columns, axis = 1 , inplace = True)


train_df.shape, train_scaled.shape


train_df = pd.concat([train_df.reset_index(drop=True), train_scaled.reset_index(drop=True)],axis = 1)


test_df = pd.concat([test_df.reset_index(drop=True), test_scaled.reset_index(drop=True)],axis = 1)


train_df.head(2)


test_df.head(2)


train_df = train_df.groupby(['sequence_id','sequence_counter','gesture'], as_index = False)[total_columns].mean()


train_df


seq_target = train_df[['sequence_id','gesture']].drop_duplicates()


test_df = test_df.groupby(['sequence_id','sequence_counter'], as_index = False)[total_columns].mean()


test_df.head(2)


def resample_sequence(group, target_length):
    """Resample one sequence (all dimensions) to target length."""
    new_time = np.linspace(0, len(group) - 1, target_length)
    resampled = {"sequence_id": group["sequence_id"].iloc[0]}
    
    for col in group.columns:
        if col not in ["sequence_id", "sequence_counter"]:
            resampled[col] = np.interp(
                new_time,
                np.arange(len(group)),
                group[col].values
            )
    return pd.DataFrame(resampled)


def make_equal_length(df, target_length=None):
    """Resample all sequences in df to the same length."""
    # If no target length specified, use max length across sequences
    if target_length is None:
        target_length = df.groupby("sequence_id").size().max()

    resampled_dfs = []
    for seq_id, group in df.groupby("sequence_id"):
        resampled_group = resample_sequence(group.sort_values("sequence_counter"), target_length)
        resampled_group["sequence_counter"] = np.arange(target_length)
        resampled_dfs.append(resampled_group)
    
    return pd.concat(resampled_dfs, ignore_index=True)


train_df_same_lenghts = make_equal_length(train_df[[col for col in train_df.columns if col not in 'gesture']])


train_df_same_lenghts.head(2)


train_df = seq_target.merge(train_df_same_lenghts, how = 'left', left_on = 'sequence_id' , right_on = 'sequence_id')


train_df.head(2)


def to_sktime_panel(df, label_col=None):
    """Convert long-format DataFrame to sktime multivariate panel format."""
    X_list = []
    y_list = []

    # Group by sequence (each gesture example)
    for seq_id, group in df.groupby("sequence_id"):
        # Drop sequence_id and counter
        group = group.sort_values("sequence_counter")

        # Collect each dimension as a pandas Series
        acc_x = pd.Series(group["acc_x"].values)
        acc_y = pd.Series(group["acc_y"].values)
        acc_z = pd.Series(group["acc_z"].values)
        rot_x = pd.Series(group["rot_x"].values)
        rot_y = pd.Series(group["rot_y"].values)
        rot_z = pd.Series(group["rot_z"].values)
        rot_w = pd.Series(group["rot_w"].values)
        thm_1 = pd.Series(group["thm_1"].values)
        thm_2 = pd.Series(group["thm_2"].values)
        thm_3 = pd.Series(group["thm_3"].values)
        thm_4 = pd.Series(group["thm_4"].values)
        thm_5 = pd.Series(group["thm_5"].values)
        Mean_tof1 = pd.Series(group["Mean_tof1"].values)
        Min_tof1 = pd.Series(group["Min_tof1"].values)
        Max_tof1 = pd.Series(group["Max_tof1"].values)
        Mean_tof2 = pd.Series(group["Mean_tof2"].values)
        Min_tof2 = pd.Series(group["Min_tof2"].values)
        Max_tof2 = pd.Series(group["Max_tof2"].values)
        Mean_tof3 = pd.Series(group["Mean_tof3"].values)
        Min_tof3 = pd.Series(group["Min_tof3"].values)
        Max_tof3 = pd.Series(group["Max_tof3"].values)
        Mean_tof4 = pd.Series(group["Mean_tof4"].values)
        Min_tof4 = pd.Series(group["Min_tof4"].values)
        Max_tof4 = pd.Series(group["Max_tof4"].values)
        Mean_tof5 = pd.Series(group["Mean_tof5"].values)
        Min_tof5 = pd.Series(group["Min_tof5"].values)
        Max_tof5 = pd.Series(group["Max_tof5"].values)

        # Each row in sktime multivariate panel = [Series_x, Series_y, Series_z]
        X_list.append([acc_x, acc_y, acc_z,rot_x, rot_y,
           rot_z, rot_w, thm_1, thm_2, thm_3, thm_4, thm_5,
           Mean_tof1, Min_tof1, Max_tof1, Mean_tof2, Min_tof2,
           Max_tof2, Mean_tof3, Min_tof3, Max_tof3, Mean_tof4,
           Min_tof4, Max_tof4, Mean_tof5, Min_tof5, Max_tof5])

        if label_col is not None:
            # Take label (same for all rows in a sequence)
            y_list.append(group[label_col].iloc[0])

    # Convert to nested DataFrame (sktime format: one row per sequence, one col per dimension)
    X = pd.DataFrame(X_list, columns=['acc_x', 'acc_y', 'acc_z', 'rot_x', 'rot_y',
       'rot_z', 'rot_w', 'thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5',
       'Mean_tof1', 'Min_tof1', 'Max_tof1', 'Mean_tof2', 'Min_tof2',
       'Max_tof2', 'Mean_tof3', 'Min_tof3', 'Max_tof3', 'Mean_tof4',
       'Min_tof4', 'Max_tof4', 'Mean_tof5', 'Min_tof5', 'Max_tof5'])

    if label_col is not None:
        y = pd.Series(y_list)
        return X, y
    else:
        return X


X,y = to_sktime_panel(train_df,'gesture')


X


y


model = ShapeletTransformClassifier(
    n_shapelet_samples=500,  # number of candidate shapelets to search
    max_shapelets=360,       # number of shapelets kept
    random_state=42,
    #estimator = RandomForestClassifier(n_estimators=200, random_state=42),
    time_limit_in_minutes = 100
)


datetime.now()


model.fit(X, y)


datetime.now()


import pickle
from pathlib import Path


out_path = Path("/kaggle/working/shapelet_model.pkl")


out_path = Path("/kaggle/working/shapelet_model.pkl")
with open(out_path, "wb") as f:
    pickle.dump(model, f)

print("Saved:", out_path)


target_length = train_df.groupby("sequence_id").size().max()


target_length


test_df_same_lenghts = make_equal_length(test_df,target_length)


test_df = to_sktime_panel(test_df_same_lenghts, label_col=None)


test_df


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    try:
        test_df = sequence.to_pandas()
        demographics_df = demographics.to_pandas()
        for col in [col for col in test_df.columns if col.startswith('tof_')]:
            test_df[col] = test_df[col].fillna(-2)
        for col in [col for col in test_df.columns if col.startswith('thm_')]:
            test_df[col] = test_df[col].fillna(-2)
        for col in [col for col in test_df.columns if col.startswith('rot_')]:
            test_df[col] = test_df[col].fillna(-2)    
        tof1 = [col for col in test_df.columns if col.startswith('tof_1')]
        tof2 = [col for col in test_df.columns if col.startswith('tof_2')]
        tof3 = [col for col in test_df.columns if col.startswith('tof_3')]
        tof4 = [col for col in test_df.columns if col.startswith('tof_4')]
        tof5 = [col for col in test_df.columns if col.startswith('tof_5')]
        d = {'tof1': tof1, 'tof2':tof2, 'tof3':tof3, 'tof4':tof4, 'tof5':tof5}
        for k, col in d.items():
            test_df[f'Mean_{k}'] = test_df[col].mean(axis = 1)
            test_df[f'Min_{k}'] = test_df[col].mean(axis = 1)
            test_df[f'Max_{k}'] = test_df[col].mean(axis = 1)
        test_df.drop(columns = [col for col in test_df.columns if col.startswith('tof')], axis = 1 , inplace = True)
        total_columns = ['acc_x','acc_y','acc_z',
                 'rot_x', 'rot_y', 'rot_z', 'rot_w',
                'thm_1','thm_2','thm_3','thm_4','thm_5',
                'Mean_tof1','Min_tof1','Max_tof1',
                 'Mean_tof2','Min_tof2','Max_tof2',
                 'Mean_tof3','Min_tof3', 'Max_tof3',
                 'Mean_tof4','Min_tof4', 'Max_tof4',
                 'Mean_tof5', 'Min_tof5', 	'Max_tof5']
        test_scaled = pd.DataFrame(scaler.transform(test_df[total_columns]), 
             columns = scaler.get_feature_names_out())
        test_df.drop(total_columns, axis = 1 , inplace = True)
        test_df = pd.concat([test_df.reset_index(drop=True), test_scaled.reset_index(drop=True)],axis = 1)
        test_df = test_df.groupby(['sequence_id','sequence_counter'], as_index = False)[total_columns].mean()
        test_df_same_lenghts = make_equal_length(test_df,71 )
        test_df = to_sktime_panel(test_df_same_lenghts, label_col=None)
        return str(model.predict(test_df)[0])
    except Exception as e:
        print(f'Error in prediction: {e}')
        return 'Forehead - pull hairline'   


predict(test_pl,test_demo_pl)


inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )


pd.read_parquet(r'/kaggle/working/submission.parquet')

