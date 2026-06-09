from pathlib import Path
import polars as pl
import numpy as np
import os
target_col = "gesture"
common_extra_cols = ["row_id","sequence_counter"]
rotation_cols = [ 'rot_w', 'rot_x',
       'rot_y', 'rot_z']
accelerometer_cols = ['acc_x', 'acc_y', 'acc_z']

thermophile_columns = [f"thm_{i}" for i in range(1, 6)]
tof_cols = [f"tof_{i}_v{j}" for i in range(1, 6) for j in range(64)]
grouping_col = "sequence_id"
subject_col = "subject"
heat_col = "total_heat"
# doing the fe
cats_to_be = ["sex","handedness","adult_child"]


#retrain the model from scratch, but first testing the model that I have
train_df = pl.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
train_demo_df = pl.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv')


def randomly_nullify_df(train_df,parts_count=2):
    row_no = train_df.height //parts_count
    random_indices = np.random.choice(np.arange(row_no))
    all_cols = train_df.columns
    for col in all_cols:
        b_df = train_df.with_columns(pl.when(pl.arange(0,train_df.height).is_in(random_indices)).
                                     then(None).otherwise(train_df[col]).alias(col))
    return b_df
        


preprocessed_demog_df = randomly_nullify_df(train_demo_df)
preprocessed_quantitative_df = randomly_nullify_df(train_df)


def randomly_sample(train_df,demog_df,idx=None):
    rows_no = train_df.height//2
    subj_range = len(train_df[subject_col].unique())
    random_idx = np.random.choice(np.arange(subj_range)) if idx is None else idx
    subjc_id = demog_df[subject_col].to_list()[random_idx]
    quant_seq_df = train_df.filter(pl.col(subject_col)==subjc_id)
    demog_seq_df = demog_df.filter(pl.col(subject_col)==subjc_id)
    return quant_seq_df,demog_seq_df


import kaggle_evaluation.cmi_inference_server


def predict(sequence,demographics) -> str:
    return 'Text on phone'


normal_seq,demog_seq = randomly_sample(preprocessed_quantitative_df,preprocessed_demog_df)
result = predict(normal_seq,demog_seq)



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

