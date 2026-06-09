import pandas as pd
import polars as pl

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

import os
from pathlib import Path

from tqdm.auto import tqdm


data_dir = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data")


train_df = pl.read_csv(data_dir / "train.csv")
test_df = pl.read_csv(data_dir / "test.csv")


train_demo = pl.read_csv(data_dir / "train_demographics.csv")
test_demo = pl.read_csv(data_dir / "test_demographics.csv")


labels_df = train_df.unique(
    subset=["sequence_id", "gesture"]
).select(
    ["sequence_id", "gesture"]
)


seq2sub = train_df.unique(
    subset=["sequence_id", "subject"]
).select(
    ["sequence_id", "subject"]
)


# we"ll only do acc since those are the not null features
train_df_subset = train_df.select([
    "sequence_id",
    "acc_x",
    "acc_y",
    "acc_z"
])


ts_features = train_df_subset.group_by("sequence_id").agg(
    pl.all().mean().name.suffix("_mean"),
    pl.all().min().name.suffix("_min"),
    pl.all().max().name.suffix("_max"),
    pl.all().var().name.suffix("_var"),
    pl.all().std().name.suffix("_std"),
    pl.all().skew().name.suffix("_skew"),
    pl.all().kurtosis().name.suffix("_kurtosis"),
    
    pl.all().diff(1).mean().name.suffix("_mean_diff1"),
    pl.all().diff(1).min().name.suffix("_min_diff1"),
    pl.all().diff(1).max().name.suffix("_max_diff1"),
    pl.all().diff(1).var().name.suffix("_var_diff1"),
    pl.all().diff(1).std().name.suffix("_std_diff1"),
    pl.all().diff(1).skew().name.suffix("_skew_diff1"),
    pl.all().diff(1).kurtosis().name.suffix("_kurtosis_diff1"),
    
    pl.all().cum_sum().mul(0.1).mean().name.suffix("_vel_mean"),
    pl.all().cum_sum().mul(0.1).min().name.suffix("_vel_min"),
    pl.all().cum_sum().mul(0.1).max().name.suffix("_vel_max"),
    pl.all().cum_sum().mul(0.1).var().name.suffix("_vel_var"),
    pl.all().cum_sum().mul(0.1).std().name.suffix("_vel_std"),
    pl.all().cum_sum().mul(0.1).skew().name.suffix("_vel_skew"),
    pl.all().cum_sum().mul(0.1).kurtosis().name.suffix("_vel_kurtosis"),

    pl.all().cum_sum().cum_sum().mul(0.1).mean().name.suffix("_pos_mean"),
    pl.all().cum_sum().cum_sum().mul(0.1).min().name.suffix("_pos_min"),
    pl.all().cum_sum().cum_sum().mul(0.1).max().name.suffix("_pos_max"),
    pl.all().cum_sum().cum_sum().mul(0.1).var().name.suffix("_pos_var"),
    pl.all().cum_sum().cum_sum().mul(0.1).std().name.suffix("_pos_std"),
    pl.all().cum_sum().cum_sum().mul(0.1).skew().name.suffix("_pos_skew"),
    pl.all().cum_sum().cum_sum().mul(0.1).kurtosis().name.suffix("_pos_kurtosis"),
)


df_total = ts_features.join(
    seq2sub, on="sequence_id"
).join(
    train_demo, on="subject"
).join(
    labels_df, on="sequence_id" 
).drop(["sequence_id", "subject"])


df_total.columns


gestures = sorted(labels_df["gesture"].unique().to_list())
gestures2id = {gesture:idx for idx,gesture in enumerate(gestures)}
id2gestures = {idx:gesture for gesture,idx in gestures2id.items()}


target_gestures = set([
    "Above ear - pull hair",
    "Cheek - pinch skin",
    "Eyebrow - pull hair",
    "Eyelash - pull hair",
    "Forehead - pull hairline",
    "Forehead - scratch",
    "Neck - pinch skin",
    "Neck - scratch",
])

non_target_gestures = set([
    "Write name on leg",
    "Wave hello",
    "Glasses on/off",
    "Text on phone",
    "Write name in air",
    "Feel around in tray and pull out an object",
    "Scratch knee/leg skin",
    "Pull air toward your face",
    "Drink from bottle/cup",
    "Pinch knee/leg skin",
])


gestures2id, id2gestures, target_gestures, non_target_gestures


df_total = df_total.with_columns(
    gesture_label=pl.col("gesture").map_elements(lambda x: gestures2id[x])
)

df_total = df_total.with_columns(
    target_gesture=pl.col("gesture").map_elements(lambda x: x in target_gestures)
)


from sklearn.linear_model import LogisticRegressionCV
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


from sklearn.model_selection import KFold, train_test_split

from sklearn.metrics import f1_score, accuracy_score


numeric_cols = [
"acc_x_mean",
"acc_y_mean",
"acc_z_mean",
"acc_x_min",
"acc_y_min",
"acc_z_min",
"acc_x_max",
"acc_y_max",
"acc_z_max",
"acc_x_var",
"acc_y_var",
"acc_z_var",
"acc_x_std",
"acc_y_std",
"acc_z_std",
"acc_x_skew",
"acc_y_skew",
"acc_z_skew",
"acc_x_kurtosis",
"acc_y_kurtosis",
"acc_z_kurtosis",
"acc_x_mean_diff1",
"acc_y_mean_diff1",
"acc_z_mean_diff1",
"acc_x_min_diff1",
"acc_y_min_diff1",
"acc_z_min_diff1",
"acc_x_max_diff1",
"acc_y_max_diff1",
"acc_z_max_diff1",
"acc_x_var_diff1",
"acc_y_var_diff1",
"acc_z_var_diff1",
"acc_x_std_diff1",
"acc_y_std_diff1",
"acc_z_std_diff1",
"acc_x_skew_diff1",
"acc_y_skew_diff1",
"acc_z_skew_diff1",
"acc_x_kurtosis_diff1",
"acc_y_kurtosis_diff1",
"acc_z_kurtosis_diff1",
"acc_x_vel_mean",
"acc_y_vel_mean",
"acc_z_vel_mean",
"acc_x_vel_min",
"acc_y_vel_min",
"acc_z_vel_min",
"acc_x_vel_max",
"acc_y_vel_max",
"acc_z_vel_max",
"acc_x_vel_var",
"acc_y_vel_var",
"acc_z_vel_var",
"acc_x_vel_std",
"acc_y_vel_std",
"acc_z_vel_std",
"acc_x_vel_skew",
"acc_y_vel_skew",
"acc_z_vel_skew",
"acc_x_vel_kurtosis",
"acc_y_vel_kurtosis",
"acc_z_vel_kurtosis",
"acc_x_pos_mean",
"acc_y_pos_mean",
"acc_z_pos_mean",
"acc_x_pos_min",
"acc_y_pos_min",
"acc_z_pos_min",
"acc_x_pos_max",
"acc_y_pos_max",
"acc_z_pos_max",
"acc_x_pos_var",
"acc_y_pos_var",
"acc_z_pos_var",
"acc_x_pos_std",
"acc_y_pos_std",
"acc_z_pos_std",
"acc_x_pos_skew",
"acc_y_pos_skew",
"acc_z_pos_skew",
"acc_x_pos_kurtosis",
"acc_y_pos_kurtosis",
"acc_z_pos_kurtosis",
"age",
"handedness",
"height_cm",
"shoulder_to_wrist_cm",
"elbow_to_wrist_cm",]

cat_cols = ["adult_child", "sex"]


X_train, X_test, y_train, y_test = train_test_split(
    df_total.drop(["gesture", "gesture_label", "target_gesture"]), 
    df_total["gesture_label", "target_gesture"], 
    test_size=0.2, 
    random_state=420, 
    stratify=df_total["gesture_label", "target_gesture"]
)


ct = ColumnTransformer([
    ("ss", StandardScaler(), numeric_cols),
    ("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01), cat_cols)
])


pipeline = Pipeline([
    ("ct", ct),
    ("clf", XGBClassifier())
])


pipeline.fit(X_train.to_pandas(), y_train["gesture_label"].to_pandas())


preds = pipeline.predict(X_test.to_pandas())
preds_target_gesture = [id2gestures[x] in target_gestures for x in preds] # shitty but works for now


f1_macro = f1_score(preds, y_test["gesture_label"], average="macro")

f1_binary = f1_score(
    y_test["target_gesture"],
    preds_target_gesture,
    pos_label=True,
    zero_division=0,
    average="binary"
) 


0.5 * f1_binary + 0.5 * f1_macro


kf = KFold(n_splits=5)


folds = {}

for i, (train_index, test_index) in tqdm(enumerate(kf.split(df_total)), total=5):
    ct = ColumnTransformer([
        ("ss", StandardScaler(), numeric_cols),
        ("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01), cat_cols)
    ])

    pipeline = Pipeline([
        ("ct", ct),
        ("clf", XGBClassifier())
    ])

    pipeline.fit(
        df_total.to_pandas().drop(["gesture", "gesture_label", "target_gesture"], axis=1).loc[train_index, :], 
        df_total.to_pandas()["gesture_label"][train_index]
    )

    # make predictions
    preds = pipeline.predict(df_total.to_pandas().drop(
        ["gesture", "gesture_label", "target_gesture"], axis=1).loc[test_index, :])
    preds_target_gesture = [id2gestures[x] in target_gestures for x in preds]

    f1_binary = f1_score(
        df_total["target_gesture"][test_index],
        preds_target_gesture,
        pos_label=True,
        zero_division=0,
        average="binary"
    )

    f1_macro = f1_score(
        df_total["gesture_label"][test_index],
        preds,
        average="macro",
    )

    print(f"FOLD {i+1}: ", 0.5 * f1_binary + 0.5 * f1_macro)

    folds[i+1] = {"ct": ct, "pipeline": pipeline}
    folds[i+1]["f1_binary"] = f1_binary
    folds[i+1]["f1_macro"] = f1_macro
    folds[i+1]["weighed_f1"] = 0.5 * f1_binary + 0.5 * f1_macro


avg_f1_binary = sum([folds[i+1]["f1_binary"] for i in range(len(folds))]) / len(folds)
avg_f1_macro = sum([folds[i+1]["f1_macro"] for i in range(len(folds))]) / len(folds)
avg_weighed_f1 = sum([folds[i+1]["weighed_f1"] for i in range(len(folds))]) / len(folds)


print(f"AVG F1 Binary: {avg_f1_binary}\n", f"AVG F1 MACRO: {avg_f1_macro}\n", f"WEIGHED F1: {avg_weighed_f1}")


import kaggle_evaluation.cmi_inference_server


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    
    test_infer_features = sequence.select(
    ["sequence_id", "acc_x", "acc_y", "acc_z"]
    ).group_by("sequence_id").agg(
        pl.all().mean().name.suffix("_mean"),
        pl.all().min().name.suffix("_min"),
        pl.all().max().name.suffix("_max"),
        pl.all().var().name.suffix("_var"),
        pl.all().std().name.suffix("_std"),
        pl.all().skew().name.suffix("_skew"),
        pl.all().kurtosis().name.suffix("_kurtosis"),
        
        pl.all().diff(1).mean().name.suffix("_mean_diff1"),
        pl.all().diff(1).min().name.suffix("_min_diff1"),
        pl.all().diff(1).max().name.suffix("_max_diff1"),
        pl.all().diff(1).var().name.suffix("_var_diff1"),
        pl.all().diff(1).std().name.suffix("_std_diff1"),
        pl.all().diff(1).skew().name.suffix("_skew_diff1"),
        pl.all().diff(1).kurtosis().name.suffix("_kurtosis_diff1"),
        
        pl.all().cum_sum().mul(0.1).mean().name.suffix("_vel_mean"),
        pl.all().cum_sum().mul(0.1).min().name.suffix("_vel_min"),
        pl.all().cum_sum().mul(0.1).max().name.suffix("_vel_max"),
        pl.all().cum_sum().mul(0.1).var().name.suffix("_vel_var"),
        pl.all().cum_sum().mul(0.1).std().name.suffix("_vel_std"),
        pl.all().cum_sum().mul(0.1).skew().name.suffix("_vel_skew"),
        pl.all().cum_sum().mul(0.1).kurtosis().name.suffix("_vel_kurtosis"),
    
        pl.all().cum_sum().cum_sum().mul(0.1).mean().name.suffix("_pos_mean"),
        pl.all().cum_sum().cum_sum().mul(0.1).min().name.suffix("_pos_min"),
        pl.all().cum_sum().cum_sum().mul(0.1).max().name.suffix("_pos_max"),
        pl.all().cum_sum().cum_sum().mul(0.1).var().name.suffix("_pos_var"),
        pl.all().cum_sum().cum_sum().mul(0.1).std().name.suffix("_pos_std"),
        pl.all().cum_sum().cum_sum().mul(0.1).skew().name.suffix("_pos_skew"),
        pl.all().cum_sum().cum_sum().mul(0.1).kurtosis().name.suffix("_pos_kurtosis"),
    )

    seq2sub_test = sequence.unique(["sequence_id", "subject"])["sequence_id", "subject"]

    df_infer = test_infer_features.join(
    seq2sub_test, on="sequence_id"
    ).join(
        demographics, on="subject"
    ).drop(["sequence_id", "subject"])

    preds = None
    for fold in folds:
        pred = folds[fold]["pipeline"].predict_proba(df_infer.to_pandas())
        
        if preds is None:
            preds = pred
        else:
            preds += pred
            
    preds /= len(folds)

    pred_cat = np.argmax(preds, axis=1)[0]
    
    return id2gestures[pred_cat]


inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            "/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv",
            "/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv",
        )
    )

