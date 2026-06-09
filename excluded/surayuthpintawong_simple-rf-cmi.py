import os
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def flatten_list(l):
    flat_list = []
    for xs in l:
        for x in xs:
            flat_list.append(x)
    return flat_list

def prep_data(train_path):
    df = pl.read_csv(train_path) 

    classes = df.select("gesture").unique() \
        .with_columns(
            pl.arange(0, pl.len())
            .alias("label")
        )
    
    selected_cols = [
        "acc_x", "acc_y", "acc_z",
        "rot_x", "rot_y", "rot_z", "rot_w"
    ]
    
    data = df \
        .fill_null(0) \
        .group_by(
            "sequence_id", "gesture", "sequence_type",
        ) \
        .agg(flatten_list([
            [
                pl.col(col).mean().alias(f"avg_{col}"),
                pl.col(col).std().alias(f"std_{col}"),
                pl.col(col).min().alias(f"min_{col}"),
                pl.col(col).max().alias(f"max_{col}"),
            ] for col in selected_cols
        ])) \
        .join(
            classes,
            on="gesture"
        ) \
        .drop("sequence_id", "gesture", "sequence_type")
    return data, classes

def split_data(data):
    X = data.drop("label").to_numpy()
    y = data.select("label").to_numpy().ravel()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, 
        random_state=42, 
        shuffle=True, stratify=y)
    
    return X_train, X_val, y_train, y_val

def train(X_train, y_train):
    clf = RandomForestClassifier(max_depth=8, random_state=0)
    clf.fit(X_train, y_train)
    return clf

def eval(clf, X_val, y_val):
    pred = clf.predict(X_val)
    acc = sum(y_val == pred) / len(pred)
    return acc


train_path = "/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv"
data, classes = prep_data(train_path)
X_train, X_val, y_train, y_val = split_data(data)
clf = train(X_train, y_train)
acc = eval(clf, X_val, y_val)


test_df = pl.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv")


selected_cols = [
    "acc_x", "acc_y", "acc_z",
    "rot_x", "rot_y", "rot_z", "rot_w"
]

test_data = test_df \
    .filter(pl.col("sequence_id") == "SEQ_000001") \
    .fill_null(0) 


test_df.filter(pl.col("sequence_id") == "SEQ_000001")


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    # Replace this function with your inference code.
    # You can return either a Pandas or Polars dataframe, though Polars is recommended.
    # Each prediction (except the very first) must be returned within 30 minutes of the batch features being provided.

    test_data = sequence \
        .fill_null(0) \
        .group_by(
            "sequence_id",
        ) \
        .agg(flatten_list([
            [
                pl.col(col).mean().alias(f"avg_{col}"),
                pl.col(col).std().alias(f"std_{col}"),
                pl.col(col).min().alias(f"min_{col}"),
                pl.col(col).max().alias(f"max_{col}"),
            ] for col in selected_cols
        ])) \
        .drop("sequence_id") \
        .to_numpy()
    pred_cls = clf.predict(test_data)[0]
    gesture = classes.filter(pl.col("label") == pred_cls).select("gesture").item()
    return gesture


predict(test_data, pl.DataFrame({"a": [1]}))


import kaggle_evaluation.cmi_inference_server

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




