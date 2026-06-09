
import os
import sys
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

DEFAULT_DATA_DIR = '/kaggle/input/london-house-price-prediction-advanced-techniques'

if 'DATA_DIR' not in globals():
    DATA_DIR = DEFAULT_DATA_DIR

print("DATA_DIR:", DATA_DIR)
assert os.path.exists(DATA_DIR), f"DATA_DIR not found: {DATA_DIR}"




train_path = os.path.join(DATA_DIR, 'train.csv')
test_path  = os.path.join(DATA_DIR, 'test.csv')

assert os.path.exists(train_path), "train.csv not found"
assert os.path.exists(test_path), "test.csv not found"

train_raw = pd.read_csv(train_path)
test_raw  = pd.read_csv(test_path)

print("Loaded:", train_raw.shape, test_raw.shape)
print("Train columns:", train_raw.columns.tolist())




def minimal_fe(df, is_train=True, ref=None):
    out = df.copy()
    eps = 1e-6

    for c in ["bathrooms","bedrooms","floorAreaSqM","livingRooms","sale_month"] + (["price"] if is_train else []):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if {"bathrooms","bedrooms","livingRooms"}.issubset(out.columns):
        out["total_rooms"] = out["bathrooms"] + out["bedrooms"] + out["livingRooms"]

    if {"bedrooms","bathrooms"}.issubset(out.columns):
        out["bed_bath_ratio"] = out["bedrooms"] / (out["bathrooms"] + eps)

    if "currentEnergyRating" in out.columns:
        rating_map = {"A":7,"B":6,"C":5,"D":4,"E":3,"F":2,"G":1}
        out["energy_rating_num"] = out["currentEnergyRating"].map(rating_map).fillna(0)

    if "outcode" in out.columns:
        if ref is None:
            counts = out["outcode"].value_counts()
        else:
            counts = ref["outcode"].value_counts() if "outcode" in ref.columns else out["outcode"].value_counts()
        out["outcode_count"] = out["outcode"].map(counts).fillna(0)

    for col in ["propertyType","tenure"]:
        if col in out.columns:
            out = pd.get_dummies(out, columns=[col], drop_first=True)

    for col in ["fullAddress","postcode","country","outcode","currentEnergyRating"]:
        if col in out.columns:
            out = out.drop(columns=[col])

    return out

use_existing_df_fe = False
try:
    df_fe  # check existence
    if isinstance(df_fe, pd.DataFrame):
        use_existing_df_fe = True
        print("Using existing df_fe from your notebook. Shape:", df_fe.shape)
except NameError:
    pass

if use_existing_df_fe:
    train_fe = df_fe.copy()
else:
    print("df_fe not found. Building minimal features from train_raw...")
    train_fe = minimal_fe(train_raw, is_train=True)

test_fe = minimal_fe(test_raw, is_train=False, ref=train_raw)

print("FE shapes => train:", train_fe.shape, " test:", test_fe.shape)




if "price" not in train_fe.columns:
    raise ValueError("'price' column not found in train features")

y_train = train_fe["price"].astype(float)
X_train = train_fe.drop(columns=["price"], errors="ignore")

X_test = test_fe.reindex(columns=X_train.columns, fill_value=0)

X_train = X_train.fillna(0)
X_test  = X_test.fillna(0)

print("X_train/X_test:", X_train.shape, X_test.shape)




model = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1.0, random_state=42))

try:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("Model trained. preds shape:", preds.shape)
except Exception as e:
    print("Model training/prediction failed:", e)
    raise




test_id_col = None
for cand in ["Id", "id", "ID"]:
    if cand in test_raw.columns:
        test_id_col = cand
        break
if test_id_col is None:
    test_id_col = test_raw.columns[0]

print("Submission ID column:", test_id_col)

submission = pd.DataFrame({test_id_col: test_raw[test_id_col], "price": preds})

save_name = "submission.csv"
err = None
try:
    submission.to_csv(save_name, index=False)
except Exception as e:
    err = e
    print("Failed to save submission:", e)

if err is None and os.path.exists(save_name):
    print(f"Saved '{save_name}' successfully with shape {submission.shape}.")
else:
    print("Submission file was not created. Please check previous logs.")


