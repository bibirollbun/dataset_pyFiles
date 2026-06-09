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


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# æŸ¥çœ‹æ‰€æœ‰è¾“å…¥æ–‡ä»¶è·¯å¾„
# Viewing all input file paths
train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")

# è¯»å�–è®­ç»ƒé›†å’Œæµ‹è¯•é›†æ•°æ�®
# Read data from training and test sets
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# è¾“å‡ºæ•°æ�®ç»´åº¦
# Output data dimensions
display(train.head())
display(test.head())

# ç®€å�•æ•°æ�®æ��è¿°ç»Ÿè®¡
# Simple data descriptive statistics
display(train.describe())
display(test.describe())


# ç¼ºå¤±å€¼ç»Ÿè®¡
# Missing value statistics
def check_missing(df, name="Dataset"):
    print(f"Missing Values in {name}:")
    display(
        df.isnull()
          .sum()
          .to_frame(name="Missing Values")
          .query("`Missing Values` > 0")
          .sort_values(by="Missing Values", ascending=False)
    )

check_missing(train, "Train Dataset")
check_missing(test, "Test Dataset")


def preprocess_data(df, is_train=True):
    df = df.copy()

    # åˆ é™¤idåˆ—
    # Delete the id column
    if "id" in df.columns:
        df.drop(columns=["id"], inplace=True)

    # åˆ—å��æ ‡å‡†åŒ–
    # Standardization of listings
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # ç¼ºå¤±å€¼å¡«å……â€œunknownâ€�
    # Missing values are filled with â€œunknownâ€�.
    categorical_cols_to_fill = ["stage_fear", "drained_after_socializing"]
    for col in categorical_cols_to_fill:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")

    # ç¼ºå¤±å€¼å¡«å……ä¸­ä½�æ•°
    # Missing value populated median
    numerical_cols = df.select_dtypes(include="number").columns
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())

    return df


from sklearn.preprocessing import LabelEncoder

# ç»Ÿä¸€LabelEncoderç¼–ç �
# Harmonize LabelEncoder encoding
def label_encoder_df(df, le_dict=None, is_train=True):
    df = df.copy()
    if le_dict is None:
        le_dict = {}
        
    for col in df.select_dtypes(include="object").columns:
        if is_train:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            le_dict[col] = le
        else:
            le = le_dict.get(col)
            if le is not None:
                df[col] = le.transform(df[col].astype(str))
            else:
                df[col] = df[col].astype(str)
                
    return df, le_dict


# é¢„å¤„ç�†
# Preprocessing
train_df = preprocess_data(train, is_train=True)
test_df = preprocess_data(test, is_train=False)

# ç¼–ç �
# Encodings
train_df, le_dict = label_encoder_df(train_df, is_train=True)
test_df, _ = label_encoder_df(test_df, le_dict, is_train=False)


from itertools import combinations

def combine_and_encode_categories(df, categorical_cols, max_comb_len=2, le_dict=None, is_train=True):
    df = df.copy()
    if le_dict is None:
        le_dict = {}

    new_cols = []

    # ç”Ÿæˆ�æ–°ç»„å�ˆåˆ—
    # Generating new combined columns
    for r in range(2, max_comb_len + 1):
        for comb in combinations(categorical_cols, r):
            new_col = "_".join(comb)
            df[new_col] = df[list(comb)].astype(str).agg("_".join, axis=1)
            new_cols.append(new_col)

    # æ–°ç»„å�ˆåˆ—ç»Ÿä¸€LabelEncoderç¼–ç �
    # Unified LabelEncoder encoding for new combo columns
    for col in new_cols:
        if is_train:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            le_dict[col] = le
        else:
            le = le_dict.get(col)
            if le is not None:
                df[col] = le.transform(df[col].astype(str))
            else:
                df[col] = df[col].astype(str)

    return df, le_dict


categorical_cols = ["stage_fear", "drained_after_socializing", "social_event_attendance"]

train_df, comb_le_dict = combine_and_encode_categories(train_df, categorical_cols, max_comb_len=3)
test_df, _ = combine_and_encode_categories(test_df, categorical_cols, max_comb_len=3)


from sklearn.preprocessing import StandardScaler

def scale_numerical_features(train_df, test_df, exclude_cols=["personality"]):
    train_df = train_df.copy()
    test_df = test_df.copy()

    # ç­›é€‰éœ€è¦�æ ‡å‡†åŒ–çš„åˆ—
    # Filter columns that need to be standardized
    num_cols = train_df.select_dtypes(include="number").columns.difference(exclude_cols)

    scaler = StandardScaler()
    train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
    test_df[num_cols] = scaler.transform(test_df[num_cols])

    return train_df, test_df


train_df, test_df = scale_numerical_features(train_df, test_df)
target_encoder = LabelEncoder()
train_df["personality"] = target_encoder.fit_transform(train_df["personality"])

X = train_df.drop(columns=["personality"])
y = train_df["personality"]


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def train_xgb_model(X, y, test_size=0.2, random_state=42):
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # åˆ�å§‹åŒ–æ¨¡å�‹
    # Initialize the model
    model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.003,
        max_depth=10,
        subsample=0.9,
        colsample_bytree=0.9,
        gamma=0.1,
        min_child_weight=5,
        scale_pos_weight=0.8,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric="logloss"
    )

    # è®­ç»ƒæ¨¡å�‹
    # Training models
    model.fit(X_train, y_train)

    # é¢„æµ‹æ¨¡å�‹
    # Predictive modeling
    y_pred = model.predict(X_valid)

    # è¾“å‡ºè¯„ä¼°æŒ‡æ ‡
    # Output evaluation indicators
    print("Classification Report:")
    print(classification_report(y_valid, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_valid, y_pred))

    return model


model = train_xgb_model(X, y)


def make_submission(model, test_df, test_raw_df, label_encoder, filename="submission.csv"):
    # é¢„æµ‹
    # Predictions
    preds = model.predict(test_df)

    # å��ç¼–ç �å›�å�Ÿå§‹æ ‡ç­¾
    # Reverse coding back to the original label
    final_preds = label_encoder.inverse_transform(preds)

    # ç”Ÿæˆ�æ��äº¤æ–‡ä»¶
    # Generate submissions
    submission = pd.DataFrame({
        "id": test_raw_df["id"],
        "personality": final_preds
    })

    submission.to_csv(filename, index=False)

    return submission


submission = make_submission(model, test_df, test, target_encoder)

