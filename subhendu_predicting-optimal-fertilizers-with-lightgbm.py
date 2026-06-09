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


import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, QuantileTransformer, KBinsDiscretizer


train_df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
print(f"{train_df.shape=} {test_df.shape=}")


train_df.head()


test_df.head()


train_cols = list(train_df.columns)
test_cols = list(test_df.columns)

assert len(train_cols) == len(test_cols) + 1
print(f"{train_cols=}\n{test_cols=}")


TARGET_COLUMN = "Fertilizer Name"
NUMERIC_COLS = ["Temparature", "Humidity", "Moisture", "Nitrogen","Potassium", "Phosphorous"]
ID_COL = "id"
CATEGORICAL_COLS = ["Soil Type","Crop Type"]


_ = train_df[NUMERIC_COLS].hist(bins=50, figsize=(12, 12))


# Check values
for col in CATEGORICAL_COLS:
    train_counts = train_df[col].value_counts()
    test_counts = test_df[col].value_counts()

    train_labels = sorted(train_counts.index)
    test_labels = sorted(test_counts.index)

    print(f"{col=} {len(train_labels)=}")

    assert train_labels == test_labels


train_df[TARGET_COLUMN].hist(bins = 15)


train_df[NUMERIC_COLS].describe()


NEW_RATIO_FEATURES = [
    ("N/K", ("Nitrogen","Potassium")),
    ("N/P", ("Nitrogen","Phosphorous")),
    ("K/P", ("Potassium", "Phosphorous")) 
]

NEW_RATIO_COLS = [name for name,_ in NEW_RATIO_FEATURES]
print(f"{NEW_RATIO_COLS=}")

def add_new_cols(df):
    for name,(a,b) in NEW_RATIO_FEATURES:
        df[name] = np.log((df[a]+1.0) / (df[b]+1.0))
    return df

train_df = add_new_cols(train_df)
test_df = add_new_cols(test_df)



train_df[NEW_RATIO_COLS].head()


train_df[NEW_RATIO_COLS].describe()


_ = train_df[NEW_RATIO_COLS].hist(bins=20, figsize=(15, 6))


# Train Encoders
CAT_LE = {}
for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    le.fit(train_df[col])
    print(f"{col=} {le.classes_=}")
    CAT_LE[col] = le
    
KBD = KBinsDiscretizer(
    n_bins=7, encode='ordinal', strategy='quantile', subsample=None,
)
KBD.fit(train_df[NUMERIC_COLS + NEW_RATIO_COLS])
print(f"{KBD.n_features_in_=}\n{KBD.n_bins_=}\n{KBD.bin_edges_=}")


def df_to_vecs(df):
    x_cat = np.hstack([CAT_LE[col].transform(df[col]).reshape(-1,1) for col in CATEGORICAL_COLS])
    print(f"{x_cat.shape=}")
    num_hot = KBD.transform(df[NUMERIC_COLS + NEW_RATIO_COLS])
    x_mat = np.hstack((x_cat,num_hot))
    return x_mat

X = df_to_vecs(train_df)
X_test = df_to_vecs(test_df)

print(f"{X.shape=} {X_test.shape=}")


X[0]


LE = LabelEncoder()
y = LE.fit_transform(train_df[TARGET_COLUMN])
classes_ = LE.classes_
num_classes = len(classes_)
print(f"{num_classes=} \n{classes_=}, \n{ y[:10]=}")


x_train, x_val, y_train, y_val = train_test_split(X,y, test_size=0.1, random_state=2025)
print(f"{x_train.shape=}, {x_val.shape=}, {y_train.shape=}, {y_val.shape=}")


n_features = x_train.shape[1]


# Initialize an empty dictionary
eval_results = {}
# Create the record_evaluation callback
record_eval_callback = lgb.record_evaluation(eval_results)

# Create model 
params={"objective": "multiclass", 
        "num_class": num_classes, 
        "metrics" : ["multi_logloss"],
        "verbosity": -1,
        "num_iterations" : 400,
        'boosting_type': 'gbdt',
        "seed": 42,
} 

model = lgb.LGBMClassifier(**params)

model.fit(x_train, y_train,
          eval_set=[(x_val, y_val)],
          eval_metric='multi_logloss',
          categorical_feature=list(range(len(CATEGORICAL_COLS))),
          callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True), record_eval_callback])

# print(eval_results)
lgb.plot_metric(eval_results)


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score


y_val_pred_raw = model.predict(x_val, raw_score=True)
print(y_val_pred_raw.shape)


def map_at_3(y_true, y_pred_raw):
    y_pred = np.argsort(y_pred_raw, axis=1)
    n = len(y_true)
    score = 0
    for t, p in zip(y_true, y_pred):
        p = p[::-1]
        for j in range(3):
            if p[j] == t:
                score += (1/(j+1))
                break
    return score / n


map_at_3_score = map_at_3(y_val, y_val_pred_raw)
print(f"{map_at_3_score=}")


y_val_pred = np.argmax(y_val_pred_raw, axis=1)
acc = accuracy_score(y_val,y_val_pred)
print(f"{acc=}")
cm = confusion_matrix(y_val, y_val_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes_)
fig, ax = plt.subplots(figsize=(10, 8)) 
disp.plot(ax=ax)
plt.show()


final_pred = np.argsort(model.predict(X_test, raw_score=True), axis=1)
print(final_pred.shape)


output_classes = [" ".join(classes_[i] for i in p[-3:][::-1]) for p in final_pred]
print(output_classes[:10])


test_df[TARGET_COLUMN] = output_classes
out_df = test_df[ [ID_COL, TARGET_COLUMN]]
out_df.head()


out_df.to_csv("submission.csv", index=False)
print("done")

