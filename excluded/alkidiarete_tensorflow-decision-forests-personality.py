!pip install ydf -q


import tensorflow_decision_forests as tfdf
import ydf
import numpy as np
import pandas as pd
import seaborn as sns
import re
import matplotlib.pyplot as plt

from IPython.core.magic import register_line_magic
from IPython.display import Javascript

try:
  from wurlitzer import sys_pipes
except:
  from colabtools.googlelog import CaptureLog as sys_pipes

import warnings
warnings.filterwarnings('ignore')




train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
sub = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")

train.drop(columns=['id'], inplace=True)
test_id = test['id']
test.drop(columns=['id'], inplace=True)


print("\nMissing Values in Train:\n", train.isnull().sum())
print("\nMissing Values in Test:\n", test.isnull().sum())


train_duplicates = train.duplicated().sum()
test_duplicates = test.duplicated().sum()

print(f"Train duplicates: {train_duplicates}")
print(f"Test duplicates: {test_duplicates}")



train.head().T


def impute_numerical(df):
    df["Time_spent_Alone"].fillna(df["Time_spent_Alone"].median(), inplace=True)
    df["Social_event_attendance"].fillna(df["Social_event_attendance"].median(), inplace=True)
    df["Going_outside"].fillna(df["Going_outside"].median(), inplace=True)
    df["Friends_circle_size"].fillna(df["Friends_circle_size"].median(), inplace=True)
    df["Post_frequency"].fillna(df["Post_frequency"].median(), inplace=True)
    return df

def impute_all_categoricals_with_missing(df):
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col].fillna("missing", inplace=True)
    return df

train = impute_numerical(train)
train = impute_all_categoricals_with_missing(train)

test = impute_numerical(test)
test = impute_all_categoricals_with_missing(test)



train["Personality"] = train["Personality"].map({"Introvert": 0, "Extrovert": 1})


train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train, label="Personality")
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test)


model = tfdf.keras.RandomForestModel()

model.compile(metrics=["accuracy"])

with sys_pipes():
  model.fit(x=train_ds)



model.make_inspector().evaluation()


model.summary()


model.make_inspector().variable_importances()


logs = model.make_inspector().training_logs()

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot([log.num_trees for log in logs], [log.evaluation.accuracy for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("Accuracy (out-of-bag)")

plt.subplot(1, 2, 2)
plt.plot([log.num_trees for log in logs], [log.evaluation.loss for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("Logloss (out-of-bag)")

plt.show()



predictions = model.predict(test_ds)
predicted_classes = [1 if prob[0] >= 0.5 else 0 for prob in predictions]


label_inverse_map = {0: "Introvert", 1: "Extrovert"}
predicted_labels = [label_inverse_map[cls] for cls in predicted_classes]



submission = pd.DataFrame({
    "id": test_id,
    "Personality": predicted_labels
})

submission.to_csv("submission.csv", index=False)
submission.head()

