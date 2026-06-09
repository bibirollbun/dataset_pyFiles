# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
original = pd.read_csv("/kaggle/input/original/Fertilizer Prediction.csv")


original


df


df = pd.concat([df, original], ignore_index=True, join="outer", sort=False)


#Divide all the features with eachother to see their correlation

df["Temp_div_Humidity"] = df["Temparature"] / df["Humidity"]
df["Temp_div_Moisture"] = df["Temparature"] / df["Moisture"]
df["Humidity_div_Moisture"] = df["Humidity"] / df["Moisture"]

df["Nitrogen_div_Potassium"] = df["Nitrogen"] / df["Potassium"]
df["Nitrogen_div_Phosphorous"] = df["Nitrogen"] / df["Phosphorous"]
df["Phosphorous_div_Potassium"] = df["Phosphorous"] / df["Potassium"]


#Doing the same for the test

df_test["Temp_div_Humidity"] = df_test["Temparature"] / df_test["Humidity"]
df_test["Temp_div_Moisture"] = df_test["Temparature"] / df_test["Moisture"]
df_test["Humidity_div_Moisture"] = df_test["Humidity"] / df_test["Moisture"]

df_test["Nitrogen_div_Potassium"] = df_test["Nitrogen"] / df_test["Potassium"]
df_test["Nitrogen_div_Phosphorous"] = df_test["Nitrogen"] / df_test["Phosphorous"]
df_test["Phosphorous_div_Potassium"] = df_test["Phosphorous"] / df_test["Potassium"]



df["NPK_total"] = df["Nitrogen"] + df["Phosphorous"] + df["Potassium"]
df_test["NPK_total"] = df_test["Nitrogen"] + df_test["Phosphorous"] + df_test["Potassium"]


# Replace all inf/-inf with NaN
df_test.replace([np.inf, -np.inf], 0, inplace=True)
df_test.fillna(0, inplace=True)


df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)


missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0]

if not missing_values.empty:
    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing_values.index, y=missing_values.values, palette='viridis')
    plt.xticks(rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Missing Values')
    plt.title('Missing Values per Feature')
    plt.tight_layout()
    plt.show()
else:
    print("✅ No missing values found in the dataset.")


y = df["Fertilizer Name"] # The target

features = ["id", "Temparature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous", "Soil Type", "Crop Type","Temp_div_Humidity","Temp_div_Moisture","Humidity_div_Moisture","Nitrogen_div_Potassium","Nitrogen_div_Phosphorous","Phosphorous_div_Potassium", "NPK_total"]

X = pd.get_dummies(df[features]) # One hot encode training set

test = pd.get_dummies(df_test) # One hot encode test set


X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,     # 20% test, 80% train
    random_state=42,   # random seed for reproducibility
    shuffle=True
)


import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# --- your existing prep ---
y_str = y.astype(str)
le = LabelEncoder().fit(y_str)
y_enc = le.transform(y_str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

seeds = [42, 33, 77, 99, 101]

# --- train models and keep them ---
models = []
for s in seeds:
    mdl = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=s
    )
    mdl.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )
    models.append(mdl)

# --- predict on your final test set and average probs ---
# Make sure column order matches training:
X_test_final = test[X.columns] if isinstance(test, pd.DataFrame) else test

probs = []
for mdl in models:
    bi = getattr(mdl, "best_iteration", None)
    if bi is not None:
        p = mdl.predict_proba(X_test_final, iteration_range=(0, bi + 1))
    else:
        p = mdl.predict_proba(X_test_final)
    probs.append(p)

avg_proba = np.mean(probs, axis=0)            # shape: (n_samples, n_classes)
best_idx = np.argmax(avg_proba, axis=1)
best_names = le.inverse_transform(best_idx)    # array of class names (strings)

# 1) Indices of the top-3 classes per row (highest → lowest)
top3_idx = np.argsort(avg_proba, axis=1)[:, -3:][:, ::-1]

# 2) Convert indices → class names
# (both of these are correct; pick one)
# top3_names = le.inverse_transform(top3_idx.ravel()).reshape(top3_idx.shape)
top3_names = le.classes_[top3_idx]   # simpler

# 3) Join names with a single space
fert_col = [" ".join(row) for row in top3_names]

# 4) Build EXACT format CSV
submission = pd.DataFrame({
    "id": test["id"],
    "Fertilizer Name": fert_col
})

submission.to_csv("submission.csv", index=False)
print("Saved submission.csv with top-3 labels concatenated")



submission

