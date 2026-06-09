# packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import math
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, matthews_corrcoef
import time
from sklearn.feature_selection import mutual_info_classif


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))




df_train = pd.read_csv('/kaggle/input/detecting-reversal-points-in-us-equities/competition_data/competition_data/train.csv')
df_train.shape


df_test = pd.read_csv('/kaggle/input/detecting-reversal-points-in-us-equities/new_comptetition_data/test.csv')
df_test.shape


df_train.head()


df_test.head()


df_train.info()


df_test.info()


#check duplicated for training data
df_train.duplicated().sum()


#check duplicated for testing data
df_test.duplicated().sum()


#show columns that have null values more for training data
for column in df_train.columns:
    if df_train[column].isnull().sum() > 0:
        print(f'{column} has {df_train[column].isnull().sum()} of null values')


#show columns that have null values more for testing data
for column in df_test.columns:
    if df_test[column].isnull().sum() > 0:
        print(f'{column} has {df_test[column].isnull().sum()} of null values')


#change datetime column type into datetime from string type
df_train['t'] = pd.to_datetime(df_train['t'])
df_test['t'] = pd.to_datetime(df_test['t'])


!pip install rapidfuzz


from rapidfuzz import fuzz

def find_string_similarities(df, threshold=80):
    results = []

    # Select only string (object) columns
    string_cols = df.select_dtypes(include="object").columns
    string_cols = [i for i in string_cols if i != 'class_label']

    values = []
    for col in string_cols:
        for idx, val in df[col].items():
            if pd.notna(val):  # ignore NaN
                values.append((idx, col, str(val)))

    # Compare all values
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            idx1, col1, val1 = values[i]
            idx2, col2, val2 = values[j]

            score = fuzz.ratio(val1, val2)
            if score >= threshold:
                results.append({
                    "row1": idx1,
                    "column1": col1,
                    "value1": val1,
                    "row2": idx2,
                    "column2": col2,
                    "value2": val2,
                    "similarity_%": score
                })

    return results
    
simm = find_string_similarities(df_train, threshold=80)
print(simm)


def find_string_similarities(df, threshold=80):
    results = []

    # Select only string (object) columns
    string_cols = df.select_dtypes(include="object").columns
    string_cols = [i for i in string_cols if i != 'class_label']

    values = []
    for col in string_cols:
        for idx, val in df[col].items():
            if pd.notna(val):  # ignore NaN
                values.append((idx, col, str(val)))

    # Compare all values
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            idx1, col1, val1 = values[i]
            idx2, col2, val2 = values[j]

            score = fuzz.ratio(val1, val2)
            if score >= threshold:
                results.append({
                    "row1": idx1,
                    "column1": col1,
                    "value1": val1,
                    "row2": idx2,
                    "column2": col2,
                    "value2": val2,
                    "similarity_%": score
                })

    return results
    
simm = find_string_similarities(df_test, threshold=80)
print(simm)


df_train['class_label'].value_counts()


# change HH and LH into H, change LL and HL into L
df_train["class_label"] = df_train["class_label"].where(
    df_train["class_label"].isna(),
    df_train["class_label"].str[-1]
)


counts = df_train['class_label'].value_counts(dropna=False)

labels = [
    f"{idx if idx == idx else 'NaN'} "
    f"(Count: {count}, {count / counts.sum() * 100:.1f}%)"
    for idx, count in counts.items()
]

plt.figure()
wedges, texts, autotexts = plt.pie(
    counts.values,
    autopct='%1.1f%%',
    startangle=50,
    pctdistance=0.75
)

plt.legend(
    wedges,
    labels,
    title="Class Label",
    loc="center left",
    bbox_to_anchor=(1, 0.5)
)

plt.title("Distribution of class_label")
plt.tight_layout()
plt.show()



ticker_ids = df_train['ticker_id'].unique().tolist()

n = len(ticker_ids)
cols = 2                         # jumlah kolom subplot
rows = math.ceil(n / cols)       # hitung baris otomatis

fig, axes = plt.subplots(rows, cols, figsize=(11, 5 * rows))
axes = axes.flatten()

for idx, ticker in enumerate(ticker_ids):
    ax = axes[idx]

    df_ticker = df_train[df_train['ticker_id'] == ticker]
    counts = df_ticker['class_label'].value_counts(dropna=False)

    labels = [
        f"{cls if cls == cls else 'NaN'} "
        f"(Count: {count}, {count / counts.sum() * 100:.1f}%)"
        for cls, count in counts.items()
    ]

    wedges, texts, autotexts = ax.pie(
        counts.values,
        autopct='%1.1f%%',
        startangle=50,
        pctdistance=0.75
    )

    ax.legend(
        wedges,
        labels,
        title="Class Label",
        loc="center left",
        bbox_to_anchor=(1, 0.5)
    )

    ax.set_title(f"Ticker ID: {ticker}")

# Hapus subplot kosong jika ada
for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()



# since every ticker_id has same pattern, it should used same model. But let us check time pattern first.
line_df = (
    df_train
    .groupby(['t', 'class_label'], dropna=False)
    .size()
    .unstack(fill_value=0)
    .sort_index()
)

# Line plot
plt.figure(figsize=(14, 6))
for col in line_df.columns:
    plt.plot(line_df.index, line_df[col], label=str(col))

plt.xlabel("Date (yyyy-mm-dd)")
plt.ylabel("Count")
plt.title("Daily Distribution by Class Label")
plt.legend(title="class_label")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# since every ticker_id has same pattern, it should used same model. But let us check time pattern first.
ticker_ids = df_train['ticker_id'].unique().tolist()
n = len(ticker_ids)

fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=True)

if n == 1:
    axes = [axes]

for ax, ticker in zip(axes, ticker_ids):
    df_ticker = df_train[df_train['ticker_id'] == ticker]

    line_df = (
        df_ticker
        .groupby(['t', 'class_label'], dropna=False)
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    for col in line_df.columns:
        ax.plot(line_df.index, line_df[col], label=str(col))

    ax.set_title(f"Ticker ID: {ticker}")
    ax.set_ylabel("Count")
    ax.legend(title="class_label")

axes[-1].set_xlabel("Date (yyyy-mm-dd)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# sort the value first by datetime
df_train = df_train.sort_values(by="t", ascending=True)


# Create X and Y coordinate for training
X = df_train.drop(columns = ['class_label', 't', 'train_id', 'ticker_id'])

y = df_train['class_label']



le = LabelEncoder()

y = le.fit_transform(y)


# # calculate variance in every column
# n_features_to_keep = 500
# variances = X.var()

# # drop NaN
# variances = variances.dropna()

# # sort & get top N
# top_cols = variances.sort_values(ascending=False).head(n_features_to_keep).index.tolist()

# print(f"Selected top {len(top_cols)} features by variance")

mi = mutual_info_classif(
    X,
    y,
    discrete_features="auto",
    random_state=0
)

top_idx = mi.argsort()[-1000:]
X_mi = X.iloc[:, top_idx]




top_cols = X_mi.columns.tolist()


# compute split index 80% for training and 20% for testing
split_idx = int(len(df_train) * 0.9)

# split
df_train_split = df_train.iloc[:split_idx].reset_index(drop=True)
df_test_split = df_train.iloc[split_idx:].reset_index(drop=True)


# Create X and Y coordinate for training
X_train = df_train_split.drop(columns = ['class_label', 't', 'train_id', 'ticker_id'])
y_train = df_train_split['class_label']


X_test = df_test_split.drop(columns = ['class_label', 't', 'train_id', 'ticker_id'])
y_test = df_test_split['class_label']


X_train.shape


le = LabelEncoder()

y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.fit_transform(y_test)


classes = np.unique(y_train_encoded)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train_encoded
)

model = CatBoostClassifier(
    loss_function="MultiClass",
    iterations=80,
    learning_rate=0.05,
    depth=2,
    class_weights=class_weights,
    l2_leaf_reg=10,
    random_seed=0,
    verbose=False
)


model.fit(
    X_train[top_cols],
    y_train_encoded
)



# predict

start = time.time()
y_pred = model.predict(X_test[top_cols])
inference_runtime = time.time() - start


# metrics
macro_f1 = f1_score(y_test_encoded, y_pred, average="macro")
macro_bal_acc = balanced_accuracy_score(y_test_encoded, y_pred)
mcc = matthews_corrcoef(y_test_encoded, y_pred)
accuracy = accuracy_score(y_test_encoded, y_pred)

results = {
    "Accuracy": accuracy,
    "Macro F1-score": macro_f1,
    "Macro Balanced Accuracy": macro_bal_acc,
    "Matthews CC": mcc,
    "Inference Runtime (s)": inference_runtime
}

for k, v in results.items():
    print(f"{k}: {v:.6f}")


test_pred_enc = model.predict(df_test[top_cols])
# decode
test_pred = le.inverse_transform(test_pred_enc)


df_test_pred = df_test.copy()
df_test_pred['class_label'] = test_pred
# get submission csv for id, class_label
df_test_pred = df_test_pred[['id', 'class_label']]


df_test_pred.to_csv("submission.csv", index=False)


df_test_pred.shape


df_test.shape




