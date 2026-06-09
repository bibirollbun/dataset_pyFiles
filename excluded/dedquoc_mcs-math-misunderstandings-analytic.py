# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


%%time
import pandas as pd

# Load training data
train_df = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")

# Load test data
test_df = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")

# Load sample submission
sample_sub = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv")

#  shortnmae for vizualize 
train = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")


%%time
import pandas as pd

# Load the data
train = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
test = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")

# View sample data
print(train.head())
print(test.head())


# Distribution of Categories
print(train['Category'].value_counts())

# Unique Misconceptions
print(train['Misconception'].nunique())


%%time
from sklearn.model_selection import train_test_split

# Prepare training data
X = train['StudentExplanation']
y = train['Category']

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)


%%time
# Use sample submission as template
sample_sub = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv")

# Fill with a baseline prediction
sample_sub['Category:Misconception'] = "True_Correct:NA False_Neither:NA False_Misconception:Incomplete"

# Save submission
sample_sub.to_csv("submission.csv", index=False)


%%time
plt.figure(figsize=(10, 6))
sns.countplot(data=train, y='Category', order=train['Category'].value_counts().index, palette='viridis')
plt.title('Distribution of Categories', fontsize=16)
plt.xlabel('Count', fontsize=12)
plt.ylabel('Category', fontsize=12)
plt.show()


%%time
# Filter out NA values
misconceptions = train[train['Misconception'] != 'NA']['Misconception']

plt.figure(figsize=(10, 8))
sns.countplot(data=train[train['Misconception'] != 'NA'], y='Misconception',
              order=misconceptions.value_counts().index[:10], palette='magma')
plt.title('Top 10 Misconceptions', fontsize=16)
plt.xlabel('Count', fontsize=12)
plt.ylabel('Misconception', fontsize=12)
plt.show()


%%time
print("Unique Misconceptions (excluding NA):", train[train['Misconception'] != 'NA']['Misconception'].nunique())
print("\nCategory Distribution:")
print(train['Category'].value_counts())


%%time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

train = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")

# Add text length column
train['text_length'] = train['StudentExplanation'].apply(lambda x: len(str(x).split()))

plt.figure(figsize=(10, 6))
sns.histplot(data=train, x='text_length', hue='Category', bins=50, kde=True)
plt.title('Student Explanation Length by Category')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.yscale('log')  # Log scale helps with class imbalance
plt.show()


%%time
# Filter rows with valid misconceptions
df_misconceptions = train[train['Misconception'] != 'NA']

# Pivot table
miscon_by_category = pd.crosstab(df_misconceptions['Category'], df_misconceptions['Misconception'])

# Show top 10 misconceptions across categories
plt.figure(figsize=(12, 6))
sns.heatmap(miscon_by_category[miscon_by_category.sum().sort_values(ascending=False).index[:10]], annot=False, cmap='Blues')
plt.title('Heatmap: Misconception Frequency by Category')
plt.show()


%%time
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load data
train_df = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
test_df = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")

# Feature engineering
def fe(df):
    df = df.copy()
    df["question_len"] = df["QuestionText"].str.len()
    df["studentexp_len"] = df["StudentExplanation"].str.len()
    df["mc_answer_len"] = df["MC_Answer"].str.len()
    df["char_ratio"] = df["studentexp_len"] / (df["question_len"] + 1)
    return df[["question_len", "studentexp_len", "mc_answer_len", "char_ratio"]]

X = fe(train_df)
X_test = fe(test_df)

# Encode labels
le_cat = LabelEncoder()
le_mis = LabelEncoder()

train_df["Category_enc"] = le_cat.fit_transform(train_df["Category"])
train_df["Misconception_enc"] = le_mis.fit_transform(train_df["Misconception"].fillna("NA"))

y_cat = train_df["Category_enc"]
y_mis = train_df["Misconception_enc"]

# Cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_cat = pd.Series(index=train_df.index, dtype=int)
oof_mis = pd.Series(index=train_df.index, dtype=int)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_cat)):
    print(f"\nğŸ”� Fold {fold+1}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train_cat, y_val_cat = y_cat.iloc[train_idx], y_cat.iloc[val_idx]
    y_train_mis, y_val_mis = y_mis.iloc[train_idx], y_mis.iloc[val_idx]

    model_cat = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        random_state=42,
        verbosity=-1
    )
    model_cat.fit(
        X_train, y_train_cat,
        eval_set=[(X_val, y_val_cat)],
        eval_metric="multi_logloss",
        callbacks=[lgb.log_evaluation(period=0)]
    )
    oof_cat.iloc[val_idx] = model_cat.predict(X_val)

    model_mis = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        random_state=42,
        verbosity=-1
    )
    model_mis.fit(
        X_train, y_train_mis,
        eval_set=[(X_val, y_val_mis)],
        eval_metric="multi_logloss",
        callbacks=[lgb.log_evaluation(period=0)]
    )
    oof_mis.iloc[val_idx] = model_mis.predict(X_val)

# Decode predictions
pred_cat = le_cat.inverse_transform(oof_cat.fillna(0).astype(int))
pred_mis = le_mis.inverse_transform(oof_mis.fillna(0).astype(int))


%%time
# Evaluation
def evaluate_predictions(true_df, pred_category, pred_misconception):
    pred_full = [
        f"{cat}:{mis}" if "Misconception" in cat else f"{cat}:NA"
        for cat, mis in zip(pred_category, pred_misconception)
    ]
    true_full = [
        f"{cat}:{mis}" if "Misconception" in cat else f"{cat}:NA"
        for cat, mis in zip(true_df["Category"], true_df["Misconception"])
    ]
    full_acc = accuracy_score(true_full, pred_full)
    cat_acc = accuracy_score(true_df["Category"], pred_category)
    mis_mask = true_df["Category"].str.contains("Misconception")
    mis_acc = accuracy_score(true_df["Misconception"][mis_mask], pd.Series(pred_misconception)[mis_mask])
    return {
        "âœ… Full Format Accuracy": round(full_acc, 4),
        "âœ… Category Accuracy": round(cat_acc, 4),
        "âœ… Misconception Accuracy": round(mis_acc, 4),
    }

# Show evaluation scores
scores = evaluate_predictions(train_df, pred_cat, pred_mis)
print("\nğŸ“Š Local CV Evaluation:")
for k, v in scores.items():
    print(f"{k}: {v}")


%%time
# ğŸ§  Predict Category
cat_preds_raw = model_cat.predict(X_test)

# ğŸ”„ Ensure we inverse transform only if model returns label indices
if isinstance(cat_preds_raw[0], (np.integer, int)):
    cat_labels = le_cat.inverse_transform(cat_preds_raw)
else:
    cat_labels = cat_preds_raw  # already strings

# ğŸ§  Predict Misconception
mis_preds_raw = model_mis.predict(X_test)

# ğŸ”„ Similarly, decode if necessary
if isinstance(mis_preds_raw[0], (np.integer, int)):
    mis_labels = le_mis.inverse_transform(mis_preds_raw)
else:
    mis_labels = mis_preds_raw

# ğŸ›  Combine into final submission format
submission_df = pd.DataFrame({
    "row_id": test["row_id"],
    "Category:Misconception": [
        f"{c}:{m}" if "Misconception" in c else f"{c}:NA"
        for c, m in zip(cat_labels, mis_labels)
    ]
})

# ğŸ’¾ Save submission
submission_df.to_csv("submission.csv", index=False)
print("âœ… submission.csv saved.")

