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


df_train=pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
df_test=pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")
sample_submission=pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv")


# Set display options to show full content of columns
pd.set_option('display.max_colwidth', None) 


df_train.head(5)


df_test.head(5)


sample_submission.head(5)


print("Category distribution:\n", df_train["Category"].value_counts())
print("\nMisconception distribution:\n", df_train["Misconception"].value_counts())


print("\n--- Missing Values in Train Data ---")
print(df_train.isnull().sum())


print("\n--- Percentage of Missing Values in Train Data ---")
print((df_train.isnull().sum() / len(df_train)) * 100)


print("\n--- Missing Values in Test Data ---")
print(df_test.isnull().sum())


# Identify categorical columns
train_cat_columns = [col for col in df_train.columns if df_train[col].dtype == 'O']
test_cat_columns = [col for col in df_test.columns if df_test[col].dtype == 'O']


print("\n--- Categorical Columns in Train Data ---")
print(train_cat_columns)
    
print("\n--- Unique Values in Categorical Columns (Train) ---")
print(df_train[train_cat_columns].nunique())
    
print("\n--- Categorical Columns in Test Data ---")
print(test_cat_columns)
    
print("\n--- Unique Values in Categorical Columns (Test) ---")
print(df_test[test_cat_columns].nunique())


print("\n--- Duplicate Rows in Train Data ---")
print(df_train.duplicated().sum())

print("\n--- Duplicate Rows in Test Data ---")
print(df_test.duplicated().sum())


for i in range(3):
    print(f"--- Sample {i+1} ---")
    print("Q:", df_train.iloc[i]['QuestionText'])
    print("A:", df_train.iloc[i]['MC_Answer'])
    print("Explanation:", df_train.iloc[i]['StudentExplanation'])
    print("Category:", df_train.iloc[i]['Category'])
    print("Misconception:", df_train.iloc[i]['Misconception'])
    print()



# Combine text fields
def combine_text(row):
    return f"{row['QuestionText']} [SEP] {row['MC_Answer']} [SEP] {row['StudentExplanation']}"

df_train["text"] = df_train.apply(combine_text, axis=1)
df_test["text"] = df_test.apply(combine_text, axis=1)

# Fill NaNs in target (just in case)
df_train["Misconception"] = df_train["Misconception"].fillna("NA")

# Combine Category and Misconception into single target label
df_train["target"] = df_train["Category"] + ":" + df_train["Misconception"]



df_train.head(1)


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df_train["target_enc"] = label_encoder.fit_transform(df_train["target"])


df_train.head(1)


def map3(y_true, y_pred):
    """
    y_true: list or array of ground truth labels (e.g. ["A:B", "C:D"])
    y_pred: 2D array or list of lists with top-3 predicted strings per row
    """
    score = 0.0
    for i in range(len(y_true)):
        preds = y_pred[i]  # this is already a list of top-3 strings
        if y_true[i] in preds:
            rank = preds.tolist().index(y_true[i]) + 1  # Convert to list in case it's np.array
            score += 1.0 / rank
    return score / len(y_true)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# TF-IDF features
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
X = tfidf.fit_transform(df_train["text"])
y = df_train["target_enc"]


print("Shape of X:", X.shape)


print("First 20 TF-IDF features:", tfidf.get_feature_names_out()[:20])


# View first 5 rows of feature vectors (dense)
print(X[:5].toarray())


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42  # stratify for balanced classes
)


# True labels
y_val_true = label_encoder.inverse_transform(y_val)


y_train.shape, y_val.shape, y.shape


# Train model
clf = LogisticRegression(max_iter=500)
clf.fit(X_train, y_train)


# Predict class probabilities
probs_clf = clf.predict_proba(X_val)

# Get top 3 predictions for each row
top3_preds_clf = np.argsort(probs_clf, axis=1)[:, -3:][:, ::-1]  # top 3, descending
val_top3_clf = label_encoder.inverse_transform(top3_preds_clf.flatten()).reshape(-1, 3)

# Calculate MAP@3
score_clf = map3(y_val_true, val_top3_clf)
print("Logistic Regression MAP@3:", score_clf)


# from catboost import CatBoostClassifier
# cat_model = CatBoostClassifier(n_estimators=100, verbose=1,random_state=42)
# cat_model.fit(X_train, y_train)

# val_preds_cat = cat_model.predict_proba(X_val)
# # Get top 3 predictions for each row
# top3_preds_cat = np.argsort(val_preds_cat, axis=1)[:, -3:][:, ::-1]  # top 3, descending
# val_top3_cat = label_encoder.inverse_transform(top3_preds_cat.flatten()).reshape(-1, 3)

# print("CatBoost MAP@3:", map3(y_val_true, val_top3_cat))



# from lightgbm import LGBMClassifier
# from sklearn.metrics import accuracy_score

# # Initialize LGBM for multiclass classification
# lgbm_model = LGBMClassifier(
#     objective='multiclass',
#     n_estimators=1000,
#     learning_rate=0.001,
#     random_state=42,verbose=-1)

# # Train
# lgbm_model.fit(X_train, y_train)

# val_preds_lgbm = lgbm_model.predict_proba(X_val)
# # Get top 3 predictions for each row
# top3_preds_lgbm = np.argsort(val_preds_lgbm, axis=1)[:, -3:][:, ::-1]  # top 3, descending
# val_top3_lgbm = label_encoder.inverse_transform(top3_preds_lgbm.flatten()).reshape(-1, 3)

# print("LGBM MAP@3:", map3(y_val_true, val_top3_lgbm))


X_test = tfidf.transform(df_test["text"])

# Predict class probabilities
probs = clf.predict_proba(X_test)

# Get top 3 predictions for each row
import numpy as np

top3_preds = np.argsort(probs, axis=1)[:, -3:][:, ::-1]  # top 3, descending
top3_labels = label_encoder.inverse_transform(top3_preds.flatten()).reshape(-1, 3)



sample_submission


sample_submission["Category:Misconception"] = [" ".join(preds) for preds in top3_labels]

sample_submission.to_csv("submission.csv", index=False)


sample_submission.head()





