# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd

# Step 2: Load the data
train_df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
test_df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')
sample_sub = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv')

# Step 3: Show basic information
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# View first few rows
train_df.head()



import re

# Function to clean response text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    return text

# Apply to train and test sets
train_df['clean_response'] = train_df['StudentExplanation'].apply(clean_text)
test_df['clean_response'] = test_df['StudentExplanation'].apply(clean_text)

# Preview cleaned text
train_df[['StudentExplanation', 'clean_response']].head()



train_df.columns



print(train_df.columns)



import re

# Clean function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Apply to the correct column
train_df['clean_response'] = train_df['StudentExplanation'].apply(clean_text)
test_df['clean_response'] = test_df['StudentExplanation'].apply(clean_text)

# Show cleaned responses
train_df[['StudentExplanation', 'clean_response']].head()



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# Combine category and misconception into one label
train_df["label"] = train_df["Category"] + ":" + train_df["Misconception"].fillna("NA")

# Encode the labels
label_encoder = LabelEncoder()
train_df["encoded_label"] = label_encoder.fit_transform(train_df["label"])

# Build a pipeline: TF-IDF + Logistic Regression
model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10000)),
    ("clf", LogisticRegression(max_iter=1000))
])
model.fit(train_df["clean_response"], train_df["encoded_label"])



import numpy as np

# Predict probabilities for each test sample
pred_probs = model.predict_proba(test_df["clean_response"])

# Get indices of top 3 predicted labels
top3 = np.argsort(-pred_probs, axis=1)[:, :3]  # negative for descending sort

# Convert label indices back to text labels row-by-row
top3_labels = []
for row in top3:
    decoded = label_encoder.inverse_transform(row)
    top3_labels.append(decoded)
# Join top 3 labels as space-separated string for each row
final_preds = [' '.join(row) for row in top3_labels]

# Prepare submission DataFrame
submission = test_df[["row_id"]].copy()
submission["Category:Misconception"] = final_preds

# Save to CSV
submission.to_csv("submission.csv", index=False)

# Preview first few rows
submission.head()


