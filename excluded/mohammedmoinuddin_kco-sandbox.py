import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder  


dataset_path = "/kaggle/input/kaggle-community-olympiad-stress-prediction/train.csv"


# Try with 'latin1' or 'ISO-8859-1'
df = pd.read_csv(dataset_path, encoding='latin1')
df.head()


# Identify all columns that have 'Unnamed' in their name
unnamed_cols = [col for col in df.columns if 'Unnamed' in col]

# Drop the identified columns from the DataFrame
# The 'columns=...' argument is the preferred way to specify columns when axis=1 is implied.
df = df.drop(columns=unnamed_cols)

# Display the first few rows to verify the columns are gone
print(df.head())


df.head()


def clean_text(text):
    return str(text).lower().strip()

df["clean_text"] = df["original_text"].apply(clean_text)


df.head()


# Dataset structure
df.info()

# Basic stats.
df.describe(include='all')

# Missing values
df.isnull().sum()


df = df.dropna()


# Dataset structure
df.info()

# Basic stats
df.describe(include='all')

# Missing values
df.isnull().sum()


import matplotlib.pyplot as plt
import seaborn as sns


# Unique emotion categories
print("Emotion Categories:", df['category'].unique())
print("\nCategory Distribution:\n")
print(df['category'].value_counts())

# Ratio visualization
plt.figure(figsize=(8,4))
sns.countplot(data=df, x='category', palette='coolwarm')
plt.title("Distribution of Emotion Categories")
plt.xlabel("Predicted Category")
plt.ylabel("Count")
plt.show()


df['text_length'] = df['cleaned_text'].apply(lambda x: len(str(x).split()))

plt.figure(figsize=(7,4))
sns.histplot(df['text_length'], bins=30, kde=True, color='orange')
plt.title("Distribution of Text Lengths")
plt.xlabel("Number of Words per Entry")
plt.ylabel("Frequency")
plt.show()

# Average text length per category
plt.figure(figsize=(8,4))
sns.barplot(data=df, x='category', y='text_length', palette='viridis')
plt.title("Average Text Length by Emotion Category")
plt.show()


test_df = pd.read_csv('/kaggle/input/kaggle-community-olympiad-stress-prediction/test.csv', encoding='latin1')
test_df.head()


pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1,2),
        min_df=2,
        max_features=50_000
    )),
    ("clf", LogisticRegression(
        C=4.0,
        max_iter=300,
        class_weight="balanced"
    ))
])



X_train = df["clean_text"]
y_train = df["category"]


# Clean category column - remove blank/whitespace categories
print(f"Before cleaning - Blank categories: {(y_train.str.strip() == '').sum()}")
print(f"Before cleaning - Whitespace-only categories: {(y_train.str.strip() == ' ').sum()}")

# Remove rows with blank or whitespace-only categories
mask = (y_train.str.strip() != '') & (y_train.str.strip() != ' ')
df_clean = df[mask].copy()
X_train = df_clean["clean_text"]
y_train = df_clean["category"]

print(f"After cleaning - Remaining rows: {len(y_train)}")
print(f"After cleaning - Unique categories: {y_train.unique()}")


# Create and fit label encoder (now on cleaned data without blanks)
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Check the mapping (optional, for debugging)
print("Category to numeric mapping:")
for i, category in enumerate(label_encoder.classes_):
    print(f"'{category}': {i}")
print(f"\nEncoded y_train shape: {y_train_encoded.shape}")
print(f"Sample encoded labels: {y_train_encoded[:10]}")


# Calculate F1 macro score on training data (for reference)
# Note: This may be optimistic due to training on the same data
# train_pred_encoded = pipeline.predict(X_train)
# f1_macro_train = f1_score(y_train_encoded, train_pred_encoded, average='macro')
# print(f"F1 Macro Score (on training data): {f1_macro_train:.4f}")

# # Print classification report
# print("\nClassification Report (on training data):")
# print(classification_report(y_train_encoded, train_pred_encoded, 
#                           target_names=label_encoder.classes_))


test_df['text'].head()


# use X_val from test.csv
X_test = test_df['text']
X_test.head()


pipeline.fit(X_train, y_train_encoded)


# Predictions are numeric, convert back to category strings
val_pred_encoded = pipeline.predict(test_df["text"])
val_pred = label_encoder.inverse_transform(val_pred_encoded)
val_pred = pd.Series(val_pred).astype(str).str.strip()

# Replace any blank/empty predictions with a default category (most common: 'achievement')
blank_mask = (val_pred == '') | (val_pred == ' ') | val_pred.isna()
if blank_mask.any():
    print(f"Warning: Found {blank_mask.sum()} blank predictions, replacing with 'achievement'")
    val_pred[blank_mask] = 'achievement'

print(f"Predictions shape: {val_pred.shape}")
print(f"Sample predictions:\n{val_pred.head(10)}")
print(f"\nUnique predictions: {val_pred.unique()}")
print(f"Blank predictions remaining: {(val_pred.str.strip() == '').sum()}")


submission = pd.DataFrame({
    'ID': test_df['ID'],  # Changed from 'Id' to 'ID' to match solutions_dataset.csv
    'category': val_pred
})

# Verify submission format is correct
print(f"Submission shape: {submission.shape}")
print(f"Submission columns: {submission.columns.tolist()}")
print(f"\nFirst few rows:")
print(submission.head(30))

# Verify no blank categories
# blank_count = (submission['category'].str.strip() == '').sum()
# print(f"\nBlank categories in submission: {blank_count}")
# if blank_count > 0:
#     print("WARNING: There are still blank categories!")


# save to outputs
submission.to_csv('/kaggle/working/output_file.csv',index=False) # save to notebook output


submission.head(30)

