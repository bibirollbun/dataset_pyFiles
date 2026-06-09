import pandas as pd
import glob
import re
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


files = glob.glob("/kaggle/input/daigt-proper-train-dataset/train_drcat_*.csv")
proper = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
print(proper.shape)



#Load all datasets 
proper   # DAIGT Proper dataset
external = pd.read_csv("/kaggle/input/daigt-external-dataset/daigt_external_dataset.csv")       # DAIGT External dataset
argugpt = pd.read_csv("/kaggle/input/argugpt/argugpt.csv")         # ArguGPT dataset

# Check columns to understand structure
print("Proper columns:", proper.columns.tolist())
print("External columns:", external.columns.tolist())
print("ArguGPT columns:", argugpt.columns.tolist())

# Prepare DAIGT Proper
# Already has text + label
proper_df = proper[['text', 'label']].copy()

#  Prepare DAIGT External 
# Only has human text (no label column)
# We'll keep the text column and assign label = 0
if 'text' in external.columns:
    external_df = external[['text']].copy()
    external_df['label'] = 0
else:
    raise ValueError("External dataset must have a 'text' column")

#Prepare ArguGPT 
# Only AI-generated text (no label column)
# Keep text column and assign label = 1
if 'text' in argugpt.columns:
    argugpt_df = argugpt[['text']].copy()
    argugpt_df['label'] = 1
else:
    raise ValueError("ArguGPT dataset must have a 'text' column")

# Combine all datasets 
combined_df = pd.concat([proper_df, external_df, argugpt_df], ignore_index=True)





combined_df.drop_duplicates(subset='text', inplace=True)
combined_df.dropna(subset=['text'], inplace=True)
combined_df.reset_index(drop=True, inplace=True)

#Check class balance
print(combined_df['label'].value_counts())

#Save combined dataset 
combined_df.to_csv("combined_dataset.csv", index=False)

print(" Combined dataset created successfully!")
print("Shape:", combined_df.shape)


# === Load Combined Dataset ===
df = pd.read_csv("/kaggle/working/combined_dataset.csv")
print("Dataset loaded successfully!")
print("Shape:", df.shape)
print(df.head())

# ===  Basic Text Cleaning Function ===
def clean_text(text):
    text = str(text).lower()                       # lowercase
    text = re.sub(r"http\S+", " ", text)           # remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)    # remove punctuation & special chars
    text = re.sub(r"\s+", " ", text).strip()       # remove extra spaces
    return text

# Apply cleaning
df['clean_text'] = df['text'].apply(clean_text)

print("\n Text cleaning completed.")
print(df[['text', 'clean_text']].head(3))

# ===  Remove very short or empty texts ===
df = df[df['clean_text'].str.len() > 10]
df.reset_index(drop=True, inplace=True)

print("\nAfter cleaning:")
print("Shape:", df.shape)


# Basic overview
print("Shape of dataset:", df.shape)
print("\nDataset info:")
df.info()

# Display first few rows
df.head()



# Count of each label
label_counts = df['label'].value_counts()

plt.figure(figsize=(6,4))
sns.barplot(x=label_counts.index, y=label_counts.values, palette='viridis')
plt.title("Label Distribution")
plt.xlabel("Label")
plt.ylabel("Count")
plt.show()





print(label_counts)


df['text_length'] = df['text'].apply(lambda x: len(x.split()))

plt.figure(figsize=(8,5))
sns.histplot(df['text_length'], bins=50, kde=True)
plt.title("Distribution of Essay Lengths (in words)")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.show()





print(df['text_length'].describe())


# Check missing values
print("Missing values:\n", df.isnull().sum())

# Check duplicates
print("Duplicate texts:", df.duplicated(subset='text').sum())

# Remove duplicates if necessary
df = df.drop_duplicates(subset='text')



# # Combine all text
# text_corpus = " ".join(df['text'].tolist())

# # Generate WordCloud
# plt.figure(figsize=(10,6))
# wordcloud = WordCloud(width=8000, height=4000, background_color='white').generate(text_corpus)
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")
# plt.title("WordCloud of Combined Text Data")
# plt.show()



plt.figure(figsize=(8,5))
sns.boxplot(x='label', y='text_length', data=df, palette='Set2')
plt.title("Text Length Distribution per Label")
plt.xlabel("Label")
plt.ylabel("Number of Words")
plt.show()



from sklearn.model_selection import train_test_split

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))



from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(
    max_features=20000,   # limit vocabulary size
    ngram_range=(1,2),    # use unigrams and bigrams
    stop_words='english'  # remove common words
)

# Fit and transform
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)

print("TF-IDF train shape:", X_train_tfidf.shape)
print("TF-IDF val shape:", X_val_tfidf.shape)



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Create a pipeline for TF-IDF + Logistic Regression
model = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=20000,
        ngram_range=(1,2),
        stop_words='english'
    )),
    ('clf', LogisticRegression(
        C=2,
        max_iter=200,
        class_weight='balanced',
        random_state=42
    ))
])

# Train the model
model.fit(X_train, y_train)

print("Model training complete!")



# Predictions on validation set
y_pred_val = model.predict(X_val)

# Metrics
val_acc = accuracy_score(y_val, y_pred_val)
val_f1 = f1_score(y_val, y_pred_val)

print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Validation F1 Score: {val_f1:.4f}")
print("\nDetailed Report:\n", classification_report(y_val, y_pred_val))



import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_val, y_pred_val)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Human','AI'], yticklabels=['Human','AI'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix on Validation Set")
plt.show()



import joblib

# Save the logistic regression model
joblib.dump(model, "ai_text_detector.pkl")
print(" Model saved as ai_text_detector.pkl")



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load datasets
train = pd.read_csv("/kaggle/input/llm-detect-ai-generated-text/train_essays.csv")
test = pd.read_csv("/kaggle/input/llm-detect-ai-generated-text/test_essays.csv")

# Split features and target
X = train["text"]
y = train["generated"]

# Text to numerical features
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X_vec = vectorizer.fit_transform(X)
test_vec = vectorizer.transform(test["text"])

# Train simple classifier
model = LogisticRegression(max_iter=200)
model.fit(X_vec, y)

# Predict on test set
preds = model.predict(test_vec)



# Load sample submission to get structure
sample = pd.read_csv("/kaggle/input/llm-detect-ai-generated-text/sample_submission.csv")

# Add predictions
sample["generated"] = preds

# Save submission file
sample.to_csv("submission.csv", index=False)
print("✅ Submission file created: submission.csv")





