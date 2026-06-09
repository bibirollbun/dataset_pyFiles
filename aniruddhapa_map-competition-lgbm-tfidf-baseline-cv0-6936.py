import pandas as pd

data = {
    "Question": ["7 Ã— 8", "12 Ã· 4", "5 + 9"],
    "Student_Answer": ["54", "3", "59"],
    "Student_Explanation": [
        "7Ã—7=49, then I just added 5",
        "12 divided by 4 is 3, because 4Ã—3=12",
        "I multiplied 5Ã—9 = 59"
    ],
    "Is_Answer_Correct": [False, True, False],
    "Explanation_Label": ["Misconception", "Correct", "Misconception"],
    "Misconception_Type": [
        "Misunderstanding_multiplication",
        "None",
        "Confusing_addition_with_multiplication"
    ]
}

df = pd.DataFrame(data)
df



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re, string
import nltk, spacy
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report



df=pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
df.sample(10)


df.shape


df.info()


df.head()


mask = df['Category'].str.contains('_Misconception', na=False) & df['Misconception'].isna()
df_missing_misconceptions = df[mask]

df_missing_misconceptions


df.isnull().sum()


pd.set_option('display.max_rows', None)

df['Misconception'] = df['Misconception'].fillna('NA').astype(str)
df['target'] = df['Category'] + ':' + df['Misconception']

# Quick check of unique labels
print("Number of unique target labels:", df['target'].nunique())
print(df['target'].value_counts())


df.sample(5)


plt.figure(figsize=(12,6))
df['target'].value_counts().head(20).plot(kind='bar')
plt.title("Top 20 Most Frequent Target Labels")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.show()


df['resp_len'] = df['StudentExplanation'].astype(str).apply(lambda x: len(x.split()))

plt.figure(figsize=(10,5))
sns.histplot(df['resp_len'], bins=50, kde=True)
plt.title("Distribution of StudentExplanation Lengths (in words)")
plt.xlabel("Number of words")
plt.ylabel("Frequency")
plt.show()


from collections import Counter
from wordcloud import WordCloud

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)  # keep only letters and spaces
    return text

all_words = " ".join(df['StudentExplanation'].dropna().apply(preprocess_text))
word_counts = Counter(all_words.split())

# Show top 20 words
print("Top 20 frequent words in explanations:")
print(word_counts.most_common(20))


# Wordcloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)
plt.figure(figsize=(12,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud of Student Explanations")
plt.show()


target_counts = df['target'].value_counts()
imbalance_ratio = target_counts.max() / target_counts.min()

print(f"Number of unique target classes: {df['target'].nunique()}")
print(f"Most frequent class count: {target_counts.max()}")
print(f"Least frequent class count: {target_counts.min()}")
print(f"Imbalance ratio (max/min): {imbalance_ratio:.2f}")

plt.figure(figsize=(12,6))
sns.histplot(target_counts, bins=50, kde=False)
plt.title("Distribution of Class Sizes (Target)")
plt.xlabel("Class Count")
plt.ylabel("Number of Classes")
plt.show()


df.head()


# --- 2. Encode the Target Variable ---
# Machine learning models need numerical targets, not strings.
# We also need these numerical labels for our MAP@3 metric.

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

print("ğŸ”¹ Encoding the string target into numerical labels...")
le = LabelEncoder()
df['target_encoded'] = le.fit_transform(df['target'])
print(f"Number of unique classes: {len(le.classes_)}")
print("Mapping from numerical label back to original class name:")
# Store the mapping for later use (e.g., in submission files)
label_to_class_map = {i: cls for i, cls in enumerate(le.classes_)}
print(label_to_class_map) # Uncomment to see the full map
print("-" * 50)


df.head()


print("ğŸ”¹ Engineering a new feature: 'full_text'...")

def safe(x):
    return '' if pd.isna(x) else str(x)

df['full_text'] = (
    "question: " + df['QuestionText'].apply(safe) +
    " [SEP] mc_answer: " + df['MC_Answer'].apply(safe) +
    " [SEP] explanation: " + df['StudentExplanation'].apply(safe)
).str.strip()


pd.set_option("display.max_colwidth", None)
df.full_text[0]


df.shape


df.target_encoded.value_counts()


# --- 3. Set Up StratifiedKFold ---
# This is the core of our validation framework.
print("ğŸ”¹ Setting up StratifiedKFold cross-validation...")
N_SPLITS = 5 # A standard choice for CV
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)


# We will split based on the dataframe's index and the encoded target
X = df.index # We can just use the index for splitting
y = df['target_encoded']


# Demonstrate how the splitter works for one fold
print(f"Demonstrating the result of one split (Fold 0):")
train_idx, val_idx = next(skf.split(X, y))
train_df = df.iloc[train_idx]
val_df = df.iloc[val_idx]

print(f"Total training samples in Fold 0: {len(train_df)}")
print(f"Total validation samples in Fold 0: {len(val_df)}")


from collections import Counter

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nFOLD {fold}")
    print("Train class counts:")
    print(Counter(y[train_idx]))
    print("Val class counts:")
    print(Counter(y[val_idx]))


# PROOF THAT STRATIFICATION WORKS: Compare distributions
print("\nOriginal target distribution (%):")
print(round(df['target'].value_counts(normalize=True) * 100, 2))
print("\nValidation set target distribution (%):")
print(round(val_df['target'].value_counts(normalize=True) * 100, 2))
print("Note how the percentages are preserved, even for the rarest classes.")
print("-" * 50)


# --- 4. Implement the MAP@3 Metric ---
# This function is your "ruler" to measure model performance.
print("ğŸ”¹ Defining the competition metric: Mean Average Precision @ 3 (MAP@3)...")

def map_at_3(y_true, y_pred_proba):
    """
    Calculates the Mean Average Precision @ 3.

    Args:
        y_true (np.array): Array of true integer-encoded labels.
        y_pred_proba (np.array): A 2D array of shape (n_samples, n_classes) with
                                 the predicted probabilities for each class.

    Returns:
        float: The MAP@3 score.
    """
    # Get the top 3 predicted class indices for each sample
    # argsort returns indices that would sort the array.
    # We use a negative sign to sort in descending order.
    top_3_preds = np.argsort(-y_pred_proba, axis=1)[:, :3]

    avg_precisions = []
    for i in range(len(y_true)):
        true_label = y_true[i]
        top_3 = top_3_preds[i]

        if true_label in top_3:
            # Find the rank (position) of the true label in the top 3 predictions
            # The rank is 1-based, so we add 1
            rank = np.where(top_3 == true_label)[0][0] + 1
            # Average Precision for this sample is 1 / rank
            avg_precisions.append(1 / rank)
        else:
            # If the true label is not in the top 3, precision is 0
            avg_precisions.append(0)

    # The final score is the mean of all average precisions
    return np.mean(avg_precisions)

print("MAP@3 function defined.")


# --- 5. Example Usage of the Metric ---
print("\nğŸ”¹ Demonstrating the MAP@3 function with dummy predictions...")
# Let's use our validation set's true labels from Fold 0
dummy_y_true = val_df['target_encoded'].values

# Create dummy predicted probabilities (n_samples, n_classes)
# In a real scenario, this comes from `model.predict_proba(X_val)`
n_samples = len(dummy_y_true)
n_classes = len(le.classes_)
# A random prediction baseline
dummy_y_pred_proba = np.random.rand(n_samples, n_classes)
# Normalize probabilities so they sum to 1 across classes
dummy_y_pred_proba = dummy_y_pred_proba / dummy_y_pred_proba.sum(axis=1, keepdims=True)


score = map_at_3(dummy_y_true, dummy_y_pred_proba)
print(f"The MAP@3 score for our dummy random predictions is: {score:.4f}")
print("-" * 50)


import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

print("ğŸ”¹ Setting up the baseline model pipeline (TF-IDF + LightGBM)...")

# --- 1. Define the components of the pipeline ---

# The text vectorizer: Converts text into a matrix of TF-IDF features.
# - stop_words='english': Removes common words like 'the', 'a', 'is'.
# - ngram_range=(1, 2): Considers both single words and two-word phrases.
# - max_features=5000: Limits the vocabulary to the top 5000 features to keep it manageable.
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)


# The classifier: LightGBM is a fast and powerful gradient boosting model.
# - objective='multiclass': Specifies that this is a multi-class classification problem.
# - class_weight='balanced': CRITICAL STEP. This tells LGBM to automatically penalize
#                            mistakes on minority classes more, directly addressing our imbalance problem.
# - random_state=42: For reproducibility.
lgbm = lgb.LGBMClassifier(objective='multiclass',
                          class_weight='balanced',
                          random_state=42)

# Create the pipeline object
pipeline = Pipeline([
    ('tfidf', tfidf),
    ('classifier', lgbm)
])

print("Pipeline created successfully.")
print("-" * 50)


# --- 2. Prepare the data for this specific fold ---
print("ğŸ”¹ Preparing data for Fold 0...")

# Get the actual dataframes for this fold using the indices from StratifiedKFold
train_df = df.iloc[train_idx]
val_df = df.iloc[val_idx]

# Define the feature column and target column
# For our baseline, we will just use the student's explanation.
feature_col = 'full_text'
target_col = 'target_encoded'

X_train = train_df[feature_col]
y_train = train_df[target_col]

X_val = val_df[feature_col]
y_val = val_df[target_col]

print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples.")
print("-" * 50)


# --- 3. Train the model and make predictions ---
print("ğŸ”¹ Training the pipeline on Fold 0 data... (This may take a minute)")
pipeline.fit(X_train, y_train)
print("Training complete.")

print("ğŸ”¹ Making predictions on the validation set...")
# We need probabilities for MAP@3, not just the final predicted class.
val_pred_probas = pipeline.predict_proba(X_val)
print("Predictions generated.")
print("-" * 50)


# --- 4. Evaluate the performance ---
print("ğŸ”¹ Calculating the MAP@3 score for Fold 0...")
fold_0_score = map_at_3(y_val.values, val_pred_probas)

print("\n" + "="*50)
print(f"ğŸ�‰ BASELINE MODEL SCORE (Fold 0): MAP@3 = {fold_0_score:.4f}")
print("="*50)


import numpy as np
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb

print("ğŸ”¹ Starting full 5-fold CV with a FIXED VOCABULARY...")

# --- 1. Create the 'full_text' feature ---
# This is done once on the entire dataframe
df['full_text'] = "question: " + df['QuestionText'].fillna('') + \
                  " [SEP] mc_answer: " + df['MC_Answer'].fillna('') + \
                  " [SEP] explanation: " + df['StudentExplanation'].fillna('')

# --- 2. Create and FIT the vectorizer on ALL text data ---
# This creates our fixed vocabulary.
print("Fitting TF-IDF on the full dataset to create a fixed vocabulary...")
tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=25000, min_df=5, max_df=0.8)
tfidf_vectorizer.fit(df['full_text'])
print(f"Vocabulary created with {len(tfidf_vectorizer.vocabulary_)} features.")

# --- 3. TRANSFORM all text data into a numerical matrix ---
print("Transforming all text data into a TF-IDF matrix...")
X_tfidf = tfidf_vectorizer.transform(df['full_text'])
y = df['target_encoded']


# --- 4. Run Cross-Validation on the numerical matrix ---
oof_scores = []
trained_models = [] # We save just the classifier now

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_tfidf, y)):
    print(f"\n===== Fold {fold} =====")

    # Get the pre-transformed train and validation sets for this fold
    X_train, X_val = X_tfidf[train_idx], X_tfidf[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Define and train ONLY the classifier
    lgbm = lgb.LGBMClassifier(objective='multiclass', class_weight='balanced', random_state=42)
    
    print("Training LightGBM model...")
    lgbm.fit(X_train, y_train)
    trained_models.append(lgbm) # Save the trained model

    # Make predictions and evaluate
    print("Making predictions...")
    val_pred_probas = lgbm.predict_proba(X_val)
    
    fold_score = map_at_3(y_val.values, val_pred_probas)
    oof_scores.append(fold_score)
    print(f"âœ… Fold {fold} MAP@3 Score: {fold_score:.4f}")


# --- 5. Save assets and calculate final score ---
print("\n" + "="*50)
print("âœ… CV finished. Saving vectorizer, models, and LabelEncoder...")
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
for fold_num, model in enumerate(trained_models):
    joblib.dump(model, f'lgbm_baseline_fold_{fold_num}.pkl')
joblib.dump(le, 'label_encoder.pkl')
print("Assets saved successfully.")
print("="*50)

print("\n" + "="*50)
print("Cross-validation complete.")
print(f"Scores for each fold: {[round(s, 4) for s in oof_scores]}")
print(f"ğŸ“ˆ Average CV MAP@3 Score: {np.mean(oof_scores):.4f}")
print(f"Standard Deviation of scores: {np.std(oof_scores):.4f}")
print("="*50)




