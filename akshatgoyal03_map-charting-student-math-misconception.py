# MAP - Charting Student Math Misunderstandings - Starter Notebook
# Cell-by-cell walkthrough for Kaggle competition

# Cell 1: Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# NLP tools
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import warnings
warnings.filterwarnings("ignore")

nltk.download('stopwords')
nltk.download('wordnet')


# Cell 2: Load data
train_df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
test_df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')
sample_submission = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv')

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("Sample Submission shape:", sample_submission.shape)

# Preview the data
train_df.head()


# Cell 3: Basic EDA
# Category distribution
plt.figure(figsize=(10,5))
train_df['Category'].value_counts().plot(kind='bar')
plt.title('Category Distribution')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# Check proportion of rows with misconceptions (i.e., non-NA Misconception)
misconception_counts = train_df['Misconception'].isna().value_counts()
plt.figure(figsize=(6,4))
sns.barplot(x=misconception_counts.index.map({True: 'NA', False: 'Labeled'}), y=misconception_counts.values)
plt.title('Presence of Misconception Labels')
plt.ylabel('Count')
plt.show()


# Top misconceptions
top_misconceptions = train_df['Misconception'].value_counts().head(10)
plt.figure(figsize=(10,5))
sns.barplot(x=top_misconceptions.values, y=top_misconceptions.index)
plt.title('Top 10 Misconceptions')
plt.xlabel('Frequency')
plt.show()


# Cell 4: Insightful EDA
# Number of unique questions and their frequency
question_counts = train_df['QuestionId'].value_counts()
plt.figure(figsize=(12,5))
sns.histplot(question_counts, bins=30, kde=True)
plt.title('Distribution of Number of Responses per Question')
plt.xlabel('Number of Responses')
plt.ylabel('Count of Questions')
plt.show()



# Check how many misconceptions occur per question
misconceptions_per_question = train_df[~train_df['Misconception'].isna()].groupby('QuestionId')['Misconception'].count()
plt.figure(figsize=(12,5))
sns.histplot(misconceptions_per_question, bins=30, kde=True)
plt.title('Distribution of Misconceptions per Question')
plt.xlabel('Number of Misconceptions')
plt.ylabel('Count of Questions')
plt.show()



from wordcloud import WordCloud


# WordCloud for misconception vs non-misconception explanations
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Preprocess explanation texts
train_df['clean_text'] = train_df['StudentExplanation'].astype(str).apply(preprocess_text)

# WordCloud for all explanations with misconceptions
text_misconception = " ".join(train_df[train_df['Category'].str.contains("Misconception", na=False)]['clean_text'])
wordcloud1 = WordCloud(width=800, height=400, background_color='white').generate(text_misconception)
plt.figure(figsize=(10,5))
plt.imshow(wordcloud1, interpolation='bilinear')
plt.axis('off')
plt.title('WordCloud - Explanations with Misconceptions')
plt.show()

# WordCloud for correct explanations
text_correct = " ".join(train_df[train_df['Category'].str.contains("Correct", na=False)]['clean_text'])
wordcloud2 = WordCloud(width=800, height=400, background_color='white').generate(text_correct)
plt.figure(figsize=(10,5))
plt.imshow(wordcloud2, interpolation='bilinear')
plt.axis('off')
plt.title('WordCloud - Correct Explanations')
plt.show()


from collections import Counter
# Common tokens in each group
def get_common_tokens(df, label):
    all_text = " ".join(df['clean_text'])
    token_counts = Counter(all_text.split())
    return pd.DataFrame(token_counts.most_common(20), columns=['Token', f'Count_{label}'])

common_mis = get_common_tokens(train_df[train_df['Category'].str.contains("Misconception", na=False)], 'misconception')
common_cor = get_common_tokens(train_df[train_df['Category'].str.contains("Correct", na=False)], 'correct')

# Merge and plot
merged_tokens = pd.merge(common_mis, common_cor, on='Token', how='outer').fillna(0)
merged_tokens.set_index('Token').plot(kind='bar', figsize=(15,6), title='Common Tokens in Misconception vs Correct Explanations')
plt.ylabel('Frequency')
plt.show()


def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Preprocess explanation texts
train_df['clean_text'] = train_df['StudentExplanation'].astype(str).apply(preprocess_text)


# WordCloud - Misconceptions
text_misconception = " ".join(train_df[train_df['Category'].str.contains("Misconception", na=False)]['clean_text'])
wordcloud1 = WordCloud(width=800, height=400, background_color='white').generate(text_misconception)
plt.figure(figsize=(10,5))
plt.imshow(wordcloud1, interpolation='bilinear')
plt.axis('off')
plt.title('WordCloud - Explanations with Misconceptions')
plt.show()


# WordCloud - Correct
text_correct = " ".join(train_df[train_df['Category'].str.contains("Correct", na=False)]['clean_text'])
wordcloud2 = WordCloud(width=800, height=400, background_color='white').generate(text_correct)
plt.figure(figsize=(10,5))
plt.imshow(wordcloud2, interpolation='bilinear')
plt.axis('off')
plt.title('WordCloud - Correct Explanations')
plt.show()


# Common tokens comparison
def get_common_tokens(df, label):
    all_text = " ".join(df['clean_text'])
    token_counts = Counter(all_text.split())
    return pd.DataFrame(token_counts.most_common(20), columns=['Token', f'Count_{label}'])

common_mis = get_common_tokens(train_df[train_df['Category'].str.contains("Misconception", na=False)], 'mis')
common_cor = get_common_tokens(train_df[train_df['Category'].str.contains("Correct", na=False)], 'cor')

merged_tokens = pd.merge(common_mis, common_cor, on='Token', how='outer').fillna(0)
merged_tokens.set_index('Token').plot(kind='bar', figsize=(15,6), title='Common Tokens in Misconception vs Correct Explanations')
plt.ylabel('Frequency')
plt.show()


# Cell 5: Text Length Analysis and TF-IDF Comparison
train_df['word_count'] = train_df['StudentExplanation'].apply(lambda x: len(str(x).split()))
train_df['char_count'] = train_df['StudentExplanation'].apply(lambda x: len(str(x)))

plt.figure(figsize=(12,5))
sns.boxplot(data=train_df, x='Category', y='word_count')
plt.title('Word Count by Category')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12,5))
sns.boxplot(data=train_df, x='Category', y='char_count')
plt.title('Character Count by Category')
plt.xticks(rotation=45)
plt.show()


# TF-IDF Discriminative Features
category_subset = train_df[train_df['Category'].isin(['True_Correct', 'False_Misconception'])]
vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1,2))
tfidf_matrix = vectorizer.fit_transform(category_subset['clean_text'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
tfidf_df['Category'] = category_subset['Category'].values

# Mean TF-IDF per term per class
mean_tfidf = tfidf_df.groupby('Category').mean().T
mean_tfidf['Diff'] = mean_tfidf['False_Misconception'] - mean_tfidf['True_Correct']


# Top differentiating terms
top_diff_terms = mean_tfidf['Diff'].sort_values(ascending=False).head(20)
top_diff_terms.plot(kind='barh', figsize=(10,8), title='Top Terms Indicative of Misconceptions')
plt.xlabel('TF-IDF Score Difference')
plt.gca().invert_yaxis()

plt.show()


# # Cell 6: Semantic Embeddings and Clustering
# from sklearn.decomposition import PCA
# from sentence_transformers import SentenceTransformer
# embed_model = SentenceTransformer('all-MiniLM-L6-v2')
# small_sample = train_df.sample(3000, random_state=42)
# embeddings = embed_model.encode(small_sample['clean_text'].tolist(), show_progress_bar=True)

# reducer = PCA(n_components=2)
# reduced = reducer.fit_transform(embeddings)

# plt.figure(figsize=(10,8))
# sns.scatterplot(x=reduced[:,0], y=reduced[:,1], hue=small_sample['Category'], palette='tab10')
# plt.title('PCA Projection of Sentence Embeddings by Category')
# plt.show()


from nltk.corpus import words
nltk.download('words')
english_vocab = set(words.words())

def count_errors(text):
    return sum(1 for word in text.split() if word not in english_vocab)

train_df['spelling_errors'] = train_df['clean_text'].apply(count_errors)

plt.figure(figsize=(12,5))
sns.boxplot(data=train_df, x='Category', y='spelling_errors')
plt.title('Spelling Errors by Category')
plt.xticks(rotation=45)
plt.show()



from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_class_weight



# Embedding and projection
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE



# Cell 8: Feature Engineering
# Basic engineered features
from sklearn.metrics import classification_report, accuracy_score
train_df['word_count'] = train_df['StudentExplanation'].apply(lambda x: len(str(x).split()))
train_df['char_count'] = train_df['StudentExplanation'].apply(lambda x: len(str(x)))
train_df['question_freq'] = train_df['QuestionId'].map(train_df['QuestionId'].value_counts())

# Spelling errors using NLTK words
english_vocab = set(words.words())
def count_errors(text):
    return sum(1 for word in text.split() if word not in english_vocab)
train_df['spelling_errors'] = train_df['clean_text'].apply(count_errors)

# Encode target: We'll classify Category first
le = LabelEncoder()
train_df['CategoryEncoded'] = le.fit_transform(train_df['Category'])

# TF-IDF features
vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = vectorizer.fit_transform(train_df['clean_text'])

# Numeric features
X_numeric = train_df[['word_count', 'char_count', 'question_freq', 'spelling_errors']].reset_index(drop=True)

# Combine features
from scipy.sparse import hstack
X_combined = hstack([X_tfidf, X_numeric])
y = train_df['CategoryEncoded']

# Cell 9: Ensemble Modeling - TFIDF + Structured Features
# Model 1: Logistic Regression (NLP heavy)
lr = LogisticRegression(max_iter=200, class_weight='balanced')

# Model 2: Gradient Boosted Trees (Structured features heavy)
gbt = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Ensemble Voting Classifier
ensemble_model = VotingClassifier(
    estimators=[('lr', lr), ('gbt', gbt)],
    voting='soft'
)

# Train-test split for evaluation
X_train, X_val, y_train, y_val = train_test_split(X_combined, y, test_size=0.2, random_state=42, stratify=y)

ensemble_model.fit(X_train, y_train)
y_pred = ensemble_model.predict(X_val)

print("Ensemble Validation Accuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_val, y_pred, target_names=le.classes_))



from xgboost import XGBClassifier
from scipy.sparse import hstack


# Clean text
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', str(text))
    text = text.lower()
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)
train_df['clean_text'] = train_df['StudentExplanation'].apply(clean_text)
test_df['clean_text'] = test_df['StudentExplanation'].apply(clean_text)


# Feature Engineering
train_df['word_count'] = train_df['StudentExplanation'].apply(lambda x: len(str(x).split()))
train_df['char_count'] = train_df['StudentExplanation'].apply(lambda x: len(str(x)))
train_df['question_freq'] = train_df['QuestionId'].map(train_df['QuestionId'].value_counts())
english_vocab = set(words.words())
def count_errors(text):
    return sum(1 for word in text.split() if word not in english_vocab)
train_df['spelling_errors'] = train_df['clean_text'].apply(count_errors)

test_df['word_count'] = test_df['StudentExplanation'].apply(lambda x: len(str(x).split()))
test_df['char_count'] = test_df['StudentExplanation'].apply(lambda x: len(str(x)))
test_df['question_freq'] = test_df['QuestionId'].map(train_df['QuestionId'].value_counts()).fillna(0)
test_df['spelling_errors'] = test_df['clean_text'].apply(count_errors)


# Encode target
le = LabelEncoder()
train_df['CategoryEncoded'] = le.fit_transform(train_df['Category'])

# TF-IDF
vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1,2))
X_train_text = vectorizer.fit_transform(train_df['clean_text'])
X_test_text = vectorizer.transform(test_df['clean_text'])

# Structured features
X_train_num = train_df[['word_count', 'char_count', 'question_freq', 'spelling_errors']].reset_index(drop=True)
X_test_num = test_df[['word_count', 'char_count', 'question_freq', 'spelling_errors']].reset_index(drop=True)

# Combine features
X_train = hstack([X_train_text, X_train_num])
X_test = hstack([X_test_text, X_test_num])
y_train = train_df['CategoryEncoded']


# Train Boosted Model
model = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# Predict on test
preds = model.predict_proba(X_test)
top_3 = np.argsort(preds, axis=1)[:, -3:][:, ::-1]  # Top 3 predictions


# Format predictions
submission = []
for row_id, pred_indices in zip(test_df.index, top_3):
    labels = [le.inverse_transform([idx])[0] + ":NA" for idx in pred_indices]
    submission.append(" ".join(labels))

sample_submission['Category:Misconception'] = submission
sample_submission.to_csv("submission.csv", index=False)




