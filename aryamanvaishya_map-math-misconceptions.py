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


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

VER=1
model_name = "/kaggle/input/huggingfacedebertav3variants/deberta-v3-xsmall"
EPOCHS = 10

DIR = f"ver_{VER}"
os.makedirs(DIR, exist_ok=True)


import pandas as pd, numpy as np
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
train.Misconception = train.Misconception.fillna('NA')
train['target'] = train.Category+":"+train.Misconception
train['label'] = le.fit_transform(train['target'])
n_classes = len(le.classes_)
print(f"Train shape: {train.shape} with {n_classes} target classes")
train.head()


idx = train.apply(lambda row: row.Category.split('_')[0],axis=1)=='True'
correct = train.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c',ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct = correct[['QuestionId','MC_Answer']]
correct['is_correct'] = 1

train = train.merge(correct, on=['QuestionId','MC_Answer'], how='left')
train.is_correct = train.is_correct.fillna(0)



from IPython.display import display, Math, Latex

# GET ANSWER CHOICES
tmp = train.groupby(['QuestionId','MC_Answer']).size().reset_index(name='count')
tmp['rank'] = tmp.groupby('QuestionId')['count'].rank(method='dense', ascending=False).astype(int) - 1
tmp = tmp.drop('count',axis=1)
tmp = tmp.sort_values(['QuestionId','rank'])

# DISPLAY QUESTION AND ANSWER CHOICES
Q = tmp.QuestionId.unique()
for q in Q:
    question = train.loc[train.QuestionId==q].iloc[0].QuestionText
    choices = tmp.loc[tmp.QuestionId==q].MC_Answer.values
    labels="ABCD"
    choice_str = " ".join([f"({labels[i]}) {choice}" for i, choice in enumerate(choices)])
    
    print()
    display(Latex(f"QuestionId {q}: {question}") )
    display(Latex(f"MC Answers: {choice_str}"))


import torch
from transformers import DebertaTokenizer, DebertaForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from datasets import Dataset
import numpy as np

tokenizer = AutoTokenizer.from_pretrained(model_name)
MAX_LEN = 256


def format_input(row):
    x = "This answer is correct."
    if not row['is_correct']:
        x = "This is answer is incorrect."
    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"{x}\n"
        f"Student Explanation: {row['StudentExplanation']}"
    )

train['text'] = train.apply(format_input,axis=1)
print("Example prompt for our LLM:")
print()
print( train.text.values[0] )


lengths = [len(tokenizer.encode(t, truncation=False)) for t in train["text"]]
import matplotlib.pyplot as plt

plt.hist(lengths, bins=50)
plt.title("Token Length Distribution")
plt.xlabel("Number of tokens")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()



L = (np.array(lengths)>MAX_LEN).sum()
print(f"There are {L} train sample(s) with more than {MAX_LEN} tokens")
np.sort( lengths )



# Split into train and validation sets
train_df, val_df = train_test_split(train, test_size=0.2, random_state=42)

# Convert to Hugging Face Dataset
COLS = ['text','label']
train_ds = Dataset.from_pandas(train_df[COLS])
val_ds = Dataset.from_pandas(val_df[COLS])


# Tokenization function
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

# Set format for PyTorch
columns = ['input_ids', 'attention_mask', 'label']
train_ds.set_format(type='torch', columns=columns)
val_ds.set_format(type='torch', columns=columns)


from transformers import DebertaV2ForSequenceClassification

model = DebertaV2ForSequenceClassification.from_pretrained(
    model_name,
    num_labels=n_classes
)


training_args = TrainingArguments(
    output_dir = f"./{DIR}",
    do_train=True,
    do_eval=True,
    eval_strategy="steps",
    save_strategy="steps", #no for no saving 
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=16*2,
    per_device_eval_batch_size=32*2,
    learning_rate=5e-5,
    logging_dir="./logs",
    logging_steps=50,
    save_steps=200,
    eval_steps=200,
    save_total_limit=1,
    metric_for_best_model="map@3",
    greater_is_better=True,
    load_best_model_at_end=True,
    report_to="none",
)


# CUSTOM MAP@3 METRIC

from sklearn.metrics import average_precision_score

def compute_map3(eval_pred):
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    
    top3 = np.argsort(-probs, axis=1)[:, :3]  # Top 3 predictions
    match = (top3 == labels[:, None])

    # Compute MAP@3 manually
    map3 = 0
    for i in range(len(labels)):
        if match[i, 0]:
            map3 += 1.0
        elif match[i, 1]:
            map3 += 1.0 / 2
        elif match[i, 2]:
            map3 += 1.0 / 3
    return {"map@3": map3 / len(labels)}


# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_map3,
)

trainer.train()


import joblib

trainer.save_model(f"{DIR}/best")
_ = joblib.dump(le, f"{DIR}/label_encoder.joblib")



tokenizer = AutoTokenizer.from_pretrained(f"{DIR}/best")
model = DebertaV2ForSequenceClassification.from_pretrained(
    f"{DIR}/best",
    num_labels=n_classes  
)
training_args = TrainingArguments(report_to="none")
trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args)
le = joblib.load(f"{DIR}/label_encoder.joblib")



test = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')
print( test.shape )
test.head()


test = test.merge(correct, on=['QuestionId','MC_Answer'], how='left')
test.is_correct = test.is_correct.fillna(0)

test['text'] = test.apply(format_input,axis=1)

test.head()


ds_test = Dataset.from_pandas(test[['text']])
ds_test = ds_test.map(tokenize, batched=True)

predictions = trainer.predict(ds_test)
probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1).numpy()


# Get top 3 predicted class indices
top3 = np.argsort(-probs, axis=1)[:, :3]   # shape: [num_samples, 3]

# Decode numeric class indices to original string labels
flat_top3 = top3.flatten()
decoded_labels = le.inverse_transform(flat_top3)
top3_labels = decoded_labels.reshape(top3.shape)

# Join 3 labels per row with space
joined_preds = [" ".join(row) for row in top3_labels]

# Save submission
sub = pd.DataFrame({
    "row_id": test.row_id.values,
    "Category:Misconception": joined_preds
})
sub.to_csv("submission.csv", index=False)
sub.head()




