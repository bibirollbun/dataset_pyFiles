!pip install pandas numpy matplotlib seaborn scikit-learn nltk wordcloud tensorflow transformers datasets evaluate


import warnings
warnings.filterwarnings('ignore')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D, GlobalMaxPooling1D

from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import evaluate

np.random.seed(42)
tf.random.set_seed(42)


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


!unzip /kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip


df = pd.read_csv("train.csv").drop(columns=["id"])
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
df = df.sample(n=8000,random_state=42)


df.shape


print(f"Total number of records: {len(df)}")

print("\nNumber of samples in each class:")
for column in df.columns[1:]:  # Skip the comment_text column
    print(f"{column}: {df[column].value_counts()[1]} toxic, {df[column].value_counts()[0]} non-toxic")


all_words = []
for comment in df['comment_text']:
    words = word_tokenize(comment.lower())
    all_words.extend(words)

print(f"Total number of words: {len(all_words)}")

word_freq = Counter(all_words)
print("\nTop 20 most frequent words and their frequencies:")
for word, freq in word_freq.most_common(20):
    print(f"{word}: {freq}")


plt.figure(figsize=(10, 6))
df.iloc[:, 1:].sum().plot(kind='bar')
plt.title('Distribution of Toxic Comment Categories')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


def clean_text(text):
    text = text.lower()
    
    text = re.sub(r'<.*?>', '', text)
    
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    tokens = word_tokenize(text)
    
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text


df['cleaned_text'] = df['comment_text'].apply(clean_text)

pd.set_option('display.max_colwidth', None)
comparison_df = df[['comment_text', 'cleaned_text']].head(5)
comparison_df


label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

X = df['cleaned_text']
y = df[label_cols]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

print("\nLabel distribution in training set:")
for col in label_cols:
    print(f"{col}: {y_train[col].sum()} positive samples ({y_train[col].sum() / len(y_train) * 100:.2f}%)")


bow_vectorizer = CountVectorizer(max_features=5000)
X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow = bow_vectorizer.transform(X_test)

print(f"BoW feature shape: {X_train_bow.shape}")


tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print(f"TF-IDF feature shape: {X_train_tfidf.shape}")


tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_length = 100
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post')

print(f"Padded sequence shape: {X_train_pad.shape}")


from sklearn.multiclass import OneVsRestClassifier

base_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Naive Bayes': MultinomialNB(),
    'SVM': LinearSVC(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

models = {}
for name, model in base_models.items():
    models[name] = OneVsRestClassifier(model)

features = {
    'BoW': (X_train_bow, X_test_bow),
    'TF-IDF': (X_train_tfidf, X_test_tfidf)
}

from sklearn.metrics import hamming_loss, jaccard_score

metrics = {
    'Accuracy': lambda y_true, y_pred: accuracy_score(y_true, y_pred),
    'Hamming Loss': lambda y_true, y_pred: hamming_loss(y_true, y_pred),
    'Jaccard Score': lambda y_true, y_pred: jaccard_score(y_true, y_pred, average='samples'),
    'Precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='micro', zero_division=0),
    'Recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='micro'),
    'F1 Score': lambda y_true, y_pred: f1_score(y_true, y_pred, average='micro')
}

results = {}


def evaluate_model(model, X_test, y_test, metrics):
    y_pred = model.predict(X_test)
    results = {}
    for metric_name, metric_func in metrics.items():
        results[metric_name] = metric_func(y_test, y_pred)
    return results

def predict_and_display(model, vectorizer, text):
    cleaned_text = clean_text(text)
    
    if isinstance(vectorizer, CountVectorizer) or isinstance(vectorizer, TfidfVectorizer):
        X = vectorizer.transform([cleaned_text])
    else: 
        X = vectorizer([cleaned_text])
    
    predictions = model.predict(X)
    
    print(f"Text: {text}")
    print(f"Cleaned text: {cleaned_text}")
    print("Predictions:")
    
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:  # Multi-label
        for i, label in enumerate(label_cols):
            print(f"{label}: {'Yes' if predictions[0][i] == 1 else 'No'}")
    else: 
        print(f"toxic: {'Yes' if predictions[0] == 1 else 'No'}")
    
    print("---")


all_results = pd.DataFrame()



lr_bow_model = OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=42))
lr_bow_model.fit(X_train_bow, y_train)

lr_bow_results = evaluate_model(lr_bow_model, X_test_bow, y_test, metrics)

print("Logistic Regression with BoW Results:")
for metric, value in lr_bow_results.items():
    print(f"{metric}: {value:.4f}")

for metric, value in lr_bow_results.items():
    all_results.loc["Logistic Regression (BoW)", metric] = value


lr_tfidf_model = OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=42))
lr_tfidf_model.fit(X_train_tfidf, y_train)

lr_tfidf_results = evaluate_model(lr_tfidf_model, X_test_tfidf, y_test, metrics)

print("Logistic Regression with TF-IDF Results:")
for metric, value in lr_tfidf_results.items():
    print(f"{metric}: {value:.4f}")

for metric, value in lr_tfidf_results.items():
    all_results.loc["Logistic Regression (TF-IDF)", metric] = value


nb_bow_model = OneVsRestClassifier(MultinomialNB())
nb_bow_model.fit(X_train_bow, y_train)

nb_bow_results = evaluate_model(nb_bow_model, X_test_bow, y_test, metrics)

print("Naive Bayes with BoW Results:")
for metric, value in nb_bow_results.items():
    print(f"{metric}: {value:.4f}")

for metric, value in nb_bow_results.items():
    all_results.loc["Naive Bayes (BoW)", metric] = value


svm_tfidf_model = OneVsRestClassifier(LinearSVC(random_state=42))
svm_tfidf_model.fit(X_train_tfidf, y_train)

svm_tfidf_results = evaluate_model(svm_tfidf_model, X_test_tfidf, y_test, metrics)

print("SVM with TF-IDF Results:")
for metric, value in svm_tfidf_results.items():
    print(f"{metric}: {value:.4f}")

for metric, value in svm_tfidf_results.items():
    all_results.loc["SVM (TF-IDF)", metric] = value


test_text = "SHUT UP YOU STUPID IDIOT!!!"
predict_and_display(svm_tfidf_model, tfidf_vectorizer, test_text)


rf_bow_model = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
rf_bow_model.fit(X_train_bow, y_train)

rf_bow_results = evaluate_model(rf_bow_model, X_test_bow, y_test, metrics)

print("Random Forest with BoW Results:")
for metric, value in rf_bow_results.items():
    print(f"{metric}: {value:.4f}")

for metric, value in rf_bow_results.items():
    all_results.loc["Random Forest (BoW)", metric] = value


test_text = "I hope you die in a fire, you worthless piece of garbage."
predict_and_display(rf_bow_model, bow_vectorizer, test_text)


print("All Classical ML Results:")
display(all_results)

plt.figure(figsize=(15, 10))
sns.heatmap(all_results, annot=True, cmap='YlGnBu', fmt='.4f')
plt.title('Classical ML Model Performance Comparison')
plt.tight_layout()
plt.show()


vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100
num_classes = len(label_cols)  # Number of classes for multi-label classification

y_train_array = np.array(y_train)
y_test_array = np.array(y_test)


import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dropout, Dense

lstm_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(32)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='sigmoid')  # Multiple outputs with sigmoid for multi-label
])

lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

lstm_model.summary()



lstm_history = lstm_model.fit(
    X_train_pad, y_train_array,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)


cnn_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    Conv1D(128, 5, activation='relu'),
    MaxPooling1D(5),
    Conv1D(128, 5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='sigmoid')  # Multiple outputs with sigmoid for multi-label
])

cnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
cnn_model.summary()


cnn_history = cnn_model.fit(
    X_train_pad, y_train_array,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)


def evaluate_dl_model(model, X_test, y_test):
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    results = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Hamming Loss': hamming_loss(y_test, y_pred),
        'Jaccard Score': jaccard_score(y_test, y_pred, average='samples'),
        'Precision': precision_score(y_test, y_pred, average='micro', zero_division=0),
        'Recall': recall_score(y_test, y_pred, average='micro'),
        'F1 Score': f1_score(y_test, y_pred, average='micro')
    }
    return results

def prepare_text_for_dl(text, tokenizer, max_length):
    cleaned_text = clean_text(text)
    
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(sequence, maxlen=max_length, padding='post')
    
    return padded



lstm_results = evaluate_dl_model(lstm_model, X_test_pad, y_test_array)

print("LSTM Model Results:")
for metric, value in lstm_results.items():
    print(f"{metric}: {value:.4f}")

for metric, value in lstm_results.items():
    all_results.loc["LSTM", metric] = value


test_text = "You are a disgusting person and I hate you!"
padded_text = prepare_text_for_dl(test_text, tokenizer, max_length)
predictions = lstm_model.predict(padded_text)
binary_predictions = (predictions > 0.5).astype(int)[0]

print(f"Text: {test_text}")
print("Predictions:")
for i, label in enumerate(label_cols):
    print(f"{label}: {'Yes' if binary_predictions[i] == 1 else 'No'} (Probability: {predictions[0][i]:.4f})")


cnn_results = evaluate_dl_model(cnn_model, X_test_pad, y_test_array)

print("CNN Model Results:")
for metric, value in cnn_results.items():
    print(f"{metric}: {value:.4f}")

for metric, value in cnn_results.items():
    all_results.loc["CNN", metric] = value


test_text = "This is a normal comment about the weather."
padded_text = prepare_text_for_dl(test_text, tokenizer, max_length)
predictions = cnn_model.predict(padded_text)
binary_predictions = (predictions > 0.5).astype(int)[0]

print(f"Text: {test_text}")
print("Predictions:")
for i, label in enumerate(label_cols):
    print(f"{label}: {'Yes' if binary_predictions[i] == 1 else 'No'} (Probability: {predictions[0][i]:.4f})")


dl_results_df = all_results.loc[["LSTM", "CNN"]]
print("Deep Learning Models Comparison:")
display(dl_results_df)

plt.figure(figsize=(15, 6))
sns.heatmap(dl_results_df, annot=True, cmap='YlGnBu', fmt='.4f')
plt.title('Deep Learning Model Performance Comparison')
plt.tight_layout()
plt.show()


bert_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = TFAutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_classes)


from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification

DISTILBERT_MODEL_NAME = 'distilbert-base-uncased'
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
NUM_LABELS = len(label_cols)

distilbert_tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_MODEL_NAME)
distilbert_model = TFDistilBertForSequenceClassification.from_pretrained(
    DISTILBERT_MODEL_NAME, 
    num_labels=NUM_LABELS
)

def tokenize_data(texts, labels=None):
    encodings = distilbert_tokenizer(
        list(texts), 
        truncation=True, 
        padding=True, 
        max_length=MAX_LENGTH, 
        return_tensors='tf'
    )
    
    if labels is not None:
        dataset = tf.data.Dataset.from_tensor_slices((dict(encodings), labels))
    else:
        dataset = tf.data.Dataset.from_tensor_slices(dict(encodings))
    
    return dataset

y_train_array = np.array(y_train)
y_test_array = np.array(y_test)

train_dataset = tokenize_data(X_train, y_train_array)
test_dataset = tokenize_data(X_test, y_test_array)

train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
train_acc_metric = tf.keras.metrics.BinaryAccuracy()
val_acc_metric = tf.keras.metrics.BinaryAccuracy()

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        outputs = distilbert_model(x, training=True)
        logits = outputs.logits
        loss_value = loss_fn(y, logits)
    
    grads = tape.gradient(loss_value, distilbert_model.trainable_weights)
    optimizer.apply_gradients(zip(grads, distilbert_model.trainable_weights))
    
    # Update metrics
    train_acc_metric.update_state(y, tf.nn.sigmoid(logits))
    return loss_value

@tf.function
def val_step(x, y):
    outputs = distilbert_model(x, training=False)
    logits = outputs.logits
    val_acc_metric.update_state(y, tf.nn.sigmoid(logits))
    return loss_fn(y, logits)

train_losses = []
val_losses = []
train_accs = []
val_accs = []

print(f"Starting DistilBERT training for {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    
    
    train_loss = 0
    num_batches = 0
    train_acc_metric.reset_state()
    
    for batch in train_dataset:
        x, y = batch
        batch_loss = train_step(x, y)
        train_loss += batch_loss
        num_batches += 1
        
        if num_batches % 10 == 0:
            print(f"Batch {num_batches}: Loss = {batch_loss:.4f}")
    
    train_loss = train_loss / num_batches
    train_acc = train_acc_metric.result()
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    val_loss = 0
    num_val_batches = 0
    val_acc_metric.reset_state()
    
    for batch in test_dataset:
        x, y = batch
        batch_val_loss = val_step(x, y)
        val_loss += batch_val_loss
        num_val_batches += 1
    
    val_loss = val_loss / num_val_batches
    val_acc = val_acc_metric.result()
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    print(f"Epoch {epoch+1}: Loss = {train_loss:.4f}, Accuracy = {train_acc:.4f}, Val Loss = {val_loss:.4f}, Val Accuracy = {val_acc:.4f}")

distilbert_history = {
    'loss': train_losses,
    'accuracy': train_accs,
    'val_loss': val_losses,
    'val_accuracy': val_accs
}

def predict_with_distilbert(text):
    cleaned_text = clean_text(text)
    
    inputs = distilbert_tokenizer(cleaned_text, return_tensors='tf', truncation=True, padding=True, max_length=MAX_LENGTH)
    outputs = distilbert_model(inputs, training=False)
    logits = outputs.logits
    probs = tf.nn.sigmoid(logits).numpy()[0]
    predictions = (probs > 0.5).astype(int)
    
    return predictions, probs

print("\nEvaluating DistilBERT model on test set...")
distilbert_preds = []
distilbert_probs = []

for batch in test_dataset:
    x, _ = batch
    outputs = distilbert_model(x, training=False)
    logits = outputs.logits
    batch_probs = tf.nn.sigmoid(logits).numpy()
    batch_preds = (batch_probs > 0.5).astype(int)
    distilbert_preds.extend(batch_preds)
    distilbert_probs.extend(batch_probs)

distilbert_preds = np.array(distilbert_preds)
y_test_array_full = np.array(y_test)[:len(distilbert_preds)]

from sklearn.metrics import accuracy_score, hamming_loss, classification_report, f1_score, precision_score, recall_score, jaccard_score

distilbert_results = {
    'Accuracy': accuracy_score(y_test_array_full, distilbert_preds),
    'Hamming Loss': hamming_loss(y_test_array_full, distilbert_preds),
    'Jaccard Score': jaccard_score(y_test_array_full, distilbert_preds, average='samples'),
    'Precision': precision_score(y_test_array_full, distilbert_preds, average='micro', zero_division=0),
    'Recall': recall_score(y_test_array_full, distilbert_preds, average='micro'),
    'F1 Score': f1_score(y_test_array_full, distilbert_preds, average='micro')
}

print("\nDistilBERT Model Results:")
for metric_name, score in distilbert_results.items():
    print(f"{metric_name}: {score:.4f}")


test_text = "You're a complete moron and I hope you die!"
distilbert_preds, distilbert_probs = predict_with_distilbert(test_text)

print(f"\nText: {test_text}")
print("Predictions:")
for i, label in enumerate(label_cols):
    print(f"{label}: {'Yes' if distilbert_preds[i] == 1 else 'No'} (Probability: {distilbert_probs[i]:.4f})")


print("All Models Comparison:")
display(all_results)

plt.figure(figsize=(15, 10))
sns.heatmap(all_results, annot=True, cmap='YlGnBu', fmt='.4f')
plt.title('Model Performance Comparison')
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
all_results['F1 Score'].sort_values().plot(kind='barh')
plt.title('F1 Scores Across All Models')
plt.xlabel('F1 Score')
plt.tight_layout()
plt.show()


best_model_name = all_results['F1 Score'].idxmax()
print(f"Best model based on F1 score: {best_model_name}")


# Function to predict toxicity of new text using the best model
def predict_toxicity(text):
    # Clean the text
    cleaned = clean_text(text)
    
    # Determine which model to use based on the best model name
    if best_model_name == 'BERT':
        # Use BERT model
        predictions, probabilities = predict_with_bert(text)
        
    elif best_model_name == 'LSTM':
        # Use LSTM model
        padded = prepare_text_for_dl(text, tokenizer, max_length)
        probabilities = lstm_model.predict(padded)[0]
        predictions = (probabilities > 0.5).astype(int)
        
    elif best_model_name == 'CNN':
        padded = prepare_text_for_dl(text, tokenizer, max_length)
        probabilities = cnn_model.predict(padded)[0]
        predictions = (probabilities > 0.5).astype(int)
        
    elif 'Logistic Regression' in best_model_name:
        if '(BoW)' in best_model_name:
            features = bow_vectorizer.transform([cleaned])
            predictions = lr_bow_model.predict(features)[0]
            probabilities = lr_bow_model.predict_proba(features)[0] if hasattr(lr_bow_model, 'predict_proba') else None
        else:  # TF-IDF
            features = tfidf_vectorizer.transform([cleaned])
            predictions = lr_tfidf_model.predict(features)[0]
            probabilities = lr_tfidf_model.predict_proba(features)[0] if hasattr(lr_tfidf_model, 'predict_proba') else None
    
    elif 'Naive Bayes' in best_model_name:
        # Use Naive Bayes with BoW
        features = bow_vectorizer.transform([cleaned])
        predictions = nb_bow_model.predict(features)[0]
        probabilities = nb_bow_model.predict_proba(features)[0] if hasattr(nb_bow_model, 'predict_proba') else None
    
    elif 'SVM' in best_model_name:
        # Use SVM with TF-IDF
        features = tfidf_vectorizer.transform([cleaned])
        predictions = svm_tfidf_model.predict(features)[0]
        probabilities = None  # SVM doesn't provide probabilities by default
    
    elif 'Random Forest' in best_model_name:
        features = bow_vectorizer.transform([cleaned])
        predictions = rf_bow_model.predict(features)[0]
        probabilities = rf_bow_model.predict_proba(features)[0] if hasattr(rf_bow_model, 'predict_proba') else None
    
    print(f"Text: {text}")
    print(f"Cleaned text: {cleaned}")
    print("Predictions:")
    
    for i, label in enumerate(label_cols):
        prediction_text = 'Yes' if predictions[i] == 1 else 'No'
        if probabilities is not None:
            prob_text = f" (Probability: {probabilities[i]:.4f})"
        else:
            prob_text = ""
        print(f"{label}: {prediction_text}{prob_text}")


test_texts = [
    "This is a normal comment about the weather.",
    "YOU ARE AN IDIOT AND I HATE YOU!!!",
    "I respectfully disagree with your opinion.",
    "This product is terrible and the company is a scam.",
    "I hope you die in a fire, you worthless piece of garbage."
]

print(f"Testing with the best model: {best_model_name}\n")
for text in test_texts:
    predict_toxicity(text)
    print("---")

