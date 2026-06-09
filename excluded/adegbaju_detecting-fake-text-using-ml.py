!pip install textstat
!pip install -q pycaret
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
!pip install  transformers
!pip install wordcloud


import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from textstat import flesch_reading_ease
from wordcloud import WordCloud
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report,accuracy_score, ConfusionMatrixDisplay,  f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tensorflow.keras.layers import GRU
import warnings


# 1. Load the train.csv mapping (id → real_text_id)
labels_df = pd.read_csv(
    r'/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv'
)

# Create a dictionary to map id to real_text_id
id_to_real = dict(zip(labels_df["id"], labels_df["real_text_id"]))

# 2. Base train directory
train_dir = Path(r"/kaggle/input/fake-or-real-the-impostor-hunt/data/train")

# 3. Prepare list of records
records = []

# 4. Iterate over each pair-folder by numeric id
for pair_id in labels_df["id"]:
    pair_folder = train_dir / f"article_{pair_id:04d}"
    real_text_id = id_to_real[pair_id]
    
    for text_id in (1, 2):
        file_path = pair_folder / f"file_{text_id}.txt"
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        
        # Label: 1 if text is real, 0 if fake
        label = 1 if text_id == real_text_id else 0
        
        records.append({
            "id": pair_id,        # Matches 'id' from train.csv
            "text": text,
            "label": label        # Binary label (1=real, 0=fake)
        })

# 5. Build DataFrame with id, text, label
df = pd.DataFrame(records)

# 6. (Optional) Save to CSV
df.to_csv("train_pair_texts.csv", index=False)

print(df.head())
print(f"Total records: {len(df)}")


label_distribution = df['label'].value_counts()
label_distribution


# Plot the label distribution
plt.bar(label_distribution.index, label_distribution.values, color=['blue', 'orange'])
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('Label Distribution')
plt.xticks(label_distribution.index, ['Label 0', 'Label 1'])
plt.show()


# Calculate readability scores for the processed text
df['readability_score'] = df['text'].apply(flesch_reading_ease)
df[['id', 'text', 'readability_score']]


# Combine all processed text into a single string
all_text = ' '.join(df['text'])

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Text')
plt.show()


# Download NLTK data if not already available
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize text
    words = word_tokenize(text)
    # Remove stopwords and apply stemming
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    # Join words back to a single string
    return ' '.join(words)

# Handle potential NaN values in the text column before preprocessing
df['text'] = df['text'].fillna('')

# Reapply preprocessing after handling NaN values
df['processed_text'] = df['text'].apply(preprocess_text)
df



# Calculate readability scores for the processed text
df['readability_score'] = df['processed_text'].apply(flesch_reading_ease)
df[['id', 'processed_text', 'readability_score']]


# Combine all processed text into a single string
all_text = ' '.join(df['processed_text'])

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Processed Text')
plt.show()


# Splitting the data into train and test sets
# text
X = df['processed_text']
# Status
y = df['label']


# Split dataset into training (70%), validation (15%), and test sets (15%)
# First split: Training and Temporary (30% of the data)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Second split: Validation and Test from the temporary set (50%-50% of the temporary set)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Print the sizes of each set
print("Train set size:", X_train.shape[0])
print("Validation set size:", X_val.shape[0])
print("Test set size:", X_test.shape[0])



print("\nLabels:")
print("Train set size:", y_train.shape[0])
print("Validation set size:", y_val.shape[0])
print("Test set size:", y_test.shape[0])


# Create a pipeline with TF-IDF vectorizer and Logistic Regression
pipeline = make_pipeline(
    TfidfVectorizer(),
    LogisticRegression(random_state=42)
)

# Train the model on the training data
pipeline.fit(X_train, y_train)

# Evaluate the model on the validation data
val_predictions = pipeline.predict(X_val)

# Generate a classification report
classification_report_val = classification_report(y_val, val_predictions)
classification_report_val


# Evaluate the model on the test set
test_predictions = pipeline.predict(X_test)


# Calculate test accuracy and F1 score
test_accuracy_lr = pipeline.score(X_test, y_test)
test_f1_score_lr = classification_report(y_test, test_predictions, output_dict=True)['weighted avg']['f1-score']

# Calculate validation accuracy
val_accuracy_lr = pipeline.score(X_val, y_val)

# Print the results
print("Test Accuracy:", test_accuracy_lr)
print("Test F1 Score:", test_f1_score_lr)
print("Validation Accuracy:", val_accuracy_lr)



# Generate and display the confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, test_predictions, display_labels=['Label 0', 'Label 1'])
plt.title('Confusion Matrix - Test Set')
plt.show()


# Random Forest Model
rf_pipeline = make_pipeline(TfidfVectorizer(), RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
rf_pipeline.fit(X_train, y_train)
rf_val_predictions = rf_pipeline.predict(X_val)
rf_test_predictions = rf_pipeline.predict(X_test)
rf_val_report = classification_report(y_val, rf_val_predictions)
rf_test_accuracy = accuracy_score(y_test, rf_test_predictions)
rf_test_f1 = f1_score(y_test, rf_test_predictions, average='weighted')

# Store Random Forest results
rf_results = {
    'Validation Report': rf_val_report,
    'Test Accuracy': rf_test_accuracy,
    'Test F1-Score': rf_test_f1
}
rf_results


# Support Vector Machine
svc_pipeline = make_pipeline(TfidfVectorizer(), SVC(kernel='linear', random_state=42))
svc_pipeline.fit(X_train, y_train)
val_predictions_svc = svc_pipeline.predict(X_val)
svc_val_report = classification_report(y_val, val_predictions_svc, output_dict=True)
test_predictions_svc = svc_pipeline.predict(X_test)
svc_test_accuracy = accuracy_score(y_test, test_predictions_svc)
svc_test_f1 = f1_score(y_test, test_predictions_svc, average='weighted')
svc_results = {
    'Validation Report': svc_val_report,
    'Test Accuracy': svc_test_accuracy,
    'Test F1-Score': svc_test_f1
}

svc_results


# Naive Bayes
nb_pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())
nb_pipeline.fit(X_train, y_train)
val_predictions_nb = nb_pipeline.predict(X_val)
nb_val_report = classification_report(y_val, val_predictions_nb, output_dict=True)
test_predictions_nb = nb_pipeline.predict(X_test)
nb_test_accuracy = accuracy_score(y_test, test_predictions_nb)
nb_test_f1 = f1_score(y_test, test_predictions_nb, average='weighted')
nb_results = {
    'Validation Report': nb_val_report,
    'Test Accuracy': nb_test_accuracy,
    'Test F1-Score': nb_test_f1
}

nb_results


# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

# Convert text to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences to ensure uniform input length
max_len = max(len(seq) for seq in X_train_seq)
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
X_val_pad = pad_sequences(X_val_seq, maxlen=max_len, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

# Encode labels
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_val_enc = label_encoder.transform(y_val)
y_test_enc = label_encoder.transform(y_test)

# Define the LSTM model
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train_pad, y_train_enc,
    validation_data=(X_val_pad, y_val_enc),
    epochs=20,
    batch_size=32,
    verbose=1
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test_pad, y_test_enc, verbose=0)

# Calculate F1 score for the test set
y_test_pred = (model.predict(X_test_pad) > 0.5).astype("int32")
test_f1_score = f1_score(y_test_enc, y_test_pred)

history, test_loss, test_accuracy, test_f1_score


# Plot training history
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


# Define a GRU-based model
model_gru = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
    GRU(128, return_sequences=True),
    Dropout(0.2),
    GRU(64),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# Compile the GRU model
model_gru.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the GRU model
history_gru = model_gru.fit(
    X_train_pad, y_train_enc,
    validation_data=(X_val_pad, y_val_enc),
    epochs=20,
    batch_size=32,
    verbose=1
)

# Evaluate the GRU model on the test set
test_loss_gru, test_accuracy_gru = model_gru.evaluate(X_test_pad, y_test_enc, verbose=0)

# Calculate the test F1 score
y_test_pred_gru = (model_gru.predict(X_test_pad) > 0.5).astype("int32")
test_f1_score_gru = f1_score(y_test_enc, y_test_pred_gru)

history_gru, test_loss_gru, test_accuracy_gru, test_f1_score_gru


# Plot training history for GRU model
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history_gru.history['accuracy'], label='Train Accuracy')
plt.plot(history_gru.history['val_accuracy'], label='Validation Accuracy')
plt.title('GRU Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history_gru.history['loss'], label='Train Loss')
plt.plot(history_gru.history['val_loss'], label='Validation Loss')
plt.title('GRU Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


from sklearn.model_selection import GridSearchCV

# Hyperparameter tuning for Logistic Regression
param_grid_lr = {
    'tfidfvectorizer__max_features': [500, 1000, 2000],
    'logisticregression__C': [0.1, 1, 10]
}
grid_search_lr = GridSearchCV(pipeline, param_grid_lr, cv=5, scoring='accuracy', verbose=1)
grid_search_lr.fit(X_train, y_train)

# Best parameters and score for Logistic Regression
best_params_lr = grid_search_lr.best_params_
best_score_lr = grid_search_lr.best_score_

best_params_lr, best_score_lr


# Hyperparameter tuning for Support Vector Machine (SVM)
param_grid_svc = {
    'tfidfvectorizer__max_features': [500, 1000, 2000],
    'svc__C': [0.1, 1, 10],
    'svc__kernel': ['linear', 'rbf']
}
grid_search_svc = GridSearchCV(svc_pipeline, param_grid_svc, cv=5, scoring='accuracy', verbose=1)
grid_search_svc.fit(X_train, y_train)

# Best parameters and score for SVM
best_params_svc = grid_search_svc.best_params_
best_score_svc = grid_search_svc.best_score_

best_params_svc, best_score_svc


# Compare the best cross-validation scores from hyperparameter tuning
if best_score_lr > best_score_svc:
    best_model = 'Logistic Regression'
    best_params = best_params_lr
    best_score = best_score_lr
else:
    best_model = 'Support Vector Machine (SVM)'
    best_params = best_params_svc
    best_score = best_score_svc

best_model, best_params, best_score


import pandas as pd
from pathlib import Path

# 1. Base test directory
test_dir = Path(r"/kaggle/input/fake-or-real-the-impostor-hunt/data/test")

# 2. Prepare list of records
records = []

# 3. Iterate over article folders (e.g., article_0000, article_0001, ...)
for pair_folder in sorted(test_dir.glob("article_*")):
    pair_id = int(pair_folder.name.split("_")[1])  # Extract numeric ID from folder name

    # 4. Read file_1.txt and file_2.txt for each pair
    for text_id in (1, 2):
        file_path = pair_folder / f"file_{text_id}.txt"
        if file_path.exists():
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            records.append({
                "id": pair_id,        # Matches the article number
                "text_id": text_id,   # Indicates if it's file_1 or file_2
                "text": text
            })
        else:
            print(f"Warning: Missing {file_path}")

# 5. Build DataFrame with id, text_id, and text
test_df = pd.DataFrame(records, columns=["id", "text_id", "text"])

# 6. Save to CSV
test_df .to_csv("test_pair_texts.csv", index=False)

# 7. Preview
print(test_df .head())
print(f"Total records: {len(test_df )}")





# Refit the best SVM model with the best parameters
best_svc_pipeline = make_pipeline(
    TfidfVectorizer(max_features=best_params['tfidfvectorizer__max_features']),
    SVC(C=best_params['svc__C'], kernel=best_params['svc__kernel'], random_state=42)
)
best_svc_pipeline.fit(X_train, y_train)

# Extract the TF-IDF vectorizer and SVM classifier
vectorizer = best_svc_pipeline.named_steps['tfidfvectorizer']
classifier = best_svc_pipeline.named_steps['svc']

# Get feature names and coefficients
feature_names = vectorizer.get_feature_names_out()
coefficients = classifier.coef_.toarray().flatten()

# Create a DataFrame for feature importance
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': coefficients
})

# Sort by absolute importance
importance_df['Absolute Importance'] = importance_df['Importance'].abs()
importance_df = importance_df.sort_values(by='Absolute Importance', ascending=False).head(20)

# Plot the top 20 features
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.title('Top 20 Important Features for SVM')
plt.gca().invert_yaxis()
plt.show()


import joblib

# Save the best SVM model to a file
joblib.dump(best_svc_pipeline, 'best_svc_model.pkl')



# Ensure NLTK stopwords are downloaded
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Reload the test data
test_df = pd.read_csv(r'/kaggle/working/test_pair_texts.csv')

# Ensure the 'text' column is treated as string
test_df['text'] = test_df['text'].astype(str)

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Preprocess the text data
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = word_tokenize(text)
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

test_df['processed_text'] = test_df['text'].apply(preprocess_text)

# Load the saved SVM model
best_svc_pipeline = joblib.load('/kaggle/working/best_svc_model.pkl')

# Make predictions
predictions = best_svc_pipeline.predict(test_df['processed_text'])

# Prepare the solution file
solution_df = test_df[['id']].copy()
solution_df['real_text_id'] = predictions
# Save the solution file
solution_df.to_csv('solution.csv', index=False, sep='\t')




