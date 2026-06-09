import pandas as pd
import numpy as np
import os


train_df = pd.read_csv('/kaggle/input/fake-news-detection-projectx/train.csv')


train_df.head()


train_df.info()


train_df.shape


train_df.isnull().sum()


train_df['text'] = train_df['text'].fillna(train_df['title'])


train_df.isnull().sum()


train_df.head()


train_df


X = train_df.drop('type', axis=1)
y = train_df['type']


X.shape


y.shape


import tensorflow as tf


tf.__version__


from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense


messages = X.copy()


messages.reset_index(inplace=True)


messages


import nltk
import re
from nltk.corpus import stopwords


nltk.download('stopwords')


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
  review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])
  review = review.lower()
  review = review.split()

  review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
  review = ' '.join(review)
  corpus.append(review)


corpus


corpus[2]


voc_size = 5000


onehot_repr = [one_hot(words, voc_size) for words in corpus]
onehot_repr


onehot_repr[2]


sent_length = 20
embedded_docs = pad_sequences(onehot_repr, padding='pre', maxlen=sent_length)
print(embedded_docs)


from tensorflow.keras.layers import Embedding, LSTM, Dense, Input

embedding_vector_features = 100
model = Sequential()
model.add(Input(shape=(sent_length, )))
model.add(Embedding(voc_size, embedding_vector_features))
model.add(LSTM(100, dropout = 0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


X_final = np.array(embedded_docs)
y_final = np.array(y)


X_final.shape, y_final.shape


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_final, y_final, test_size=0.2, random_state=42)


history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=64)


import matplotlib.pyplot as plt

# Plot accuracy
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Predictions on training data
y_train_pred_prob = model.predict(X_train)
y_train_pred = (y_train_pred_prob > 0.5).astype(int)

# Calculate metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)

print("=" * 50)
print("TRAINING SET METRICS")
print("=" * 50)
print(f"Accuracy:  {train_accuracy:.4f}")
print(f"Precision: {train_precision:.4f}")
print(f"Recall:    {train_recall:.4f}")
print(f"F1-Score:  {train_f1:.4f}")
print("\n" + classification_report(y_train, y_train_pred, target_names=['Real', 'Fake']))


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Get predictions on validation set
y_val_pred_prob = model.predict(X_val)
y_val_pred = (y_val_pred_prob > 0.5).astype(int)

# Calculate metrics
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred, target_names=['Real', 'Fake']))

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_val, y_val_pred)
print(cm)

# Visualize confusion matrix
import seaborn as sns
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Real', 'Fake'],
            yticklabels=['Real', 'Fake'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()


# Predictions on validation data
y_val_pred_prob = model.predict(X_val)
y_val_pred = (y_val_pred_prob > 0.5).astype(int)

# Calculate metrics
val_accuracy = accuracy_score(y_val, y_val_pred)
val_precision = precision_score(y_val, y_val_pred)
val_recall = recall_score(y_val, y_val_pred)
val_f1 = f1_score(y_val, y_val_pred)

print("=" * 50)
print("VALIDATION SET METRICS")
print("=" * 50)
print(f"Accuracy:  {val_accuracy:.4f}")
print(f"Precision: {val_precision:.4f}")
print(f"Recall:    {val_recall:.4f}")
print(f"F1-Score:  {val_f1:.4f}")
print("\n" + classification_report(y_val, y_val_pred, target_names=['Real', 'Fake']))


test_df = pd.read_csv('/kaggle/input/fake-news-detection-projectx/test.csv')


test_df.head()


test_df


test_df.shape


test_df.info()


test_df.isna().sum()


print("\n" + "=" * 60)
print("STEP 2: PREPROCESSING TEST DATA (SAME AS TRAINING)")
print("=" * 60)

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize stemmer (same as training)
ps = PorterStemmer()

# Create corpus from test data titles (same process as training)
test_corpus = []

print("\nProcessing test titles...")
for i in range(0, len(test_df)):
    # Remove non-alphabetic characters
    review = re.sub('[^a-zA-Z]', ' ', test_df['title'].iloc[i])
    # Convert to lowercase
    review = review.lower()
    # Split into words
    review = review.split()
    # Stem and remove stopwords
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    # Join back to string
    review = ' '.join(review)
    test_corpus.append(review)

print(f"✓ Processed {len(test_corpus)} test titles")

# Show samples of processed text
print("\n" + "=" * 60)
print("SAMPLE PROCESSED TITLES")
print("=" * 60)
print("\nOriginal vs Processed:")
for i in range(3):
    print(f"\n{i+1}. Original: {test_df['title'].iloc[i][:70]}...")
    print(f"   Processed: {test_corpus[i][:70]}...")

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)


from tensorflow.keras.preprocessing.text import one_hot

print("\n" + "=" * 60)
print("STEP 3: ONE-HOT ENCODING TEST DATA")
print("=" * 60)

# Use the same vocabulary size as training
voc_size = 5000

# Apply one-hot encoding to test corpus
test_onehot_repr = [one_hot(words, voc_size) for words in test_corpus]

print(f"✓ One-hot encoding completed")
print(f"  Vocabulary size: {voc_size}")
print(f"  Encoded samples: {len(test_onehot_repr)}")

# Show sample encoded sequences
print("\n" + "=" * 60)
print("SAMPLE ENCODED SEQUENCES")
print("=" * 60)
for i in range(3):
    print(f"\n{i+1}. Processed text: {test_corpus[i][:50]}...")
    print(f"   Encoded sequence: {test_onehot_repr[i][:15]}...")  # Show first 15 numbers
    print(f"   Sequence length: {len(test_onehot_repr[i])} words")

print("\n" + "=" * 60)
print("ONE-HOT ENCODING COMPLETE")
print("=" * 60)


print("\n" + "=" * 60)
print("STEP 4: PADDING SEQUENCES")
print("=" * 60)

# Use the same sentence length as training
sent_length = 20

# Pad sequences (same parameters as training)
test_embedded_docs = pad_sequences(test_onehot_repr, padding='pre', maxlen=sent_length)

print(f"✓ Padding completed")
print(f"  Padding type: 'pre' (zeros added at the beginning)")
print(f"  Max length: {sent_length}")
print(f"  Shape: {test_embedded_docs.shape}")

# Show sample padded sequences
print("\n" + "=" * 60)
print("SAMPLE PADDED SEQUENCES")
print("=" * 60)
for i in range(3):
    print(f"\n{i+1}. Original length: {len(test_onehot_repr[i])} words")
    print(f"   Padded length: {len(test_embedded_docs[i])} words")
    print(f"   Padded sequence: {test_embedded_docs[i]}")
    print(f"   (Note: Zeros added at the beginning if length < {sent_length})")

# Convert to numpy array (final format for model)
X_test_final = np.array(test_embedded_docs)

print("\n" + "=" * 60)
print("FINAL TEST DATA PREPARED")
print("=" * 60)
print(f"✓ X_test_final shape: {X_test_final.shape}")
print(f"  - {X_test_final.shape[0]} samples")
print(f"  - {X_test_final.shape[1]} features (padded length)")
print(f"\n✓ Test data is ready for prediction!")



print("\n" + "=" * 60)
print("STEP 5: MAKING PREDICTIONS ON TEST DATA")
print("=" * 60)

# Make predictions (get probabilities)
print("\nGenerating predictions...")
y_test_pred_prob = model.predict(X_test_final, verbose=1)

print(f"\n✓ Predictions completed!")
print(f"  Prediction shape: {y_test_pred_prob.shape}")
print(f"  Total predictions: {len(y_test_pred_prob)}")

# Convert probabilities to binary predictions (0 or 1)
# Threshold: 0.5 (if probability > 0.5, predict Fake=1, else Real=0)
y_test_pred = (y_test_pred_prob > 0.5).astype(int).flatten()

print("\n" + "=" * 60)
print("PREDICTION SUMMARY")
print("=" * 60)
print(f"Total predictions: {len(y_test_pred)}")
print(f"Predicted as Real (0): {(y_test_pred == 0).sum()} ({(y_test_pred == 0).sum()/len(y_test_pred)*100:.1f}%)")
print(f"Predicted as Fake (1): {(y_test_pred == 1).sum()} ({(y_test_pred == 1).sum()/len(y_test_pred)*100:.1f}%)")

# Show sample predictions
print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS (First 10)")
print("=" * 60)
print(f"{'ID':<8} {'Title':<50} {'Probability':<12} {'Prediction':<12}")
print("-" * 82)
for i in range(10):
    title = test_df['title'].iloc[i][:47] + "..." if len(test_df['title'].iloc[i]) > 50 else test_df['title'].iloc[i]
    prob = y_test_pred_prob[i][0]
    pred = "Fake" if y_test_pred[i] == 1 else "Real"
    print(f"{test_df['id'].iloc[i]:<8} {title:<50} {prob:<12.4f} {pred:<12}")

print("\n" + "=" * 60)
print("PREDICTIONS COMPLETE")
print("=" * 60)


print("\n" + "=" * 60)
print("STEP 6: CREATING SUBMISSION FILE")
print("=" * 60)

# Create submission dataframe (id and prediction only)
submission = pd.DataFrame({
    'id': test_df['id'],
    'prediction': y_test_pred
})

# Show first few rows
print("\nSubmission file preview (first 15 rows):")
print(submission.head(15))

print(f"\n✓ Submission shape: {submission.shape}")
print(f"  - {submission.shape[0]} predictions")
print(f"  - {submission.shape[1]} columns (id, prediction)")

# Check distribution
print("\n" + "=" * 60)
print("PREDICTION DISTRIBUTION")
print("=" * 60)
print(submission['prediction'].value_counts().sort_index())
print(f"\nReal (0): {(submission['prediction'] == 0).sum()} ({(submission['prediction'] == 0).sum()/len(submission)*100:.1f}%)")
print(f"Fake (1): {(submission['prediction'] == 1).sum()} ({(submission['prediction'] == 1).sum()/len(submission)*100:.1f}%)")

# Validation checks
print("\n" + "=" * 60)
print("VALIDATION CHECKS")
print("=" * 60)
print(f"✓ All IDs present: {len(submission) == len(test_df)}")
print(f"✓ No missing values: {submission.isnull().sum().sum() == 0}")
print(f"✓ Predictions are binary (0 or 1): {set(submission['prediction'].unique()) == {0, 1}}")
print(f"✓ No duplicate IDs: {submission['id'].nunique() == len(submission)}")

# Save submission file
submission.to_csv('submission.csv', index=False)

print("\n" + "=" * 60)
print("✅ SUBMISSION FILE CREATED SUCCESSFULLY!")
print("=" * 60)
print("File name: 'submission.csv'")
print("Format: CSV with 'id' and 'prediction' columns")
print("Total rows:", len(submission))




