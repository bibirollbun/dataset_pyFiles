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


import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping



train_df = pd.read_csv('/kaggle/input/news-classification-challenge/Train.csv')
test_df = pd.read_csv('/kaggle/input/news-classification-challenge/Test.csv')


print("Training data shape:", train_df.shape)
print("Test data shape:", test_df.shape)
train_df.head()


print(train_df['Category'].value_counts())


#Text preprocessing function
def clean_text(text):
    text = text.lower()                                  # Lowercase
    text = re.sub(r'http\S+|www\S+', '', text)           # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)              # Remove punctuation/numbers
    text = re.sub(r'\s+', ' ', text).strip()             # Remove extra spaces
    return text

train_df['clean_headline'] = train_df['Headline'].apply(clean_text)
test_df['clean_headline'] = test_df['Headline'].apply(clean_text)



#Encode labels
label_encoder = LabelEncoder()
train_df['label'] = label_encoder.fit_transform(train_df['Category'])
num_classes = len(label_encoder.classes_)

print("Classes:", label_encoder.classes_)
print("Number of classes:", num_classes)


#Tokenization and Padding
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_df['clean_headline'])

X = tokenizer.texts_to_sequences(train_df['clean_headline'])
X_test = tokenizer.texts_to_sequences(test_df['clean_headline'])

max_length = 20  # Based on typical news headline length
X = pad_sequences(X, maxlen=max_length, padding='post', truncating='post')
X_test = pad_sequences(X_test, maxlen=max_length, padding='post', truncating='post')

y = to_categorical(train_df['label'], num_classes=num_classes)


#Split train/validation data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


#Build LSTM Model
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=max_length),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


#Train the model
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)


model.summary()


#Visualize accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.show()



plt.figure(figsize=(12,5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()



from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()



from collections import Counter

all_words = " ".join(train_df['clean_headline']).split()
word_freq = Counter(all_words)
common_words = word_freq.most_common(20)

words, counts = zip(*common_words)
plt.figure(figsize=(10,5))
sns.barplot(x=list(words), y=list(counts), palette="coolwarm")
plt.xticks(rotation=45)
plt.title("Top 20 Most Frequent Words in Headlines")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.show()



# Predict probabilities for validation data
y_pred_probs = model.predict(X_val)

# Convert probabilities to class labels
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_val, axis=1)



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Compute evaluation metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print("ğŸ“ˆ Model Evaluation Metrics:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")



# Print class-wise precision, recall, and f1-score
print("\nDetailed Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))



cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()



metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-score'],
    'Score': [accuracy, precision, recall, f1]
})

plt.figure(figsize=(8,5))
sns.barplot(x='Metric', y='Score', data=metrics_df, palette='viridis')
plt.title("Model Performance Comparison")
plt.ylim(0, 1)
for i, v in enumerate(metrics_df['Score']):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=10)
plt.show()



best_metric = metrics_df.loc[metrics_df['Score'].idxmax()]
print(f"ğŸ�… Best Performing Metric: {best_metric['Metric']} ({best_metric['Score']:.4f})")



#Predict on test data
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)
predicted_categories = label_encoder.inverse_transform(predicted_labels)


#Preparing submission file
submission = pd.DataFrame({
    'id': test_df['Id'],
    'category': predicted_categories
})

submission.to_csv('/kaggle/working/submission.csv', index=False)
print("âœ… Submission file created: /kaggle/working/submission.csv")


#Display first few predictions
submission.head()




