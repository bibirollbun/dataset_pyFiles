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


import pandas as pd

# Load zipped CSVs directly
train = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')
test = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip')

# Check shapes
print('Train:', train.shape)
print('Test:', test.shape)

# Preview
train.head()


print(train.isnull().sum())


import matplotlib.pyplot as plt
import seaborn as sns

train['comment_length'] = train['comment_text'].apply(len)
train['word_count'] = train['comment_text'].apply(lambda x: len(x.split()))

train[['comment_length', 'word_count']].hist(bins=50, figsize=(12,5))
plt.suptitle('Comment Length Distributions')
plt.show()


labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

label_counts = train[labels].sum()
none_count = (train[labels].sum(axis=1) == 0).sum()
label_counts['none'] = none_count


label_counts.sort_values().plot(kind='bar', title='Label Counts')
plt.show()


plt.figure(figsize=(8,6))
sns.heatmap(train[labels].corr(), annot=True, cmap='Blues')
plt.title('Label Correlation')
plt.show()


from wordcloud import WordCloud

toxic_text = ' '.join(train[train['toxic']==1]['comment_text'])
non_toxic_text = ' '.join(train[train['toxic']==0]['comment_text'])

# plt.figure(figsize=(12,6))
# plt.subplot(1,2,1)
# plt.imshow(WordCloud(max_words=100).generate(toxic_text))
# plt.axis('off')
# plt.title('Toxic Comments')

# plt.subplot(1,2,2)
# plt.imshow(WordCloud(max_words=100).generate(non_toxic_text))
# plt.axis('off')
# plt.title('Non-Toxic Comments')

# plt.show()


train['label_count'] = train[labels].sum(axis=1)

print("\n✅ Example Non-Toxic Comments:\n", train[train['label_count']==0]['comment_text'].sample(2).values)
print("\n✅ Example Toxic Comments:\n", train[train['label_count']>=1]['comment_text'].sample(2).values)


train['has_mention'] = train['comment_text'].str.contains(r'@\w+')
print(train['has_mention'].sum(), "comments contain mentions (@...)")


train['has_hashtag'] = train['comment_text'].str.contains(r'#\w+')
print(train['has_hashtag'].sum(), "comments contain hashtags (#...)")


import re
import string

def clean_text(text):
    text = text.lower() # Lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+|#\w+', '', text) # Remove mentions and hashtags
    text = text.translate(str.maketrans('', '', string.punctuation)) # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'\\n', ' ', text)  # Remove literal \n
    text = re.sub(r'\\r', ' ', text)  # Remove literal \r
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace (new lines, tabs, etc.)
    return text

# Apply to the dataset
train['clean_comment'] = train['comment_text'].apply(clean_text)
test['clean_comment'] = test['comment_text'].apply(clean_text)


print("\n✅ Example Non-Toxic Comments:\n", train[train['label_count']==0]['clean_comment'].sample(2).values)
print("\n✅ Example Toxic Comments:\n", train[train['label_count']>=1]['clean_comment'].sample(2).values)


comment_lengths = train['clean_comment'].apply(lambda x: len(x.split()))
comment_lengths.describe()


from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Step 1: Tokenization and Padding
max_vocab = 15000
max_len = 250 # About two standard deviations from the mean.

tokenizer = Tokenizer(num_words=max_vocab, oov_token="<OOV>")
tokenizer.fit_on_texts(train['clean_comment'])

train_sequences = tokenizer.texts_to_sequences(train['clean_comment'])

X = pad_sequences(train_sequences, maxlen=max_len, padding='post', truncating='post')

X_train, X_val, y_train, y_val = train_test_split(
    X, train[labels], test_size=0.2, random_state=42)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

simple_model = Sequential([
    Embedding(input_dim=max_vocab, output_dim=128),
    SimpleRNN(64),
    Dense(6, activation='sigmoid')
])
simple_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'auc'])

simple_model.summary()


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, GRU, Dense, Dropout

from tensorflow.keras.optimizers import Adam

gru_model = Sequential([
    Embedding(input_dim=max_vocab, output_dim=128),
    Bidirectional(GRU(64, dropout=0.2, recurrent_dropout=0.2)),
    Dense(6, activation='sigmoid')
])

gru_model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.0005),  # smaller LR for stability
    metrics=['accuracy', 'auc']
)

gru_model.summary()


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

lstm_model = Sequential([
    Embedding(input_dim=max_vocab, output_dim=128),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dense(6, activation='sigmoid')
])

lstm_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy', 'auc'])

lstm_model.summary()


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

simple_callbacks_list = [
    # This helps stop the model early
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    # Reduces learning rate when plateued
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    ),
    # save best model
    ModelCheckpoint(
        filepath='simple_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

simple_history = simple_model.fit(
    X_train, 
    y_train, 
    epochs=15, 
    batch_size=128,
    validation_data=(X_val, y_val),
    callbacks=simple_callbacks_list
)


import matplotlib.pyplot as plt

# Create a 1x3 plot layout
plt.figure(figsize=(18, 5))

# Accuracy plot
plt.subplot(1, 3, 1)
plt.plot(simple_history.history['accuracy'], label='Train Accuracy')
plt.plot(simple_history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 3, 2)
plt.plot(simple_history.history['loss'], label='Train Loss')
plt.plot(simple_history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# AUC plot
plt.subplot(1, 3, 3)
plt.plot(simple_history.history['auc'], label='Train AUC')
plt.plot(simple_history.history['val_auc'], label='Val AUC')
plt.title('Model AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()

plt.tight_layout()
plt.show()


from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Predict on validation set
y_pred_probs = simple_model.predict(X_val)
y_pred = (y_pred_probs > 0.5).astype(int)

# Step 2: Evaluation metrics
print("\nClassification Report:\n", classification_report(y_val, y_pred, target_names=labels, zero_division=0))

fig, axes = plt.subplots(3, 2, figsize=(12, 12))
axes = axes.flatten()

# Step 3: Confusion Matrix
for i, label in enumerate(labels):
    cm = confusion_matrix(y_val[label], y_pred[:, i])

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['0', '1'], yticklabels=['0', '1'], ax=axes[i])
    axes[i].set_title(f'Confusion Matrix for "{label}"')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

plt.tight_layout()
plt.show()


from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

for i, label in enumerate(labels):
    fpr, tpr, _ = roc_curve(y_val[label], y_pred_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Each Label')
plt.legend(loc='lower right')
plt.grid()
plt.show()

auc_score = roc_auc_score(y_val, y_pred_probs, average='macro')
print(f"Average AUC is {auc_score}.")


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

gru_callbacks_list = [
    EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=2, 
        min_lr=1e-6, 
        verbose=1
    ),
    ModelCheckpoint(
        filepath='gru_model.keras', 
        monitor='val_loss', 
        save_best_only=True, 
        verbose=1
    )
]

gru_history = gru_model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=128,
    validation_data=(X_val, y_val),
    callbacks=gru_callbacks_list
)


import matplotlib.pyplot as plt

# Create a 1x3 plot layout
plt.figure(figsize=(18, 5))

# Accuracy plot
plt.subplot(1, 3, 1)
plt.plot(gru_history.history['accuracy'], label='Train Accuracy')
plt.plot(gru_history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 3, 2)
plt.plot(gru_history.history['loss'], label='Train Loss')
plt.plot(gru_history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# AUC plot
plt.subplot(1, 3, 3)
plt.plot(gru_history.history['auc'], label='Train AUC')
plt.plot(gru_history.history['val_auc'], label='Val AUC')
plt.title('Model AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()

plt.tight_layout()
plt.show()


from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Predict on validation set
y_pred_probs_gru = gru_model.predict(X_val)
y_pred_gru = (y_pred_probs_gru > 0.5).astype(int)

# Step 2: Evaluation metrics
print("\nClassification Report:\n", classification_report(y_val, y_pred_gru, target_names=labels, zero_division=0))

fig, axes = plt.subplots(3, 2, figsize=(12, 12))
axes = axes.flatten()

# Step 3: Confusion Matrix
for i, label in enumerate(labels):
    cm = confusion_matrix(y_val[label], y_pred_gru[:, i])

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['0', '1'], yticklabels=['0', '1'], ax=axes[i])
    axes[i].set_title(f'Confusion Matrix for "{label}"')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

plt.tight_layout()
plt.show()


from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

for i, label in enumerate(labels):
    fpr, tpr, _ = roc_curve(y_val[label], y_pred_probs_gru[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Each Label')
plt.legend(loc='lower right')
plt.grid()
plt.show()

auc_score_gru = roc_auc_score(y_val, y_pred_probs_gru, average='macro')
print(f"Average AUC is {auc_score_gru}.")


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

lstm_callbacks_list = [
    # This helps stop the model early
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    # Reduces learning rate when plateued
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    ),
    # save best model
    ModelCheckpoint(
        filepath='lstm_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

lstm_history = lstm_model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=128,
    validation_data=(X_val, y_val),
    callbacks=lstm_callbacks_list
)


import matplotlib.pyplot as plt

# Create a 1x3 plot layout
plt.figure(figsize=(18, 5))

# Accuracy plot
plt.subplot(1, 3, 1)
plt.plot(lstm_history.history['accuracy'], label='Train Accuracy')
plt.plot(lstm_history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 3, 2)
plt.plot(lstm_history.history['loss'], label='Train Loss')
plt.plot(lstm_history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# AUC plot
plt.subplot(1, 3, 3)
plt.plot(lstm_history.history['auc'], label='Train AUC')
plt.plot(lstm_history.history['val_auc'], label='Val AUC')
plt.title('Model AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()

plt.tight_layout()
plt.show()


from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Predict on validation set
y_pred_probs_lstm = lstm_model.predict(X_val)
y_pred_lstm = (y_pred_probs_lstm > 0.5).astype(int)

# Step 2: Evaluation metrics
print("\nClassification Report:\n", classification_report(y_val, y_pred_lstm, target_names=labels, zero_division=0))

fig, axes = plt.subplots(3, 2, figsize=(12, 12))
axes = axes.flatten()

# Step 3: Confusion Matrix
for i, label in enumerate(labels):
    cm = confusion_matrix(y_val[label], y_pred_lstm[:, i])

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['0', '1'], yticklabels=['0', '1'], ax=axes[i])
    axes[i].set_title(f'Confusion Matrix for "{label}"')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

plt.tight_layout()
plt.show()


from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

for i, label in enumerate(labels):
    fpr, tpr, _ = roc_curve(y_val[label], y_pred_probs_lstm[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Each Label')
plt.legend(loc='lower right')
plt.grid()
plt.show()

auc_score_lstm = roc_auc_score(y_val, y_pred_probs_lstm, average='macro')
print(f"Average AUC is {auc_score_lstm}.")


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, GRU, Dense, Dropout
from tensorflow.keras.losses import BinaryFocalCrossentropy
from tensorflow.keras.optimizers import Adam

gru_model_v2 = Sequential([
    Embedding(input_dim=max_vocab, output_dim=128),
    Bidirectional(GRU(64, dropout=0.2, recurrent_dropout=0.2)),
    Dense(6, activation='sigmoid')
])

gru_model_v2.compile(
    loss=BinaryFocalCrossentropy(apply_class_balancing=True),
    optimizer=Adam(learning_rate=0.0005),  # smaller LR for stability
    metrics=['accuracy', 'auc']
)

gru_model_v2.summary()


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

gru_callbacks_list = [
    EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=2, 
        min_lr=1e-6, 
        verbose=1
    ),
    ModelCheckpoint(
        filepath='gru_model_v2.keras', 
        monitor='val_loss', 
        save_best_only=True, 
        verbose=1
    )
]

gru_history_v2 = gru_model_v2.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=128,
    validation_data=(X_val, y_val),
    callbacks=gru_callbacks_list
)


import matplotlib.pyplot as plt

# Create a 1x3 plot layout
plt.figure(figsize=(18, 5))

# Accuracy plot
plt.subplot(1, 3, 1)
plt.plot(gru_history_v2.history['accuracy'], label='Train Accuracy')
plt.plot(gru_history_v2.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 3, 2)
plt.plot(gru_history_v2.history['loss'], label='Train Loss')
plt.plot(gru_history_v2.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# AUC plot
plt.subplot(1, 3, 3)
plt.plot(gru_history_v2.history['auc'], label='Train AUC')
plt.plot(gru_history_v2.history['val_auc'], label='Val AUC')
plt.title('Model AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()

plt.tight_layout()
plt.show()


from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Predict on validation set
y_pred_probs_gru2 = gru_model_v2.predict(X_val)
y_pred_gru2 = (y_pred_probs_gru2 > 0.5).astype(int)

# Step 2: Evaluation metrics
print("\nClassification Report:\n", classification_report(y_val, y_pred_gru2, target_names=labels, zero_division=0))

fig, axes = plt.subplots(3, 2, figsize=(12, 12))
axes = axes.flatten()

# Step 3: Confusion Matrix
for i, label in enumerate(labels):
    cm = confusion_matrix(y_val[label], y_pred_gru2[:, i])

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['0', '1'], yticklabels=['0', '1'], ax=axes[i])
    axes[i].set_title(f'Confusion Matrix for "{label}"')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

plt.tight_layout()
plt.show()


from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

for i, label in enumerate(labels):
    fpr, tpr, _ = roc_curve(y_val[label], y_pred_probs_gru2[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Each Label')
plt.legend(loc='lower right')
plt.grid()
plt.show()

auc_score_gru2 = roc_auc_score(y_val, y_pred_probs_gru2, average='macro')
print(f"Average AUC is {auc_score_gru2}.")


test_sequences = tokenizer.texts_to_sequences(test['clean_comment'])
test_padded = pad_sequences(test_sequences, maxlen=max_len, padding='post', truncating='post')

# Predict probabilities
predictions = gru_model_v2.predict(test_padded)

# Create submission DataFrame
submission = pd.DataFrame({
    'id': test['id'],
    'toxic': predictions[:,0],
    'severe_toxic': predictions[:,1],
    'obscene': predictions[:,2],
    'threat': predictions[:,3],
    'insult': predictions[:,4],
    'identity_hate': predictions[:,5]
})

# Preview
submission.head()


# Save to CSV
submission.to_csv('submission.csv', index=False)


results = {
    'Model': ['Simple Model', 'LSTM Model', 'GRU Model', 'GRU Model v2'],
    'AUC': [auc_score, auc_score_lstm, auc_score_gru, auc_score_gru2]
}

df_result = pd.DataFrame(results).sort_values(by=['AUC'], ascending=False)
df_result


from sklearn.metrics import classification_report
import pandas as pd

# List of model predictions and their names
model_preds = {
    'GRU Model v2': y_pred_gru2,
    'GRU Model': y_pred_gru,
    'LSTM Model': y_pred_lstm,
    'Simple Model': y_pred
}

# Collect f1 scores
f1_results = []

for model_name, y_pred_model in model_preds.items():
    report = classification_report(y_val, y_pred_model, target_names=labels, output_dict=True, zero_division=0)
    f1_threat = report.get('threat', {}).get('f1-score', 0)
    f1_identity = report.get('identity_hate', {}).get('f1-score', 0)
    
    f1_results.append({
        'Model': model_name,
        'Threat F1': f1_threat,
        'Identity Hate F1': f1_identity
    })

# Create a DataFrame
f1_df = pd.DataFrame(f1_results)
f1_df = f1_df.sort_values(by='Threat F1', ascending=False).reset_index(drop=True)
f1_df

