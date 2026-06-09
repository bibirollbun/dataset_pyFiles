import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
df = pd.read_csv("/kaggle/input/quora-insincere-questions-classification/train.csv")
df = df.drop("qid", axis=1)


df.head()


x = df["question_text"]
y = df["target"]

token = Tokenizer()
token.fit_on_texts(x)
seq = token.texts_to_sequences(x)
seq[1:5] 


pad_seq = pad_sequences(seq,maxlen=300)
pad_seq[1:5]
vocab_size = len(token.word_index)+1
vocab_size


from tqdm import tqdm

embedding_vector = {}
with open('/kaggle/working/embeddings/glove.840B.300d/glove.840B.300d.txt', encoding='utf8') as f:
    for line in tqdm(f, total=2196017): 
        values = line.strip().split(' ')
        word = values[0]
        try:
            coef = np.array(values[1:], dtype='float32')
            embedding_vector[word] = coef
        except ValueError:
            continue 


embedding_matrix = np.zeros((vocab_size,300))
for word,i in tqdm(token.word_index.items()):
    embedding_value = embedding_vector.get(word)
    if embedding_value is not None:
        embedding_matrix[i] = embedding_value


embedding_matrix


from sklearn.utils.class_weight import compute_class_weight
import numpy as np

classWeights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y),
    y=y
)

classWeightDict = dict(enumerate(classWeights))
plt.figure(figsize=(6, 4))
plt.bar(classWeightDict.keys(), classWeightDict.values(), color='skyblue')
plt.xlabel('Class')
plt.ylabel('Weight')
plt.title('Class Weights')
plt.xticks(list(classWeightDict.keys()))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


model = Sequential()
model.add(Embedding(vocab_size, 300, weights=[embedding_matrix], trainable=False))
model.add(Bidirectional(LSTM(64, return_sequences=False)))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()


history = model.fit(
    pad_seq,
    y,
    epochs=5,
    batch_size=256,     
    validation_split=0.2,
    verbose=1
)


# Get the history values for accuracy and loss
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Create two subplots: one for accuracy and one for loss
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot training and validation accuracy
ax1.plot(range(1, len(train_accuracy) + 1), train_accuracy, label='Train Accuracy', color='blue')
ax1.plot(range(1, len(val_accuracy) + 1), val_accuracy, label='Validation Accuracy', color='orange')
ax1.set_title('Training and Validation Accuracy over Epochs')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

# Plot training and validation loss
ax2.plot(range(1, len(train_loss) + 1), train_loss, label='Train Loss', color='blue')
ax2.plot(range(1, len(val_loss) + 1), val_loss, label='Validation Loss', color='orange')
ax2.set_title('Training and Validation Loss over Epochs')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

# Show the plots
plt.tight_layout()
plt.show()


y_pred_probs = model.predict(pad_seq)
y_pred = np.argmax(y_pred_probs, axis=1) 
cm = confusion_matrix(y, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


testing = pd.read_csv('/kaggle/input/quora-insincere-questions-classification/test.csv')

x_test = testing['question_text']
x_test = token.texts_to_sequences(x_test)
testing_seq = pad_sequences(x_test,maxlen=300)
predict = model.predict(testing_seq)
pred = (predict > 0.5).astype(int)
testing['label'] = pred
testing.head()


submit_df = pd.DataFrame({"qid": testing["qid"], "prediction": testing['label']})
submit_df.to_csv("/kaggle/working/submission.csv", index=False)

submit_df

