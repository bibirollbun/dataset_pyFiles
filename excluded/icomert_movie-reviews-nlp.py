import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report


df = pd.read_csv('/kaggle/input/sentiment-analysis-on-movie-reviews/train.tsv.zip' , sep = '\t')


df.head()


df.shape


df.info()


#distribution of sentiment labels
sentiment_counts = df['Sentiment'].value_counts()

sns.set_theme(style="whitegrid")
palette = sns.color_palette("pastel")
plt.figure(figsize=(8, 6))
ax = sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=palette)
plt.title("Distribution of Sentiment Labels", fontsize=16, weight='bold')
plt.xlabel("Sentiment Categories", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=10, color='black')
plt.tight_layout()
plt.show()


#cleaning the text
df['Phrase'] = df['Phrase'].str.lower().str.replace(r'[^\w\s]|\d+|\n|\r', '', regex=True)


#TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=10000,ngram_range=(1, 2))
X = tfidf.fit_transform(df['Phrase']).toarray()
y = tf.keras.utils.to_categorical(df['Sentiment'], num_classes=len(df['Sentiment'].unique()))

#splitting data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#defining the deep learning model
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y_train.shape[1], activation='softmax')
])

#compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


#training the model
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val),verbose=0)


model.summary()


#evaluate the model
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Model Accuracy: {accuracy:.2f}")


#classification report
y_pred_probs = model.predict(X_val)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_val, axis=1)

target_names = ['Negative', 'Somewhat Negative', 'Neutral', 'Somewhat Positive', 'Positive']
print(classification_report(y_true, y_pred, target_names=target_names))


#confusion matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

