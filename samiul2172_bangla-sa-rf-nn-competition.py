import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from tensorflow import keras
from tensorflow.keras import layers


train_data = pd.read_csv("/kaggle/input/aiquest-bangla-sentiment-analysis-competition/train.csv")


label_encoder = LabelEncoder()
train_data['sentiment'] = label_encoder.fit_transform(train_data['sentiment'])

X = train_data['text']
y = train_data['sentiment']


vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = vectorizer.fit_transform(X).toarray()

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("classification report:\n", classification_report(y_test, y_pred_rf))
print("accuracy:", accuracy_score(y_test, y_pred_rf))



model = keras.Sequential([layers.Input(shape=(X_train.shape[1],)), layers.Dense(128, activation='relu'), layers.Dropout(0.5),layers.Dense(64, activation='relu'), layers.Dropout(0.5), layers.Dense(3, activation='softmax')])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, batch_size=16)


loss, accuracy = model.evaluate(X_test, y_test)
print("neural accuracy:", accuracy)


nn_predictions = model.predict(X_test)
nn_pred_labels = np.argmax(nn_predictions, axis=1)
print("classification report:\n", classification_report(y_test, nn_pred_labels))


submission = pd.DataFrame({'id': train_data['id'], 'sentiment': label_encoder.inverse_transform(y)})
submission.to_csv('submission.csv', index=False)
print("file created.")

