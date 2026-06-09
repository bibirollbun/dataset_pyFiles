# Step 1: Import libraries
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score





df = pd.read_csv('/kaggle/input/testreview/train-reviews-gmaps.csv')
print(df.columns)
print(df.head())


texts = df['reviews'] 
labels = df['label'].map({'Positive': 1, 'Negative': 0}) 


#Split into train and test
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

#Tokenize and pad texts
max_words = 10000  
max_len = 100      



tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')


# deep learning model
model = Sequential([
    Embedding(input_dim=max_words, output_dim=64, input_length=max_len),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid') 
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


#Train_model
history = model.fit(X_train_pad, y_train, epochs=5, batch_size=32, validation_split=0.2)



y_pred_prob = model.predict(X_test_pad)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


import pandas as pd

#Load your test dataset (replace column/file names if needed)
test_df = pd.read_csv('/kaggle/input/testreview/test-review-gmaps-new.csv')

#Prepare the reviews (again, use actual column name if different)
test_reviews = test_df['reviews']

test_pad = pad_sequences(tokenizer.texts_to_sequences(test_reviews), maxlen=max_len, padding='post')

y_test_pred_prob = model.predict(test_pad)
y_test_pred = (y_test_pred_prob > 0.5).astype(int).flatten()


submission = pd.DataFrame({
    'id': test_df['id'],           
    'Sentiment': y_test_pred     
})


submission.to_csv('submission.csv', index=False)



print(submission.head())            
print(submission)                   



import pandas as pd

# Create your custom Hinglish test data (with example sentiments and unique IDs)
custom_data = [
    {'id': 10001, 'reviews': "Yeh restaurant bahut accha tha, loved the food!", 'Sentiment': 1},
    {'id': 10002, 'reviews': "Staff bilkul rude hai, not satisfied.", 'Sentiment': 0},
    {'id': 10003, 'reviews': "Ambience acha hai but food thik-thak tha.", 'Sentiment': 0},
    {'id': 10004, 'reviews': "Parking nahi mili par rest sab kuch theek tha.", 'Sentiment': 1},
    {'id': 10005, 'reviews': "Service was too slow, next time nahi aayenge.", 'Sentiment': 0}
]

custom_df = pd.DataFrame(custom_data)

# Tokenize and pad the reviews
test_pad = pad_sequences(tokenizer.texts_to_sequences(custom_df['reviews']), maxlen=max_len, padding='post')

# Predict sentiment using your model
y_pred_prob = model.predict(test_pad)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Add predictions to DataFrame
custom_df['Predicted'] = y_pred

# Print results
print(custom_df[['id', 'reviews', 'Sentiment', 'Predicted']])

# Optionally, save as a CSV for submission or further analysis
custom_df[['id', 'Predicted']].rename(columns={'Predicted': 'Sentiment'}).to_csv('custom_hinglish_submission.csv', index=False)


