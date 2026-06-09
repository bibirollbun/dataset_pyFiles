import numpy as np 
import pandas as pd 
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import os
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split



df_train = pd.read_csv('/kaggle/input/news-classification-challenge/Train.csv')
df_test =  pd.read_csv('/kaggle/input/news-classification-challenge/Test.csv')



df_train.info


df_train.info()


df_train.drop('Id', axis =1, inplace= True)


df_train.info()


df_train.Category.unique()


df_test.head(2)


x = df_train['Headline']
y = df_train['Category']


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 2, random_state = 10)


xtrain.head()


tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(xtrain)

sequences = tokenizer.texts_to_sequences(xtrain)


padded_sequences = pad_sequences(sequences, padding='post', maxlen=10)






model = Sequential([
    Embedding(input_dim=1000, output_dim=64, input_length=10),
    LSTM(50, activation='tanh', return_sequences=True, input_shape=(10, 5)),  
    LSTM(30, activation='tanh'), 
    Dense(10, activation='sigmoid')  
])

model.compile(optimizer= 'adam', loss='binary_crossentropy', metrics=['accuracy', 'mae'])



model.summary()


label_encoder = LabelEncoder()
label_encoder.fit(ytrain)
# label_encoder.fit(ytest)

y_train_e = label_encoder.transform(ytrain)
y_test_e = label_encoder.transform(ytest)


y_train_encoded = to_categorical(y_train_e, num_classes=10)




history = model.fit(padded_sequences, y_train_encoded, epochs=10)

    


df_test.info()


sample_text = df_test.Headline[0]



sample_seq = tokenizer.texts_to_sequences(sample_text)

sample_padded = pad_sequences(sample_seq, padding='post', maxlen=10)



prediction = model.predict(sample_padded)


predicted_class = prediction.argmax(axis=-1)[0]


predicted = label_encoder.inverse_transform([predicted_class])[0]
print("Input Text:", sample_text)
print("predicted:", predicted)




loss = history.history['loss']
mae = history.history['mae']
accuracy = history.history['accuracy']

epochs_range = range(len(loss))  


plt.plot(epochs_range, mae, label='Training MAE')
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, accuracy, label='Training accuracy')
plt.legend(loc='upper right')
plt.title('Training MAE, loss and accuracy')
plt.xlabel('Epoch')
plt.ylabel('value')
plt.grid(True)
plt.show()




test_sequences = tokenizer.texts_to_sequences(df_test['Headline'])
test_padded = pad_sequences(test_sequences, padding='post', maxlen=10)


raw_predictions = model.predict(test_padded)

predicted_class_indices = np.argmax(raw_predictions, axis=1)


final = label_encoder.inverse_transform(predicted_class_indices)


submission = pd.DataFrame({
    "Id": df_test["Id"],
    # "Headline": df_test["Headline"],
    "Predicted_Category": final
})


submission.to_csv("submission.csv", index=False)

print("Saved.")




