# Import necessary libraries
import pandas as pd

# Load the datasets
try:
    train_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
    test_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')
    sample_submission_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv')

    print("Data loaded successfully!")
    print("\nTraining data head:")
    # Showing the correct text column 'body'
    print(train_df[['row_id', 'body', 'rule_violation']].head())
    print("\nTraining data info:")
    train_df.info()

    print("\nTest data head:")
    # Showing the correct text column 'body'
    print(test_df[['row_id', 'body']].head())
    print("\nTest data info:")
    test_df.info()

    print("\nSample Submission head:")
    print(sample_submission_df.head())

except FileNotFoundError:
    print("Error: Make sure 'train.csv', 'test.csv', and 'sample_submission.csv' are in the same directory.")
except Exception as e:
    print(f"An error occurred while loading data: {e}")




import re
import nltk
from nltk.corpus import stopwords



# Get English stopwords
stop_words = set(stopwords.words('english'))


from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words] # Apply lemmatization
    text = ' '.join(words)
    return text


# Define the preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return "" # Handle non-string inputs gracefully, e.g., NaNs in 'body'
    text = text.lower() # Lowercase the text
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove punctuation and numbers
    words = text.split() # Tokenize
    words = [word for word in words if word not in stop_words] # Remove stopwords
    text = ' '.join(words) # Join words back into a string
    return text


print("Applying preprocessing to training data...")

train_df['processed_text'] = train_df['body'].apply(preprocess_text)
print("Preprocessing applied to training data.")

print("\nApplying preprocessing to test data...")

test_df['processed_text'] = test_df['body'].apply(preprocess_text)
print("Preprocessing applied to test data.")

print("\nTraining data with processed text head:")
print(train_df[['body', 'processed_text', 'rule_violation']].head())

print("\nTest data with processed text head:")

print(test_df[['body', 'processed_text']].head())



import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define parameters for tokenization and sequencing
max_words = 10000  # Max number of words to keep based on frequency
maxlen = 100       # Max length of sequences (comments). Shorter will be padded, longer truncated.

# Initialize the Tokenizer
print("Initializing Tokenizer...")
tokenizer = Tokenizer(num_words=max_words, oov_token="<unk>") # <unk> for out-of-vocabulary words

# Fit the tokenizer on the training data's processed text
# This builds the vocabulary
print("Fitting Tokenizer on training data...")
tokenizer.fit_on_texts(train_df['processed_text'])
print("Tokenizer fitted.")

# Get the vocabulary size (number of unique words plus 1 for padding/Oov)
vocab_size = len(tokenizer.word_index) + 1
print(f"Vocabulary size: {vocab_size}")

# Convert text to sequences of integers
print("Converting training text to sequences...")
X_train_sequences = tokenizer.texts_to_sequences(train_df['processed_text'])
print("Converting test text to sequences...")
X_test_sequences = tokenizer.texts_to_sequences(test_df['processed_text'])

# Pad the sequences to ensure uniform length
print(f"Padding sequences to maxlen={maxlen}...")
X_train_padded = pad_sequences(X_train_sequences, maxlen=maxlen, padding='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=maxlen, padding='post')
print("Sequences padded.")

# Define the target variable for training
y_train = train_df['rule_violation']

print(f"\nShape of X_train_padded: {X_train_padded.shape}")
print(f"Shape of X_test_padded: {X_test_padded.shape}")
print(f"Shape of y_train: {y_train.shape}")

print("\nRNN-specific feature extraction complete.")



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Define model parameters
embedding_dim = 128  # Dimension of the dense embedding
lstm_units = 128     # Number of LSTM units
epochs = 5           # Number of training epochs (start low, can increase)
batch_size = 32      # Number of samples per gradient update

# Build the Sequential Model
print("Building the RNN (LSTM) model...")
model = Sequential([
    # Embedding layer: Turns positive integers (word IDs) into dense vectors of fixed size.
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen),
    # LSTM layer: The core RNN layer to learn sequential patterns.
    LSTM(units=lstm_units, return_sequences=False), # return_sequences=False for classification output
    # Dropout layer: Helps prevent overfitting by randomly setting a fraction of inputs to 0.
    Dropout(0.5),
    # Dense output layer: Single neuron with sigmoid activation for binary classification.
    Dense(1, activation='sigmoid')
])

# Compile the model
print("Compiling the model...")
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.AUC(name='auc')])

# Print model summary
model.summary()
print("Model compiled.")

# Train the model
print(f"\nTraining the model for {epochs} epochs with batch size {batch_size}...")
history = model.fit(
    X_train_padded, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.2, # Use 20% of training data for validation
    verbose=1 # Show progress bar during training
)
print("Model training complete.")

# You can plot training history to see loss and AUC over epochs
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['auc'], label='Train AUC')
plt.plot(history.history['val_auc'], label='Validation AUC')
plt.title('AUC over Epochs')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()

plt.tight_layout()
plt.show()



# Make predictions on the preprocessed test data
# The model directly outputs probabilities for the positive class
print("Generating predictions on the test data...")
predictions = model.predict(X_test_padded).flatten() # .flatten() converts shape (N,1) to (N,)
print("Predictions generated.")

# Create the submission DataFrame
submission_df = sample_submission_df.copy()

# Assign the predictions to the 'prediction' column
submission_df['prediction'] = predictions

# Display the first few rows of the submission file
print("\nSubmission file head:")
print(submission_df.head())

# Save the submission file
submission_file_name = 'submission_rnn.csv' # Changed filename to reflect RNN model
submission_df.to_csv(submission_file_name, index=False)

print(f"\nSubmission file '{submission_file_name}' created successfully!")
print("You can now submit this file to the Kaggle competition.")


