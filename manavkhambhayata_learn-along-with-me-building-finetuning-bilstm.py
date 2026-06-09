# Import necessary libraries
import pandas as pd
import zipfile

# Define file paths
train_data_path = "/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip"
test_data_path = "/kaggle/input/word2vec-nlp-tutorial/testData.tsv.zip"
unlabeled_data_path = "/kaggle/input/word2vec-nlp-tutorial/unlabeledTrainData.tsv.zip"

# Load the labeled training data
with zipfile.ZipFile(train_data_path, 'r') as z:
    with z.open("labeledTrainData.tsv") as f:
        train_df = pd.read_csv(f, delimiter="\t")

# Display basic information
print("Dataset Shape:", train_df.shape)
print("First 5 Rows:")
display(train_df.head())

# Check for missing values
print("\nMissing Values:")
print(train_df.isnull().sum())

# Check for duplicate rows
print("\nDuplicate Rows:", train_df.duplicated().sum())

# Check unique sentiments
print("\nSentiment Value Counts:")
print(train_df["sentiment"].value_counts())



import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords (only needed once)
nltk.download('stopwords')

def clean_review(text):
    """
    Function to clean the text data:
    1. Remove HTML tags.
    2. Remove non-alphabetical characters.
    3. Convert text to lowercase.
    4. Remove stopwords.
    """
    # 1. Remove HTML tags using BeautifulSoup
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # 2. Remove non-alphabetic characters (keep only letters)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    
    # 3. Convert to lowercase
    text = text.lower()
    
    # 4. Remove stopwords
    words = text.split()
    stop_words = set(stopwords.words("english"))
    meaningful_words = [word for word in words if word not in stop_words]
    
    # Join words back into one string
    return " ".join(meaningful_words)

# Apply the function to clean reviews
train_df["cleaned_review"] = train_df["review"].apply(clean_review)

# Display the first few cleaned reviews
print("Original Review:\n", train_df["review"][0])
print("\nCleaned Review:\n", train_df["cleaned_review"][0])



train_df.head()


import nltk
from nltk.tokenize import word_tokenize

# Download NLTK tokenizer (only needed once)
nltk.download('punkt')

def tokenize_review(review):
    """
    Tokenizes a cleaned review into words.
    """
    return word_tokenize(review)

# Apply tokenization to all cleaned reviews
train_df["tokenized_review"] = train_df["cleaned_review"].apply(tokenize_review)

# Display first few tokenized reviews
print("Original Cleaned Review:\n", train_df["cleaned_review"][0])
print("\nTokenized Review:\n", train_df["tokenized_review"][0])



from gensim.models import Word2Vec

# Train Word2Vec model
word2vec_model = Word2Vec(
    sentences=train_df["tokenized_review"],  # Tokenized reviews
    vector_size=100,  # Embedding size (100D vectors)
    window=5,  # Context window size
    min_count=2,  # Ignore words appearing less than 2 times
    workers=4,  # Use 4 CPU threads for training
    sg=1  # 1 for Skip-gram, 0 for CBOW
)

# Save model for later use
word2vec_model.save("word2vec_model.bin")

# Print the number of words in the vocabulary
print(f"Vocabulary size: {len(word2vec_model.wv)}")



# Example: Get the vector for the word "good"
word_vector = word2vec_model.wv["good"]
print("Vector representation for 'good':\n", word_vector)


# Find words most similar to "good"
similar_words = word2vec_model.wv.most_similar("good", topn=5)
print("Words most similar to 'good':", similar_words)


# Analogy: "king" - "man" + "woman" = ?
analogy_result = word2vec_model.wv.most_similar(positive=["king", "woman"], negative=["man"], topn=1)
print("king - man + woman = ?", analogy_result)



import numpy as np

def get_average_word_vectors(reviews, model, vector_size):
    """
    Converts a list of reviews into a feature matrix where each review is represented 
    as the average of its word vectors.

    Parameters:
    - reviews: List of tokenized reviews (list of lists of words)
    - model: Trained Word2Vec model
    - vector_size: Size of word vectors

    Returns:
    - A NumPy array where each row is a feature vector for a review.
    """
    review_vectors = []  # List to store review feature vectors

    for review in reviews:
        word_vectors = [model.wv[word] for word in review if word in model.wv]
        
        if len(word_vectors) > 0:
            review_vector = np.mean(word_vectors, axis=0)  # Average word vectors
        else:
            review_vector = np.zeros(vector_size)  # If no known words, use zero vector
        
        review_vectors.append(review_vector)
    
    return np.array(review_vectors)



train_df.head()


# Convert labeled training data
train_review_vectors = get_average_word_vectors(train_df["tokenized_review"], word2vec_model, vector_size=100)

# Check shape
print("Shape of train_review_vectors:", train_review_vectors.shape)



train_review_vectors[0] #One example of how it looks


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(train_review_vectors, train_df["sentiment"], test_size=0.2, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)



import pandas as pd
import zipfile

# Path to the test dataset
test_data_path = "/kaggle/input/word2vec-nlp-tutorial/testData.tsv.zip"

# Unzip and read the test dataset
with zipfile.ZipFile(test_data_path, 'r') as z:
    with z.open("testData.tsv") as f:
        test_df = pd.read_csv(f, delimiter="\t")

# Display basic information
print("Test Dataset Shape:", test_df.shape)
print(test_df.head())



# Apply cleaning function
test_df["cleaned_review"] = test_df["review"].apply(clean_review)

# Display first few cleaned reviews
print("\nOriginal Review:\n", test_df["review"][0])
print("\nCleaned Review:\n", test_df["cleaned_review"][0])



# Apply tokenization
test_df["tokenized_review"] = test_df["cleaned_review"].apply(tokenize_review)

# Display first few tokenized reviews
print("\nTokenized Review:\n", test_df["tokenized_review"][0])



# Convert test reviews into word vectors
test_review_vectors = get_average_word_vectors(test_df["tokenized_review"], word2vec_model, vector_size=100)

# Check the shape
print("Shape of test_review_vectors:", test_review_vectors.shape)



import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout, Embedding, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split



# Labels (Sentiments)
y = train_df["sentiment"].values  # Convert to NumPy array

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_review_vectors, y, test_size=0.2, random_state=42)

print("Training set shape:", X_train.shape, y_train.shape)
print("Validation set shape:", X_val.shape, y_val.shape)



# Define the model
model = Sequential([
    Input(shape=(100,)),  # Input shape = 100-dimensional word vectors
    Dense(256, activation="relu"),  # Fully connected layer
    Dropout(0.3),  # Prevent overfitting
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")  # Binary classification (sentiment: 0 or 1)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0005), loss="binary_crossentropy", metrics=["accuracy"])

# Print model summary
model.summary()



# Train the model
history = model.fit(
    X_train, y_train,
    epochs=10,  # You can increase epochs for better performance
    batch_size=64,
    validation_data=(X_val, y_val),
    verbose=1
)



# Predict sentiment for test data
test_predictions = model.predict(test_review_vectors)

# Convert probabilities to binary labels (0 or 1)
test_labels = (test_predictions > 0.5).astype(int).flatten()

# Check predictions
print("Sample Predictions:", test_labels[:10])



# Create submission dataframe
submission = pd.DataFrame({"id": test_df["id"], "sentiment": test_labels})

# Save to CSV
submission.to_csv("submission.csv", index=False)

print("Submission file saved as submission.csv")



from tensorflow.keras.layers import BatchNormalization

# Define improved model
model = Sequential([
    Dense(512, activation="relu", input_shape=(100,)),  # Increase neurons
    BatchNormalization(),
    Dropout(0.4),  # Increase dropout to prevent overfitting
    
    Dense(256, activation="relu"),
    BatchNormalization(),
    Dropout(0.4),

    Dense(128, activation="relu"),
    Dropout(0.3),
    
    Dense(1, activation="sigmoid")  # Output layer for binary classification
])

# Compile model with Adam optimizer and a reduced learning rate
model.compile(optimizer=Adam(learning_rate=0.0003), loss="binary_crossentropy", metrics=["accuracy"])

# Print model summary
model.summary()



from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Reduce learning rate if validation loss doesn't improve for 3 epochs
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Stop training early if no improvement in 5 epochs
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)



# Train the model with callbacks
history = model.fit(
    X_train, y_train,
    epochs=20,  # Increase epochs
    batch_size=128,  # Increase batch size
    validation_data=(X_val, y_val),
    callbacks=[lr_scheduler, early_stopping],
    verbose=1
)



# Predict sentiment for test data
test_predictions = model.predict(test_review_vectors)

# Convert probabilities to binary labels (0 or 1)
test_labels = (test_predictions > 0.5).astype(int).flatten()

# Save predictions
submission = pd.DataFrame({"id": test_df["id"], "sentiment": test_labels})
submission.to_csv("submission_tuned.csv", index=False)

print("Fine-tuned submission file saved as submission_tuned.csv")





