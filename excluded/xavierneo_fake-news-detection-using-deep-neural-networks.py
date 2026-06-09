import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


import psutil
def check_memory():
    print(f"RAM Usage: {psutil.virtual_memory().used >> 30}GB({psutil.virtual_memory().percent}%)/{psutil.virtual_memory().total>>30}GB")


check_memory()


try:
  import google.colab
  IN_COLAB = True
except:
  IN_COLAB = False


if IN_COLAB:
  from google.colab import drive
  drive.mount('/content/drive')
  %cd /content/drive/MyDrive/ML/


import pandas as pd
import numpy as np


df = pd.read_csv('/kaggle/input/depi-r-2-competition-1/xy_train.csv')


df.shape


df.info()


labels = df["label"].value_counts()
print(labels)


df.loc[df["label"] == 2].head()


df = df.loc[df["label"] != 2]
print(df.shape)


import re
# Tokenization
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
# Stopwords
from nltk.corpus import stopwords
nltk.download('stopwords')
# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer


def to_lower(text):
    return text.lower()


def remove_punctuation(text):
    return re.sub(r'[^a-zA-Z\s]', '', text)


stop_words = set(stopwords.words("english"))

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]


def preprocess_text(text):
    text = to_lower(text)  # Step 1: Lowercasing
    text = remove_punctuation(text)  # Step 2: Remove punctuation
    tokens = word_tokenize(text)  # Step 3: Tokenization
    tokens = remove_stopwords(tokens)  # Step 4: Stop word removal
    return " ".join(tokens)  # Convert tokens back into text for vectorization


df['processed_text']= df['text'].apply(preprocess_text)
df['processed_text'].head()


class_counts = [df.loc[df['label']== i].shape[0] for i in range(2)]

print("Number of genuine posts:" , class_counts[0])
print("Number of fake news posts:" ,class_counts[1])
print("Class imbalance ratio: ", max(class_counts)/min(class_counts))


from tensorflow.keras.metrics import F1Score

METRICS = ['accuracy','recall', 'precision', F1Score(name='f1_score', average='macro', threshold=0.5)]


# Assuming y_train contains the true labels (0 = genuine, 1 = fake)
num_fake = class_counts[1]  # Count of fake (1) samples
num_genuine = class_counts[0]  # Count of genuine (0) samples
num_samples = sum(class_counts)

majority_class_prediction = 0 if num_genuine > num_fake else 1
majority_class_accuracy = max(num_fake, num_genuine) / num_samples

print(f"Majority Class Baseline Accuracy: {majority_class_accuracy:.2%} (Predicting all '{majority_class_prediction}')")


vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=20000)


from sklearn.model_selection import train_test_split

# Training and Test split
X, X_test, y, y_test = train_test_split(df['processed_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])

# Vectorizing splits with TF-IDF
X_tfidf = vectorizer.fit_transform(X)
X_test_tfidf = vectorizer.transform(X_test) # Test set transformed seperately to avoid data leakage

# Training and Validation split
X_train_tfidf, X_val_tfidf, y_train, y_val = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train_tfidf.shape}, Validation set: {X_val_tfidf.shape}, Test set: {X_test_tfidf.shape}")


from sklearn.feature_selection import SelectKBest, chi2

# Define the number of features to keep (adjust based on dataset)
NUM_FEATURES = 10000  # Keeping top 10,000 most relevant features

# Applying Chi-Square test for feature selection
chi2_selector = SelectKBest(chi2, k=NUM_FEATURES)
X_train_tfidf_reduced = chi2_selector.fit_transform(X_train_tfidf, y_train)
X_val_tfidf_reduced = chi2_selector.transform(X_val_tfidf)
X_test_tfidf_reduced = chi2_selector.transform(X_test_tfidf)

# Print new dataset sizes
print(f"Reduced Training Set: {X_train_tfidf_reduced.shape}")
print(f"Reduced Validation Set: {X_val_tfidf_reduced.shape}")
print(f"Reduced Test Set: {X_test_tfidf_reduced.shape}")


# Get the chi-square scores for all features
chi2_scores = chi2_selector.scores_

# Indices of selected features
selected_indices = chi2_selector.get_support(indices=True)

# Extract the chi-square scores for selected features
selected_scores = chi2_scores[selected_indices]

# Retrieve feature names
selected_feature_names = np.array(vectorizer.get_feature_names_out())[selected_indices]

# Sort the features by chi-square score in descending order
sorted_indices = np.argsort(selected_scores)[::-1]  # Reverse
sorted_feature_names = selected_feature_names[sorted_indices]
sorted_scores = selected_scores[sorted_indices]


from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Get the top 50 words based on their Chi-Square scores
top_words = sorted_feature_names[:50]
top_scores = sorted_scores[:50]

# Create a dictionary of words and their corresponding Chi-Square scores
word_scores = {word: score for word, score in zip(top_words, top_scores)}

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color="white", colormap="viridis",
                      max_words=50).generate_from_frequencies(word_scores)

# Show the word cloud
plt.figure(figsize=(8, 4))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")  # Hide axes
plt.title("Word Cloud of Top 50 Features Based on Chi-Square Scores", pad=20)
plt.show()


from tensorflow.keras import models, layers


def get_slp_model():
    model = models.Sequential([
    layers.Dense(1, activation='sigmoid', input_shape=(NUM_FEATURES,)) # Single layer sigmoid activation
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics = METRICS)

    return model


from tensorflow.keras.callbacks import EarlyStopping

'''
Inputs:
  - model: Tensorflow model
  - epochs: Number of epoch to use for training (int)
  - batch_size: Batch size for training (int)
  - silent: silent or verbose output for training and evaluation (Default=False)
  - early_stop: Disable or enable early stopping (Default=True)

Returns:
  - history: Training history
  - results: Evaluation results
'''
def train_and_evaluate_model(model, epochs, batch_size, silent=False, validation=False, early_stop=True):

    # Sparse TF-IDF matrix to dense array
    X_train = X_train_tfidf_reduced.toarray()
    X_val = X_val_tfidf_reduced.toarray()
    X_test = X_test_tfidf_reduced.toarray()

    # Implement early stopping
    callbacks=[]
    if early_stop:
      callbacks.append(EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True))

    # Train model
    history = model.fit(X_train, y_train,
            epochs=epochs, batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=0 if silent else 1)

    # To use validation or test data for evaluation
    if validation:
      results = model.evaluate(X_val, y_val, verbose=0 if silent else 1)

    else:
      results = model.evaluate(X_test, y_test, verbose=0 if silent else 1)

    del X_train, X_val, X_test
    return history, results


# Prints the evaluation metrics provided by model.evalutate()
def print_results(results):
  print("Test Loss:", f"{results[0]:.2%}")
  print("Test Accuracy:", f"{results[1]:.2%}")
  print("Test Recall:", f"{results[2]:.2%}")
  print("Test Precision:", f"{results[3]:.2%}")
  print("Test F1 Score:", f"{results[4]:.2%}")


# Plots the training loss and accuracy of a model
def plot_training(history):
    # Extract loss and accuracy history
    history_dict = history.history

    # Plot training & validation loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history_dict['loss'], label='Training Loss')
    plt.plot(history_dict['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()

    # Plot training & validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history_dict['accuracy'], label='Training Accuracy')
    plt.plot(history_dict['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.legend()

    plt.show()


model = get_slp_model()
history_baseline, results_baseline = train_and_evaluate_model(model, 25, 128)
del model
print_results(results_baseline)
plot_training(history_baseline)


def get_wider_model(no_of_neurons):
    model = models.Sequential([
    layers.Dense(no_of_neurons, activation='relu', input_shape=(NUM_FEATURES,)),
    layers.Dense(1, activation='sigmoid')  # Binary classification
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=METRICS)

    return model


number_of_neurons = [16, 32, 64]
model_results = []
model_histories = []
for neurons in number_of_neurons:
    model = get_wider_model(neurons)
    history, results = train_and_evaluate_model(model, 25, 128)
    model_histories.append(history)
    model_results.append(results)
    del model


for neurons, results in zip(number_of_neurons, model_results):
    print(f"{neurons} Neuron model - Accuracy: {results[1]:.2%} | Recall: {results[2]:.2%} | Precision: {results[3]:.2%} | F1Score: {results[4]:.2%}")


# Compares the training history of 2 models
def compare_training_history(histories, model_names):
    # Check if number of models and model names match
    if len(histories) != len(model_names):
        raise ValueError("Number of models and model names should match.")

    # Extract history dictionaries
    model_histories = [history.history for history in histories]

    # Calculate number of plots (2 plots per model, loss & validation)
    num_models = len(model_histories)
    num_plots = num_models * 2

    # Figure size
    plt.figure(figsize=(14, 10))

    for plot_no in range(1, num_plots+1):
        plt.subplot(num_models, 2, plot_no)
        if plot_no % 2 == 1:
            # Change model history and name every 2 plots (change of model)
            history_dict = model_histories[(plot_no-1) // 2]
            model_name = model_names[(plot_no-1) // 2]

            # Draw loss plot
            plt.plot(history_dict['loss'], label=f'Training Loss ({model_name})', linestyle='--')
            plt.plot(history_dict['val_loss'], label=f'Validation Loss ({model_name})')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title(f'Training & Validation Loss ({model_name})')
            plt.legend()
        else:
            # Draw accuracy plot
            plt.plot(history_dict['accuracy'], label=f'Training Accuracy ({model_name})', linestyle='--')
            plt.plot(history_dict['val_accuracy'], label=f'Validation Accuracy ({model_name})')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.title(f'Training & Validation Accuracy ({model_name})')
            plt.legend()

    plt.tight_layout()
    plt.show()


model_names = [f"{neurons}-Neuron Model" for neurons in number_of_neurons]
compare_training_history(model_histories, model_names)


hyperparameters = {
    "neurons": [16],
    "num_layers" : [1],
    "dropout": [0.0],  # Dropout rate
    "learning_rate":[1e-5, 3e-5, 1e-4, 3e-4], # Learning rate
    "alpha" : [1e-5, 1e-4, 1e-3], # Regularization factor
    "reg_type": ["l1", "l2", "l1+l2"],  # Type of regularization
}


from tensorflow import keras
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras import backend as K


def build_model(num_layers, neurons, dropout, learning_rate, reg_type, alpha):
    model = models.Sequential()

    # Add hidden layers based on number of layers
    for _ in range(num_layers):
        model.add(layers.Dense(neurons, activation='relu'))
        model.add(layers.Dropout(dropout))

    # Set the input shape of the first layer
    model.layers[0].input_shape = NUM_FEATURES

    # Add output layer
    model.add(layers.Dense(1, activation='sigmoid'))

    # Regularization selector
    if reg_type == 'l1':
        regularizer = l1(alpha)
    elif reg_type == 'l2':
        regularizer = l2(alpha)
    else:
        regularizer = l1_l2(l1=alpha, l2=alpha)

    # Apply regularization to all layers
    for layer in model.layers[:-1]:  # Except output layer
        if isinstance(layer, layers.Dense):
            layer.kernel_regularizer = regularizer

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss='binary_crossentropy',
                  metrics=METRICS)
    return model


import os
import pickle
import itertools
import gc

'''
Iterates through each hyperparameter combination, storing them in a dictionary.
Keys are the hyperparameter combination
Values are the results of the model evaluation

Returns:
  - results: Dictionary of hyperparameter combinations and their results

This function attempts to be conservative in its memory management by deleteing each model after evaluation and using gc.collect()
'''
def tune_hyperparameters(hyperparameters, epochs=8,batch_size = 128, results_file="hyperparam_results.pkl"):
    # Load existing results if previous results file is found
    if os.path.exists(results_file):
        with open(results_file, "rb") as f:
            results = pickle.load(f)
    else:
        results = {}

    # Converting to dense array
    X_train = X_train_tfidf_reduced.toarray()
    X_val = X_val_tfidf_reduced.toarray()
    X_test = X_test_tfidf_reduced.toarray()

    # Generate all hyperparameter combinations
    keys, values = zip(*hyperparameters.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for params in param_combinations:
        param_key = tuple(sorted(params.items()))  # Convert dict to a unique hashable key

        # Skip if this combination has been tested before
        if param_key in results:
            print(f"Skipping {param_key}, already tested.")
            continue

        # Print hyperparameter combination
        for i in range(len(param_key)-1):
            print(f"{param_key[i][0]}: {param_key[i][1]}", end=' | ')
        print(f"{param_key[-1][0]}: {param_key[-1][1]}")

        check_memory()
        # Free up memory from previous models
        K.clear_session()
        gc.collect()

        # Create and train the model
        model = build_model(**params)
        print("Model built")

        history = model.fit(X_train, y_train,
                epochs=epochs, batch_size=batch_size,
                validation_data=(X_val, y_val),
                verbose=0)

        print("Results:")
        eval_results = model.evaluate(X_val, y_val, verbose=1)

        # Store results (loss + success metrics)
        results[param_key] = {
            "loss": eval_results[0],
            "accuracy": eval_results[1],
            "recall": eval_results[2],
            "precision": eval_results[3],
            "f1_score": eval_results[4]
        }

        # Delete model to free memory
        del model
        gc.collect()

        # Save results after each iteration to prevent data loss
        with open(results_file, "wb") as f:
            pickle.dump(results, f)

    del X_train, X_val, X_test

    return results


import pickle
with open("hyperparam_results.pkl", "rb") as f:
            results = pickle.load(f)


results = tune_hyperparameters(hyperparameters)


# Sort all hyperparameter combinations by accuracy in descending order
top_5_models = sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True)[:5]

# Print the top 5 hyperparameter configurations and their metrics
for rank, (best_params, best_metrics) in enumerate(top_5_models, start=1):
    print(f"Rank {rank}: Hyperparameters: {dict(best_params)}")
    print(f"Accuracy: {best_metrics['accuracy']:.4f}, "
          f"Loss: {best_metrics['loss']:.4f}, "
          f"Recall: {best_metrics['recall']:.4f}, "
          f"Precision: {best_metrics['precision']:.4f}, "
          f"F1 Score: {best_metrics['f1_score']:.4f}\n")


best_params = dict(top_5_models[0][0]) # get the parameters for the best model (rank 1)


model = build_model(**best_params) # Rebuild the model
history, results = train_and_evaluate_model(model, 50, 128)
del model
print_results(results)
plot_training(history)


print(best_params)


model = build_model(
    best_params['num_layers'],
    32, # Increased number of neurons
    best_params['dropout'],
    best_params['learning_rate'],
    best_params['reg_type'],
    best_params['alpha']
    )


history, results = train_and_evaluate_model(model, 50, 128)
del model
print_results(results)
plot_training(history)


hyperparameters_wider = {
    "neurons": [32, 64],
    "num_layers" : [1],
    "dropout": [0.0, 0.1],
    "alpha" : [1e-5, 1e-4, 1e-3], # Regularization factor
    "learning_rate": [1e-5, 3e-5, 1e-4, 3e-4]
}
hyperparameters_wider['reg_type'] = [best_params['reg_type']] # Use L2 regularization


results_wider = tune_hyperparameters(hyperparameters_wider, epochs=8,batch_size=128, results_file="hyperparam_results_wider.pkl")


with open("hyperparam_results_wider.pkl", "rb") as f:
            results_wider = pickle.load(f)


best_model_wider = sorted(results_wider.items(), key=lambda x: x[1]["accuracy"], reverse=True)[0]
best_params_wider = dict(best_model_wider[0])
best_results_wider = best_model_wider[1]
print(f"Best Hyperparameters: {best_params_wider}")
print(f"Accuracy: {best_results_wider['accuracy']:.4f}, "
      f"Loss: {best_results_wider['loss']:.4f}, "
      f"Recall: {best_results_wider['recall']:.4f}, "
      f"Precision: {best_results_wider['precision']:.4f}, "
      f"F1 Score: {best_results_wider['f1_score']:.4f}\n")


model = build_model(**best_params_wider)
history_wider, results_wider = train_and_evaluate_model(model, 50, 128)
del model
print_results(results_wider)
plot_training(history_wider)


print(best_params)


model = build_model(
    2, # Increased number of layers
    best_params['neurons'],
    best_params['dropout'],
    best_params['learning_rate'],
    best_params['reg_type'],
    best_params['alpha']
    )


history, results = train_and_evaluate_model(model, 50, 128)
del model
print_results(results)
plot_training(history)


hyperparameters_deeper = {
    "neurons": [16, 32],
    "num_layers" : [2, 3], # Exclude 1 layer as its was already explored in the previous section
    "dropout": [0.0, 0.1, 0.2, 0.3],
    "learning_rate": [1e-05, 3e-05, 1e-04, 3e-04],
    "alpha" : [1e-5, 1e-4, 1e-3], # Regularization factor
}
hyperparameters_deeper['reg_type'] = [best_params['reg_type']]


results_deeper = tune_hyperparameters(hyperparameters_deeper, epochs=8,batch_size=128, results_file="hyperparam_results_deeper.pkl")


import pickle
with open("hyperparam_results_deeper.pkl", "rb") as f:
    results_deeper = pickle.load(f)


best_model_deeper = sorted(results_deeper.items(), key=lambda x: x[1]["accuracy"], reverse=True)[0]
best_params_deeper = dict(best_model_deeper[0])
best_results_deeper = best_model_deeper[1]
print(f"Best Hyperparameters: {best_params_deeper}")
print(f"Accuracy: {best_results_deeper['accuracy']:.4f}, "
      f"Loss: {best_results_deeper['loss']:.4f}, "
      f"Recall: {best_results_deeper['recall']:.4f}, "
      f"Precision: {best_results_deeper['precision']:.4f}, "
      f"F1 Score: {best_results_deeper['f1_score']:.4f}\n")


model = build_model(**best_params_deeper)
history_deeper, results_deeper = train_and_evaluate_model(model, 50, 128)
del model
print_results(results_deeper)
plot_training(history_deeper)


!wget http://nlp.stanford.edu/data/glove.twitter.27B.zip
!unzip glove.twitter.27B.zip


from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.callbacks import EarlyStopping


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define hyperparameters
MAX_VOCAB_SIZE = 20000  # Number of words to keep (Same as TFIDF, before feature selection)
MAX_SEQUENCE_LENGTH = 100  # Max words per sample

# Tokenizer initialization
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="?") # Words out of vocab are replaced with ?
tokenizer.fit_on_texts(df["processed_text"])

# Convert text to sequences
X_sequences = tokenizer.texts_to_sequences(df["processed_text"])
print(f"Example of sequence: {X_sequences[0]}")

# Pad sequences
X_padded = pad_sequences(X_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")

print(f"Shape of tokenized data: {X_padded.shape}")  # (num_samples, max_length)


# Load GloVe embeddings
embedding_dim = 100  # Use 100D GloVe embeddings
glove_path = "./glove.twitter.27B.100d.txt"  # Update path if using Twitter embeddings

embeddings_index = {}
with open(glove_path, "r", encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]  # Word
        coeffs = np.asarray(values[1:], dtype="float32")  # Vector
        embeddings_index[word] = coeffs

print(f"Loaded {len(embeddings_index)} word vectors.")

# Create embedding matrix for our vocabulary
embedding_matrix = np.zeros((MAX_VOCAB_SIZE, embedding_dim))

# Mapping glove word vectors to our vocabulary
for word, i in tokenizer.word_index.items():
    if i < MAX_VOCAB_SIZE:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector  # Assign GloVe vector

print(f"Shape of embedding matrix: {embedding_matrix.shape}")


# Step 1: Split into Training & Test Set
X_train, X_test, y_train, y_test = train_test_split(X_padded, df["label"], test_size=0.2, random_state=42, stratify=df["label"])

# Step 2: Further Split Training into Train & Validation Set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")


def build_glove_model(num_layers, neurons, dropout, learning_rate, reg_type, alpha):
    model = models.Sequential()

    embedding_layer = tf.keras.layers.Embedding(
        input_dim=MAX_VOCAB_SIZE,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        trainable=False
    )
    model.add(embedding_layer)

    # Reduce dimensionality to 1D
    model.add(layers.GlobalAveragePooling1D())

    # Add hidden layers based on number of layers
    for _ in range(num_layers):
        model.add(layers.Dense(neurons, activation='relu'))
        model.add(layers.Dropout(dropout))

    # Add output layer
    model.add(layers.Dense(1, activation='sigmoid'))

    # Regularization selector
    if reg_type == 'l1':
        regularizer = l1(alpha)
    elif reg_type == 'l2':
        regularizer = l2(alpha)
    else:
        regularizer = l1_l2(l1=alpha, l2=alpha)

    # Apply regularization to all layers
    for layer in model.layers[1:-1]:  # Except embedding and output layer
        if isinstance(layer, layers.Dense):
            layer.kernel_regularizer = regularizer

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss='binary_crossentropy',
                  metrics=METRICS)
    return model


# Create model
glove_model = build_glove_model(**best_params_deeper)
glove_model.summary()


history_glove = glove_model.fit(X_train, y_train,
        epochs= 100,
        batch_size=128,
        validation_data=(X_val, y_val),
        callbacks=EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True),
        verbose=1)
results_glove = glove_model.evaluate(X_test, y_test, verbose=1)


print_results(results_glove)


import matplotlib.pyplot as plt


plot_training(history_glove)


data = [
    ["Baseline Model"] + results_baseline,
    ["Tuned Model"] + results,
    ["Widened Model"] + results_wider,
    ["Deeper Model"] + results_deeper,
    ["GloVe Model"] + results_glove
]

df = pd.DataFrame(data, columns=["Model"] + ['loss', 'accuracy', 'recall', 'precision', 'f1_score'])
df.set_index("Model", inplace=True)
df = pd.DataFrame.map(df, lambda x: f"{x*100:.2f}%")

df




