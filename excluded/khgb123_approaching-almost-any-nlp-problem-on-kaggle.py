import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, GRU, Dense, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from nltk.stem import PorterStemmer  # Doesn't require WordNet

# Custom stopwords list (common English stopwords)
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 
    'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
    'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
    'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
    'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've",
    'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn',
    "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
    "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
    "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn',
    "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}
CONTRACTION_MAP = {
    "don't": "do not",
    "can't": "cannot",
    "won't": "will not",
    "it's": "it is",
    "i'm": "i am",
    "you're": "you are",
    "they're": "they are",
    "that's": "that is",
    "he's": "he is",
    "she's": "she is",
    "we're": "we are",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "aren't": "are not",
    "isn't": "is not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "doesn't": "does not",
    "didn't": "did not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "let's": "let us",
    "i'll": "i will",
    "you'll": "you will",
    "they'll": "they will",
    "it'll": "it will"
}

def expand_contractions(text):
    for contraction, expansion in CONTRACTION_MAP.items():
        text = text.replace(contraction, expansion)
    return text

# Initialize stemmer
stemmer = PorterStemmer()
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-5
)
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    min_delta=0.001
)

# Load data
train_df = pd.read_csv('/kaggle/input/spooky/train.csv')
test_df = pd.read_csv('/kaggle/input/spooky/test.csv')

# Preprocess text
def preprocess_text(text):
    # Fix contractions using local rules
    text = expand_contractions(text)
    
    # Clean text
    text = text.lower()
    text = ''.join([c if c.isalnum() or c == ' ' else ' ' for c in text])
    text = ' '.join(text.split())
    
    # Tokenize and stem
    tokens = text.split()
    tokens = [stemmer.stem(token) for token in tokens]
    
    # Remove stopwords using local list
    tokens = [token for token in tokens if token not in STOPWORDS]
    
    return ' '.join(tokens)

train_df['text'] = train_df['text'].apply(preprocess_text)
test_df['text'] = test_df['text'].apply(preprocess_text)

# Determine max sequence length
train_lengths = train_df['text'].apply(lambda x: len(x.split()))
maxlen = int(np.percentile(train_lengths, 95))

# Tokenization for neural models
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_df['text'])
vocab_size = len(tokenizer.word_index) + 1

X_train = tokenizer.texts_to_sequences(train_df['text'])
X_train = pad_sequences(X_train, maxlen=maxlen, padding='post')

X_test = tokenizer.texts_to_sequences(test_df['text'])
X_test = pad_sequences(X_test, maxlen=maxlen, padding='post')

# Encode labels
le = LabelEncoder()
y = le.fit_transform(train_df['author'])
y = to_categorical(y)

# Split into training and validation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y, test_size=0.2, random_state=42)

# GeMPooling layer
class GeMPooling(tf.keras.layers.Layer):
    def __init__(self, p=3., train_p=False, **kwargs):
        super(GeMPooling, self).__init__(**kwargs)
        self.p = p
        self.train_p = train_p
        if self.train_p:
            self.p = tf.Variable(initial_value=p, trainable=True, dtype=tf.float32)
        else:
            self.p = tf.constant(p, dtype=tf.float32)

    def call(self, inputs):
        safe_p = tf.maximum(self.p, 1e-6)
        x = tf.abs(inputs)
        x = tf.pow(x, safe_p)
        x = tf.reduce_mean(x, axis=1)
        x = tf.pow(x, 1.0/safe_p)
        return x

# Define model building functions
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2

# Attention model components
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1), initializer='glorot_normal', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        score = tf.matmul(inputs, self.W)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = tf.reduce_sum(attention_weights * inputs, axis=1)
        return context_vector

def build_lstm_model(vocab_size, maxlen):
    inputs = Input(shape=(maxlen,))
    x = Embedding(vocab_size, 100)(inputs)
    x = LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)(x)
    x = Dropout(0.4)(x)
    x = GeMPooling(p=3)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.3)(x)
    outputs = Dense(3, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_gru_model(vocab_size, maxlen):
    inputs = Input(shape=(maxlen,))
    x = Embedding(vocab_size, 100)(inputs)
    x = GRU(128, return_sequences=True)(x)
    x = GeMPooling(p=3)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(3, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_deep_lstm_model(vocab_size, maxlen):
    inputs = Input(shape=(maxlen,))
    x = Embedding(vocab_size, 100)(inputs)
    x = LSTM(128, return_sequences=True)(x)
    x = LSTM(64, return_sequences=True)(x)
    x = GeMPooling(p=3)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(3, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_deep_gru_model(vocab_size, maxlen):
    inputs = Input(shape=(maxlen,))
    x = Embedding(vocab_size, 100)(inputs)
    x = GRU(128, return_sequences=True)(x)
    x = GRU(64, return_sequences=True)(x)
    x = GeMPooling(p=3)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(3, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Attention model components
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1), initializer='glorot_normal', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        score = tf.matmul(inputs, self.W)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = tf.reduce_sum(attention_weights * inputs, axis=1)
        return context_vector

def build_attention_model(vocab_size, maxlen):
    inputs = Input(shape=(maxlen,))
    x = Embedding(vocab_size, 100)(inputs)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = AttentionLayer()(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(3, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Add to imports
from tensorflow.keras.layers import Bidirectional
from sklearn.naive_bayes import MultinomialNB

# ======================
# BiLSTM Model
# ======================
def build_bilstm_model(vocab_size, maxlen):
    inputs = Input(shape=(maxlen,))
    x = Embedding(vocab_size, 100)(inputs)
    x = Bidirectional(LSTM(64, return_sequences=True, 
                         dropout=0.4, recurrent_dropout=0.3))(x)
    x = Dropout(0.4)(x)
    x = GeMPooling(p=3)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.3)(x)
    outputs = Dense(3, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    return model

# Update models list


# ======================
# Naive Bayes + CountVectorizer
# ======================

# Train each neural model
models = [
    ('lstm', build_lstm_model),
    ('gru', build_gru_model),
    ('deep_lstm', build_deep_lstm_model),
    ('deep_gru', build_deep_gru_model),
    ('attention', build_attention_model),
    ('bilstm', build_bilstm_model)
]
for model_name, builder in models:
    print(f'Training {model_name}...')
    model = builder(vocab_size, maxlen)
    
    # Train model and capture history
    history = model.fit(
        X_train_split, y_train_split,
        validation_data=(X_val_split, y_val_split),
        epochs=20,  # Increase max epochs
        batch_size=64,
        callbacks=[early_stopping, lr_scheduler],
        verbose=1
    )
    # Plot training history
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name.upper()} Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'loss_{model_name}.png', bbox_inches='tight')
    plt.close()
    
    # Generate predictions
    y_pred = model.predict(X_test)
    submission = pd.DataFrame({
        'id': test_df['id'],
        'EAP': y_pred[:, 0],
        'HPL': y_pred[:, 1],
        'MWS': y_pred[:, 2]
    })
    submission.to_csv(f'submission_{model_name}.csv', index=False)

# Traditional ML model
X_ml = train_df['text']
y_ml = train_df['author']

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000))
])

parameters = {
    'vect__max_features': [1000, 5000],
    'vect__ngram_range': [(1, 1), (1, 2)],
    'clf__C': [0.1, 1, 10]
}

grid_search = GridSearchCV(pipeline, parameters, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_ml, y_ml)

test_probs = grid_search.predict_proba(test_df['text'])
submission_ml = pd.DataFrame({
    'id': test_df['id'],
    'EAP': test_probs[:, 0],
    'HPL': test_probs[:, 1],
    'MWS': test_probs[:, 2]
})
submission_ml.to_csv('submission_ml.csv', index=False)

def train_naive_bayes():
    # Use same preprocessing as neural models
    X_ml = train_df['text']
    y_ml = train_df['author']
    
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('clf', MultinomialNB())
    ])
    
    parameters = {
        'vect__max_features': [2000, 5000],
        'vect__ngram_range': [(1, 1), (1, 2)],
        'clf__alpha': [0.1, 0.5, 1.0]
    }
    
    grid_search = GridSearchCV(pipeline, parameters, 
                             cv=5, scoring='accuracy',
                             verbose=1)
    grid_search.fit(X_ml, y_ml)
    
    print(f"Best Naive Bayes params: {grid_search.best_params_}")
    print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")
    
    # Generate predictions
    test_probs = grid_search.predict_proba(test_df['text'])
    submission = pd.DataFrame({
        'id': test_df['id'],
        'EAP': test_probs[:, 0],
        'HPL': test_probs[:, 1],
        'MWS': test_probs[:, 2]
    })
    submission.to_csv('submission_naive_bayes.csv', index=False)
print('training naive bayes')
train_naive_bayes()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# 1. Define vectorizers with different configurations
vectorizers = {
    'v1': CountVectorizer(
        ngram_range=(1,1), 
        min_df=3, 
        max_df=0.85, 
        stop_words='english'
    ),
    'v2': CountVectorizer(
        ngram_range=(1,2),
        min_df=2,
        max_df=0.9
    ),
    'v3': CountVectorizer(
        ngram_range=(1,3),
        min_df=3,
        max_df=0.8
    ),
    'v4': CountVectorizer(
        ngram_range=(2,2),
        min_df=2,
        max_df=0.9
    )
}

# 2. Model configurations with corrected parameters
configurations = [
    ('LR3_v3', LogisticRegression(
        C=1.0, 
        solver='liblinear',
        class_weight='balanced',
        penalty='l2',
        max_iter=500
    ), 'v3'),
    
    ('LR4_v3', LogisticRegression(
        C=2.0,
        solver='liblinear',
        class_weight='balanced',
        penalty='l2',
        max_iter=500
    ), 'v3'),
    
    ('LR5_v3', LogisticRegression(
        C=1.0,
        penalty='l1',
        solver='liblinear',
        class_weight='balanced',
        max_iter=500
    ), 'v3'),
    
    ('LR10_v3', LogisticRegression(
        C=5.0,
        solver='liblinear',
        class_weight='balanced',
        penalty='l2',
        max_iter=500
    ), 'v3'),
    
    ('NB1_v1', MultinomialNB(alpha=0.1), 'v1'),
    ('NB5_v2', MultinomialNB(alpha=5.0), 'v2')
]

# 3. Training and prediction function
def train_ensemble():
    results = []
    
    for model_name, clf, vec_name in configurations:
        print(f"\nTraining {model_name} with {vec_name}...")
        
        # Create pipeline
        pipeline = Pipeline([
            ('vectorizer', vectorizers[vec_name]),
            ('classifier', clf)
        ])
        
        # Train
        pipeline.fit(train_df['text'], train_df['author'])
        
        # Predict probabilities
        test_probs = pipeline.predict_proba(test_df['text'])
        
        # Save submission
        submission = pd.DataFrame({
            'id': test_df['id'],
            'EAP': test_probs[:, 0],
            'HPL': test_probs[:, 1],
            'MWS': test_probs[:, 2]
        })
        submission.to_csv(f'submission_{model_name}.csv', index=False)
        
        # Store results
        results.append({
            'model': model_name,
            'vectorizer': vec_name,
            'pipeline': pipeline
        })
    
    return results

# Execute training
print("Training ensemble models...")
model_results = train_ensemble()


df = pd.read_csv("/kaggle/input/spooky/train.csv")
df.shape


# display_images.py
"""
Simple script to display all images in /kaggle/working.
Run this in a Jupyter notebook or Python environment.
"""
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt

# Directory containing images
dir_path = '/kaggle/working'
# Supported image extensions
extensions = ['png', 'jpg', 'jpeg', 'gif', 'bmp']

# Gather all image file paths
image_paths = []
for ext in extensions:
    pattern = os.path.join(dir_path, f'*.{ext}')
    image_paths.extend(glob.glob(pattern))

# Display each image
for img_path in image_paths:
    try:
        img = Image.open(img_path)
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title(os.path.basename(img_path))
        plt.axis('off')
    except Exception as e:
        print(f"Could not open {img_path}: {e}")

plt.show()



!cp /kaggle/input/spooky/sample_submission.csv /kaggle/working/sample.csv




