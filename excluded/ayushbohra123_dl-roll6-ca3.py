!pip install transformers==4.41.2 --quiet


!nvidia-smi


import os
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging (0=all, 1=info, 2=warning, 3=error)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

# Suppress other warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers
from tensorflow.keras.optimizers import Adam
import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split

# Optionally, set TensorFlow logging level
tf.get_logger().setLevel('ERROR')

# Verify GPU is available (optional)
print("GPU Available:", tf.config.list_physical_devices('GPU'))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tf.config.set_visible_devices([], 'GPU')

def create_sentiment_model(input_dim=768):
    """
    Create a neural network for binary sentiment classification
    Input: BERT embeddings (768 dimensions)
    Output: Sigmoid probability (0=negative, 1=positive)
    """
    model = Sequential([
        layers.Input(shape=(input_dim,)),
        
        # First hidden layer with dropout
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        # Second hidden layer with dropout
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Third hidden layer
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Third hidden layer
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        
        # Output layer with sigmoid activation
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model

# Create and compile model
model_nn = create_sentiment_model(input_dim=768)

model_nn.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)

# Model summary
model_nn.summary()


import warnings
warnings.filterwarnings('ignore')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model_bert = BertModel.from_pretrained('bert-base-uncased')
model_bert.eval()
model_bert.to(device)


print(tokenizer)
print(model_bert)


df = pd.read_csv("/kaggle/input/kcvanguard-deep-learning-assignment/train-reviews-gmaps.csv")
df['target'] = pd.CategoricalIndex(df['label']).codes
df.head(), df.describe(), df.dtypes


test = tokenizer(df['reviews'].iloc[10], return_tensors='pt').to(device)
model_bert(**test)[0][:,0,:].shape


def get_bert_embeddings(texts, batch_size=32):
    """Extract BERT embeddings for a list of texts"""
    embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            tokens = tokenizer(batch_texts, 
                             padding=True, 
                             truncation=True, 
                             max_length=128,
                             return_tensors='pt').to(device)
            
            outputs = model_bert(**tokens)
            # Use [CLS] token embedding (first token) for classification
            cls_embeddings = outputs[0][:, 0, :].cpu().numpy()
            embeddings.append(cls_embeddings)
    
    return np.vstack(embeddings)
print("Extracting BERT embeddings...")
X = get_bert_embeddings(df['reviews'].tolist())
y = df['target'].values

print(f"Embeddings shape: {X.shape}")
print(f"Labels shape: {y.shape}")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))


history = model_nn.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=16,
    verbose=1,
    class_weight=class_weight_dict
)


X_test = get_bert_embeddings(pd.read_csv('/kaggle/input/kcvanguard-deep-learning-assignment/test-review-gmaps-new.csv')['reviews'].tolist())
# y_pred = df['target'].values

print(f"Embeddings shape: {X_test.shape}")
# print(f"Labels shape: {y.shape}")


output = model_nn.predict(X_test)


output = pd.DataFrame(output)
output



df_reset = df_reset.rename(columns={'index': 'id', 0:'label'})


df_reset


df = df_reset


# Convert values above 0.5 to 'Positive' and below/equal to 0.5 to 'Negative'
df['label'] = df['label'].apply(lambda x: 'Positive' if x > 0.5 else 'Negative')

# Save to CSV file
df.to_csv('submission.csv', index=False)

# Display the result
print(df.head(10))
print(f"\nTotal rows: {len(df)}")
print(f"\nValue counts:\n{df['label'].value_counts()}")




