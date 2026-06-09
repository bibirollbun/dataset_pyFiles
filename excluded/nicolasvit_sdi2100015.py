
# Import necessary libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import random
from tqdm import tqdm

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# Load datasets
train_df = pd.read_csv('/kaggle/input/ai-2-dl-for-nlp-2025-homework-2/train_dataset.csv')
val_df = pd.read_csv('/kaggle/input/ai-2-dl-for-nlp-2025-homework-2/val_dataset.csv')
test_df = pd.read_csv('/kaggle/input/ai-2-dl-for-nlp-2025-homework-2/test_dataset.csv')

# Display information about the datasets
print(f"Training set size: {train_df.shape[0]} samples")
print(f"Validation set size: {val_df.shape[0]} samples")
print(f"Test set size: {test_df.shape[0]} samples")

# Check for class distribution in the training set
train_class_dist = train_df['Label'].value_counts(normalize=True)
print("\nClass distribution in training set:")
print(train_class_dist)


# Visualize class distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='Label', data=train_df)
plt.title('Class Distribution in Training Data')
plt.xlabel('Sentiment (0: Negative, 1: Positive)')
plt.ylabel('Count')
plt.show()

# Display a few examples from each class
print("\nNegative sentiment examples:")
print(train_df[train_df['Label'] == 0]['Text'].head(3).values)

print("\nPositive sentiment examples:")
print(train_df[train_df['Label'] == 1]['Text'].head(3).values)


# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Function to clean and preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Apply preprocessing to each dataset
train_df['Processed_Text'] = train_df['Text'].apply(preprocess_text)
val_df['Processed_Text'] = val_df['Text'].apply(preprocess_text)
test_df['Processed_Text'] = test_df['Text'].apply(preprocess_text)

# Function to tokenize text
def tokenize(text):
    return word_tokenize(text)

# Apply tokenization
train_df['Tokens'] = train_df['Processed_Text'].apply(tokenize)
val_df['Tokens'] = val_df['Processed_Text'].apply(tokenize)
test_df['Tokens'] = test_df['Processed_Text'].apply(tokenize)


# Display sample preprocessed and tokenized data
print("Sample preprocessed and tokenized data:")
sample_idx = 10  # Choose a sample index
print("Original text:", train_df.loc[sample_idx, 'Text'])
print("Processed text:", train_df.loc[sample_idx, 'Processed_Text'])
print("Tokenized text:", train_df.loc[sample_idx, 'Tokens'])


# Prepare sentences for Word2Vec training
all_sentences = list(train_df['Tokens']) + list(val_df['Tokens'])

# Define Word2Vec parameters
vector_size = 300  # Dimensionality of the word vectors
window = 5  # Maximum distance between the current and predicted word
min_count = 2  # Ignore words that appear less than this
workers = 4  # Number of threads to run in parallel
sg = 1  # Training algorithm: 1 for skip-gram; 0 for CBOW

# Train Word2Vec model
print("Training Word2Vec model...")
w2v_model = Word2Vec(sentences=all_sentences, 
                     vector_size=vector_size, 
                     window=window, 
                     min_count=min_count,
                     workers=workers,
                     sg=sg)

print(f"Word2Vec model trained on {len(w2v_model.wv.key_to_index)} words")


# Function to get average embedding for a text
def get_avg_embedding(tokens, model, vector_size):
    embeddings = [model.wv[word] for word in tokens if word in model.wv.key_to_index]
    if len(embeddings) == 0:
        return np.zeros(vector_size)
    return np.mean(embeddings, axis=0)

# Generate embeddings for each dataset
print("Generating embeddings for datasets...")
train_df['Embedding'] = train_df['Tokens'].apply(lambda x: get_avg_embedding(x, w2v_model, vector_size))
val_df['Embedding'] = val_df['Tokens'].apply(lambda x: get_avg_embedding(x, w2v_model, vector_size))
test_df['Embedding'] = test_df['Tokens'].apply(lambda x: get_avg_embedding(x, w2v_model, vector_size))


# Create a custom dataset class for PyTorch
class SentimentDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        embedding = torch.tensor(self.embeddings[idx], dtype=torch.float)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return embedding, label

# Convert lists to numpy arrays for faster processing
train_embeddings = np.stack(train_df['Embedding'].values)
train_labels = train_df['Label'].values

val_embeddings = np.stack(val_df['Embedding'].values)
val_labels = val_df['Label'].values

test_embeddings = np.stack(test_df['Embedding'].values)

# Create datasets
train_dataset = SentimentDataset(train_embeddings, train_labels)
val_dataset = SentimentDataset(val_embeddings, val_labels)

# Create dataloaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Define the neural network architecture
class SentimentClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.5):
        super(SentimentClassifier, self).__init__()
        
        # Define architecture with multiple hidden layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_prob)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout_prob)
        
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x

# Set hyperparameters
input_size = vector_size  # Size of word embeddings
hidden_size = 128  # Size of hidden layer
output_size = 2  # Binary classification (0 or 1)
learning_rate = 0.0001
num_epochs = 20
dropout_prob = 0.5

# Initialize the model
model = SentimentClassifier(input_size, hidden_size, output_size, dropout_prob)

# Check if GPU is available and move model to GPU if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # L2 regularization

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)


# Define training and evaluation functions
def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    for embeddings, labels in tqdm(data_loader, desc="Training"):
        # Move data to device
        embeddings = embeddings.to(device)
        labels = labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(embeddings)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * embeddings.size(0)
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)
    
    epoch_loss = running_loss / total_predictions
    epoch_acc = correct_predictions / total_predictions
    
    return epoch_loss, epoch_acc

def evaluate(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for embeddings, labels in tqdm(data_loader, desc="Evaluating"):
            # Move data to device
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * embeddings.size(0)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    eval_loss = running_loss / len(data_loader.dataset)
    acc = accuracy_score(all_labels, all_predictions)
    prec = precision_score(all_labels, all_predictions)
    rec = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    return eval_loss, acc, prec, rec, f1, conf_matrix, all_predictions


# Train the model
train_losses = []
train_accs = []
val_losses = []
val_accs = []

best_val_acc = 0.0
best_model_state = None

print(f"Training on device: {device}")

for epoch in range(num_epochs):
    # Train one epoch
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    
    # Evaluate on validation set
    val_loss, val_acc, val_prec, val_rec, val_f1, val_conf_matrix, _ = evaluate(model, val_loader, criterion, device)
    
    # Print progress
    print(f"Epoch {epoch+1}/{num_epochs}:")
    print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
    print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, Val Precision: {val_prec:.4f}")
    print(f"  Val Recall: {val_rec:.4f}, Val F1 Score: {val_f1:.4f}")
    
    # Update learning rate scheduler
    scheduler.step(val_acc)
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict().copy()
        print(f"  New best model saved with validation accuracy: {best_val_acc:.4f}")
    
    # Record losses and accuracies
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

# Load the best model
model.load_state_dict(best_model_state)


# Plot training and validation metrics
def plot_metrics(train_values, val_values, y_label, title):
    epochs = range(1, len(train_values) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_values, 'bo-', label='Training')
    plt.plot(epochs, val_values, 'ro-', label='Validation')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot loss and accuracy
plot_metrics(train_losses, val_losses, 'Loss', 'Training and Validation Loss')
plot_metrics(train_accs, val_accs, 'Accuracy', 'Training and Validation Accuracy')

# Plot confusion matrix
_, _, _, _, _, val_conf_matrix, _ = evaluate(model, val_loader, criterion, device)

plt.figure(figsize=(8, 6))
sns.heatmap(val_conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


# We'll define a simple grid search function for hyperparameter tuning
def hyperparameter_tuning(hidden_sizes, learning_rates, dropout_probs, epochs=10):
    results = []
    
    for hidden_size in hidden_sizes:
        for lr in learning_rates:
            for dropout in dropout_probs:
                print(f"\nTraining model with: hidden_size={hidden_size}, lr={lr}, dropout={dropout}")
                
                # Initialize model with current hyperparameters
                model = SentimentClassifier(input_size, hidden_size, output_size, dropout)
                model.to(device)
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
                
                # Train for specified number of epochs
                for epoch in range(epochs):
                    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
                    
                    if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                        val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(model, val_loader, criterion, device)
                        print(f"Epoch {epoch+1}/{epochs}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
                
                # Final evaluation on validation set
                val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(model, val_loader, criterion, device)
                
                # Record results
                results.append({
                    'hidden_size': hidden_size,
                    'learning_rate': lr,
                    'dropout': dropout,
                    'val_accuracy': val_acc,
                    'val_precision': val_prec,
                    'val_recall': val_rec,
                    'val_f1': val_f1
                })
                
    # Convert results to DataFrame for easy analysis
    results_df = pd.DataFrame(results)
    return results_df

# Define hyperparameter grid
hidden_sizes = [64, 128, 256]
learning_rates = [0.01, 0.001, 0.0001]
dropout_probs = [0.3, 0.5]

# Comment out this section if you want to skip hyperparameter tuning
# This can be time-consuming depending on the dataset size
'''
tuning_results = hyperparameter_tuning(hidden_sizes, learning_rates, dropout_probs, epochs=5)

# Display results sorted by validation accuracy
print("\nHyperparameter tuning results (sorted by validation accuracy):")
print(tuning_results.sort_values(by='val_accuracy', ascending=False))

# Get best hyperparameters
best_row = tuning_results.loc[tuning_results['val_accuracy'].idxmax()]
best_hidden_size = int(best_row['hidden_size'])
best_lr = best_row['learning_rate']
best_dropout = best_row['dropout']

print(f"\nBest hyperparameters: hidden_size={best_hidden_size}, lr={best_lr}, dropout={best_dropout}")
print(f"Best validation accuracy: {best_row['val_accuracy']:.4f}")
'''


# Create a test dataset without labels
class TestDataset(Dataset):
    def __init__(self, embeddings):
        self.embeddings = embeddings
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        embedding = torch.tensor(self.embeddings[idx], dtype=torch.float)
        return embedding

# Create test dataset and dataloader
test_dataset = TestDataset(test_embeddings)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Generate predictions on the test set
model.eval()
predictions = []

with torch.no_grad():
    for embeddings in tqdm(test_loader, desc="Predicting on test set"):
        embeddings = embeddings.to(device)
        outputs = model(embeddings)
        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())

# Create submission DataFrame
submission = pd.DataFrame({
    'ID': test_df['ID'],
    'Label': predictions
})

# Display sample predictions
print("\nSample predictions:")
print(submission.head())

# Save submission file
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("\nSubmission file created successfully!")


# Analyze model performance on validation set
_, val_acc, val_prec, val_rec, val_f1, val_conf_matrix, val_predictions = evaluate(model, val_loader, criterion, device)

print("\nFinal Model Performance on Validation Set:")
print(f"Accuracy: {val_acc:.4f}")
print(f"Precision: {val_prec:.4f}")
print(f"Recall: {val_rec:.4f}")
print(f"F1 Score: {val_f1:.4f}")

# Plot confusion matrix as a percentage
val_conf_matrix_norm = val_conf_matrix.astype('float') / val_conf_matrix.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10, 8))
sns.heatmap(val_conf_matrix_norm, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Normalized Confusion Matrix')
plt.show()


# Error analysis: Look at some misclassified examples
val_df_copy = val_df.copy()
val_df_copy['Predicted'] = val_predictions
val_df_copy['Correct'] = val_df_copy['Label'] == val_df_copy['Predicted']

# Get misclassified examples
misclassified = val_df_copy[~val_df_copy['Correct']]

print(f"\nNumber of misclassified examples: {len(misclassified)} out of {len(val_df_copy)}")

# Display some misclassified examples
print("\nSome misclassified examples:")
for i, row in misclassified.head(5).iterrows():
    true_label = "Positive" if row['Label'] == 1 else "Negative"
    pred_label = "Positive" if row['Predicted'] == 1 else "Negative"
    print(f"Text: {row['Text']}")
    print(f"True label: {true_label}, Predicted label: {pred_label}")
    print("-" * 80)

