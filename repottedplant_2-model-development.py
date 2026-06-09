# Install required packages
!pip install -q transformers datasets torch scikit-learn scipy
# Install specific scipy and seaborn versions for compatibility
!pip install -q seaborn==0.11.2
!pip install textstat


# Import necessary libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm

# Text processing libraries
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import textstat

# Machine learning libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# For transformer models
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset

# Set random seed for reproducibility
np.random.seed(42)


# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')


# Define paths and reproducibility settings
BASE      = "../input/fake-or-real-the-impostor-hunt/data"  # Adjust this path based on your setup
TRAIN_CSV = os.path.join(BASE, "train.csv")
TRAIN_DIR = os.path.join(BASE, "train")
TEST_DIR  = os.path.join(BASE, "test")
SUB_PATH  = "submission.csv"

RND = 42
np.random.seed(RND)

# Function to read text files
def read_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except:
        print(f"Error reading file: {file_path}")
        return ""

# Improved function to load data
def load_data(csv_path, data_dir):
    try:
        meta = pd.read_csv(csv_path)
    except:
        print(f"Error: Could not find or read CSV file at {csv_path}")
        print("Using sample data for demonstration purposes...")
        # Create sample data if CSV file is not available
        meta = pd.DataFrame({
            'id': range(5),
            'real_text_id': [1, 2, 1, 2, 1]
        })
    
    rows = []
    for _, r in tqdm(meta.iterrows(), total=len(meta), desc="Loading data"):
        art    = f"article_{r['id']:04d}"
        real_i = int(r['real_text_id'])
        fake_i = 1 if real_i == 2 else 2
        
        # Load real text
        real_path = os.path.join(data_dir, art, f"file_{real_i}.txt")
        try:
            real_text = read_text_file(real_path)
        except:
            # If file doesn't exist, use placeholder text
            real_text = "This is a placeholder for real text. File could not be found."
        
        # Load fake text
        fake_path = os.path.join(data_dir, art, f"file_{fake_i}.txt")
        try:
            fake_text = read_text_file(fake_path)
        except:
            # If file doesn't exist, use placeholder text
            fake_text = "This is a placeholder for fake text. File could not be found."
        
        # Add to rows
        rows.append({
            "article_id": r['id'],
            "real_text": real_text,
            "fake_text": fake_text,
            "real_text_id": real_i
        })
    
    return pd.DataFrame(rows)

# Load the training data
print("Loading training data...")
try:
    train_df = load_data(TRAIN_CSV, TRAIN_DIR)
    print(f"Loaded {len(train_df)} article pairs.")
except Exception as e:
    print(f"Error loading data: {e}")
    # Create a simple example dataset
    train_df = pd.DataFrame({
        'article_id': [0, 1, 2],
        'real_text': [
            "The study examined various linguistic patterns across academic disciplines.",
            "Climate change research indicates significant shifts in global weather patterns.",
            "Economic analysis shows correlation between unemployment and inflation rates."
        ],
        'fake_text': [
            "The examination looked at different word patterns across schools of thought.",
            "Weather patterns are changing dramatically according to recent studies.",
            "Joblessness and price increases appear to be linked in recent economic trends."
        ],
        'real_text_id': [1, 2, 1]
    })
    print("Created sample dataset for demonstration.")

train_df.head()


# Add basic text statistics
train_df['real_text_length'] = train_df['real_text'].apply(len)
train_df['fake_text_length'] = train_df['fake_text'].apply(len)
train_df['real_text_words'] = train_df['real_text'].apply(lambda x: len(word_tokenize(x)))
train_df['fake_text_words'] = train_df['fake_text'].apply(lambda x: len(word_tokenize(x)))

# Check for empty texts
empty_texts = train_df[(train_df['fake_text_length'] == 0) | (train_df['fake_text_words'] == 0)]
print(f"Number of articles with empty fake texts: {len(empty_texts)}")

# Basic summary info
print(f"Dataset contains {len(train_df)} article pairs")
print(f"Distribution of real_text_id: {train_df['real_text_id'].value_counts().to_dict()}")


# Define enhanced feature extraction function with empty text handling
def extract_basic_features(text):
    # Check if text is empty or nearly empty
    if text is None or len(text.strip()) == 0:
        # Return default values for empty text
        return {
            'is_empty': 1,  # New feature to flag empty texts
            'num_chars': 0,
            'num_words': 0,
            'num_sentences': 0,
            'avg_word_length': 0,
            'avg_sentence_length': 0,
            'lexical_diversity': 0,
            'flesch_reading_ease': 0,
            'flesch_kincaid_grade': 0,
            'smog_index': 0,
            'automated_readability_index': 0,
            'coleman_liau_index': 0,
            'punctuation_per_word': 0,
            'function_word_ratio': 0
        }
    
    # For non-empty texts, proceed as normal
    try:
        # Tokenize text
        words = word_tokenize(text.lower())
        sentences = sent_tokenize(text)
        
        # Basic statistics
        num_chars = len(text)
        num_words = len(words)
        num_sentences = len(sentences)
        avg_word_length = sum(len(word) for word in words) / max(1, num_words)
        avg_sentence_length = num_words / max(1, num_sentences)
        
        # Lexical diversity (unique words / total words)
        lexical_diversity = len(set(words)) / max(1, num_words)
        
        # Readability metrics - handle potential errors
        try:
            flesch_reading_ease = textstat.flesch_reading_ease(text)
            flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
            smog_index = textstat.smog_index(text)
            automated_readability_index = textstat.automated_readability_index(text)
            coleman_liau_index = textstat.coleman_liau_index(text)
        except Exception as e:
            # If readability metrics fail, use default values
            print(f"Warning: Could not calculate readability metrics: {e}")
            flesch_reading_ease = 0
            flesch_kincaid_grade = 0
            smog_index = 0
            automated_readability_index = 0
            coleman_liau_index = 0
        
        # Punctuation frequency
        punctuation_count = sum(1 for char in text if char in '.,;:!?"()')
        punctuation_per_word = punctuation_count / max(1, num_words)
        
        # Part of speech tags (requires NLTK's pos_tag)
        # We'll use a simpler approach with common function words
        stop_words = set(stopwords.words('english'))
        function_words = sum(1 for word in words if word in stop_words)
        function_word_ratio = function_words / max(1, num_words)
        
        return {
            'is_empty': 0,  # New feature - not empty
            'num_chars': num_chars,
            'num_words': num_words,
            'num_sentences': num_sentences,
            'avg_word_length': avg_word_length,
            'avg_sentence_length': avg_sentence_length,
            'lexical_diversity': lexical_diversity,
            'flesch_reading_ease': flesch_reading_ease,
            'flesch_kincaid_grade': flesch_kincaid_grade,
            'smog_index': smog_index,
            'automated_readability_index': automated_readability_index,
            'coleman_liau_index': coleman_liau_index,
            'punctuation_per_word': punctuation_per_word,
            'function_word_ratio': function_word_ratio
        }
    except Exception as e:
        print(f"Error processing text: {e}")
        # Return default values in case of unexpected errors
        return {
            'is_empty': 0,
            'num_chars': len(text),
            'num_words': 0,
            'num_sentences': 0,
            'avg_word_length': 0,
            'avg_sentence_length': 0,
            'lexical_diversity': 0,
            'flesch_reading_ease': 0,
            'flesch_kincaid_grade': 0,
            'smog_index': 0,
            'automated_readability_index': 0,
            'coleman_liau_index': 0,
            'punctuation_per_word': 0,
            'function_word_ratio': 0
        }


# Define function to get embeddings from a pre-trained model
def get_transformer_embeddings(texts, model_name='distilbert-base-uncased', max_length=512):
    # Load pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    embeddings = []
    
    # Process texts in batches
    batch_size = 8
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize and prepare inputs
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get model outputs
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Use the [CLS] token embeddings (first token of last hidden state)
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.extend(batch_embeddings)
    
    return np.array(embeddings)


# Sample a subset for embedding extraction (optional, to save time)
sample_size = min(200, len(train_df))  # Adjust based on your computational resources
sampled_df = train_df.sample(sample_size, random_state=42)

# Extract embeddings
print("Extracting embeddings for real and fake texts...")
real_embeddings = get_transformer_embeddings(sampled_df['real_text'].tolist())
fake_embeddings = get_transformer_embeddings(sampled_df['fake_text'].tolist())

print(f"Real embeddings shape: {real_embeddings.shape}")
print(f"Fake embeddings shape: {fake_embeddings.shape}")

# Combine embeddings with labels for training
X_real = real_embeddings
X_fake = fake_embeddings
y_real = np.ones(len(X_real))
y_fake = np.zeros(len(X_fake))

# Combine datasets
X_combined = np.vstack([X_real, X_fake])
y_combined = np.concatenate([y_real, y_fake])

# Split data for training and validation
X_train, X_val, y_train, y_val = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined)
print(f"Training set shape: {X_train.shape}, Validation set shape: {X_val.shape}")


# Train and evaluate transformer embeddings-based models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    val_preds = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_preds)
    
    print(f"{name} Validation Accuracy: {val_accuracy:.4f}")
    print(classification_report(y_val, val_preds))
    
    results[name] = {'model': model, 'val_accuracy': val_accuracy}

# Identify the best model
best_model_name = max(results, key=lambda x: results[x]['val_accuracy'])
best_model = results[best_model_name]['model']
best_accuracy = results[best_model_name]['val_accuracy']

print(f"Best model: {best_model_name} with validation accuracy: {best_accuracy:.4f}")


# Define a function to calculate pairwise accuracy
def calculate_pairwise_accuracy(model, real_texts, fake_texts):
    correct_predictions = 0
    total_pairs = len(real_texts)
    
    for i in range(total_pairs):
        real_prob = model.predict_proba(real_texts[i].reshape(1, -1))[0][1]
        fake_prob = model.predict_proba(fake_texts[i].reshape(1, -1))[0][1]
        
        # If model gives higher probability to real text being real
        if real_prob > fake_prob:
            correct_predictions += 1
    
    return correct_predictions / total_pairs


# Create test pairs for pairwise evaluation
val_real_indices = np.where(y_val == 1)[0]
val_fake_indices = np.where(y_val == 0)[0]

# Ensure equal number of real and fake samples
min_samples = min(len(val_real_indices), len(val_fake_indices))
val_real_indices = val_real_indices[:min_samples]
val_fake_indices = val_fake_indices[:min_samples]

# Create arrays of real and fake texts
val_real_texts = X_val[val_real_indices]
val_fake_texts = X_val[val_fake_indices]

# Evaluate each model with pairwise accuracy
pairwise_results = {}

for name, model_info in results.items():
    model = model_info['model']
    pairwise_acc = calculate_pairwise_accuracy(model, val_real_texts, val_fake_texts)
    pairwise_results[name] = pairwise_acc
    print(f"{name} Pairwise Accuracy: {pairwise_acc:.4f}")

# Visualize pairwise accuracy results
plt.figure(figsize=(10, 6))
plt.bar(pairwise_results.keys(), pairwise_results.values())
plt.title('Pairwise Accuracy by Model')
plt.ylabel('Pairwise Accuracy')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


def predict_with_empty_text_handling(model, text1, text2, embeddings_func=None):
    """
    Custom prediction function that handles empty text edge cases before using the model.
    
    Args:
        model: The trained model for prediction
        text1: First text in the pair
        text2: Second text in the pair
        embeddings_func: Function to convert texts to embeddings (if needed)
        
    Returns:
        real_text_id: 1 if text1 is more likely real, 2 if text2 is more likely real
        confidence: Confidence level for the prediction
    """
    # Check for empty or near-empty texts (define a minimal threshold)
    text1_empty = len(text1.strip()) < 10
    text2_empty = len(text2.strip()) < 10
    
    # Case 1: Both texts are empty or near-empty
    if text1_empty and text2_empty:
        # In this unusual case, we default to text1 with low confidence
        return 1, 0.51
    
    # Case 2: Only text1 is empty - text2 must be real
    if text1_empty and not text2_empty:
        return 2, 0.99
    
    # Case 3: Only text2 is empty - text1 must be real
    if not text1_empty and text2_empty:
        return 1, 0.99
    
    # Case 4: Neither text is empty - use the model for prediction
    if embeddings_func:
        # If we need to convert texts to embeddings first
        text1_embedding = embeddings_func([text1])
        text2_embedding = embeddings_func([text2])
        
        # Get probability of each text being real
        text1_prob = model.predict_proba(text1_embedding)[0][1]
        text2_prob = model.predict_proba(text2_embedding)[0][1]
    else:
        # For models that accept text directly (like fine-tuned transformers)
        # This assumes your model accepts raw text input
        text1_prob = model.predict_proba(text1)[0][1]
        text2_prob = model.predict_proba(text2)[0][1]
    
    # Determine which text is more likely to be real
    real_text_id = 1 if text1_prob > text2_prob else 2
    confidence = max(text1_prob, text2_prob)
    
    return real_text_id, confidence


# Fine-tune a transformer model for classification
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback, DataCollatorWithPadding
from datasets import Dataset
import torch
import os
from sklearn.metrics import accuracy_score, f1_score

# Disable wandb and other reporting completely
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings

# Sample from the training data for fine-tuning (adjust based on your computational resources)
sample_size = min(500, len(train_df))  # Reduced sample size for faster training
print(f"Using {sample_size} samples for fine-tuning")
fine_tune_df = train_df.sample(sample_size, random_state=42)

# Prepare dataset for training
train_texts = fine_tune_df['real_text'].tolist() + fine_tune_df['fake_text'].tolist()
train_labels = [1] * len(fine_tune_df) + [0] * len(fine_tune_df)

# Create a Dataset object
train_dataset = Dataset.from_dict({
    'text': train_texts,
    'label': train_labels
})

# Split into training and validation sets
train_val_dataset = train_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_val_dataset['train']
val_dataset = train_val_dataset['test']

print(f"Training dataset: {len(train_dataset)} samples")
print(f"Validation dataset: {len(val_dataset)} samples")

# Load tokenizer and model
model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Tokenize function - with progress display
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)  # Reduced max length

# Tokenize datasets
print("Tokenizing training dataset...")
tokenized_train = train_dataset.map(tokenize_function, batched=True)
print("Tokenizing validation dataset...")
tokenized_val = val_dataset.map(tokenize_function, batched=True)

# Define training arguments - with minimal settings for stability
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,  # Reduced epochs
    per_device_train_batch_size=4,  # Smaller batch size
    per_device_eval_batch_size=4,
    warmup_steps=0,  # Simplified warmup
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy='steps',  # Evaluate more frequently
    eval_steps=50,  # Check validation every 50 steps
    save_strategy='steps',
    save_steps=50,
    load_best_model_at_end=True,
    report_to=[],  # Disable wandb and all other integrations
    disable_tqdm=False,  # Show progress bars
    no_cuda=torch.cuda.is_available() == False,  # Only use GPU if available
)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
    data_collator=data_collator
)
trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=2))

print("Starting fine-tuning...")
# Train the model with a try-except to catch interruptions
try:
    trainer.train()
    print("Fine-tuning completed successfully!")
except KeyboardInterrupt:
    print("Training was interrupted. Saving current model state...")
    trainer.save_model("./interrupted_model")
    print("Model saved to ./interrupted_model")
except Exception as e:
    print(f"An error occurred during training: {str(e)}")


# Evaluate the fine-tuned model

# Function to get predictions from the model
def get_predictions(model, tokenizer, texts, batch_size=16):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    all_preds = []
    all_probs = []
    
    # Process in batches to avoid memory issues
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:min(i+batch_size, len(texts))]
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get probabilities and predictions
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        preds = torch.argmax(outputs.logits, dim=-1)
        
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of being real (class 1)
    
    return np.array(all_preds), np.array(all_probs)

# Create a validation dataset for evaluation
val_size = min(50, len(train_df))
eval_df = train_df.sample(val_size, random_state=43)  # Different random state from training

# Prepare texts and labels
real_texts = eval_df['real_text'].tolist()
fake_texts = eval_df['fake_text'].tolist()
all_texts = real_texts + fake_texts
all_labels = [1] * len(real_texts) + [0] * len(fake_texts)

# Get predictions
print("Getting predictions for validation set...")
predictions, probabilities = get_predictions(model, tokenizer, all_texts)

# Calculate standard accuracy
accuracy = accuracy_score(all_labels, predictions)
print(f"Standard Accuracy: {accuracy:.4f}")
print(classification_report(all_labels, predictions))

# Calculate pairwise accuracy (competition metric)
correct_pairs = 0
for i in range(len(real_texts)):
    real_prob = probabilities[i]  # Probability of real text being real
    fake_prob = probabilities[i + len(real_texts)]  # Probability of fake text being real
    
    if real_prob > fake_prob:
        correct_pairs += 1

transformer_pairwise_accuracy = correct_pairs / len(real_texts)
print(f"Transformer Model Pairwise Accuracy: {transformer_pairwise_accuracy:.4f}")

# Compare with the embedding-based models
print("\nComparison with embedding-based models:")
for name, acc in pairwise_results.items():
    print(f"{name} Pairwise Accuracy: {acc:.4f}")


# Custom prediction function for transformer model with empty text handling
def predict_with_fine_tuned_model_handling_empty(model, tokenizer, text1, text2):
    """
    Custom prediction function for transformer models that handles empty text edge cases
    """
    # Check for empty or near-empty texts
    text1_empty = len(text1.strip()) < 10
    text2_empty = len(text2.strip()) < 10
    
    # Handle empty text cases
    if text1_empty and text2_empty:
        return 0.51, 0.49  # Slightly favor text1 with low confidence
    elif text1_empty and not text2_empty:
        return 0.01, 0.99  # Text2 is real with high confidence
    elif not text1_empty and text2_empty:
        return 0.99, 0.01  # Text1 is real with high confidence
    
    # For non-empty texts, proceed with the transformer model
    try:
        # Ensure model is in evaluation mode
        model.eval()
        
        # Move model to appropriate device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Tokenize inputs
        inputs1 = tokenizer(text1, return_tensors='pt', padding=True, truncation=True, max_length=256)
        inputs2 = tokenizer(text2, return_tensors='pt', padding=True, truncation=True, max_length=256)
        
        # Move inputs to device
        inputs1 = {k: v.to(device) for k, v in inputs1.items()}
        inputs2 = {k: v.to(device) for k, v in inputs2.items()}
        
        # Get model outputs
        with torch.no_grad():
            outputs1 = model(**inputs1)
            outputs2 = model(**inputs2)
        
        # Get probabilities using softmax
        probs1 = torch.nn.functional.softmax(outputs1.logits, dim=-1)
        probs2 = torch.nn.functional.softmax(outputs2.logits, dim=-1)
        
        # Return probability of being real (class 1)
        return probs1[0][1].item(), probs2[0][1].item()
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        # Return default values that slightly favor the first text
        return 0.51, 0.49


# Load test data
def load_test_data(data_dir, test_folder_pattern="article_*"):
    # Get list of test article folders
    try:
        test_folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f)) and f.startswith("article_")])
    except:
        print(f"Error: Could not list directories in {data_dir}")
        # Create sample data for demonstration purposes
        test_folders = [f"article_{i:04d}" for i in range(5)]
    
    print(f"Number of test articles: {len(test_folders)}")
    
    rows = []
    for folder in tqdm(test_folders, desc="Loading test data"):
        # Extract article ID from folder name
        if folder.startswith('article_'):
            try:
                article_id = int(folder.split('_')[1])
            except ValueError:
                article_id = folder
        else:
            try:
                article_id = int(folder)
                folder = f"article_{article_id:04d}"  # Convert to standard format
            except ValueError:
                article_id = folder
        
        # Load texts
        file_1_path = os.path.join(data_dir, folder, "file_1.txt")
        file_2_path = os.path.join(data_dir, folder, "file_2.txt")
        
        try:
            if os.path.exists(file_1_path) and os.path.exists(file_2_path):
                text_1 = read_text_file(file_1_path)
                text_2 = read_text_file(file_2_path)
                
                rows.append({
                    "article_id": article_id,
                    "text_1": text_1,
                    "text_2": text_2
                })
            else:
                # For demonstration, create sample texts if files don't exist
                rows.append({
                    "article_id": article_id,
                    "text_1": f"Sample text 1 for article {article_id}",
                    "text_2": f"Sample text 2 for article {article_id}"
                })
        except Exception as e:
            print(f"Error processing folder {folder}: {e}")
            # Add placeholder for error cases
            rows.append({
                "article_id": article_id,
                "text_1": "Error loading text 1",
                "text_2": "Error loading text 2"
            })
    
    return pd.DataFrame(rows)

# Load the test data
try:
    test_df = load_test_data(TEST_DIR)
    print(f"Loaded {len(test_df)} test article pairs")
except Exception as e:
    print(f"Error loading test data: {e}")
    # Create sample test data
    test_df = pd.DataFrame({
        'article_id': range(5),
        'text_1': [f"Sample test text 1 for article {i}" for i in range(5)],
        'text_2': [f"Sample test text 2 for article {i}" for i in range(5)]
    })
    print("Created sample test dataset for demonstration.")

# Display basic statistics about the test data
print(f"Test data shape: {test_df.shape}")
print(f"Number of unique article IDs: {test_df['article_id'].nunique()}")

# Calculate text lengths
test_df['text_1_length'] = test_df['text_1'].apply(len)
test_df['text_2_length'] = test_df['text_2'].apply(len)

print("\nText length statistics:")
print(test_df[['text_1_length', 'text_2_length']].describe())


# Function to make predictions using our best embedding-based model
def generate_embedding_model_predictions(model, test_df, embeddings_func):
    submission_data = []
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Making embedding model predictions"):
        article_id = row['article_id']
        
        # Use our custom prediction function with empty text handling
        real_text_id, confidence = predict_with_empty_text_handling(
            model,
            row['text_1'],
            row['text_2'],
            embeddings_func=embeddings_func
        )
        
        submission_data.append({
            'id': article_id,
            'real_text_id': real_text_id,
            'confidence': confidence
        })
    
    return pd.DataFrame(submission_data)

def get_embeddings_for_prediction(texts):
    
    return get_transformer_embeddings(texts)

# Function to make predictions using the fine-tuned transformer model
def generate_transformer_model_predictions(model, tokenizer, test_df):
    submission_data = []
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Making transformer model predictions"):
        article_id = row['article_id']
        
        # Get probabilities using our custom function with empty text handling
        text1_prob, text2_prob = predict_with_fine_tuned_model_handling_empty(
            model,
            tokenizer,
            row['text_1'],
            row['text_2']
        )
        
        # Determine which text is more likely to be real
        real_text_id = 1 if text1_prob > text2_prob else 2
        confidence = max(text1_prob, text2_prob)
        
        submission_data.append({
            'id': article_id,
            'real_text_id': real_text_id,
            'confidence': confidence
        })
    
    return pd.DataFrame(submission_data)


# Generate predictions with the transformer model on the full dataset
# Based on our analysis, the transformer model performs best, so we'll use it directly

# Generate predictions
print("Generating predictions with the transformer model on the full test dataset...")
print(f"Processing all {len(test_df)} test samples - this may take some time...")

# Using only the transformer model for predictions (faster and better performing)
transformer_preds = generate_transformer_model_predictions(model, tokenizer, test_df)

print("\nSample transformer model predictions:")
print(transformer_preds.head(10))

#For embedding-based models (if needed)
# best_ml_model = results[best_model_name]['model']  # Get the best embedding-based model
# embedding_preds = generate_embedding_model_predictions(best_ml_model, test_df, get_embeddings_for_prediction)


# Create final submission file using transformer model predictions
transformer_preds[['id', 'real_text_id']].to_csv(SUB_PATH, index=False)
print(f"Created submission file '{SUB_PATH}' with {len(transformer_preds)} predictions")

# Verify submission format
print("\nSubmission file format:")
print(pd.read_csv(SUB_PATH).dtypes)
print("\nSample predictions:")
print(pd.read_csv(SUB_PATH).head(10))

# Validate that all real_text_id values are either 1 or 2
valid_ids = pd.read_csv(SUB_PATH)['real_text_id'].isin([1, 2]).all()
print(f"All real_text_id values are valid (1 or 2): {valid_ids}")

# Show distribution of predictions
text1_pred = (transformer_preds['real_text_id'] == 1).sum()
text2_pred = (transformer_preds['real_text_id'] == 2).sum()
print(f"\nPrediction distribution: Text 1 as real: {text1_pred} ({text1_pred/len(transformer_preds)*100:.1f}%), Text 2 as real: {text2_pred} ({text2_pred/len(transformer_preds)*100:.1f}%)")


# Visualize prediction distribution by the transformer model
plt.figure(figsize=(8, 6))
labels = ['Text 1 is Real', 'Text 2 is Real']
counts = [
    (transformer_preds['real_text_id'] == 1).sum(),
    (transformer_preds['real_text_id'] == 2).sum()
]

# Pie chart for prediction distribution
plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#5DA5DA', '#FAA43A'])
plt.axis('equal')
plt.title('Transformer Model Prediction Distribution')
plt.tight_layout()
plt.show()

# Confidence distribution analysis using matplotlib instead of seaborn
plt.figure(figsize=(10, 6))
# Use matplotlib's hist function instead of seaborn's histplot
plt.hist(transformer_preds['confidence'], bins=20, alpha=0.7, color='#5DA5DA')
plt.title('Distribution of Confidence Scores in Transformer Model Predictions')
plt.xlabel('Confidence Score')
plt.ylabel('Count')
plt.axvline(x=0.9, linestyle='--', color='r', label='High Confidence Threshold (0.9)')
plt.legend()
plt.tight_layout()
plt.show()

# Calculate confidence statistics
high_confidence = (transformer_preds['confidence'] > 0.9).sum()
print(f"Predictions with high confidence (>0.9): {high_confidence} ({high_confidence/len(transformer_preds)*100:.1f}%)")
print(f"Average confidence score: {transformer_preds['confidence'].mean():.4f}")
print(f"Median confidence score: {transformer_preds['confidence'].median():.4f}")


# After training completes, save your transformer model
print("Saving transformer model...")
trainer.save_model("model-distilbert")

# Save the tokenizer separately
tokenizer.save_pretrained("model-distilbert")
print("Transformer model and tokenizer saved successfully.")

