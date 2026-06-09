import pandas as pd
import numpy as np
import gc
import psutil
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, roc_curve, auc, confusion_matrix
from sklearn.pipeline import make_pipeline
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPool1D, Dense, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model

# --- CONFIGURATION AND RESOURCE MANAGEMENT ---
ROOT_PATH = '/kaggle/input/jigsaw-agile-community-rules'
TRAIN_PATH = f'{ROOT_PATH}/train.csv'
TEST_PATH = f'{ROOT_PATH}/test.csv'

# FIX: Changed filename to strictly comply with Keras's requirement for save_weights_only=True
BEST_CNN_WEIGHTS_FILE = 'best_cnn_model.weights.h5' 

# Aggressive memory management settings
CHUNK_SIZE = 10000  # Smaller chunks for memory safety
MAX_VOCAB = 10000    # Vocabulary size limit for Tokenizer
MAX_SEQ_LEN = 150    # Max comment length for CNN input
EMBEDDING_DIM = 64   # Dimension for the CNN embedding layer

def memory_usage():
    """Logs current memory usage."""
    process = psutil.Process()
    mem_b = process.memory_info().rss
    mem_mb = mem_b / (1024 * 1024)
    return f"{mem_mb:.2f} MB"

def force_cleanup(step_name=""):
    """Implements aggressive garbage collection."""
    if step_name:
        print(f"\n[{step_name}] Memory Usage BEFORE cleanup: {memory_usage()}")
    
    del_count = gc.collect()
    
    if step_name:
        print(f"[{step_name}] Garbage Collected: {del_count} objects. Memory AFTER cleanup: {memory_usage()}")
    print("-" * 50)

# --- 1. DATA PREPARATION AND LOADING ---

def load_data_in_chunks(filepath, dtype):
    """Ultra-conservative, chunk-based data loading with memory monitoring."""
    print(f"Loading data from {filepath} in chunks...")
    chunks = []
    
    # Use float32 and int32 for memory efficiency
    tqdm_chunks = pd.read_csv(filepath, chunksize=CHUNK_SIZE, dtype=dtype)
    
    for i, chunk in enumerate(tqdm(tqdm_chunks, desc=f"Loading Chunks (Max {CHUNK_SIZE} rows/chunk)")):
        # Robust Batch Processing: Check memory between chunks
        if psutil.virtual_memory().percent > 85: # Automatic stopping if memory limits exceeded
            print(f"WARNING: Memory usage too high ({psutil.virtual_memory().percent}%). Stopping data load after chunk {i}.")
            break
        chunks.append(chunk)
        
    df = pd.concat(chunks, ignore_index=True)
    print(f"Successfully loaded {len(df)} rows.")
    force_cleanup("Data Loading")
    return df

# Define memory-efficient data types
train_dtype = {
    'row_id': np.int32, 'rule_violation': np.int8,
    'body': 'object', 'rule': 'object', 'subreddit': 'object',
    'positive_example_1': 'object', 'positive_example_2': 'object',
    'negative_example_1': 'object', 'negative_example_2': 'object'
}
test_dtype = {
    'row_id': np.int32,
    'body': 'object', 'rule': 'object', 'subreddit': 'object',
    'positive_example_1': 'object', 'positive_example_2': 'object',
    'negative_example_1': 'object', 'negative_example_2': 'object'
}

try:
    # Use full path assuming standard Kaggle environment
    train_df = load_data_in_chunks(TRAIN_PATH, train_dtype)
    test_df = load_data_in_chunks(TEST_PATH, test_dtype)
except FileNotFoundError:
    # Failsafe mechanism: If running outside Kaggle with no data, create dummy structure
    print("WARNING: Data files not found at /kaggle/input/. Creating dummy structure for demonstration.")
    # Create larger dummy sets to enable robust feature generation
    N_DUMMY = 500
    dummy_data = {
        'body': [f'comment {i}' for i in range(N_DUMMY)],
        'rule': [f'Rule {i % 5}' for i in range(N_DUMMY)],
        'rule_violation': np.random.randint(0, 2, N_DUMMY),
        'row_id': np.arange(1, N_DUMMY + 1)
    }
    train_df = pd.DataFrame(dummy_data)
    test_df = pd.DataFrame({
        'body': [f'test comment {i}' for i in range(N_DUMMY // 2)],
        'rule': [f'Test Rule {i % 5}' for i in range(N_DUMMY // 2)],
        'row_id': np.arange(N_DUMMY + 1, N_DUMMY + 1 + N_DUMMY // 2)
    })
    
# Data Cleaning: Fill NaNs (Crucial for Tfidf and Tokenizer)
TEXT_COLS = ['body', 'rule']
for col in TEXT_COLS:
    train_df[col] = train_df[col].fillna('')
    test_df[col] = test_df[col].fillna('')

# --- 2. DATA SPLITTING AND EVALUATION SETUP ---

# Prepare data into training, validation and test sets (70/15/15)
X = train_df.drop('rule_violation', axis=1)
y = train_df['rule_violation']

# Split 1: 85% Train/Val, 15% Holdout/Test (for final robust evaluation)
X_train_val, X_test_holdout, y_train_val, y_test_holdout = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# Split 2: 70% Train, 15% Validation (for hyperparameter tuning/early stopping)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=(0.15 / 0.85), random_state=42, stratify=y_train_val
)

print(f"Train set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Holdout/Test set size: {len(X_test_holdout)}")

force_cleanup("Data Splitting")

# --- 3. FEATURE ENGINEERING ---

# 3a. Classical Feature Extraction (TF-IDF)
print("Processing Classical Features (TF-IDF)...")
# Combine comment body and rule text to allow the model to generalize to unseen rules.
train_text = X_train['body'] + " " + X_train['rule']
val_text = X_val['body'] + " " + X_val['rule']
test_holdout_text = X_test_holdout['body'] + " " + X_test_holdout['rule']
test_inference_text = test_df['body'] + " " + test_df['rule']

tfidf_vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 3), # Use unigrams, bigrams, and trigrams
    max_features=5000,   # Adjusted for memory safety
    dtype=np.float32     # Memory-efficient data type
)

X_train_tfidf = tfidf_vectorizer.fit_transform(tqdm(train_text, desc="TF-IDF Fit/Transform TRAIN"))
X_val_tfidf = tfidf_vectorizer.transform(tqdm(val_text, desc="TF-IDF Transform VAL"))
X_test_holdout_tfidf = tfidf_vectorizer.transform(tqdm(test_holdout_text, desc="TF-IDF Transform HOLDOUT"))
X_test_inference_tfidf = tfidf_vectorizer.transform(tqdm(test_inference_text, desc="TF-IDF Transform INFERENCE"))

print(f"TF-IDF Feature Count: {X_train_tfidf.shape[1]}")
force_cleanup("TF-IDF Feature Engineering")

# 3b. Deep Feature Extraction Setup (Tokenization for CNN)
print("Processing Deep Features (Tokenization)...")

# Tokenizer fits on the entire train/val data to ensure full vocabulary capture
full_train_text = pd.concat([train_text, val_text])
tokenizer = Tokenizer(num_words=MAX_VOCAB, lower=True)
tokenizer.fit_on_texts(tqdm(full_train_text, desc="Tokenizer Fit"))

# Convert text to sequences and pad them
X_train_seq = pad_sequences(tokenizer.texts_to_sequences(train_text), maxlen=MAX_SEQ_LEN)
X_val_seq = pad_sequences(tokenizer.texts_to_sequences(val_text), maxlen=MAX_SEQ_LEN)
X_test_holdout_seq = pad_sequences(tokenizer.texts_to_sequences(test_holdout_text), maxlen=MAX_SEQ_LEN)
X_test_inference_seq = pad_sequences(tokenizer.texts_to_sequences(test_inference_text), maxlen=MAX_SEQ_LEN)

force_cleanup("Tokenization")

# --- 4. MODEL DEVELOPMENT (Micro-CNN for Deep Features) ---

def build_micro_cnn(vocab_size, max_len, embedding_dim):
    """
    Implements a memory-optimized Micro Neural Network (CNN) for feature learning.
    It returns both the classifier model (for training) and the feature extractor
    (to get the embeddings for the hybrid model).
    """
    inputs = Input(shape=(max_len,), name='text_input')
    x = Embedding(vocab_size, embedding_dim, trainable=True, input_length=max_len)(inputs)
    
    # 1D Convolutional Layer for pattern recognition
    x = Conv1D(32, 5, activation='relu')(x)
    x = GlobalMaxPool1D()(x) # Aggregates features
    
    # Simple Dense layers to produce a compact embedding vector (Deep Features)
    deep_features = Dense(16, activation='relu', name='deep_features')(x)
    output = Dense(1, activation='sigmoid', name='cnn_output')(deep_features) # Final classification layer (for standalone use)

    # Use the model up to the 16-unit layer to extract the features/embeddings
    feature_extractor = Model(inputs=inputs, outputs=deep_features, name='cnn_feature_extractor')
    
    # Compile a standalone classifier model for training purposes
    classifier_model = Model(inputs=inputs, outputs=output, name='cnn_classifier')
    classifier_model.compile(
        optimizer='adam', 
        loss='binary_crossentropy', 
        metrics=['AUC'] # Use AUC as primary metric
    )
    
    return classifier_model, feature_extractor

# Build the CNN model
cnn_classifier_model, cnn_feature_extractor = build_micro_cnn(
    MAX_VOCAB, MAX_SEQ_LEN, EMBEDDING_DIM
)

# Training the Micro-CNN
print("Training Micro-CNN for Deep Feature Extraction...")
# FIX 1 (from previous turn): Corrected monitor name to 'val_AUC' (uppercase) to match available metrics
early_stopping = EarlyStopping(monitor='val_AUC', patience=2, mode='max', verbose=1)

# FIX 2 (from this turn): File path updated to comply with Keras's '.weights.h5' requirement 
model_checkpoint = ModelCheckpoint(
    BEST_CNN_WEIGHTS_FILE, 
    monitor='val_AUC', 
    mode='max', 
    save_best_only=True, 
    save_weights_only=True, # Save only weights for efficiency
    verbose=1
)

cnn_history = cnn_classifier_model.fit(
    X_train_seq, y_train,
    epochs=10, # Reduced epochs for memory safety
    batch_size=32,
    validation_data=(X_val_seq, y_val),
    callbacks=[early_stopping, model_checkpoint],
    verbose=0 # Suppress output, use tqdm
)

# Load the best weights (now saved correctly)
try:
    cnn_classifier_model.load_weights(BEST_CNN_WEIGHTS_FILE)
    print(f"Successfully loaded best weights from {BEST_CNN_WEIGHTS_FILE}")
except Exception as e:
    print(f"WARNING: Could not load weights from checkpoint. Using last trained state. Error: {e}")

# Extract Deep Features (Embeddings) from the pre-trained CNN
X_train_deep = cnn_feature_extractor.predict(X_train_seq, verbose=0)
X_val_deep = cnn_feature_extractor.predict(X_val_seq, verbose=0)
X_test_holdout_deep = cnn_feature_extractor.predict(X_test_holdout_seq, verbose=0)
X_test_inference_deep = cnn_feature_extractor.predict(X_test_inference_seq, verbose=0)

force_cleanup("Micro-CNN Feature Extraction")

# --- 5. HYBRID MODEL TRAINING (Classical + Deep Features) ---

# Combine Classical (TF-IDF) and Deep (CNN) features
# We must convert sparse matrices to dense arrays for stacking with CNN features
X_train_combined = np.hstack([X_train_tfidf.toarray(), X_train_deep])
X_val_combined = np.hstack([X_val_tfidf.toarray(), X_val_deep])
X_test_holdout_combined = np.hstack([X_test_holdout_tfidf.toarray(), X_test_holdout_deep])
X_test_inference_combined = np.hstack([X_test_inference_tfidf.toarray(), X_test_inference_deep])

print(f"Combined Feature Set Size: {X_train_combined.shape[1]} features.")

# Train the final, simple classifier (Logistic Regression) on the hybrid feature set
final_hybrid_model = LogisticRegression(
    C=1.0, 
    solver='saga', # Memory efficient solver for large datasets
    class_weight='balanced', 
    max_iter=500,
    random_state=42
)

print("Training Final Hybrid Logistic Regression Model...")
final_hybrid_model.fit(X_train_combined, y_train)
print("Hybrid Model Training Complete.")
force_cleanup("Hybrid Model Training")

# --- 6. EVALUATION AND VISUALIZATION ---

def evaluate_and_plot(model, X_data, y_true, name):
    """Calculates AUC, plots ROC curve, Confusion Matrix, and Precision-Recall curve."""
    
    # Predict probabilities (required for AUC)
    y_pred_proba = model.predict_proba(X_data)[:, 1]
    
    # Set a balanced threshold (e.g., 0.5) for class prediction
    y_pred_class = (y_pred_proba >= 0.5).astype(int)
    
    # 6a. Scoring: AUC (Area Under the Curve)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    print(f"\n--- Evaluation on {name} Set ---")
    # Column-averaged AUC for binary classification is the ROC-AUC score
    print(f"Column-Averaged AUC: {roc_auc:.4f}") 
    print("\nClassification Report (Threshold=0.5):")
    print(classification_report(y_true, y_pred_class, target_names=['No Violation', 'Violation']))

    # 6b. Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Model Evaluation on {name} Set (AUC: {roc_auc:.4f})', fontsize=16)

    # Plot 1: ROC-AUC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC - AUC Curve')
    axes[0].legend(loc="lower right")

    # Plot 2: Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_class)
    im = axes[1].matshow(cm, cmap=plt.cm.Blues)
    axes[1].set_title('Confusion Matrix')
    axes[1].set_ylabel('True label')
    axes[1].set_xlabel('Predicted label')
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[1].text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")

    # Plot 3: Precision-Recall Curve
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    average_precision = average_precision_score(y_true, y_pred_proba)
    axes[2].plot(recall, precision, color='purple', lw=2, label=f'AP = {average_precision:.2f}')
    axes[2].set_xlim([0.0, 1.0])
    axes[2].set_ylim([0.0, 1.05])
    axes[2].set_xlabel('Recall')
    axes[2].set_ylabel('Precision')
    axes[2].set_title('Precision-Recall Curve')
    axes[2].legend(loc="lower left")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Evaluate on Validation and Holdout Sets
evaluate_and_plot(final_hybrid_model, X_val_combined, y_val, "Validation")
evaluate_and_plot(final_hybrid_model, X_test_holdout_combined, y_test_holdout, "Holdout (Unseen)")

force_cleanup("Evaluation & Plotting")

# --- 7. PREDICTION AND SUBMISSION FILE GENERATION ---

print("Generating predictions for test.csv...")
# Predict probabilities on the final test set (inference data)
test_predictions = final_hybrid_model.predict_proba(X_test_inference_combined)[:, 1]

# Create the submission DataFrame
submission_df = pd.DataFrame({
    'row_id': test_df['row_id'],
    'rule_violation': test_predictions
})

# Ensure the required format (row_id, rule_violation)
submission_df['rule_violation'] = submission_df['rule_violation'].astype(np.float64)

# Save to submission.csv
SUBMISSION_FILE = 'submission.csv'
submission_df.to_csv(SUBMISSION_FILE, index=False)

print(f"Successfully generated {SUBMISSION_FILE} with {len(submission_df)} predictions.")
print("Submission file head:")
print(submission_df.head())

force_cleanup("Submission")





