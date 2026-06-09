import pandas as pd
import tensorflow as tf # Import tensorflow

# Load the datasets
try:
    train_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
    test_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')
    sample_submission_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv')

    print("Data loaded successfully!")
    print("\nTraining data head (showing 'body' as the text column):")
    print(train_df[['row_id', 'body', 'rule_violation']].head())
    print("\nTraining data info:")
    train_df.info()

    print("\nTest data head (showing 'body' as the text column):")
    print(test_df[['row_id', 'body']].head())
    print("\nTest data info:")
    test_df.info()

    print("\nSample Submission head:")
    print(sample_submission_df.head())

except FileNotFoundError:
    print("Error: Make sure 'train.csv', 'test.csv', and 'sample_submission.csv' are in the same directory.")
except Exception as e:
    print(f"An error occurred while loading data: {e}")

# Verify GPU availability
print("\nChecking for GPU availability:")
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
    print("GPU is available and will be used.")
else:
    print("No GPU available. Training will be slow on CPU.")




from transformers import AutoTokenizer

# Choose a pre-trained transformer model
# Good options: 'distilbert-base-uncased', 'bert-base-uncased', 'roberta-base'
# 'distilbert-base-uncased' is a good balance of speed and performance for a start.
model_name = 'distilbert-base-uncased'
max_length = 128 # Max sequence length for the transformer. Adjust based on average comment length.

# Load the tokenizer
print(f"Loading tokenizer for {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Tokenizer loaded.")

# Prepare training data inputs
print("Tokenizing training data...")
train_encodings = tokenizer(
    train_df['body'].tolist(),
    truncation=True,
    padding='max_length',
    max_length=max_length,
    return_tensors='tf'
)
y_train = train_df['rule_violation'].values # Get labels as a NumPy array

# Prepare test data inputs
print("Tokenizing test data...")
test_encodings = tokenizer(
    test_df['body'].tolist(),
    truncation=True,
    padding='max_length',
    max_length=max_length,
    return_tensors='tf'
)

# Create TensorFlow Datasets for efficient training and prediction
# Inputs are dicts containing 'input_ids', 'attention_mask', (and sometimes 'token_type_ids')
# Labels are y_train for the training dataset
BATCH_SIZE = 16 # Batch size for training. Can be 16, 32, 64 etc. Adjust based on GPU memory.

print("\nCreating TensorFlow Datasets...")
train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), y_train)).shuffle(1000).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(dict(test_encodings)).batch(BATCH_SIZE)

print(f"Train dataset created with batch size: {BATCH_SIZE}")
print("Test dataset created.")

print("\nTransformer-specific feature extraction complete.")



from transformers import TFAutoModelForSequenceClassification
import tensorflow as tf # Ensure tensorflow is imported

# Load the pre-trained model with a classification head
# For binary classification, num_labels=1 for sigmoid output
print(f"Loading pre-trained model for fine-tuning: {model_name}...")
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, from_pt=False)
print("Model loaded.")

# Define optimizer (critical for transformers)
# Use a very low learning rate for fine-tuning
learning_rate = 5e-5 # Common learning rate for transformer fine-tuning
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Define loss function and metrics
# from_logits=True if the model output is logits (raw scores), False if it's probabilities (sigmoid activated)
# TFAutoModelForSequenceClassification typically outputs logits, so we use from_logits=True
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics = [tf.keras.metrics.AUC(name='auc')] # AUC is the competition metric

# Compile the model
print("Compiling the model...")
model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
model.summary()
print("Model compiled.")

# Fine-tune the model
epochs = 3 # Number of epochs. 2-5 is common for fine-tuning transformers.
print(f"\nFine-tuning the model for {epochs} epochs...")

# For validation during training, you would split train_dataset into train_subset and val_subset.
# For simplicity and direct training for submission, we'll train on the full 'train_dataset'.
# In a real scenario, you'd use a validation split or K-fold cross-validation
# to monitor overfitting and pick the best epoch.

history = model.fit(
    train_dataset,
    epochs=epochs,
    # validation_data=val_dataset, # Uncomment if you create a validation_dataset
    verbose=1
)
print("Model fine-tuning complete.")

# Optional: You can plot training history (loss and AUC) if you had a validation set
# (requires modifications to 'model.fit' to include validation_data)
# import matplotlib.pyplot as plt
# if 'val_loss' in history.history: # Check if validation data was used
#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(history.history['loss'], label='Train Loss')
#     plt.plot(history.history['val_loss'], label='Validation Loss')
#     plt.title('Loss over Epochs')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.subplot(1, 2, 2)
#     plt.plot(history.history['auc'], label='Train AUC')
#     plt.plot(history.history['val_auc'], label='Validation AUC')
#     plt.title('AUC over Epochs')
#     plt.xlabel('Epoch')
#     plt.ylabel('AUC')
#     plt.legend()
#     plt.tight_layout()
#     plt.show()




import numpy as np # Import numpy for array operations

# Make predictions on the preprocessed test data
print("Generating predictions on the test data...")
# The model outputs logits, which need to be converted to probabilities using sigmoid
logits = model.predict(test_dataset).logits # Access the 'logits' attribute from the model output
predictions = tf.nn.sigmoid(logits).numpy().flatten() # Apply sigmoid and flatten
print("Predictions generated.")

# Create the submission DataFrame
submission_df = sample_submission_df.copy()

# Assign the predictions to the 'prediction' column
submission_df['prediction'] = predictions

# Display the first few rows of the submission file
print("\nSubmission file head:")
print(submission_df.head())

# Save the submission file
submission_file_name = 'submission.csv' # Changed filename
submission_df.to_csv(submission_file_name, index=False)

print(f"\nSubmission file '{submission_file_name}' created successfully!")
print("You can now submit this file to the Kaggle competition.")


