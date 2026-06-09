import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset


# Check if GPU is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load training and test datasets
train = pd.read_csv('/kaggle/input/train-and-test-files/train.csv')  # Training data with text and labels
test = pd.read_csv('/kaggle/input/train-and-test-files/test_features.csv')  # Test data for predictions


# Define a custom dataset class for PyTorch
class SentimentDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=128):
        self.texts = texts  # List of input texts
        self.labels = labels  # Optional labels for training/validation
        self.tokenizer = tokenizer  # Tokenizer for text processing
        self.max_len = max_len  # Maximum sequence length

    def __len__(self):
        return len(self.texts)  # Number of samples

    def __getitem__(self, idx):
        # Tokenize a single text
        text = self.texts[idx]
        inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        # Prepare input dictionary
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        if self.labels is not None:  # Include labels if available
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")  # Pre-trained tokenizer
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)  # Pre-trained model with 4 output labels
model.to(device)  # Move model to the selected device (CPU/GPU)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",  # Directory to save model checkpoints
    evaluation_strategy="epoch",  # Evaluate after every epoch
    save_strategy="epoch",  # Save model after every epoch
    learning_rate=2e-5,  # Learning rate for optimization
    per_device_train_batch_size=16,  # Batch size for training
    per_device_eval_batch_size=32,  # Batch size for evaluation
    num_train_epochs=3,  # Number of training epochs
    weight_decay=0.01,  # Regularization to prevent overfitting
    logging_dir="./logs",  # Directory for logs
    logging_steps=10,  # Log progress every 10 steps
    load_best_model_at_end=True,  # Automatically load the best model
    metric_for_best_model="accuracy",  # Metric to determine the best model
    greater_is_better=True,  # Higher accuracy is better
    report_to="none"
)

# Function to calculate evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred  # Model outputs and true labels
    predictions = logits.argmax(axis=1)  # Get the predicted label
    acc = accuracy_score(labels, predictions)  # Compute accuracy
    return {"accuracy": acc}

# Perform Stratified K-Fold Cross-Validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Ensure balanced label distribution across folds
all_labels = train["Category"].tolist()  # Labels for the dataset
texts = train["Text"].tolist()  # Text data

accuracy_scores = []  # Store accuracy for each fold
confusion_matrices = []  # Store confusion matrices for each fold


for fold, (train_idx, val_idx) in enumerate(kf.split(texts, all_labels)):
    print(f"Starting Fold {fold + 1}")
    
    # Split data into training and validation for the current fold
    train_texts = [texts[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    train_labels = [all_labels[i] for i in train_idx]
    val_labels = [all_labels[i] for i in val_idx]

    # Create datasets for training and validation
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)



    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics  # Use the defined metrics
    )

    # Train the model
    trainer.train()

    # Evaluate the model on validation data
    results = trainer.predict(val_dataset)
    predictions = results.predictions.argmax(axis=1)  # Predicted labels
    accuracy = accuracy_score(val_labels, predictions)  # Compute accuracy
    accuracy_scores.append(accuracy)
    print(f"Fold {fold + 1} Accuracy: {accuracy}")
    
    # Generate confusion matrix for the current fold
    cm = confusion_matrix(val_labels, predictions)
    confusion_matrices.append(cm)

# Compute and print the average accuracy across all folds
print(f"Mean Accuracy: {sum(accuracy_scores) / len(accuracy_scores)}")


# Visualize the confusion matrix for the final fold
sns.heatmap(confusion_matrices[-1], annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1, 2, 3], yticklabels=[0, 1, 2, 3])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix (Final Fold)")
plt.show()

# Prepare the test dataset
test_dataset = SentimentDataset(test["Text"].tolist(), tokenizer=tokenizer)

# Make predictions on the test dataset
test_results = trainer.predict(test_dataset)
predictions = test_results.predictions.argmax(axis=1)  # Predicted labels for test data



# Create and save the submission file
submission = pd.DataFrame({
    "ID": test["ID"],  # Test sample IDs
    "Prediction": predictions  # Predicted labels
})
submission.to_csv("final_submission.csv", index=False)
print("Submission file saved!")

