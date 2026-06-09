import pandas as pd

# Load the train.csv file
train_df = pd.read_csv('./train.csv', encoding='latin-1')

# --- 1. Get unique categories ---
print("Unique relevance categories:")
print(train_df['relevance'].unique())

# --- 2. Get the count of each category (distribution) ---
print("\nDistribution of relevance categories:")
print(train_df['relevance'].value_counts())

# --- 3. Get the percentage of each category ---
print("\nPercentage distribution of relevance categories:")
print(train_df['relevance'].value_counts(normalize=True) * 100)

# --- 4. Visualize the distribution (optional, but good for EDA) ---
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6, 4))
sns.countplot(x='relevance', data=train_df, palette='viridis')
plt.title('Distribution of Relevance Scores in train.csv')
plt.xlabel('Relevance Score')
plt.ylabel('Count')
plt.show()


import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# --- Configuration for Reduced Dataset ---
# Set this to a smaller number (e.g., 10000, 50000) for faster training
# Set to None or a very large number to use the full dataset
SAMPLE_SIZE = 10000 # Use 10,000 samples for quick testing

# --- NLTK Downloads ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- Data Loading ---
print("Loading data...")
train_df = pd.read_csv('train.csv', encoding='latin-1')
prod_desc_df = pd.read_csv('product_descriptions.csv', encoding='latin-1')
attributes_df = pd.read_csv('attributes.csv', encoding='latin-1')

# --- Apply Sampling if SAMPLE_SIZE is set ---
if SAMPLE_SIZE is not None and SAMPLE_SIZE < len(train_df):
    print(f"Reducing dataset to {SAMPLE_SIZE} samples...")
    train_df = train_df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    print(f"Reduced train_df size: {len(train_df)}")

# --- Data Merging and Preprocessing for Attributes ---
print("Merging data and processing attributes...")
df = pd.merge(train_df, prod_desc_df, on='product_uid', how='left')

# Fill NaN values in 'name' and 'value' columns with empty strings before concatenation
attributes_df['name'] = attributes_df['name'].fillna('')
attributes_df['value'] = attributes_df['value'].fillna('')

attributes_df['attribute_text'] = attributes_df['name'] + ": " + attributes_df['value']
# Filter attributes_df to only include product_uids present in the sampled train_df
product_attributes = attributes_df[attributes_df['product_uid'].isin(df['product_uid'])].groupby('product_uid')['attribute_text'].apply(lambda x: "; ".join(x)).reset_index()
product_attributes.rename(columns={'attribute_text': 'product_attributes'}, inplace=True)

df = pd.merge(df, product_attributes, on='product_uid', how='left')
df['product_attributes'] = df['product_attributes'].fillna('')

# Combine product description and attributes
df['full_product_description'] = df['product_description'].fillna('') + " " + df['product_attributes']
df['product_title'] = df['product_title'].fillna('') # Ensure product_title is not NaN
df['search_term'] = df['search_term'].fillna('') # Ensure search_term is not NaN

# --- Text Preprocessing Function ---
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text, use_stemming=False, use_lemmatization=True):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stop_words]

    if use_stemming:
        tokens = [stemmer.stem(word) for word in tokens]
    elif use_lemmatization:
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

print("Applying text preprocessing...")
df['search_term_processed'] = df['search_term'].apply(preprocess_text)
df['product_title_processed'] = df['product_title'].apply(preprocess_text)
df['full_product_description_processed'] = df['full_product_description'].apply(preprocess_text)

print("Preprocessing complete. Displaying sample:")
print(df[['search_term', 'search_term_processed', 'product_title', 'product_title_processed', 'full_product_description', 'full_product_description_processed', 'relevance']].head())



# --- Exploratory Data Analysis (EDA) ---
print("\n--- Starting EDA ---")

# 1. Relevance Score Distribution (Original)
plt.figure(figsize=(8, 5))
sns.histplot(df['relevance'], bins=20, kde=True)
plt.title('Distribution of Original Relevance Scores (Sampled Data)')
plt.xlabel('Relevance Score')
plt.ylabel('Count')
plt.xticks(np.arange(1, 3.1, 0.25)) # Show more ticks for float values
plt.show()

# 2. Text Length Distributions (Original)
df['search_term_len'] = df['search_term'].apply(len)
df['product_title_len'] = df['product_title'].apply(len)
df['full_product_description_len'] = df['full_product_description'].apply(len)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.histplot(df['search_term_len'], bins=30, kde=True)
plt.title('Search Term Length (Sampled Data)')
plt.subplot(1, 3, 2)
sns.histplot(df['product_title_len'], bins=30, kde=True)
plt.title('Product Title Length (Sampled Data)')
plt.subplot(1, 3, 3)
sns.histplot(df['full_product_description_len'], bins=30, kde=True)
plt.title('Full Product Description Length (Sampled Data)')
plt.tight_layout()
plt.show()

# 3. Word Clouds
def generate_wordcloud(text_series, title):
    text = " ".join(text_series.dropna().astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

generate_wordcloud(df['search_term_processed'], 'Word Cloud for Processed Search Terms (Sampled Data)')
generate_wordcloud(df['product_title_processed'], 'Word Cloud for Processed Product Titles (Sampled Data)')

print("--- EDA Complete ---")


# --- Model Development ---
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.utils.class_weight import compute_class_weight

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# Combine text fields for model input
df['text_input'] = df['search_term_processed'] + " [SEP] " + df['product_title_processed'] + " [SEP] " + df['full_product_description_processed']

# --- Map original float relevance values to 0-indexed integer labels ---
# Get all unique relevance values and sort them
unique_relevance_values = sorted(df['relevance'].unique())
print(f"\nOriginal unique relevance values: {unique_relevance_values}")

# Create a mapping from float value to 0-indexed integer
relevance_to_label = {val: i for i, val in enumerate(unique_relevance_values)}
label_to_relevance = {i: val for i, val in enumerate(unique_relevance_values)}

df['labels'] = df['relevance'].map(relevance_to_label)
num_labels = len(unique_relevance_values)
print(f"Number of classes for classification: {num_labels}")
print(f"Mapping: {relevance_to_label}")

# --- Calculate Class Weights for Imbalanced Data ---
# This will be used in the model's loss function
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(num_labels),
    y=df['labels']
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
print(f"\nCalculated Class Weights: {class_weights.cpu().numpy()}")

# --- Custom Trainer to apply class weights ---
# The default Trainer doesn't directly support class_weights for AutoModelForSequenceClassification
# We need to subclass it and override the compute_loss method
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=0):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# Split data (stratified to maintain class proportions)
# Ensure all classes are present in both train and test sets for stratified split
# Filter out classes with only one sample if necessary, or handle them
# For simplicity, we'll assume enough samples for now, but for very rare classes,
# you might need to group them or ensure they appear in both splits.
try:
    train_df_split, val_df_split = train_test_split(df, test_size=0.2, random_state=42, stratify=df['labels'])
except ValueError as e:
    print(f"Warning: Stratified split failed due to {e}. Falling back to non-stratified split.")
    print("This usually happens if some classes have too few samples to be split.")
    train_df_split, val_df_split = train_test_split(df, test_size=0.2, random_state=42)


# Convert to Hugging Face Dataset format
train_hf_dataset = Dataset.from_pandas(train_df_split[['text_input', 'labels']])
val_hf_dataset = Dataset.from_pandas(val_df_split[['text_input', 'labels']])

# Choose a pre-trained model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# --- Optimize max_length based on data ---
print("\nAnalyzing tokenized input lengths to optimize max_length...")
sample_texts = df['text_input'].sample(min(10000, len(df)), random_state=42).tolist()
tokenized_sample = tokenizer(sample_texts, truncation=False, padding=False)
token_lengths = [len(ids) for ids in tokenized_sample['input_ids']]

p95 = np.percentile(token_lengths, 95)
p99 = np.percentile(token_lengths, 99)
print(f"95th percentile token length: {p95:.0f}")
print(f"99th percentile token length: {p99:.0f}")

MAX_LENGTH = min(int(p99 * 1.1), 512)
print(f"Setting MAX_LENGTH for tokenization to: {MAX_LENGTH}")

def tokenize_function(examples):
    return tokenizer(examples["text_input"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

print("Tokenizing datasets...")
tokenized_train_dataset = train_hf_dataset.map(tokenize_function, batched=True, num_proc=2)
tokenized_val_dataset = val_hf_dataset.map(tokenize_function, batched=True, num_proc=2)

tokenized_train_dataset = tokenized_train_dataset.remove_columns(["text_input"])
tokenized_val_dataset = tokenized_val_dataset.remove_columns(["text_input"])

# Load model for sequence classification (now with num_labels classes)
print(f"Loading model: {model_name} for {num_labels}-class classification...")
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
model.to(device)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1", # Monitor F1-score for best model with imbalanced classes
    greater_is_better=True, # F1 is better when higher
    report_to="none",
    fp16=torch.cuda.is_available(),
)

# Define compute_metrics for evaluation (Classification Metrics)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    # Use zero_division=0 to handle cases where a class has no true samples or no predicted samples
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro', zero_division=0)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# Create CustomTrainer
trainer = CustomTrainer( # Use CustomTrainer here
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)



# Train the model
print("\n--- Starting Model Training (Multi-class Classification) ---")
trainer.train()
print("--- Model Training Complete ---")

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_relevance_multi_classifier_model")
tokenizer.save_pretrained("./fine_tuned_relevance_multi_classifier_model")
print("Model saved to ./fine_tuned_relevance_multi_classifier_model")




# --- Evaluation ---
print("\n--- Starting Model Evaluation (Multi-class Classification) ---")
eval_results = trainer.evaluate()
print(f"Evaluation Results: {eval_results}")

# Get predictions for further analysis
predictions_output = trainer.predict(tokenized_val_dataset)
predicted_logits = predictions_output.predictions
predicted_labels = np.argmax(predicted_logits, axis=-1)
actual_labels = predictions_output.label_ids

# Convert back to original relevance scores for plotting/reporting
predicted_relevance_scores = np.array([label_to_relevance[label] for label in predicted_labels])
actual_relevance_scores = np.array([label_to_relevance[label] for label in actual_labels])

# --- Accuracy Graph (Actual vs. Predicted) ---
plt.figure(figsize=(10, 6))
sns.scatterplot(x=actual_relevance_scores, y=predicted_relevance_scores, alpha=0.6, s=20)
plt.plot([1, 3], [1, 3], 'r--', label='Perfect Prediction')
plt.xlabel("Actual Relevance")
plt.ylabel("Predicted Relevance")
plt.title("Actual vs. Predicted Relevance Scores on Validation Set (Multi-class Classification)")
plt.xticks(unique_relevance_values, rotation=45, ha='right') # Show all unique values
plt.yticks(unique_relevance_values)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

# --- Confusion Matrix ---
print("\n--- Generating Confusion Matrix ---")
# Labels for the confusion matrix (original float values)
cm_display_labels = [f"{val:.2f}" for val in unique_relevance_values] # Format for display

cm = confusion_matrix(actual_labels, predicted_labels, labels=np.arange(num_labels)) # Use 0-indexed labels for confusion_matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cm_display_labels)

fig, ax = plt.subplots(figsize=(12, 12)) # Larger figure for more classes
disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# --- Classification Report ---
print("\n--- Classification Report ---")
# Use np.arange(num_labels) for the 'labels' parameter to ensure all possible classes are considered
# This tells classification_report to expect num_labels classes, even if some are missing in the current batch
print(classification_report(actual_labels, predicted_labels,
                            labels=np.arange(num_labels), # Specify all possible 0-indexed labels
                            target_names=cm_display_labels,
                            zero_division=0))
# --- Analyze Specific Cases (Misclassifications) ---
print("\n--- Analyzing Misclassified Examples ---")
error_analysis_df = pd.DataFrame({
    'search_term': val_df_split['search_term'].values,
    'product_title': val_df_split['product_title'].values,
    'product_description': val_df_split['product_description'].values,
    'actual_relevance': actual_relevance_scores,
    'predicted_relevance': predicted_relevance_scores,
    'is_correct': (actual_relevance_scores == predicted_relevance_scores)
})

print("\nTop 5 misclassified examples (Actual vs. Predicted):")
misclassified_df = error_analysis_df[~error_analysis_df['is_correct']].sort_values(by='actual_relevance').head(5)
print(misclassified_df[['search_term', 'product_title', 'actual_relevance', 'predicted_relevance']])

print("\n--- Evaluation Complete ---")

