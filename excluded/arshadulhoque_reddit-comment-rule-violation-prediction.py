!pip install transformers datasets accelerate torch


def print_header(text, width=55, char="â”€"):
    print(f"\n{char * width}")
    print(f"{text:^{width}}")
    print(f"{char * width}\n")


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, TrainingArguments, Trainer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from accelerate import PartialState
from transformers import EarlyStoppingCallback
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')


# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('vader_lexicon', quiet=True)


distributed_state = PartialState()


def url_to_semantics(text):
    """Extract meaningful keywords from URLs in text and capitalize them."""
    if not isinstance(text, str):
        return ""
    
    url_pattern = re.compile(
        r'https?://'  # http:// or https://
        r'(?:www\.)?'  # optional www.
        r'([^/?]+)'  # domain (group 1)
        r'(?:[^/]*)'  # optional TLD and port
        r'(/[^/?]*)'  # path (group 2)
    )
    
    urls = url_pattern.findall(text)
    
    if not urls:
        return ""
    
    keywords = []
    
    for domain, path in urls:
        domain = domain.split('.')[0]
        if domain and domain not in ['www', 'http', 'https']:
            keywords.append(f"domain:{domain}")
        
        if path and len(path) > 1:
            path_parts = path.strip('/').split('/')
            for part in path_parts:
                if part and not part.isdigit() and len(part) > 2:
                    if part.lower() not in ['jpg', 'jpeg', 'png', 'gif', 'html', 'php', 'asp', 'aspx']:
                        keywords.append(f"path:{part}")
                        break
    
    seen = set()
    unique_keywords = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique_keywords.append(kw)
    
    if unique_keywords:
        # Capitalize the keywords and the prefix
        capitalized_keywords = [kw.upper() for kw in unique_keywords]
        return "URL KEYWORDS: " + " ".join(capitalized_keywords)
    else:
        return ""


def extract_keywords(text):
    """Extract important keywords from text."""
    if not isinstance(text, str):
        return []
    
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    keywords = [word for word in tokens if word.isalpha() and word not in stop_words and len(word) > 2]
    return keywords

def calculate_jaccard_similarity(list1, list2):
    """Calculate Jaccard similarity between two lists."""
    set1 = set(list1)
    set2 = set(list2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0

def get_sentiment_features(text):
    """Extract sentiment features from text."""
    if not isinstance(text, str):
        return 0, 0
    
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    return scores['compound'], scores['pos'] - scores['neg']


# --- This is your ORIGINAL function. It remains unchanged. ---
# It's excellent for creating a dense, numerical feature summary.
def create_example_features(row):
    """
    Create a string of engineered features from positive and negative examples.
    This function is perfect as is and will be used by the new function.
    """
    # Extract keywords from all examples
    pos1_keywords = extract_keywords(row['positive_example_1'])
    pos2_keywords = extract_keywords(row['positive_example_2'])
    neg1_keywords = extract_keywords(row['negative_example_1'])
    neg2_keywords = extract_keywords(row['negative_example_2'])
    
    # Combine positive and negative keywords
    all_pos_keywords = pos1_keywords + pos2_keywords
    all_neg_keywords = neg1_keywords + neg2_keywords
    
    # Extract keywords from comment
    comment_keywords = extract_keywords(row['body'])
    
    # Calculate keyword overlaps
    pos_overlap = calculate_jaccard_similarity(comment_keywords, all_pos_keywords)
    neg_overlap = calculate_jaccard_similarity(comment_keywords, all_neg_keywords)
    
    # Calculate sentiment features
    comment_compound, comment_sent_diff = get_sentiment_features(row['body'])
    pos1_compound, _ = get_sentiment_features(row['positive_example_1'])
    pos2_compound, _ = get_sentiment_features(row['positive_example_2'])
    neg1_compound, _ = get_sentiment_features(row['negative_example_1'])
    neg2_compound, _ = get_sentiment_features(row['negative_example_2'])
    
    # Average sentiment of positive and negative examples
    avg_pos_sentiment = (pos1_compound + pos2_compound) / 2
    avg_neg_sentiment = (neg1_compound + neg2_compound) / 2
    
    # Sentiment differences
    pos_sent_diff = abs(comment_compound - avg_pos_sentiment)
    neg_sent_diff = abs(comment_compound - avg_neg_sentiment)
    
    # Length features
    comment_len = len(row['body'].split())
    pos1_len = len(row['positive_example_1'].split())
    pos2_len = len(row['positive_example_2'].split())
    neg1_len = len(row['negative_example_1'].split())
    neg2_len = len(row['negative_example_2'].split())
    
    avg_pos_len = (pos1_len + pos2_len) / 2
    avg_neg_len = (neg1_len + neg2_len) / 2
    
    # Length ratios
    pos_len_ratio = comment_len / avg_pos_len if avg_pos_len > 0 else 0
    neg_len_ratio = comment_len / avg_neg_len if avg_neg_len > 0 else 0
    
    # Create feature string
    features = (
        f"Example Features: "
        f"pos_overlap:{pos_overlap:.3f} "
        f"neg_overlap:{neg_overlap:.3f} "
        f"pos_sent_diff:{pos_sent_diff:.3f} "
        f"neg_sent_diff:{neg_sent_diff:.3f} "
        f"pos_len_ratio:{pos_len_ratio:.3f} "
        f"neg_len_ratio:{neg_len_ratio:.3f} "
        f"comment_sentiment:{comment_sent_diff:.3f}"
    )
    
    return features


def create_combined_text(row):
    """
    Creates a combined text string using a hybrid approach with capitalization.
    It includes the subreddit, raw text of the rule and examples for semantic understanding,
    and appends the engineered features as a dense summary. All text from the dataset is capitalized.
    """
    # Capitalize all relevant text fields from the dataset
    subreddit_text = str(row['subreddit']).upper()
    rule_text = str(row['rule']).upper()
    pos_examples = f"{str(row['positive_example_1']).upper()} {str(row['positive_example_2']).upper()}"
    neg_examples = f"{str(row['negative_example_1']).upper()} {str(row['negative_example_2']).upper()}"
    body_text = str(row['body']).upper()

    # Get the URL semantics (now capitalized by the updated function above)
    url_info = url_to_semantics(row['body'])

    # Call the original function to get the engineered feature string
    features_str = create_example_features(row)

    # Build the combined string with SUBREDDIT first and all labels capitalized
    combined_text = (
        f"SUBREDDIT: {subreddit_text} [SEP] "
        f"RULE: {rule_text} [SEP] "
        f"POSITIVE EXAMPLES: {pos_examples} [SEP] "
        f"NEGATIVE EXAMPLES: {neg_examples} [SEP] "
        f"COMMENT: {body_text} {url_info} [SEP] "
        f"{features_str.upper()}" # Capitalize the engineered feature string for consistency
    )
    return combined_text


# Load data
train_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')


print("Creating combined text with hybrid approach...")
train_df['combined_text'] = train_df.apply(create_combined_text, axis=1)

# Prepare data for k-fold
texts = train_df['combined_text'].tolist()
labels = train_df['rule_violation'].tolist()


class RedditDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=1024):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


# Initialize tokenizer
model_name = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create configuration
config = AutoConfig.from_pretrained(model_name)
config.hidden_dropout_prob = 0.3
config.attention_probs_dropout_prob = 0.3
config.num_labels = 2
config.problem_type = "single_label_classification"


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    auc = roc_auc_score(labels, probs)
    return {"auc": auc}


# Fixed CustomTrainer
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_losses = []
        self.validation_losses = []
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        labels = labels.long()
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        
        self.training_losses.append(loss.item())
        
        return (loss, outputs) if return_outputs else loss
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_output = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        if "eval_loss" in eval_output:
            self.validation_losses.append(eval_output["eval_loss"])
        
        return eval_output


def analyze_kfold_performance(all_fold_results):
    """
    Analyze model performance across k-fold cross-validation to detect underfitting/overfitting
    """
    print("\n" + "="*60)
    print("K-FOLD CROSS-VALIDATION ANALYSIS")
    print("="*60)
    
    # Extract AUC scores
    auc_scores = [result['auc'] for result in all_fold_results]
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    
    print(f"Mean AUC: {mean_auc:.4f} Â± {std_auc:.4f}")
    print(f"Individual Fold AUCs: {[f'{auc:.4f}' for auc in auc_scores]}")
    
    # Find the maximum number of epochs across folds
    max_epochs = max(result['epochs'] for result in all_fold_results)
    
    # Initialize arrays to store average losses per epoch across folds
    avg_train_loss = np.zeros(max_epochs)
    avg_val_loss = np.zeros(max_epochs)
    epoch_counts = np.zeros(max_epochs)
    
    # Calculate average losses per epoch across folds
    for result in all_fold_results:
        train_losses = result['train_loss']  # list of training losses per step
        val_losses = result['val_loss']      # list of validation losses per epoch
        num_epochs = len(val_losses)
        
        # Split training losses into epochs
        # We need to handle the case where the number of training losses might not be evenly divisible
        if len(train_losses) > 0:
            # Calculate approximate number of training steps per epoch
            steps_per_epoch = len(train_losses) // num_epochs
            if steps_per_epoch == 0:
                steps_per_epoch = 1
                
            # Split training losses into epochs
            train_losses_per_epoch = []
            for i in range(num_epochs):
                start_idx = i * steps_per_epoch
                end_idx = (i + 1) * steps_per_epoch
                if i == num_epochs - 1:  # Last epoch takes all remaining
                    end_idx = len(train_losses)
                epoch_losses = train_losses[start_idx:end_idx]
                if epoch_losses:  # Only add if there are losses
                    train_losses_per_epoch.append(np.mean(epoch_losses))
                else:
                    train_losses_per_epoch.append(0.0)
            
            # Ensure we have the same number of epochs as validation losses
            if len(train_losses_per_epoch) < num_epochs:
                train_losses_per_epoch.extend([0.0] * (num_epochs - len(train_losses_per_epoch)))
        else:
            train_losses_per_epoch = [0.0] * num_epochs
        
        for epoch in range(num_epochs):
            if epoch < max_epochs:
                avg_train_loss[epoch] += train_losses_per_epoch[epoch]
                avg_val_loss[epoch] += val_losses[epoch]
                epoch_counts[epoch] += 1
    
    # Calculate averages
    for epoch in range(max_epochs):
        if epoch_counts[epoch] > 0:
            avg_train_loss[epoch] /= epoch_counts[epoch]
            avg_val_loss[epoch] /= epoch_counts[epoch]
    
    # Find best epoch (lowest validation loss)
    best_epoch = np.argmin(avg_val_loss)
    best_train_loss = avg_train_loss[best_epoch]
    best_val_loss = avg_val_loss[best_epoch]
    
    print(f"\nBest Performance at Epoch {best_epoch+1}:")
    print(f"Training Loss: {best_train_loss:.4f}")
    print(f"Validation Loss: {best_val_loss:.4f}")
    print(f"Loss Gap (Val - Train): {best_val_loss - best_train_loss:.4f}")
    
    # Analyze underfitting/overfitting
    print("\n" + "-"*60)
    print("MODEL STATUS ANALYSIS")
    print("-"*60)
    
    # Check for underfitting
    if best_train_loss > 0.5 and best_val_loss > 0.5:
        print("Status: UNDERFITTING")
        print("Reasons: Both training and validation losses are high")
        print("Recommendations:")
        print("- Increase model capacity (use larger model)")
        print("- Train for more epochs")
        print("- Reduce regularization")
        print("- Increase learning rate")
    
    # Check for overfitting
    elif (best_val_loss - best_train_loss) > 0.1:
        print("Status: OVERFITTING")
        print("Reasons: Validation loss is significantly higher than training loss")
        print("Recommendations:")
        print("- Add more dropout")
        print("- Increase weight decay")
        print("- Use early stopping with lower patience")
        print("- Add data augmentation")
        print("- Reduce model complexity")
    
    # Check for good fit
    elif best_train_loss < 0.4 and best_val_loss < 0.4 and abs(best_val_loss - best_train_loss) < 0.05:
        print("Status: GOOD FIT")
        print("Reasons: Both losses are low and close to each other")
        print("Recommendations:")
        print("- Continue monitoring for overfitting")
        print("- Consider fine-tuning hyperparameters")
        print("- Try ensemble methods for further improvement")
    
    # Check for under-optimized
    else:
        print("Status: UNDER-OPTIMIZED")
        print("Reasons: Model has potential but needs more optimization")
        print("Recommendations:")
        print("- Train for more epochs")
        print("- Adjust learning rate")
        print("- Try different optimizers")
        print("- Tune hyperparameters more carefully")
    
    # Analyze consistency across folds
    print("\n" + "-"*60)
    print("FOLD CONSISTENCY ANALYSIS")
    print("-"*60)
    
    if std_auc < 0.02:
        print("Consistency: VERY HIGH")
        print("The model performs consistently across all folds")
    elif std_auc < 0.05:
        print("Consistency: HIGH")
        print("The model performs quite consistently across folds")
    elif std_auc < 0.1:
        print("Consistency: MODERATE")
        print("There's some variation in performance across folds")
    else:
        print("Consistency: LOW")
        print("Performance varies significantly across folds")
        print("Recommendations:")
        print("- Check for data distribution issues")
        print("- Consider stratified sampling")
        print("- Increase training data")
    
    # Plot loss curves
    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")
    
    epochs = range(1, max_epochs + 1)
    sns.lineplot(x=epochs, y=avg_train_loss, label='Average Training Loss', marker='o', linewidth=2)
    sns.lineplot(x=epochs, y=avg_val_loss, label='Average Validation Loss', marker='s', linewidth=2)
    
    # Mark best epoch
    plt.axvline(x=best_epoch+1, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch+1})')
    
    plt.title('Average Training vs Validation Loss Across Folds', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.xticks(epochs)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('./loss_analysis.png')
    plt.show()
    
    # Plot individual fold performances
    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")
    
    fold_nums = [result['fold'] for result in all_fold_results]
    fold_aucs = [result['auc'] for result in all_fold_results]
    
    sns.barplot(x=fold_nums, y=fold_aucs, palette='viridis')
    plt.axhline(y=mean_auc, color='red', linestyle='--', label=f'Mean AUC ({mean_auc:.4f})')
    
    plt.title('AUC Scores Across Folds', fontsize=16)
    plt.xlabel('Fold', fontsize=14)
    plt.ylabel('AUC', fontsize=14)
    plt.ylim(0.5, 1.0)  # AUC range
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('./fold_performance.png')
    plt.show()
    
    return {
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'best_epoch': best_epoch,
        'best_train_loss': best_train_loss,
        'best_val_loss': best_val_loss,
        'loss_gap': best_val_loss - best_train_loss
    }


# Set up k-fold cross-validation
n_splits = 5 
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Track best model across folds
best_auc = 0
best_fold = None
best_model_path = "./results/best_model"  # Final best model
best_fold_model_dir = None  # For backup of best fold's full dir

# Store results for analysis
all_fold_results = []

# Create output directories
os.makedirs("./results", exist_ok=True)

print_header("Starting K-Fold Cross-Validation")


# --- K-Fold Loop ---
for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
    print_header(f"Training Fold {fold+1}/{n_splits}")

    # Split data
    train_texts_fold = [texts[i] for i in train_idx]
    train_labels_fold = [labels[i] for i in train_idx]
    val_texts_fold = [texts[i] for i in val_idx]
    val_labels_fold = [labels[i] for i in val_idx]

    # Create datasets
    train_dataset = RedditDataset(train_texts_fold, train_labels_fold, tokenizer)
    val_dataset = RedditDataset(val_texts_fold, val_labels_fold, tokenizer)

    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    # Output dir per fold
    fold_output_dir = f"./results/fold_{fold}"
    os.makedirs(fold_output_dir, exist_ok=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"{fold_output_dir}/checkpoints",
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.1,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        logging_dir=f"{fold_output_dir}/logs",
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="auc", 
        greater_is_better=True,
        report_to="none",
        fp16=True,
        bf16=False,
        local_rank=-1,
        ddp_find_unused_parameters=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # --- Training ---
    try:
        trainer.train()
    except Exception as e:
        print(f"â�Œ Training failed on fold {fold+1}: {e}")
        del model, trainer
        torch.cuda.empty_cache()
        continue

    # --- Evaluation ---
    eval_result = trainer.evaluate()
    fold_auc = eval_result["eval_auc"]
    print(f"Fold {fold+1} AUC: {fold_auc:.4f}")

    # Extract losses
    train_losses = [log["loss"] for log in trainer.state.log_history if "loss" in log and "eval_loss" not in log]
    val_losses = [log["eval_loss"] for log in trainer.state.log_history if "eval_loss" in log]

    # Save fold results
    fold_result = {
        'fold': fold + 1,
        'auc': fold_auc,
        'train_loss': train_losses,
        'val_loss': val_losses,
        'epochs': len(val_losses),
        'early_stopped': len(val_losses) < training_args.num_train_epochs
    }
    all_fold_results.append(fold_result)

    # --- Save Best Model Across Folds ---
    if fold_auc > best_auc:
        if os.path.exists(best_model_path):
            backup_path = f"{best_model_path}_backup_fold{best_fold}"
            if os.path.exists(backup_path):
                shutil.rmtree(backup_path)
            shutil.move(best_model_path, backup_path)
            print(f"ğŸ“¦ Previous best model backed up to {backup_path}")

        if os.path.exists(best_model_path):
            shutil.rmtree(best_model_path)
        trainer.save_model(best_model_path)
        tokenizer.save_pretrained(best_model_path)
        best_auc = fold_auc
        best_fold = fold
        print(f"âœ… New best model saved! Fold {fold+1} with AUC: {fold_auc:.4f}")

    # --- Cleanup ---
    checkpoint_dir = f"{fold_output_dir}/checkpoints"
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)

    del model, trainer
    torch.cuda.empty_cache()
    print(f"ğŸ§  Memory cleaned for fold {fold+1}")

# --- End of Loop ---
print_header("K-Fold CV Completed!")

# --- Final Summary ---
print(f"\nğŸ�† Best Fold: {best_fold + 1} | AUC: {best_auc:.4f}")
print(f"ğŸ“Š All Fold AUCs: {[r['auc'] for r in all_fold_results]}")
print(f"ğŸ“ˆ Mean AUC: {np.mean([r['auc'] for r in all_fold_results]):.4f} Â± {np.std([r['auc'] for r in all_fold_results]):.4f}")


# --- Final Analysis ---
analysis_results = analyze_kfold_performance(all_fold_results)

print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"Best Model: Fold {best_fold+1}")
print(f"Best AUC: {best_auc:.4f}")
print(f"Mean AUC across folds: {analysis_results['mean_auc']:.4f}")
print(f"Best Epoch: {analysis_results['best_epoch']+1}")
print(f"Loss Gap: {analysis_results['loss_gap']:.4f}")

# Save analysis results
with open('./analysis_results.txt', 'w') as f:
    f.write(f"Best Fold: {best_fold+1}\n")
    f.write(f"Best AUC: {best_auc:.4f}\n")
    f.write(f"Mean AUC: {analysis_results['mean_auc']:.4f}\n")
    f.write(f"Std AUC: {analysis_results['std_auc']:.4f}\n")
    f.write(f"Best Epoch: {analysis_results['best_epoch']+1}\n")
    f.write(f"Best Training Loss: {analysis_results['best_train_loss']:.4f}\n")
    f.write(f"Best Validation Loss: {analysis_results['best_val_loss']:.4f}\n")
    f.write(f"Loss Gap: {analysis_results['loss_gap']:.4f}\n")

print("âœ… Pipeline complete. Best model at:", best_model_path)
print("ğŸ“� Full results and logs in ./results/")

