#
import os
# CUDA_LAUNCH_BLOCKING = "1" # Debug environment variable to see proper traces.
# os.environ["CUDA_VISIBLE_DEVICES"]="0" # GPU to used. 0 means use GPU 1.
os.environ["WANDB_DISABLED"] = "true" # The primary switch
os.environ["WANDB_MODE"] = "offline"

import re
import random
import logging
from tqdm import tqdm
from pathlib import Path
from functools import lru_cache

#
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split 
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, roc_auc_score, roc_curve
#
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, Adam

from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification
# from transformers import DistilBertModel, DistilBertTokenizer
# from transformers import BertTokenizer, BertModel


# ignore warnings
from transformers import logging as trans_logging
trans_logging.set_verbosity_error()

import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)


"""This is equivalent to config.py. On google colab this class is used to hold
all the configurations to be used throughout the project.
"""
class Config:
    VER = 1
    SEED = 3407
    CUDA_AVAILABLE = False #  gpu or cpu? to be set later
    DEVICE = None # based on CUDA_AVAILABLE
    DATA_DIR = Path("/kaggle/input/jigsaw-agile-community-rules")
    OUTPUT_DIR = Path("Output")
    LOGS_DIR = Path(OUTPUT_DIR, "logs")
    MODELS_DIR = Path(OUTPUT_DIR, f"Jigsaw_agile_community_rule_classifier_ver_{VER}")
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOAD_TOKENS_FROM = None
    LOAD_MODEL_FROM = None
    DOWNLOADED_MODEL_PATH = None


    # Cross-validation settings
    N_SPLITS = 5  # Number of folds for cross-validation
    RANDOM_STATE = 42


def initialize_logger(log_file: str = Path(Config.LOGS_DIR, "info.log")):

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    file_handler = logging.FileHandler(filename=log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger

Config.logger = initialize_logger()


Config.CUDA_AVAILABLE = torch.cuda.is_available()
Config.DEVICE = torch.device("cuda" if Config.CUDA_AVAILABLE else "cpu")
Config.logger.info(f"We are using {Config.DEVICE}")


# https://odsc.medium.com/properly-setting-the-random-seed-in-ml-experiments-not-as-simple-as-you-might-imagine-219969c84752

def set_seed(seed: int = Config.SEED) -> None:
    """Seed all random number generators."""
    os.environ["PYTHONHASHSEED"] = str(seed)  # set PYTHONHASHSEED env var at fixed value
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)  # pytorch (both CPU and CUDA)
    np.random.seed(seed)  # for numpy pseudo-random generator

    # set fixed value for python built-in pseudo-random generator
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    Config.logger.info(f"Using Seed Number: {seed}")


set_seed()


"""
The classes below tracks different parameters to be used through out the project.
The idea is to make a change only here and not all the part where these
variables can be used.
"""
class FilePaths:
    train_csv = Path(Config.DATA_DIR, "train.csv")
    test_csv = Path(Config.DATA_DIR, "test.csv")
    submit_csv = Path(Config.DATA_DIR, "sample_submission.csv")


class ModelParams:
    MAX_LEN = 512 # TODO: Verify this, 512
    model_name = "/kaggle/input/huggingfacedebertav3variants/deberta-v3-large"
    output_len = 2



class DataLoaderParams:
    TRAIN_BATCH_SIZE = 2 #8
    VALID_BATCH_SIZE = 2 #4
    train_loader = {
            "batch_size": TRAIN_BATCH_SIZE,
            "num_workers": 4,
            "pin_memory": False,
            "drop_last": True,
            "shuffle": True,
            "collate_fn": None
    }

    valid_loader = {
            "batch_size": VALID_BATCH_SIZE,
            "num_workers": 4,
            "pin_memory": False,
            "drop_last": False,
            "shuffle": False,
            "collate_fn": None
    }

    test_loader = {
            "batch_size": VALID_BATCH_SIZE,
            "num_workers": 4,
            "pin_memory": True,
            "drop_last": False,
            "shuffle": False,
            "collate_fn": None
    }


class GlobalTrainParams:
    debug: bool = False
    epochs: int = 7


class CriterionParams:
    loss_function_name = "CrossEntropyLoss"


class OptimizerParams:
    """A class to track optimizer parameters.
    """
    optimizer_name = "Adam"
    lr = [1e-5, 3e-5, 2e-5, 2.5e-5, 2.5e-6, 2.5e-6, 2.5e-7]



class TokenizerParams:
    LOAD_TOKENS_FROM = None
    tokenizer_name = ModelParams().model_name
    # lower_case = False # for bert_base_cased
    lower_case = True # for distillbert_base_uncased
    max_length = ModelParams().MAX_LEN
    truncation = True
    padding = "max_length"


FILES = FilePaths()
LOADER_PARAMS = DataLoaderParams()
TRAIN_PARAMS = GlobalTrainParams()
CRITERION_PARAMS = CriterionParams()
OPTIMIZER_PARAMS = OptimizerParams()
MODEL_PARAMS = ModelParams()
TOKENIZER_PARAMS = TokenizerParams()


# Download tokenizer if not already downloaded and saved.
if Config.DOWNLOADED_MODEL_PATH is None:
  # TOKENIZER = BertTokenizer.from_pretrained(TOKENIZER_PARAMS.tokenizer_name)
  # TOKENIZER = DistilBertTokenizer.from_pretrained(TOKENIZER_PARAMS.tokenizer_name, truncation=True, do_lower_case=True)
  # TOKENIZER.save_pretrained(Config.MODELS_DIR)
  from transformers import AutoTokenizer
  TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_PARAMS.tokenizer_name)



def url_to_semantics(text: str) -> str:
    """
    Input : A text string (possibly containing URLs)
    Output: String containing keywords like 'domain:reddit path:comment'
    Logic :
        - Find URLs with regex
        - Extract domain and path parts
        - Add as 'domain:' and 'path:' tokens
        - Helps model learn URL meaning without raw URL noise
    """
    if not isinstance(text, str):
        return ""

    url_pattern = r'https?://[^\s/$.?#].[^\s]*'
    urls = re.findall(url_pattern, text)
    
    if not urls:
        return "" 

    all_semantics = []
    seen_semantics = set()

    for url in urls:
        url_lower = url.lower()
        
        domain_match = re.search(r"(?:https?://)?([a-z0-9\-\.]+)\.[a-z]{2,}", url_lower)
        if domain_match:
            full_domain = domain_match.group(1)
            parts = full_domain.split('.')
            for part in parts:
                if part and part not in seen_semantics and len(part) > 3: # Avoid short parts like 'www'
                    all_semantics.append(f"domain:{part}")
                    seen_semantics.add(part)

        # 2. Extract path parts
        path = re.sub(r"^(?:https?://)?[a-z0-9\.-]+\.[a-z]{2,}/?", "", url_lower)
        path_parts = [p for p in re.split(r'[/_.-]+', path) if p and p.isalnum()] # Split by common delimiters

        for part in path_parts:
            # Clean up potential file extensions or query params
            part_clean = re.sub(r"\.(html?|php|asp|jsp)$|#.*|\?.*", "", part)
            if part_clean and part_clean not in seen_semantics and len(part_clean) > 3:
                all_semantics.append(f"path:{part_clean}")
                seen_semantics.add(part_clean)

    if not all_semantics:
        return ""

    return f"\nURL Keywords: {' '.join(all_semantics)}"


def get_dataframe_to_train(data_path):
    """
    Input : Folder path containing 'train.csv' and 'test.csv'
    Output: Cleaned Pandas DataFrame with ['body','rule','subreddit','rule_violation']
    Logic :
        - Read both train/test CSVs
        - Flatten positive and negative examples into a single frame
        - Label positive â†’ 1, negative â†’ 0
        - Drop NA/empty texts
        - Remove duplicates by ['body','rule','subreddit']
        - Shuffle dataframe with random_state=42
    """
    train_dataset = pd.read_csv(f"{data_path}") 
    test_dataset = pd.read_csv(f"{data_path}")

    flatten = []

    # flatten.append(train_dataset[["body", "rule", "subreddit","rule_violation"]].copy())

    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            col_name = f"{violation_type}_example_{i}"
            
            if col_name in train_dataset.columns:
                sub_dataset = train_dataset[[col_name, "rule", "subreddit"]].copy()
                sub_dataset = sub_dataset.rename(columns={col_name: "body"})
                sub_dataset["rule_violation"] = 1 if violation_type == "positive" else 0
                
                sub_dataset.dropna(subset=['body'], inplace=True)
                sub_dataset = sub_dataset[sub_dataset['body'].str.strip().str.len() > 0]
                
                if not sub_dataset.empty:
                    flatten.append(sub_dataset)
    
    for violation_type in ["positive", "negative"]:
        for i in range(1, 3):
            col_name = f"{violation_type}_example_{i}"
            
            if col_name in test_dataset.columns:
                sub_dataset = test_dataset[[col_name, "rule", "subreddit"]].copy()
                sub_dataset = sub_dataset.rename(columns={col_name: "body"})
                sub_dataset["rule_violation"] = 1 if violation_type == "positive" else 0
                
                sub_dataset.dropna(subset=['body'], inplace=True)
                sub_dataset = sub_dataset[sub_dataset['body'].str.strip().str.len() > 0]
                
                if not sub_dataset.empty:
                    flatten.append(sub_dataset)
    
    dataframe = pd.concat(flatten, axis=0)
    dataframe = dataframe.drop_duplicates(subset=['body', 'rule', 'subreddit'], ignore_index=True)
    dataframe.drop_duplicates(subset=['body','rule'],keep='first',inplace=True)
    
    return dataframe.sample(frac=1, random_state=42).reset_index(drop=True)


class JigsawDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


training_data_df = get_dataframe_to_train(FILES.train_csv)
test_df_for_prediction = pd.read_csv(FILES.test_csv)

print(training_data_df.shape)
print(test_df_for_prediction.shape)


training_data_df['body_with_url'] = training_data_df['body'].apply(lambda x: x + url_to_semantics(x))
training_data_df['input_text'] = training_data_df['rule'] + "[SEP]" + training_data_df['body_with_url']


def create_comprehensive_dashboard(cv_results, fold_predictions, filename='cv_dashboard.png'):
    """Create a comprehensive dashboard with all plots in one image"""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Define the grid layout
    gs = plt.GridSpec(3, 3, figure=fig)
    
    # Plot 1: CV Metrics across folds (top left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    folds = list(range(1, len(cv_results) + 1))
    metrics = ['f1', 'precision', 'recall', 'auc']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    x = np.arange(len(folds))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [result[metric] for result in cv_results]
        ax1.bar(x + i*width, values, width, label=metric.upper(), color=colors[i], alpha=0.8)
        
        # Add value labels
        for j, v in enumerate(values):
            ax1.text(j + i*width, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('Score')
    ax1.set_title('Cross-Validation Metrics Across Folds', fontsize=14, fontweight='bold')
    ax1.set_xticks(x + width*1.5)
    ax1.set_xticklabels(folds)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)
    
    # Plot 2: ROC Curves for all folds (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    for i, fold_pred in enumerate(fold_predictions):
        fpr, tpr, _ = roc_curve(fold_pred['true_labels'], fold_pred['probabilities'])
        auc_score = cv_results[i]['auc']
        ax2.plot(fpr, tpr, alpha=0.7, linewidth=2, label=f'Fold {i+1} (AUC = {auc_score:.3f})')
    
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curves - All Folds', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Plot 3: Metric distributions (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    metric_data = {metric: [result[metric] for result in cv_results] for metric in metrics}
    box_plot = ax3.boxplot([metric_data[metric] for metric in metrics], 
                          labels=[m.upper() for m in metrics],
                          patch_artist=True)
    
    # Color the boxes
    colors_box = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    for patch, color in zip(box_plot['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax3.set_ylabel('Score')
    ax3.set_title('Metric Distributions Across Folds', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # Plot 4: Performance summary table (middle center)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('tight')
    ax4.axis('off')
    
    # Calculate summary statistics
    summary_data = []
    for metric in metrics:
        values = [result[metric] for result in cv_results]
        summary_data.append([
            metric.upper(),
            f'{np.mean(values):.4f}',
            f'{np.std(values):.4f}',
            f'{np.min(values):.4f}',
            f'{np.max(values):.4f}'
        ])
    
    table = ax4.table(cellText=summary_data,
                     colLabels=['Metric', 'Mean', 'Std', 'Min', 'Max'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax4.set_title('Performance Summary', fontsize=14, fontweight='bold')
    
    # Plot 5: Training loss curves if available (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    # This would require storing loss history during training
    ax5.text(0.5, 0.5, 'Training Loss Curves\n(Enable logging to see)', 
             ha='center', va='center', transform=ax5.transAxes, fontsize=12)
    ax5.set_title('Training Progress', fontsize=14, fontweight='bold')
    ax5.axis('off')
    
    # Plot 6: Confusion matrix for best fold (bottom left)
    ax6 = fig.add_subplot(gs[2, 0])
    best_fold_idx = np.argmax([result['f1'] for result in cv_results])
    best_fold_pred = fold_predictions[best_fold_idx]
    
    cm = confusion_matrix(best_fold_pred['true_labels'], best_fold_pred['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax6,
                xticklabels=['No Violation', 'Violation'],
                yticklabels=['No Violation', 'Violation'])
    ax6.set_title(f'Confusion Matrix - Best Fold (Fold {best_fold_idx + 1})', 
                 fontsize=14, fontweight='bold')
    ax6.set_xlabel('Predicted')
    ax6.set_ylabel('Actual')
    
    # Plot 7: Class distribution (bottom center)
    ax7 = fig.add_subplot(gs[2, 1])
    all_true_labels = np.concatenate([fp['true_labels'] for fp in fold_predictions])
    class_counts = [np.sum(all_true_labels == 0), np.sum(all_true_labels == 1)]
    colors_pie = ['#FF6B6B', '#4ECDC4']
    ax7.pie(class_counts, labels=['No Violation', 'Violation'], autopct='%1.1f%%',
            colors=colors_pie, startangle=90)
    ax7.set_title('Overall Class Distribution', fontsize=14, fontweight='bold')
    
    # Plot 8: Fold-wise sample sizes (bottom right)
    ax8 = fig.add_subplot(gs[2, 2])
    train_sizes = [result['train_size'] for result in cv_results]
    val_sizes = [result['val_size'] for result in cv_results]
    
    x = np.arange(len(folds))
    ax8.bar(x - 0.2, train_sizes, 0.4, label='Training', color='#2E86AB', alpha=0.8)
    ax8.bar(x + 0.2, val_sizes, 0.4, label='Validation', color='#A23B72', alpha=0.8)
    
    ax8.set_xlabel('Fold')
    ax8.set_ylabel('Number of Samples')
    ax8.set_title('Sample Sizes per Fold', fontsize=14, fontweight='bold')
    ax8.set_xticks(x)
    ax8.set_xticklabels(folds)
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # Add overall title
    plt.suptitle('DeBERTa Cross-Validation Analysis Dashboard', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"âœ“ Comprehensive dashboard saved as {filename}")


def print_cv_summary(cv_results):
    """Print comprehensive cross-validation summary"""
    
    print("\n" + "="*60)
    print("CROSS-VALIDATION SUMMARY")
    print("="*60)
    
    metrics = ['precision', 'recall', 'f1', 'auc']
    
    for metric in metrics:
        values = [result[metric] for result in cv_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        print(f"\nğŸ“Š {metric.upper()}:")
        print(f"  Mean: {mean_val:.4f} Â± {std_val:.4f}")
        print(f"  Range: {min(values):.4f} - {max(values):.4f}")
        print(f"  Fold values: {[f'{v:.4f}' for v in values]}")
    
    # Overall summary
    f1_scores = [result['f1'] for result in cv_results]
    auc_scores = [result['auc'] for result in cv_results]
    
    print(f"\nğŸ�¯ OVERALL PERFORMANCE:")
    print(f"  Mean F1-Score: {np.mean(f1_scores):.4f} Â± {np.std(f1_scores):.4f}")
    print(f"  Mean AUC-ROC:  {np.mean(auc_scores):.4f} Â± {np.std(auc_scores):.4f}")
    
    stability = np.std(f1_scores)
    if stability < 0.03:
        stability_text = "Excellent"
    elif stability < 0.06:
        stability_text = "Good"
    elif stability < 0.1:
        stability_text = "Moderate"
    else:
        stability_text = "Variable"
    print(f"  Model Stability: {stability_text} (std: {stability:.4f})")


def evaluate_model(trainer, dataset, true_labels, rules, rule_names):
    """Comprehensive evaluation of the model"""
    
    predictions = trainer.predict(dataset)
    pred_probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    
    # Convert to numpy arrays for indexing
    true_labels = np.array(true_labels)
    rules = np.array(rules)
    
    # Overall metrics
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='binary')
    auc_score = roc_auc_score(true_labels, pred_probs[:, 1].numpy())
    cm = confusion_matrix(true_labels, pred_labels)
    
    return {
        'predictions': pred_labels,
        'probabilities': pred_probs[:, 1].numpy(),
        'true_labels': true_labels,
        'confusion_matrix': cm,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc_score,
        'classification_report': classification_report(true_labels, pred_labels)
    }


from torch.optim import AdamW
from transformers import Trainer, TrainingArguments, get_scheduler

def get_optimizer_grouped_parameters(model, base_lr, weight_decay, lr_decay):
    """
    Create optimizer parameter groups with layer-wise learning rate decay.
    """
    layers = [model.deberta.encoder.layer[i] for i in range(model.config.num_hidden_layers)]
    layers = list(reversed(layers))  # Top layers first (get higher LR)

    optimizer_parameters = []

    # Embedding layer
    optimizer_parameters += [{
        "params": model.deberta.embeddings.parameters(),
        "lr": base_lr * (lr_decay ** (len(layers) + 1)),
        "weight_decay": weight_decay
    }]

    # Transformer layers
    for i, layer in enumerate(layers):
        lr = base_lr * (lr_decay ** i)
        optimizer_parameters += [{
            "params": layer.parameters(),
            "lr": lr,
            "weight_decay": weight_decay
        }]

    # Classifier head (final layer)
    optimizer_parameters += [{
        "params": model.classifier.parameters(),
        "lr": base_lr,
        "weight_decay": 0.0
    }]

    return optimizer_parameters




# Run cross-validation
skf = StratifiedKFold(n_splits=Config.N_SPLITS, shuffle=True, random_state=Config.RANDOM_STATE)
cv_results = []
fold_predictions = []

print(f"Starting {Config.N_SPLITS}-fold cross-validation...")
print("=" * 60)

for fold, (train_idx, val_idx) in enumerate(skf.split(training_data_df, training_data_df['rule']), 1):
    print(f"\nğŸ�¯ FOLD {fold}/{Config.N_SPLITS}")
    print("-" * 40)
    
    # Split data
    train_df = training_data_df.iloc[train_idx].reset_index(drop=True)
    val_df = training_data_df.iloc[val_idx].reset_index(drop=True)

    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Class balance - Train: {train_df['rule_violation'].value_counts().to_dict()}")
    print(f"Class balance - Val: {val_df['rule_violation'].value_counts().to_dict()}")


    # Tokenize
    train_encodings = TOKENIZER(
        train_df['input_text'].tolist(), 
        truncation=TOKENIZER_PARAMS.truncation,
        padding=TOKENIZER_PARAMS.padding,
        max_length=TOKENIZER_PARAMS.max_length)
    val_encodings = TOKENIZER(
        val_df['input_text'].tolist(), 
        truncation=TOKENIZER_PARAMS.truncation,
        padding=TOKENIZER_PARAMS.padding,
        max_length=TOKENIZER_PARAMS.max_length)

    train_dataset = JigsawDataset(train_encodings, train_df['rule_violation'].tolist())
    val_dataset = JigsawDataset(val_encodings, val_df['rule_violation'].tolist())

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PARAMS.model_name, num_labels=MODEL_PARAMS.output_len)




    # *************************************
        # === Example setup ===
    base_lr = 3e-5
    lr_decay = 0.96
    weight_decay = 0.01
    
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(model, base_lr, weight_decay, lr_decay)
    optimizer = AdamW(optimizer_grouped_parameters)
    
    num_training_steps = int(len(training_data_df) / LOADER_PARAMS.TRAIN_BATCH_SIZE * TRAIN_PARAMS.epochs)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )

# ****************************************

    # For deberta-v3-large
    # model.gradient_checkpointing_enable()  # <-- saves memory

    # Log the training loss at each epoch
    # logging_steps = len(training_data_df["rule_violation"])//LOADER_PARAMS.TRAIN_BATCH_SIZE
    logging_steps = max(1, len(training_data_df) // (LOADER_PARAMS.TRAIN_BATCH_SIZE * 10))

    

    training_args = TrainingArguments(
        report_to="none",
        output_dir=Config.OUTPUT_DIR,
        num_train_epochs=TRAIN_PARAMS.epochs,
        per_device_train_batch_size=LOADER_PARAMS.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=LOADER_PARAMS.VALID_BATCH_SIZE,
        learning_rate=OptimizerParams.lr[1],
        # load_best_model_at_end=True,
        metric_for_best_model="f1 (macro)",
        weight_decay=0.01, # TODO: Also make this as a param
        # eval_strategy="steps",
        save_strategy="steps",
        eval_steps = 100, #100
        logging_steps=logging_steps,
        disable_tqdm=False,

        # For deberta-v3-large
        fp16=True, # Make it train fast.
        gradient_accumulation_steps = 4 # 4
    
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        # compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        # eval_dataset=test_dataset,
        # tokenizer=TOKENIZER
        optimizers=(optimizer, scheduler)  # Custom optimizer & scheduler
    )

    # Train
    trainer.train()
    
    # Evaluate
    fold_results = evaluate_model(
        trainer, val_dataset, 
        val_df['rule_violation'].tolist(), 
        val_df['rule'].tolist(),
        val_df['rule'].unique()
    )
    
    # Store results
    cv_results.append({
        'fold': fold,
        'precision': fold_results['precision'],
        'recall': fold_results['recall'],
        'f1': fold_results['f1'],
        'auc': fold_results['auc'],
        'train_size': len(train_df),
        'val_size': len(val_df)
    })
    
    # Store predictions for this fold
    fold_predictions.append({
        'fold': fold,
        'true_labels': fold_results['true_labels'],
        'predictions': fold_results['predictions'],
        'probabilities': fold_results['probabilities'],
        'rules': val_df['rule'].tolist()
    })
    
    print(f"Fold {fold} Results:")
    print(f"  Precision: {fold_results['precision']:.4f}")
    print(f"  Recall:    {fold_results['recall']:.4f}")
    print(f"  F1-Score:  {fold_results['f1']:.4f}")
    print(f"  AUC-ROC:   {fold_results['auc']:.4f}")



print_cv_summary(cv_results)
# Create comprehensive dashboard
print(f"\nğŸ�¨ GENERATING COMPREHENSIVE DASHBOARD...")
create_comprehensive_dashboard(cv_results, fold_predictions, 'cv_dashboard.png')

# Save detailed results
cv_df = pd.DataFrame(cv_results)
cv_df.to_csv('cross_validation_results.csv', index=False)

# Save all predictions
all_predictions = []
for fold_pred in fold_predictions:
    fold_df = pd.DataFrame({
        'fold': fold_pred['fold'],
        'true_label': fold_pred['true_labels'],
        'predicted_label': fold_pred['predictions'],
        'probability': fold_pred['probabilities'],
        'rule': fold_pred['rules']
    })
    all_predictions.append(fold_df)

predictions_df = pd.concat(all_predictions, ignore_index=True)
predictions_df.to_csv('cv_predictions.csv', index=False)

print(f"\nâœ… CROSS-VALIDATION COMPLETED SUCCESSFULLY!")
print(f"ğŸ“Š Dashboard saved as: cv_dashboard.png")
print(f"ğŸ“� Results saved as: cross_validation_results.csv")
print(f"ğŸ“� Predictions saved as: cv_predictions.csv")




