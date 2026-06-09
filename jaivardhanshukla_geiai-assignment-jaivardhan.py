# =====================================================================================
#  Home Depot Product Search Relevance - Generative AI Scorer with Hyperparameter Tuning
#  Author: Jaivardhan Shukla
#  Date: 2025-9-1
# =====================================================================================

# --- Core Libraries ---
import pandas as pd
import numpy as np
import torch
import re
import spacy
import os
import random
import gc
import optuna  # Import Optuna for hyperparameter tuning

# --- PyTorch & Hugging Face ---
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup

# --- Scikit-learn ---
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# --- Visualization ---
import matplotlib.pyplot as plt
import seaborn as sns

# Set a consistent plot style
plt.style.use('seaborn-v0_8-whitegrid')
# This magic command ensures plots are displayed inline in notebooks
%matplotlib inline


# =====================================================================================
#  1. Project Configuration & Initialization
# =====================================================================================
class ProjectConfig:
    """Stores all hyperparameters and configuration settings for the project."""
    # --- Data Paths ---
    TRAIN_PATH = '/kaggle/input/home-depot-product-search-relevance/train.csv.zip'
    DESCRIPTIONS_PATH = '/kaggle/input/home-depot-product-search-relevance/product_descriptions.csv.zip'
    ATTRIBUTES_PATH = '/kaggle/input/home-depot-product-search-relevance/attributes.csv.zip'
    
    # --- Model & Tokenizer ---
    MODEL_CHECKPOINT = 't5-small'
    MAX_INPUT_LENGTH = 256
    
    # --- Training Parameters ---
    EPOCHS = 3 # Fewer epochs for each trial to speed up tuning
    
    # --- Hyperparameter Tuning with Optuna ---
    N_TUNING_TRIALS = 15 # Number of different hyperparameter combinations to try
    TUNING_RANGES = {
        'learning_rate': (1e-5, 9e-5),
        'train_batch_size': [8, 16, 32],
        'warmup_steps': (0, 500)
    }

    # --- Environment & Splitting ---
    RANDOM_SEED = 101
    VALIDATION_SPLIT_SIZE = 0.15 # Slightly larger validation set for more stable tuning
    NUM_DEV_ROWS = 74067 # Set to None to use the full dataset

    # --- Output ---
    SAVED_MODEL_PATH = '/kaggle/working/t5_relevance_generator_optimized.pt'

def initialize_environment(seed_value):
    """Sets random seeds for reproducibility across all relevant libraries."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"*** Environment initialized with random seed: {seed_value} ***")

# Initialize
initialize_environment(ProjectConfig.RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"*** Computation device set to: {DEVICE} ***")

# Load SpaCy model for text processing
try:
    nlp_lemmatizer = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    print("*** SpaCy lemmatizer 'en_core_web_sm' loaded. ***")
except OSError:
    print("Downloading SpaCy model...")
    spacy.cli.download("en_core_web_sm")
    nlp_lemmatizer = spacy.load("en_core_web_sm", disable=['parser', 'ner'])


# =====================================================================================
#  2. Data Loading and Preprocessing
# =====================================================================================
def normalize_text_field(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    doc = nlp_lemmatizer(text)
    return " ".join([token.lemma_ for token in doc])

def load_and_merge_data(config):
    print("--- Starting Data Loading and Merging ---")
    df_train = pd.read_csv(config.TRAIN_PATH, encoding='latin-1', nrows=config.NUM_DEV_ROWS)
    df_desc = pd.read_csv(config.DESCRIPTIONS_PATH, encoding='latin-1')
    df_attr = pd.read_csv(config.ATTRIBUTES_PATH, encoding='latin-1')
    df_attr['attribute_text'] = df_attr['name'].astype(str) + " " + df_attr['value'].astype(str)
    df_agg_attr = df_attr.groupby('product_uid')['attribute_text'].apply(lambda x: ' '.join(x)).reset_index()
    merged_data = pd.merge(df_train, df_desc, on='product_uid', how='left')
    merged_data = pd.merge(merged_data, df_agg_attr, on='product_uid', how='left')
    merged_data.fillna('', inplace=True)
    print("Applying text normalization...")
    for col in ['search_term', 'product_title', 'product_description', 'attribute_text']:
        merged_data[f'processed_{col}'] = merged_data[col].apply(normalize_text_field)
    merged_data['combined_text'] = ("title: " + merged_data['processed_product_title'] + " | attributes: " + merged_data['processed_attribute_text'] + " | description: " + merged_data['processed_product_description'])
    print(f"Data preparation complete. Total records: {len(merged_data)}")
    return merged_data


# =====================================================================================
#  3. PyTorch Dataset
# =====================================================================================
class SearchRelevanceDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.dataframe)
    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        source_text = f"relevance score for search: '{row['processed_search_term']}' and product: '{row['combined_text']}'"
        target_text = str(row['relevance'])
        source_encoding = self.tokenizer(source_text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        target_encoding = self.tokenizer(target_text, max_length=8, padding='max_length', truncation=True, return_tensors='pt')
        labels = target_encoding['input_ids']
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {'input_ids': source_encoding['input_ids'].flatten(), 'attention_mask': source_encoding['attention_mask'].flatten(), 'labels': labels.flatten()}


# =====================================================================================
#  4. Optuna Objective Function for Hyperparameter Search
# =====================================================================================
def objective(trial, train_df, val_df, config):
    """
    This is the core function for Optuna. It defines one trial of the HPO process.
    """
    # --- 1. Suggest Hyperparameters ---
    params = {
        'learning_rate': trial.suggest_float('learning_rate', *config.TUNING_RANGES['learning_rate'], log=True),
        'train_batch_size': trial.suggest_categorical('train_batch_size', config.TUNING_RANGES['train_batch_size']),
        'warmup_steps': trial.suggest_int('warmup_steps', *config.TUNING_RANGES['warmup_steps'])
    }
    
    # --- 2. Setup Model and Data for this Trial ---
    tokenizer = T5Tokenizer.from_pretrained(config.MODEL_CHECKPOINT)
    train_dataset = SearchRelevanceDataset(train_df, tokenizer, config.MAX_INPUT_LENGTH)
    val_dataset = SearchRelevanceDataset(val_df, tokenizer, config.MAX_INPUT_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=params['train_batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['train_batch_size'] * 2, shuffle=False)
    
    model = T5ForConditionalGeneration.from_pretrained(config.MODEL_CHECKPOINT).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=params['warmup_steps'],
        num_training_steps=len(train_loader) * config.EPOCHS
    )

    # --- 3. Training and Validation Loop for the Trial ---
    for epoch in range(config.EPOCHS):
        model.train()
        for data_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=data_batch['input_ids'].to(DEVICE),
                attention_mask=data_batch['attention_mask'].to(DEVICE),
                labels=data_batch['labels'].to(DEVICE)
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

        # --- Validation and Pruning ---
        val_metrics = assess_performance(model, val_loader, config.MODEL_CHECKPOINT)
        val_mse = val_metrics['mse']
        
        # Report intermediate results to Optuna
        trial.report(val_mse, epoch)
        
        # Handle pruning (stop unpromising trials early)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # Clean up memory after each trial
    del model, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()
    
    return val_mse # Return the final metric for Optuna to optimize


# =====================================================================================
#  5. Core Training & Evaluation Logic
# =====================================================================================
def assess_performance(model, data_loader, model_checkpoint):
    """Evaluates the model on a given dataset and returns performance metrics."""
    model.eval()
    all_predictions, all_true_labels = [], []
    tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
    with torch.no_grad():
        for data_batch in data_loader:
            generated_ids = model.generate(input_ids=data_batch['input_ids'].to(DEVICE), attention_mask=data_batch['attention_mask'].to(DEVICE), max_length=8)
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            labels = data_batch['labels'].clone(); labels[labels == -100] = tokenizer.pad_token_id
            targets = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in labels]
            for p in preds:
                try: all_predictions.append(float(p))
                except ValueError: all_predictions.append(1.0)
            all_true_labels.extend([float(t) for t in targets])
    predicted_scores = np.clip(np.array(all_predictions), 1.0, 3.0)
    true_scores = np.array(all_true_labels)
    return {"mse": mean_squared_error(true_scores, predicted_scores), "rmse": np.sqrt(mean_squared_error(true_scores, predicted_scores)), "r2": r2_score(true_scores, predicted_scores), "mae": mean_absolute_error(true_scores, predicted_scores), "predicted_scores": predicted_scores, "true_scores": true_scores}

def train_final_model(params, train_df, val_df, config):
    """Trains the final model using the best hyperparameters found by Optuna."""
    print("\n--- Training Final Model with Best Hyperparameters ---")
    print(f"Best Params: {params}")
    
    # Setup
    tokenizer = T5Tokenizer.from_pretrained(config.MODEL_CHECKPOINT)
    train_dataset = SearchRelevanceDataset(train_df, tokenizer, config.MAX_INPUT_LENGTH)
    val_dataset = SearchRelevanceDataset(val_df, tokenizer, config.MAX_INPUT_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=params['train_batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['train_batch_size'] * 2, shuffle=False)
    
    model = T5ForConditionalGeneration.from_pretrained(config.MODEL_CHECKPOINT).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=params['warmup_steps'], num_training_steps=len(train_loader) * config.EPOCHS)

    # Training Loop
    best_val_mse = float('inf')
    for epoch in range(config.EPOCHS):
        model.train()
        for data_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(input_ids=data_batch['input_ids'].to(DEVICE), attention_mask=data_batch['attention_mask'].to(DEVICE), labels=data_batch['labels'].to(DEVICE))
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        val_metrics = assess_performance(model, val_loader, config.MODEL_CHECKPOINT)
        print(f"Epoch {epoch+1}/{config.EPOCHS}, Val MSE: {val_metrics['mse']:.4f}")
        if val_metrics['mse'] < best_val_mse:
            best_val_mse = val_metrics['mse']
            torch.save(model.state_dict(), config.SAVED_MODEL_PATH)
            print(f"  -> Best model saved to {config.SAVED_MODEL_PATH}")
    return model


# =====================================================================================
#  6. Visualization and Reporting
# =====================================================================================
def visualize_predictions(true_scores, predicted_scores):
    plt.figure(figsize=(8, 8))
    sns.regplot(x=true_scores, y=predicted_scores, scatter_kws={'alpha':0.3, 'color':'teal'}, line_kws={'color':'crimson', 'linestyle':'--'})
    plt.plot([1, 3], [1, 3], color='black', linestyle='--', label='Ideal Prediction')
    plt.title('True vs. Predicted Relevance Scores', fontsize=14); plt.xlabel('Actual Relevance Score'); plt.ylabel('Predicted Relevance Score')
    plt.xlim(0.9, 3.1); plt.ylim(0.9, 3.1); plt.legend(); plt.show()

def analyze_errors(val_df, eval_results):
    print("\n--- Qualitative Error Analysis ---")
    analysis_df = val_df.copy(); analysis_df['predicted_relevance'] = eval_results['predicted_scores']; analysis_df['error'] = abs(analysis_df['relevance'] - analysis_df['predicted_relevance'])
    pd.set_option('display.max_colwidth', 200)
    print("\n*** Top 5 BEST Predictions (Lowest Error) ***"); print(analysis_df.nsmallest(5, 'error')[['search_term', 'product_title', 'relevance', 'predicted_relevance', 'error']])
    print("\n*** Top 5 WORST Predictions (Highest Error) ***"); print(analysis_df.nlargest(5, 'error')[['search_term', 'product_title', 'relevance', 'predicted_relevance', 'error']])


# =====================================================================================
#  7. Main Execution Pipeline
# =====================================================================================
def execute_pipeline():
    """Main function to run the entire ML pipeline."""
    # --- 1. Data Preparation ---
    full_data = load_and_merge_data(ProjectConfig)
    train_data, val_data = train_test_split(full_data, test_size=ProjectConfig.VALIDATION_SPLIT_SIZE, random_state=ProjectConfig.RANDOM_SEED, stratify=full_data['relevance'].round())
    
    # --- 2. Hyperparameter Search using Optuna ---
    print("\n--- Starting Hyperparameter Search with Optuna ---")
    print(f"WARNING: This will run {ProjectConfig.N_TUNING_TRIALS} trials and may be computationally expensive.")
    
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, train_data, val_data, ProjectConfig), n_trials=ProjectConfig.N_TUNING_TRIALS)
    
    print("\n--- Hyperparameter Search Complete ---")
    print(f"Best trial finished with value (MSE): {study.best_trial.value:.4f}")
    print("Best hyperparameters found:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")

    # --- 3. Train Final Model with Best Parameters ---
    best_model = train_final_model(study.best_params, train_data, val_data, ProjectConfig)
    
    # --- 4. Final Evaluation and Reporting ---
    print("\n--- Final Evaluation on Validation Set using Optimized Model ---")
    val_loader = DataLoader(SearchRelevanceDataset(val_data, T5Tokenizer.from_pretrained(ProjectConfig.MODEL_CHECKPOINT), ProjectConfig.MAX_INPUT_LENGTH), batch_size=study.best_params['train_batch_size'] * 2)
    final_eval_metrics = assess_performance(best_model, val_loader, ProjectConfig.MODEL_CHECKPOINT)
    
    print(f"  - Mean Squared Error (MSE): {final_eval_metrics['mse']:.4f}")
    print(f"  - Root Mean Squared Error (RMSE): {final_eval_metrics['rmse']:.4f}")
    print(f"  - R-squared (R2): {final_eval_metrics['r2']:.4f}")
    print(f"  - Mean Absolute Error (MAE): {final_eval_metrics['mae']:.4f}")
    
    # --- 5. Visualization and Analysis ---
    visualize_predictions(final_eval_metrics['true_scores'], final_eval_metrics['predicted_scores'])
    analyze_errors(val_data, final_eval_metrics)

if __name__ == "__main__":
    execute_pipeline()




