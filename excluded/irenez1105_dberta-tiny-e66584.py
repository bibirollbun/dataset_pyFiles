# stack_ensemble.py
import os
import sys
import gc
import time
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    logging as hf_logging
)
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.utils import resample # For balancing subsets

# --- Configuration ---
# General
BASE_MODEL_NAME = "distilroberta-base" 
OUTPUT_DIR_BASE = "/kaggle/input/my-dataset" 
KAGGLE_DATA_PATH = "/kaggle/input/llm-detect-ai-generated-text/train_essays.csv"
HF_DATASET_NAME = "dmitva/human_ai_generated_text"
TEST_SPLIT_SIZE = 0.2
RANDOM_STATE = 42
MAX_SAMPLES_PER_CLASS = 7500 # Should match finetune_models.py

# Training Hyperparameters (KEEP LOW FOR DEMO, INCREASE FOR REAL RUNS)
NUM_EPOCHS_KFOLD = 1 # Epochs for models trained during K-Fold OOF generation
NUM_EPOCHS_FINAL = 1 # Epochs for final base models trained on full data
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 2
LEARNING_RATE = 2e-5
FP16_ENABLED = torch.cuda.is_available()

# K-Fold / Stacking
N_SPLITS = 2 # Number of folds for K-Fold
WORD_COUNT_THRESHOLD = 280 # Threshold for long/short subsets
SUBSET_RANDOM_FRACTION = 0.7 # Fraction for random subset

# --- Setup ---
hf_logging.set_verbosity_error() # Reduce transformers logging noise
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)
os.environ["HF_DATASETS_CACHE"] = "/kaggle/working/hf_datasets"



def load_and_prepare_data():
    """Loads, combines, splits train/test, adds word count.
       Saves HuggingFace train split to CSV.
    """
    print("Loading and preparing data...")
    try:
        kaggle_dataset = pd.read_csv(KAGGLE_DATA_PATH)
        hf_train_path = "/kaggle/input/meta-learner-aidetect/hf_train.csv"
        hf_dataset = pd.read_csv(hf_train_path)

    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # 构造训练数据
    n_samples = min(
        MAX_SAMPLES_PER_CLASS,
        len(hf_dataset['human_text']),
        len(hf_dataset['ai_text'])
    )

    human_texts = pd.concat([
        pd.Series(hf_dataset['human_text'])[:n_samples],
        kaggle_dataset[kaggle_dataset['generated'] == 0]['text'][:n_samples]
    ], ignore_index=True)
    ai_texts = pd.concat([
        pd.Series(hf_dataset['ai_text'])[:n_samples],
        kaggle_dataset[kaggle_dataset['generated'] == 1]['text'][:n_samples]
    ], ignore_index=True)

    human_df = pd.DataFrame({'text': human_texts, 'generated': 0})
    ai_df = pd.DataFrame({'text': ai_texts, 'generated': 1})

    combined_df = pd.concat([human_df, ai_df], ignore_index=True)
    combined_df['text'] = combined_df['text'].fillna("").astype(str).str.strip()
    combined_df['word_count'] = combined_df['text'].apply(lambda x: len(x.split()))
    combined_df = combined_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # 构造 train/test 划分
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(
        combined_df,
        test_size=TEST_SPLIT_SIZE,
        random_state=RANDOM_STATE,
        stratify=combined_df['generated']
    )
    print(f"   - Train shape = {train_df.shape}")
    print(f"   - Test shape = {test_df.shape}")

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)



def load_true_test():
    return pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/test_essays.csv')


# --- Data Subset Creation Functions ---
def get_full_subset(df):
    return df

def get_random_subset(df, fraction, random_state):
    # Stratified sampling by 'generated' label
    return df.groupby('generated', group_keys=False).apply(
        lambda x: x.sample(frac=fraction, random_state=random_state)
    )

def get_long_subset(df, threshold, random_state):
    long_df = df[df['word_count'] > threshold]
    # Balance the subset (simple downsampling of majority class within long texts)
    min_class_count = long_df['generated'].value_counts().min()
    if min_class_count > 0:
        return long_df.groupby('generated', group_keys=False).apply(
             lambda x: x.sample(n=min_class_count, random_state=random_state)
        )
    else:
        return pd.DataFrame(columns=df.columns) # Return empty if one class is missing

def get_short_subset(df, threshold, random_state):
    short_df = df[df['word_count'] <= threshold]
    min_class_count = short_df['generated'].value_counts().min()
    if min_class_count > 0:
        return short_df.groupby('generated', group_keys=False).apply(
             lambda x: x.sample(n=min_class_count, random_state=random_state)
        )
    else:
        return pd.DataFrame(columns=df.columns) # Return empty



from transformers import AutoModelForSequenceClassification

# --- Model Training Function ---
def train_distilbert(train_df, eval_df, model_output_dir, num_epochs):
    """Fine-tunes DistilBERT on the provided train/eval dataframes."""
    if len(train_df) == 0 or len(eval_df) == 0:
        print(f"Skipping training for {model_output_dir}: Empty train or eval dataframe.")
        return None # Cannot train if data is empty

    print(f"Starting training for: {model_output_dir}, Train size: {len(train_df)}, Eval size: {len(eval_df)}, Epochs: {num_epochs}")
    start_time = time.time()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_output_dir)
    
    # 如果模型目录下已有模型，直接加载
    if os.path.exists(os.path.join(model_output_dir, "model.safetensors")):
        print(f"Loading pretrained model from: {model_output_dir}")
        model = AutoModelForSequenceClassification.from_pretrained(model_output_dir).to(device)
        return model  #跳过重新训练，直接返回
    else:
        model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_NAME, num_labels=1).to(device)

    # Prepare datasets
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df) # Use eval_df directly for eval dataset

    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    eval_tokenized = eval_dataset.map(tokenize_function, batched=True)

    # Add labels (need to be float for regression/sigmoid)
    train_tokenized = train_tokenized.map(lambda examples: {'labels': [float(l) for l in examples['generated']]}, batched=True)
    eval_tokenized = eval_tokenized.map(lambda examples: {'labels': [float(l) for l in examples['generated']]}, batched=True)


    # Define training arguments (minimalistic for speed)
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        eval_strategy="no", # No evaluation during k-fold training for speed
        save_strategy="no",      # No saving checkpoints during k-fold
        # evaluation_strategy="epoch", # Enable for monitoring, slows down
        # save_strategy="epoch",       # Enable for monitoring, slows down
        # load_best_model_at_end=True, # Enable for monitoring
        # metric_for_best_model="loss",# Enable for monitoring
        fp16=FP16_ENABLED,
        logging_steps=500,       # Log less often
        report_to="none",         # Disable reporting (wandb, etc.)
        disable_tqdm=True         # Disable progress bars for cleaner logs in loops
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized, # Provide eval dataset
        tokenizer=tokenizer,
        # compute_metrics=compute_metrics, # Can add if doing evaluation
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] # Can add if doing evaluation
    )

    # Train
    try:
        trainer.train()
        print(f"Finished training for {model_output_dir}. Time: {time.time() - start_time:.2f}s")
        model.save_pretrained(model_output_dir)
        tokenizer.save_pretrained(model_output_dir)

        return model # Return the trained model object
    except Exception as e:
         print(f"Error during training for {model_output_dir}: {e}")
         return None # Return None if training failed


def predict_proba(model, tokenizer, df):
    """Generates sigmoid probabilities for the dataframe."""
    if model is None or len(df) == 0:
        print("Skipping prediction: Model is None or dataframe is empty.")
        # Return array of 0.5 (neutral prediction) of correct shape if df is empty or model failed
        return np.full(len(df), 0.5)

    print(f"Generating predictions for {len(df)} samples...")
    start_time = time.time()
    dataset = Dataset.from_pandas(df)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    # No labels needed for prediction

    # Manual prediction loop (Trainer.predict is sometimes tricky with custom setups)
    model.eval()
    model.to(device)
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(tokenized_dataset), EVAL_BATCH_SIZE):
            batch_indices = range(i, min(i + EVAL_BATCH_SIZE, len(tokenized_dataset)))
            batch = tokenized_dataset[batch_indices]

            # Manually create input tensors
            input_ids = torch.tensor(batch['input_ids']).to(device)
            attention_mask = torch.tensor(batch['attention_mask']).to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            if probs.ndim == 0: # Handle single prediction case
                all_probs.append(float(probs))
            else:
                all_probs.extend(probs.tolist())

    print(f"Finished prediction. Time: {time.time() - start_time:.2f}s")
    return np.array(all_probs)


# --- Main Stacking Workflow ---
def run_stacking():
    train_df, test_df = load_and_prepare_data()
    if train_df.empty or test_df.empty:
        print("Exiting due to data loading issues.")
        return

    tokenizer_main = AutoTokenizer.from_pretrained('/kaggle/input/distilroberta/transformers/default/1/kaggle/working/results_distilroberta/checkpoint-489') # Load once for predictions

    # --- K-Fold OOF Prediction Generation ---
    '''
    print(f"\n--- Starting {N_SPLITS}-Fold OOF Prediction Generation ---")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = {} # Dictionary to store OOF preds for each model type
    base_model_types = ["Full", "Random", "Long", "Short"]
    for model_type in base_model_types:
        oof_preds[model_type] = np.zeros(len(train_df)) # Initialize OOF arrays

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['generated'])):
        print(f"\n===== Fold {fold+1}/{N_SPLITS} =====")
        fold_start_time = time.time()
        # Create train/validation splits for this fold
        train_fold_df = train_df.iloc[train_idx]
        val_fold_df = train_df.iloc[val_idx]

        # --- Train/Predict for each base model type within the fold ---
        trained_models_fold = {} # Store models trained in this fold

        # 1. Full Model
        print(f"\n-- Fold {fold+1}: Training Full Model --")
        model_fold_dir = os.path.join(OUTPUT_DIR_BASE, f"fold_{fold+1}", "full")
        trained_model = train_distilbert(train_fold_df, val_fold_df, model_fold_dir, NUM_EPOCHS_KFOLD)
        if trained_model:
            oof_preds["Full"][val_idx] = predict_proba(trained_model, tokenizer_main, val_fold_df)
            trained_models_fold["Full"] = trained_model # Keep model only if needed later, otherwise del
        else:
            oof_preds["Full"][val_idx] = np.full(len(val_idx), 0.5) # Neutral prediction on failure
        del trained_model; gc.collect(); torch.cuda.empty_cache() # Cleanup

        # 2. Random Subset Model
        print(f"\n-- Fold {fold+1}: Training Random Subset Model --")
        subset_random = get_random_subset(train_fold_df, SUBSET_RANDOM_FRACTION, RANDOM_STATE + fold)
        model_fold_dir = os.path.join(OUTPUT_DIR_BASE, f"fold_{fold+1}", "random")
        trained_model = train_distilbert(subset_random, val_fold_df, model_fold_dir, NUM_EPOCHS_KFOLD)
        if trained_model:
            oof_preds["Random"][val_idx] = predict_proba(trained_model, tokenizer_main, val_fold_df)
            trained_models_fold["Random"] = trained_model
        else:
             oof_preds["Random"][val_idx] = np.full(len(val_idx), 0.5)
        del trained_model; gc.collect(); torch.cuda.empty_cache() # Cleanup


        # 3. Long Subset Model
        print(f"\n-- Fold {fold+1}: Training Long Subset Model --")
        subset_long = get_long_subset(train_fold_df, WORD_COUNT_THRESHOLD, RANDOM_STATE + fold)
        model_fold_dir = os.path.join(OUTPUT_DIR_BASE, f"fold_{fold+1}", "long")
        trained_model = train_distilbert(subset_long, val_fold_df, model_fold_dir, NUM_EPOCHS_KFOLD)
        if trained_model:
            oof_preds["Long"][val_idx] = predict_proba(trained_model, tokenizer_main, val_fold_df)
            trained_models_fold["Long"] = trained_model
        else:
             oof_preds["Long"][val_idx] = np.full(len(val_idx), 0.5)
        del trained_model; gc.collect(); torch.cuda.empty_cache() # Cleanup


        # 4. Short Subset Model
        print(f"\n-- Fold {fold+1}: Training Short Subset Model --")
        subset_short = get_short_subset(train_fold_df, WORD_COUNT_THRESHOLD, RANDOM_STATE + fold)
        model_fold_dir = os.path.join(OUTPUT_DIR_BASE, f"fold_{fold+1}", "short")
        trained_model = train_distilbert(subset_short, val_fold_df, model_fold_dir, NUM_EPOCHS_KFOLD)
        if trained_model:
             oof_preds["Short"][val_idx] = predict_proba(trained_model, tokenizer_main, val_fold_df)
             trained_models_fold["Short"] = trained_model
        else:
             oof_preds["Short"][val_idx] = np.full(len(val_idx), 0.5)
        del trained_model; gc.collect(); torch.cuda.empty_cache() # Cleanup


        print(f"===== Fold {fold+1} finished. Time: {time.time() - fold_start_time:.2f}s =====")
        # Optional: Could save trained_models_fold[model_type].state_dict() here if needed

    print("\n--- OOF Prediction Generation Complete ---")

    # Create Level-1 training data for Meta-Learner
    oof_df = pd.DataFrame(oof_preds)
    oof_df['generated'] = train_df['generated'] # Add true labels
    print("OOF Predictions Head:")
    print(oof_df.head())

    # --- Train Meta-Learner ---
    print("\n--- Training Meta-Learner (Logistic Regression) ---")
    meta_features = base_model_types
    X_meta_train = oof_df[meta_features]
    y_meta_train = oof_df['generated']
    meta_model_path = '/kaggle/input/meta-learner-aidetect/stacking_checkpoints_distilroberta/meta_learner.pkl'

    if os.path.exists(meta_model_path):
        print(f"Loading meta-learner from: {meta_model_path}")
        meta_learner = joblib.load(meta_model_path)
    else:
        meta_learner = LogisticRegression(random_state=RANDOM_STATE, C=1.0, solver='liblinear') # Basic LogReg
        meta_learner.fit(X_meta_train, y_meta_train)
        print("Meta-Learner Coefficients:", meta_learner.coef_)
        print("Meta-Learner Intercept:", meta_learner.intercept_)
        import joblib
        
        meta_model_path = os.path.join(OUTPUT_DIR_BASE, "meta_learner.pkl")
        joblib.dump(meta_learner, meta_model_path)
        print(f"Meta-learner saved to: {meta_model_path}")

    # Evaluate Meta-Learner on OOF predictions (as a proxy for CV performance)
    oof_final_preds = meta_learner.predict(X_meta_train)
    oof_final_probs = meta_learner.predict_proba(X_meta_train)[:, 1]
    print("\nMeta-Learner Performance on OOF Data:")
    print(f"Accuracy: {accuracy_score(y_meta_train, oof_final_preds):.4f}")
    try:
        print(f"AUC: {roc_auc_score(y_meta_train, oof_final_probs):.4f}")
    except ValueError as e:
        print(f"AUC calculation failed on OOF: {e}") # May happen if OOF preds are constant for a class

   '''
    import joblib
    meta_model_path = '/kaggle/input/meta-learner-aidetect/stacking_checkpoints_distilroberta/meta_learner.pkl'
    meta_learner = joblib.load(meta_model_path)
    # --- Train Final Base Models on Full Training Data ---
    print("\n--- Training Final Base Models on Full Training Data ---")
    final_base_models = {}

    # 1. Full Model
    print("\n-- Training Final Full Model --")
    final_model_dir = os.path.join(OUTPUT_DIR_BASE, "final", "full")
    subset_full_final = train_df
    final_base_models["Full"] = train_distilbert(subset_full_final, test_df, final_model_dir, NUM_EPOCHS_FINAL)
    # Save the final model if needed:
    # if final_base_models["Full"]: final_base_models["Full"].save_pretrained(final_model_dir)

    # 2. Random Subset Model
    print("\n-- Training Final Random Subset Model --")
    final_model_dir = os.path.join(OUTPUT_DIR_BASE, "final", "random")
    subset_random_final = get_random_subset(train_df, SUBSET_RANDOM_FRACTION, RANDOM_STATE)
    final_base_models["Random"] = train_distilbert(subset_random_final, test_df, final_model_dir, NUM_EPOCHS_FINAL)
    # if final_base_models["Random"]: final_base_models["Random"].save_pretrained(final_model_dir)


    # 3. Long Subset Model
    print("\n-- Training Final Long Subset Model --")
    final_model_dir = os.path.join(OUTPUT_DIR_BASE, "final", "long")
    subset_long_final = get_long_subset(train_df, WORD_COUNT_THRESHOLD, RANDOM_STATE)
    final_base_models["Long"] = train_distilbert(subset_long_final, test_df, final_model_dir, NUM_EPOCHS_FINAL)
    # if final_base_models["Long"]: final_base_models["Long"].save_pretrained(final_model_dir)


    # 4. Short Subset Model
    print("\n-- Training Final Short Subset Model --")
    final_model_dir = os.path.join(OUTPUT_DIR_BASE, "final", "short")
    subset_short_final = get_short_subset(train_df, WORD_COUNT_THRESHOLD, RANDOM_STATE)
    final_base_models["Short"] = train_distilbert(subset_short_final, test_df, final_model_dir, NUM_EPOCHS_FINAL)
    # if final_base_models["Short"]: final_base_models["Short"].save_pretrained(final_model_dir)

    print("\n--- Final Base Model Training Complete ---")

    # --- Generate Test Set Predictions from Final Base Models ---
    print("\n--- Generating Test Set Predictions from Final Base Models ---")
    test_preds = {}
    true_test_df = load_true_test()
    base_model_types = ["Full", "Random", "Long", "Short"]
    meta_features = base_model_types
    for model_type in base_model_types:
        model = final_base_models.get(model_type)
        test_preds[model_type] = predict_proba(model, tokenizer_main, true_test_df)
        # Cleanup model from memory if not needed anymore
        if model_type in final_base_models: del final_base_models[model_type]; gc.collect(); torch.cuda.empty_cache()


    test_preds_df = pd.DataFrame(test_preds)
    print("Test Set Base Predictions Head:")
    print(test_preds_df.head())

    # --- Make Final Predictions using Meta-Learner ---
    print("\n--- Making Final Predictions using Meta-Learner ---")
    X_meta_test = test_preds_df[meta_features]
    final_predictions = meta_learner.predict(X_meta_test)
    final_probabilities = meta_learner.predict_proba(X_meta_test)[:, 1]
    submission_df = pd.DataFrame({
        'id': true_test_df['id'],  # Assuming 'id' column is in the test data
        'generated': final_probabilities
    })
    submission_df.to_csv('submission.csv', index=False)

    # --- Evaluate Final Stacking Model ---
    print("\n--- Final Stacking Model Evaluation on Test Set ---")
    #y_test_true = true_test_df['generated']
    #accuracy = accuracy_score(y_test_true, final_predictions)
    #try:
    #    auc = roc_auc_score(y_test_true, final_probabilities)
    #except ValueError as e:
    #    auc = f"Calculation Failed ({e})"


    #print(f"Stacking Model Test Accuracy: {accuracy:.4f}")
    #print(f"Stacking Model Test AUC: {auc if isinstance(auc, str) else f'{auc:.4f}'}")
    #print("\nClassification Report:")
    #print(classification_report(y_test_true, final_predictions, target_names=["Human (0)", "AI (1)"]))

    # You can also save the final predictions
    # test_df['stacking_probs'] = final_probabilities
    # test_df['stacking_preds'] = final_predictions
    # test_df.to_csv("test_predictions_stacking.csv", index=False)
    # print("Saved test predictions with stacking results to test_predictions_stacking.csv")

if __name__ == "__main__":
    run_stacking()

