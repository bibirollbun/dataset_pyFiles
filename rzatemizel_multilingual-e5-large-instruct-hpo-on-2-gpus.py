#!pip install -q -U peft bitsandbytes accelerate


%%writefile hpo_trial_cv_runner.py

import os
import gc
import json
import pandas as pd
import torch
import random
import numpy as np
import argparse
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from peft import get_peft_model, LoraConfig, TaskType

# --- CONFIGURATION ---
class CFG:
    COMPETITION_NAME = "jigsaw-agile-community-rules"
    DATA_PATH = f"/kaggle/input/{COMPETITION_NAME}/"
    MODEL_ID = "/kaggle/input/multilingual-e5-large-instruct/content/multilingual-e5-large-instruct"
    MODEL_OUTPUT_BASE = "/kaggle/working/hpo_models/" # Directory to save all models
    MAX_LENGTH = 512
    PER_DEVICE_BATCH_SIZE = 8
    GRAD_ACCUMULATION_STEPS = 2
    BF16 = True
    SEED = 42
    HPO_EPOCHS = 2
    NUM_FOLDS = 5

def seed_everything(seed=CFG.SEED):
    # Unchanged
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def prepare_training_data():
    # Unchanged
    # (Same function as before)
    train_df = pd.read_csv(CFG.DATA_PATH + "train.csv")
    test_df = pd.read_csv(CFG.DATA_PATH + "test.csv")
    all_data = []
    train_main = train_df[['rule', 'subreddit', 'body', 'rule_violation']].copy()
    all_data.append(train_main)
    for i in range(1, 3):
        for prefix in ['positive', 'negative']:
            label = 1 if prefix == 'positive' else 0
            df = train_df[['rule', 'subreddit', f'{prefix}_example_{i}']].copy()
            df.rename(columns={f'{prefix}_example_{i}': 'body'}, inplace=True)
            df['rule_violation'] = label
            all_data.append(df)
    for i in range(1, 3):
        for prefix in ['positive', 'negative']:
            label = 1 if prefix == 'positive' else 0
            df = test_df[['rule', 'subreddit', f'{prefix}_example_{i}']].copy()
            df.rename(columns={f'{prefix}_example_{i}': 'body'}, inplace=True)
            df['rule_violation'] = label
            all_data.append(df)
    full_train_df = pd.concat(all_data, ignore_index=True)
    full_train_df.dropna(subset=['body'], inplace=True)
    full_train_df['body'] = full_train_df['body'].astype(str)
    full_train_df.drop_duplicates(subset=['rule', 'body'], keep='first', inplace=True)
    full_train_df['text_input'] = ("Instruct: Given the rule: '" + full_train_df['rule'] + "', determine if the following comment violates it.\nQuery: " + full_train_df['body'])
    full_train_df.rename(columns={'rule_violation': 'labels'}, inplace=True)
    return full_train_df[['text_input', 'labels', 'rule']].reset_index(drop=True)

def train_single_fold(params, trial_num, fold_num, train_dataset, val_dataset, tokenizer, val_df_for_metric):
    # This function now saves the best model permanently for its trial
    trial_output_dir = os.path.join(CFG.MODEL_OUTPUT_BASE, f"trial_{trial_num}", f"fold_{fold_num}")
    
    lora_config = LoraConfig(
        r=params["lora_r"], lora_alpha=params["lora_alpha"],
        target_modules=["query", "key", "value", "dense"],
        lora_dropout=params["lora_dropout"], bias="none", task_type=TaskType.SEQ_CLS,
        modules_to_save=["classifier", "pooler"],
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        CFG.MODEL_ID, num_labels=2, torch_dtype=torch.bfloat16
    )
    model = get_peft_model(model, lora_config)

    def compute_averaged_auc(eval_pred):
        predictions, labels = eval_pred
        probs = torch.softmax(torch.tensor(predictions), dim=-1)[:, 1].numpy()
        results_df = pd.DataFrame({'labels': labels, 'preds': probs, 'rule': val_df_for_metric['rule']})
        auc_scores = [roc_auc_score(g['labels'], g['preds']) for _, g in results_df.groupby('rule') if len(g['labels'].unique()) > 1]
        return {"averaged_auc": np.mean(auc_scores) if auc_scores else 0.0}

    training_args = TrainingArguments(
        output_dir=trial_output_dir, num_train_epochs=CFG.HPO_EPOCHS,
        learning_rate=params["learning_rate"], weight_decay=params["weight_decay"],
        lr_scheduler_type=params["lr_scheduler_type"], warmup_ratio=params["warmup_ratio"],
        per_device_train_batch_size=CFG.PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=CFG.PER_DEVICE_BATCH_SIZE * 2,
        gradient_accumulation_steps=CFG.GRAD_ACCUMULATION_STEPS, bf16=CFG.BF16,
        optim="adamw_8bit", logging_steps=9999, eval_strategy="epoch",
        save_strategy="epoch", save_total_limit=1, # <-- IMPORTANT: We now save the model
        load_best_model_at_end=False, # <-- Keep False as per original design
        ddp_find_unused_parameters=False,
        metric_for_best_model="averaged_auc", greater_is_better=True, report_to="none",
        seed=CFG.SEED, data_seed=CFG.SEED,
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=val_dataset, tokenizer=tokenizer,
        compute_metrics=compute_averaged_auc,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
    )
    
    trainer.train()
    
    best_score = trainer.state.best_metric if trainer.state.best_metric is not None else 0.0
    
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    # DO NOT delete the output directory anymore
    return best_score

if __name__ == "__main__":
    # This main block is identical to the previous version
    parser = argparse.ArgumentParser()
    parser.add_argument("--params_json", type=str, required=True)
    parser.add_argument("--trial_num", type=int, required=True)
    args = parser.parse_args()
    params = json.loads(args.params_json)
    seed_everything(CFG.SEED)
    full_df = prepare_training_data()
    tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_ID)
    def tokenize_function(ex): return tokenizer(ex['text_input'], truncation=True, max_length=CFG.MAX_LENGTH)
    skf = StratifiedKFold(n_splits=CFG.NUM_FOLDS, shuffle=True, random_state=CFG.SEED)
    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(full_df, full_df['labels'])):
        if os.environ.get("LOCAL_RANK", "0") == "0": print(f"--- [Trial {args.trial_num}] Running Fold {fold+1}/{CFG.NUM_FOLDS} ---")
        train_df, val_df = full_df.iloc[train_idx], full_df.iloc[val_idx]
        train_dataset = Dataset.from_pandas(train_df).map(tokenize_function, batched=True, remove_columns=['rule'])
        val_dataset = Dataset.from_pandas(val_df).map(tokenize_function, batched=True, remove_columns=['rule'])
        score = train_single_fold(params, args.trial_num, fold, train_dataset, val_dataset, tokenizer, val_df)
        fold_scores.append(score)
    if os.environ.get("LOCAL_RANK", "0") == "0":
        avg_score = np.mean(fold_scores)
        print(f"TRIAL {args.trial_num} FINAL AVG SCORE: {avg_score}")
        score_file_dir = "/kaggle/working/hpo_scores/"
        os.makedirs(score_file_dir, exist_ok=True)
        score_file = os.path.join(score_file_dir, f"trial_{args.trial_num}_score.json")
        with open(score_file, 'w') as f: json.dump({"averaged_auc": avg_score}, f)


%%writefile run_hpo_manager.py

import os
import json
import optuna
from optuna.samplers import TPESampler
import shutil

class CFG:
    SCORE_DIR = "/kaggle/working/hpo_scores/"
    MODEL_OUTPUT_BASE = "/kaggle/working/hpo_models/" # Same as worker
    N_TRIALS = 10
    SEED = 42

def run_hpo_manager():
    os.makedirs(CFG.SCORE_DIR, exist_ok=True)
    os.makedirs(CFG.MODEL_OUTPUT_BASE, exist_ok=True)
    
    sampler = TPESampler(seed=CFG.SEED) 
    study = optuna.create_study(direction="maximize", study_name="E5-Large-LoRA-Jigsaw-CV-Integrated")

    for i in range(CFG.N_TRIALS):
        trial = study.ask()
        # Parameter suggestion logic is unchanged
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 5e-5, 6e-4, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 0.001, 0.1, log=True),
            "warmup_ratio": trial.suggest_categorical("warmup_ratio", [0.05, 0.1, 0.15]),
            "lr_scheduler_type": trial.suggest_categorical("lr_scheduler_type", ["linear", "cosine"]),
            "lora_r": trial.suggest_categorical("lora_r", [16, 32, 64]),
            "alpha_multiplier": trial.suggest_categorical("alpha_multiplier", [2, 4]),
            "lora_dropout": trial.suggest_float("lora_dropout", 1e-4, 0.1, log=True)
        }
        params["lora_alpha"] = params["lora_r"] * params["alpha_multiplier"]
        del params["alpha_multiplier"]
        
        trial_num = trial.number
        print(f"\n{'='*20} LAUNCHING TRIAL {trial_num} {'='*20}")
        params_json_str = json.dumps(params)
        command = (
            f'accelerate launch --multi_gpu --num_processes=2 hpo_trial_cv_runner.py '
            f'--trial_num {trial_num} --params_json \'{params_json_str}\''
        )
        exit_code = os.system(command)

        score = 0.0
        if exit_code == 0:
            try:
                score_file = os.path.join(CFG.SCORE_DIR, f"trial_{trial_num}_score.json")
                with open(score_file, 'r') as f:
                    result = json.load(f); score = result.get("averaged_auc", 0.0)
                print(f"TRIAL {trial_num} SUCCEEDED. Average CV Score: {score}")
            except Exception as e:
                print(f"TRIAL {trial_num} finished but could not read score file. Error: {e}")
        else:
            print(f"TRIAL {trial_num} FAILED with a non-zero exit code ({exit_code}).")
        
        study.tell(trial, score)
    
    print("\n--- HPO Complete ---")
    best_trial = study.best_trial
    
    # --- CRITICAL CLEANUP STEP ---
    print(f"\nBest trial is #{best_trial.number}. Cleaning up models from other trials...")
    for trial_dir in os.listdir(CFG.MODEL_OUTPUT_BASE):
        if trial_dir != f"trial_{best_trial.number}":
            full_path = os.path.join(CFG.MODEL_OUTPUT_BASE, trial_dir)
            shutil.rmtree(full_path)
            print(f"  - Deleted: {full_path}")
    
    print("\nBest Parameters:")
    print(json.dumps(best_trial.params, indent=2))
    
    # Save results for the inference script
    results_path = "/kaggle/working/hpo_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            "best_trial_number": best_trial.number,
            "best_value_auc": best_trial.value,
            "best_params": best_trial.params
        }, f, indent=4)
    print(f"\nBest trial info saved to: {results_path}")

if __name__ == "__main__":
    run_hpo_manager()


%%writefile final_inference_from_hpo.py

import os
import gc
import json
import pandas as pd
import torch
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator

class CFG:
    BASE_MODEL_ID = "/kaggle/input/multilingual-e5-large-instruct/content/multilingual-e5-large-instruct"
    DATA_PATH = "/kaggle/input/jigsaw-agile-community-rules/"
    HPO_RESULTS_PATH = "/kaggle/working/hpo_results.json"
    MODEL_OUTPUT_BASE = "/kaggle/working/hpo_models/"
    MAX_LENGTH = 512
    BATCH_SIZE = 8
    NUM_FOLDS = 5

def find_best_checkpoint_for_fold(trial_num, fold_num):
    # This finds the single saved checkpoint inside a fold's directory
    fold_path = os.path.join(CFG.MODEL_OUTPUT_BASE, f"trial_{trial_num}", f"fold_{fold_num}")
    checkpoints = [d for d in os.listdir(fold_path) if d.startswith('checkpoint-')]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found for trial {trial_num}, fold {fold_num}")
    return os.path.join(fold_path, checkpoints[0])

def run_inference():
    accelerator = Accelerator()
    
    with open(CFG.HPO_RESULTS_PATH, 'r') as f:
        best_trial_info = json.load(f)
    best_trial_num = best_trial_info["best_trial_number"]
    
    if accelerator.is_main_process:
        print(f"--- Running Inference using models from BEST HPO Trial: #{best_trial_num} ---")

    all_fold_predictions = []
    
    for fold in range(CFG.NUM_FOLDS):
        accelerator.wait_for_everyone() # Sync before loading each model
        
        checkpoint_path = find_best_checkpoint_for_fold(best_trial_num, fold)
        if accelerator.is_main_process:
            print(f"  - Loading model for fold {fold+1} from: {checkpoint_path}")

        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            CFG.BASE_MODEL_ID, num_labels=2, torch_dtype=torch.bfloat16
        )
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        model = accelerator.prepare(model)
        model.eval()

        # Inference logic is the same, just inside the loop
        test_df = pd.read_csv(os.path.join(CFG.DATA_PATH, "test.csv"))
        test_dataset = Dataset.from_pandas(test_df)
        test_dataset = test_dataset.map(lambda ex: {'text_input': ("Instruct: Given the rule: '" + str(ex['rule']) + "', determine if the following comment violates it.\nQuery: " + str(ex['body']))})
        def tokenize_function(ex): return tokenizer(ex['text_input'], truncation=True, max_length=CFG.MAX_LENGTH)
        tokenized_test = test_dataset.map(tokenize_function, batched=True, remove_columns=test_dataset.column_names)
        tokenized_test.set_format('torch')
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        test_loader = DataLoader(tokenized_test, batch_size=CFG.BATCH_SIZE*2, shuffle=False, collate_fn=data_collator)
        test_loader = accelerator.prepare(test_loader)
        
        fold_probs = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Infer Fold {fold+1}", disable=not accelerator.is_main_process):
                outputs = model(**batch)
                probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
                fold_probs.append(accelerator.gather_for_metrics(probs))
        
        fold_probs_cat = torch.cat(fold_probs).cpu().float().numpy()
        temp_df = pd.DataFrame({'row_id': test_dataset['row_id'], 'preds': fold_probs_cat[:len(test_dataset)]})
        all_fold_predictions.append(temp_df.sort_values('row_id')['preds'].values)

        del model, base_model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    if accelerator.is_main_process:
        avg_preds = np.mean(all_fold_predictions, axis=0)
        test_df = pd.read_csv(os.path.join(CFG.DATA_PATH, "test.csv"))
        submission_df = pd.DataFrame({'row_id': test_df['row_id'], 'rule_violation': avg_preds})
        submission_df.to_csv("submission.csv", index=False)
        print("\n--- Final Submission Created ---")
        print(submission_df.head())

if __name__ == "__main__":
    run_inference()


import pandas as pd
import os
import shutil

COMP_DATA_PATH = "/kaggle/input/jigsaw-agile-community-rules/"
test_df = pd.read_csv(os.path.join(COMP_DATA_PATH, "test.csv"))

if len(test_df) <= 1:
    print("Toy test set detected. Creating dummy submission.")
    pd.DataFrame({'row_id': test_df['row_id'], 'rule_violation': 0.5}).to_csv("submission.csv", index=False)
else:
    print("Full test set detected. Proceeding with integrated HPO and inference.")

    # Clean up from previous full runs
    if os.path.exists("/kaggle/working/hpo_scores/"): shutil.rmtree("/kaggle/working/hpo_scores/")
    if os.path.exists("/kaggle/working/hpo_models/"): shutil.rmtree("/kaggle/working/hpo_models/")
    if os.path.exists("/kaggle/working/hpo_results.json"): os.remove("/kaggle/working/hpo_results.json")
        
    # --- Step 1: Run the HPO Manager ---
    # This will train and save models for all trials, then clean up the non-best ones.
    print("\n--- Starting HPO Manager (Saving models from all trials temporarily) ---")
    os.system("python run_hpo_manager.py")

    # --- Step 2: Final Inference ---
    # This uses the models saved during the best HPO trial.
    print("\n--- Starting Final Inference using models from the BEST trial ---")
    os.system("accelerate launch --multi_gpu --num_processes=2 final_inference_from_hpo.py")

    print("\n--- EFFICIENT HPO + INFERENCE PIPELINE COMPLETE ---")


































