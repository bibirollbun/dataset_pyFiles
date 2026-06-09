#!pip install -q -U torch torchvision peft bitsandbytes accelerate trl


%%writefile train_e5_cv_model.py

import os
import gc
import pandas as pd
import torch
import random
import numpy as np
import argparse # For command-line arguments
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from peft import get_peft_model, LoraConfig, TaskType

class CFG:
    COMPETITION_NAME = "jigsaw-agile-community-rules"
    DATA_PATH = f"/kaggle/input/{COMPETITION_NAME}/"
    MODEL_ID = "/kaggle/input/multilingual-e5-large-instruct/content/multilingual-e5-large-instruct"
    MAX_LENGTH = 512
    PER_DEVICE_BATCH_SIZE = 8
    GRAD_ACCUMULATION_STEPS = 2
    BF16 = True
    SEED = 1988
    NUM_EPOCHS = 2
    NUM_FOLDS = 6 # Define K-Fold splits here

BEST_PARAMS = {
    "learning_rate": 0.00021465174041305742,
    "weight_decay": 0.06321291624614574,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "linear",
    "lora_r": 32, "lora_alpha": 64, "lora_dropout": 0.0001797014425213347
}

def seed_everything(seed=CFG.SEED):
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

def train_model(train_dataset, val_dataset, tokenizer, val_df_for_metric, fold):
    output_dir = f"/kaggle/working/model_fold_{fold}/"
    os.makedirs(output_dir, exist_ok=True)
    
    lora_config = LoraConfig(
        r=BEST_PARAMS["lora_r"], lora_alpha=BEST_PARAMS["lora_alpha"],
        target_modules=["query", "key", "value", "dense"],
        lora_dropout=BEST_PARAMS["lora_dropout"], bias="none", task_type=TaskType.SEQ_CLS,
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
        output_dir=output_dir,
        num_train_epochs=CFG.NUM_EPOCHS,
        learning_rate=BEST_PARAMS["learning_rate"],
        weight_decay=BEST_PARAMS["weight_decay"],
        lr_scheduler_type=BEST_PARAMS["lr_scheduler_type"],
        warmup_ratio=BEST_PARAMS["warmup_ratio"],
        per_device_train_batch_size=CFG.PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=CFG.PER_DEVICE_BATCH_SIZE * 2,
        gradient_accumulation_steps=CFG.GRAD_ACCUMULATION_STEPS,
        bf16=CFG.BF16, optim="adamw_8bit", logging_steps=100,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=False, # <-- KEPT AS FALSE, PER YOUR DESIGN
        ddp_find_unused_parameters=False,
        metric_for_best_model="averaged_auc", greater_is_better=True,
        report_to="none", seed=CFG.SEED, data_seed=CFG.SEED,
    )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=val_dataset, tokenizer=tokenizer,
        compute_metrics=compute_averaged_auc,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
    )
    
    trainer.train()
    
    # Store the final validation metric
    final_log = [log for log in trainer.state.log_history if 'eval_averaged_auc' in log]
    if final_log:
        last_score = final_log[-1]['eval_averaged_auc']
        with open(f"/kaggle/working/score_fold_{fold}.txt", "w") as f:
            f.write(str(last_score))

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, help="Fold number for CV")
    args = parser.parse_args()
    
    seed_everything(CFG.SEED)
    full_df = prepare_training_data()
    
    skf = StratifiedKFold(n_splits=CFG.NUM_FOLDS, shuffle=True, random_state=CFG.SEED)
    train_idx, val_idx = list(skf.split(full_df, full_df['labels']))[args.fold]
    
    train_df = full_df.iloc[train_idx]
    val_df = full_df.iloc[val_idx]

    tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_ID)
    def tokenize_function(examples):
        return tokenizer(examples['text_input'], truncation=True, max_length=CFG.MAX_LENGTH)
    
    train_dataset = Dataset.from_pandas(train_df).map(tokenize_function, batched=True, remove_columns=['rule'])
    val_dataset = Dataset.from_pandas(val_df).map(tokenize_function, batched=True, remove_columns=['rule'])
    
    train_model(train_dataset, val_dataset, tokenizer, val_df, args.fold)


%%writefile inference_e5_cv_model.py

import os
import gc
import glob
import pandas as pd
import torch
import argparse # For command-line arguments
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator

class CFG:
    BASE_MODEL_ID = "/kaggle/input/multilingual-e5-large-instruct/content/multilingual-e5-large-instruct"
    DATA_PATH = "/kaggle/input/jigsaw-agile-community-rules/"
    MAX_LENGTH = 512
    BATCH_SIZE = 8

def find_best_checkpoint_path(fold): # <-- Now takes fold number
    base_path = f"/kaggle/working/model_fold_{fold}/" # <-- Looks in fold-specific dir
    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"Model directory not found for fold {fold}: {base_path}.")
    
    checkpoints = [d for d in os.listdir(base_path) if d.startswith('checkpoint-') and os.path.isdir(os.path.join(base_path, d))]
    
    if not checkpoints:
        raise FileNotFoundError(f"Could not find any 'checkpoint-*' directory in {base_path}.")
        
    # Your original logic: since save_total_limit=1, there's only one, which is the best.
    best_checkpoint_name = checkpoints[0]
    best_model_path = os.path.join(base_path, best_checkpoint_name)
    return best_model_path

def run_inference(fold):
    accelerator = Accelerator()
    
    best_model_path = find_best_checkpoint_path(fold)
    print(f"--- Using checkpoint for fold {fold}: {best_model_path} ---")
    
    tokenizer = AutoTokenizer.from_pretrained(best_model_path)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        CFG.BASE_MODEL_ID, num_labels=2, torch_dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(base_model, best_model_path)
    model = accelerator.prepare(model)
    model.eval()

    test_df = pd.read_csv(os.path.join(CFG.DATA_PATH, "test.csv"))
    full_dataset = Dataset.from_pandas(test_df)

    # Your original inference logic using accelerator.split_between_processes
    # This part is complex and works for you, so it's preserved.
    with accelerator.split_between_processes(full_dataset) as data_slice:
        if data_slice:
            slice_dataset = Dataset.from_dict(data_slice.to_dict())
            slice_dataset = slice_dataset.map(lambda ex: {'text_input': ("Instruct: Given the rule: '" + str(ex['rule']) + "', determine if the following comment violates it.\nQuery: " + str(ex['body']))})
            
            def tokenize_function(ex): return tokenizer(ex['text_input'], truncation=True, max_length=CFG.MAX_LENGTH)
            
            tokenized_data = slice_dataset.map(tokenize_function, batched=True, remove_columns=slice_dataset.column_names)
            tokenized_data.set_format('torch')
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
            loader = DataLoader(tokenized_data, batch_size=CFG.BATCH_SIZE, shuffle=False, collate_fn=data_collator)
            
            all_probs = []
            row_ids = slice_dataset['row_id']
            with torch.no_grad():
                for batch in tqdm(loader, desc=f"Proc {accelerator.process_index}", disable=not accelerator.is_main_process):
                    outputs = model(**batch)
                    probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
                    all_probs.extend(probs.cpu().float().numpy())
            
            result_df = pd.DataFrame({'row_id': row_ids, 'rule_violation': all_probs})
            # Save partial predictions to a unique file to avoid clashes
            result_df.to_csv(f"preds_part_{fold}_{accelerator.process_index}.csv", index=False)

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        part_files = glob.glob(f"preds_part_{fold}_*.csv")
        all_dfs = [pd.read_csv(f) for f in part_files]
        if all_dfs:
            final_preds = pd.concat(all_dfs).sort_values('row_id').reset_index(drop=True)
            # Save the aggregated predictions for THIS FOLD
            final_preds.to_csv(f"preds_fold_{fold}.csv", index=False)
            for f in part_files: os.remove(f)
            
    del model, base_model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, help="Fold number for CV")
    args = parser.parse_args()
    run_inference(args.fold)


import pandas as pd
import os
import shutil
import numpy as np

COMP_DATA_PATH = "/kaggle/input/jigsaw-agile-community-rules/"
NUM_FOLDS = 6 # Should match CFG.NUM_FOLDS in train script
test_df = pd.read_csv(os.path.join(COMP_DATA_PATH, "test.csv"))

if len(test_df) <= 10:
    print("Toy test set detected. Creating dummy submission.")
    pd.DataFrame({'row_id': test_df['row_id'], 'rule_violation': 0.5}).to_csv("submission_e5.csv", index=False)
else:
    print("Full test set detected. Starting 5-Fold CV training and inference.")
    
    for fold in range(NUM_FOLDS):
        print("\n" + "="*50)
        print(f"===== PROCESSING FOLD {fold+1}/{NUM_FOLDS} =====")
        print("="*50)
        
        # --- Step 1: Train the model for the current fold ---
        print(f"\n--- Training Fold {fold} ---")
        train_command = f"accelerate launch --multi_gpu --num_processes=2 train_e5_cv_model.py --fold {fold}"
        os.system(train_command)

        # --- Step 2: Run inference using the trained model for the current fold ---
        print(f"\n--- Inferencing with Fold {fold} Model ---")
        inference_command = f"accelerate launch --multi_gpu --num_processes=2 inference_e5_cv_model.py --fold {fold}"
        os.system(inference_command)

        # --- Step 3: Clean up the model directory for this fold to save space ---
        model_dir = f"/kaggle/working/model_fold_{fold}/"
        print(f"Cleaning up directory: {model_dir}")
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)

    # --- Step 4: Aggregate predictions and report scores ---
    print("\n" + "="*50)
    print("===== AGGREGATION AND FINAL SUBMISSION =====")
    print("="*50)
    all_preds = []
    all_scores = []
    for fold in range(NUM_FOLDS):
        pred_df = pd.read_csv(f"preds_fold_{fold}.csv")
        all_preds.append(pred_df['rule_violation'].values)
        
        # Read the score saved by the training script
        try:
            with open(f"/kaggle/working/score_fold_{fold}.txt", "r") as f:
                score = float(f.read().strip())
                all_scores.append(score)
                print(f"Fold {fold+1} Validation Averaged AUC: {score:.5f}")
        except FileNotFoundError:
            print(f"Score file for fold {fold+1} not found.")

    if all_scores:
        mean_auc = np.mean(all_scores)
        std_auc = np.std(all_scores)
        print("-" * 50)
        print(f"Overall CV Averaged AUC: {mean_auc:.5f} +/- {std_auc:.5f}")
        print("-" * 50)

    # Average predictions across folds
    avg_preds = np.mean(all_preds, axis=0)
    submission_df = pd.DataFrame({'row_id': test_df['row_id'], 'rule_violation': avg_preds})
    submission_df.to_csv("submission_e5.csv", index=False)
    
    print("\n--- Pipeline Complete. submission_e5.csv created. ---")
    print(submission_df.head())


%%writefile train_dblarge_cv_model.py

import os
import gc
import pandas as pd
import torch
import random
import numpy as np
import argparse # For command-line arguments
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from peft import get_peft_model, LoraConfig, TaskType

class CFG:
    COMPETITION_NAME = "jigsaw-agile-community-rules"
    DATA_PATH = f"/kaggle/input/{COMPETITION_NAME}/"
    MODEL_ID = "/kaggle/input/huggingfacedebertav3variants/deberta-v3-large" 
    MAX_LENGTH = 512
    PER_DEVICE_BATCH_SIZE = 8
    GRAD_ACCUMULATION_STEPS = 2
    BF16 = True
    SEED = 1988
    NUM_EPOCHS = 2
    NUM_FOLDS = 6 # Define K-Fold splits here

BEST_PARAMS = {
  "learning_rate": 0.0002074655083182706,
  "weight_decay": 0.018670989156360767,
  "warmup_ratio": 0.05,
  "lr_scheduler_type": "linear",
  "lora_r": 32,
  "lora_dropout": 0.02287183019267056,
  "lora_alpha": 128
}



def seed_everything(seed=CFG.SEED):
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
    full_train_df['text_input'] = ("Rule: " + full_train_df['rule'] + "\n" + "Comment: " + full_train_df['body'])
    full_train_df.rename(columns={'rule_violation': 'labels'}, inplace=True)
    return full_train_df[['text_input', 'labels', 'rule']].reset_index(drop=True)

def train_model(train_dataset, val_dataset, tokenizer, val_df_for_metric, fold):
    output_dir = f"/kaggle/working/model_fold_{fold}/"
    os.makedirs(output_dir, exist_ok=True)
    
    lora_config = LoraConfig(
            r=BEST_PARAMS["lora_r"], lora_alpha=BEST_PARAMS["lora_alpha"],
            target_modules=["query_proj", "key_proj", "value_proj", "dense", "o_proj"], # <-- DeBERTa style modules
            lora_dropout=BEST_PARAMS["lora_dropout"], bias="none", task_type=TaskType.SEQ_CLS,
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
        output_dir=output_dir,
        num_train_epochs=CFG.NUM_EPOCHS,
        learning_rate=BEST_PARAMS["learning_rate"],
        weight_decay=BEST_PARAMS["weight_decay"],
        lr_scheduler_type=BEST_PARAMS["lr_scheduler_type"],
        warmup_ratio=BEST_PARAMS["warmup_ratio"],
        per_device_train_batch_size=CFG.PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=CFG.PER_DEVICE_BATCH_SIZE * 2,
        gradient_accumulation_steps=CFG.GRAD_ACCUMULATION_STEPS,
        bf16=CFG.BF16, optim="adamw_8bit", logging_steps=100,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=False, # <-- KEPT AS FALSE, PER YOUR DESIGN
        ddp_find_unused_parameters=False,
        metric_for_best_model="averaged_auc", greater_is_better=True,
        report_to="none", seed=CFG.SEED, data_seed=CFG.SEED,
    )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=val_dataset, tokenizer=tokenizer,
        compute_metrics=compute_averaged_auc,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
    )
    
    trainer.train()
    
    # Store the final validation metric
    final_log = [log for log in trainer.state.log_history if 'eval_averaged_auc' in log]
    if final_log:
        last_score = final_log[-1]['eval_averaged_auc']
        with open(f"/kaggle/working/score_fold_{fold}.txt", "w") as f:
            f.write(str(last_score))

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, help="Fold number for CV")
    args = parser.parse_args()
    
    seed_everything(CFG.SEED)
    full_df = prepare_training_data()
    
    skf = StratifiedKFold(n_splits=CFG.NUM_FOLDS, shuffle=True, random_state=CFG.SEED)
    train_idx, val_idx = list(skf.split(full_df, full_df['labels']))[args.fold]
    
    train_df = full_df.iloc[train_idx]
    val_df = full_df.iloc[val_idx]

    tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_ID)
    def tokenize_function(examples):
        return tokenizer(examples['text_input'], truncation=True, max_length=CFG.MAX_LENGTH)
    
    train_dataset = Dataset.from_pandas(train_df).map(tokenize_function, batched=True, remove_columns=['rule'])
    val_dataset = Dataset.from_pandas(val_df).map(tokenize_function, batched=True, remove_columns=['rule'])
    
    train_model(train_dataset, val_dataset, tokenizer, val_df, args.fold)


%%writefile inference_dblarge_cv_model.py

import os
import gc
import glob
import pandas as pd
import torch
import argparse # For command-line arguments
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator

class CFG:
    BASE_MODEL_ID = "/kaggle/input/huggingfacedebertav3variants/deberta-v3-large"
    DATA_PATH = "/kaggle/input/jigsaw-agile-community-rules/"
    MAX_LENGTH = 512
    BATCH_SIZE = 8

def find_best_checkpoint_path(fold): # <-- Now takes fold number
    base_path = f"/kaggle/working/model_fold_{fold}/" # <-- Looks in fold-specific dir
    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"Model directory not found for fold {fold}: {base_path}.")
    
    checkpoints = [d for d in os.listdir(base_path) if d.startswith('checkpoint-') and os.path.isdir(os.path.join(base_path, d))]
    
    if not checkpoints:
        raise FileNotFoundError(f"Could not find any 'checkpoint-*' directory in {base_path}.")
        
    # Your original logic: since save_total_limit=1, there's only one, which is the best.
    best_checkpoint_name = checkpoints[0]
    best_model_path = os.path.join(base_path, best_checkpoint_name)
    return best_model_path

def run_inference(fold):
    accelerator = Accelerator()
    
    best_model_path = find_best_checkpoint_path(fold)
    print(f"--- Using checkpoint for fold {fold}: {best_model_path} ---")
    
    tokenizer = AutoTokenizer.from_pretrained(best_model_path)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        CFG.BASE_MODEL_ID, num_labels=2, torch_dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(base_model, best_model_path)
    model = accelerator.prepare(model)
    model.eval()

    test_df = pd.read_csv(os.path.join(CFG.DATA_PATH, "test.csv"))
    full_dataset = Dataset.from_pandas(test_df)

    # Your original inference logic using accelerator.split_between_processes
    # This part is complex and works for you, so it's preserved.
    with accelerator.split_between_processes(full_dataset) as data_slice:
        if data_slice:
            slice_dataset = Dataset.from_dict(data_slice.to_dict())
            slice_dataset = slice_dataset.map(lambda ex: {'text_input': ("Rule: " + str(ex['rule']) + "\n" + "Comment: " + str(ex['body']))})            
            def tokenize_function(ex): return tokenizer(ex['text_input'], truncation=True, max_length=CFG.MAX_LENGTH)
            
            tokenized_data = slice_dataset.map(tokenize_function, batched=True, remove_columns=slice_dataset.column_names)
            tokenized_data.set_format('torch')
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
            loader = DataLoader(tokenized_data, batch_size=CFG.BATCH_SIZE, shuffle=False, collate_fn=data_collator)
            
            all_probs = []
            row_ids = slice_dataset['row_id']
            with torch.no_grad():
                for batch in tqdm(loader, desc=f"Proc {accelerator.process_index}", disable=not accelerator.is_main_process):
                    outputs = model(**batch)
                    probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
                    all_probs.extend(probs.cpu().float().numpy())
            
            result_df = pd.DataFrame({'row_id': row_ids, 'rule_violation': all_probs})
            # Save partial predictions to a unique file to avoid clashes
            result_df.to_csv(f"preds_part_{fold}_{accelerator.process_index}.csv", index=False)

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        part_files = glob.glob(f"preds_part_{fold}_*.csv")
        all_dfs = [pd.read_csv(f) for f in part_files]
        if all_dfs:
            final_preds = pd.concat(all_dfs).sort_values('row_id').reset_index(drop=True)
            # Save the aggregated predictions for THIS FOLD
            final_preds.to_csv(f"preds_fold_{fold}.csv", index=False)
            for f in part_files: os.remove(f)
            
    del model, base_model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, help="Fold number for CV")
    args = parser.parse_args()
    run_inference(args.fold)


import pandas as pd
import os
import shutil
import numpy as np

COMP_DATA_PATH = "/kaggle/input/jigsaw-agile-community-rules/"
NUM_FOLDS = 6 # Should match CFG.NUM_FOLDS in train script
test_df = pd.read_csv(os.path.join(COMP_DATA_PATH, "test.csv"))

if len(test_df) <= 10:
    print("Toy test set detected. Creating dummy submission.")
    pd.DataFrame({'row_id': test_df['row_id'], 'rule_violation': 0.5}).to_csv("submission_deberta_large.csv", index=False)
else:
    print("Full test set detected. Starting 5-Fold CV training and inference.")
    
    for fold in range(NUM_FOLDS):
        print("\n" + "="*50)
        print(f"===== PROCESSING FOLD {fold+1}/{NUM_FOLDS} =====")
        print("="*50)
        
        # --- Step 1: Train the model for the current fold ---
        print(f"\n--- Training Fold {fold} ---")
        train_command = f"python train_dblarge_cv_model.py --fold {fold}"
        os.system(train_command)

        # --- Step 2: Run inference using the trained model for the current fold ---
        print(f"\n--- Inferencing with Fold {fold} Model ---")
        inference_command = f"accelerate launch --multi_gpu --num_processes=2 inference_dblarge_cv_model.py --fold {fold}"
        os.system(inference_command)

        # --- Step 3: Clean up the model directory for this fold to save space ---
        model_dir = f"/kaggle/working/model_fold_{fold}/"
        print(f"Cleaning up directory: {model_dir}")
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)

    # --- Step 4: Aggregate predictions and report scores ---
    print("\n" + "="*50)
    print("===== AGGREGATION AND FINAL SUBMISSION =====")
    print("="*50)
    all_preds = []
    all_scores = []
    for fold in range(NUM_FOLDS):
        pred_df = pd.read_csv(f"preds_fold_{fold}.csv")
        all_preds.append(pred_df['rule_violation'].values)
        
        # Read the score saved by the training script
        try:
            with open(f"/kaggle/working/score_fold_{fold}.txt", "r") as f:
                score = float(f.read().strip())
                all_scores.append(score)
                print(f"Fold {fold+1} Validation Averaged AUC: {score:.5f}")
        except FileNotFoundError:
            print(f"Score file for fold {fold+1} not found.")

    if all_scores:
        mean_auc = np.mean(all_scores)
        std_auc = np.std(all_scores)
        print("-" * 50)
        print(f"Overall CV Averaged AUC: {mean_auc:.5f} +/- {std_auc:.5f}")
        print("-" * 50)

    # Average predictions across folds
    avg_preds = np.mean(all_preds, axis=0)
    submission_df = pd.DataFrame({'row_id': test_df['row_id'], 'rule_violation': avg_preds})
    submission_df.to_csv("submission_deberta_large.csv", index=False)
    
    print("\n--- Pipeline Complete. submission_deberta_large.csv created. ---")
    print(submission_df.head())


%%writefile train_dbbase_cv_model.py

import os
import gc
import pandas as pd
import torch
import random
import numpy as np
import argparse # For command-line arguments
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from peft import get_peft_model, LoraConfig, TaskType

class CFG:
    COMPETITION_NAME = "jigsaw-agile-community-rules"
    DATA_PATH = f"/kaggle/input/{COMPETITION_NAME}/"
    # <<< CHANGE 1: Updated the model ID to 'base' >>>
    MODEL_ID = "/kaggle/input/huggingfacedebertav3variants/deberta-v3-base" 
    MAX_LENGTH = 512
    PER_DEVICE_BATCH_SIZE = 8
    GRAD_ACCUMULATION_STEPS = 2
    BF16 = True
    SEED = 1988
    # <<< CHANGE 2: Updated epochs to match HPO settings >>>
    NUM_EPOCHS = 3
    NUM_FOLDS = 6 

# <<< CHANGE 3: Replaced with new hyperparameters from HPO >>>
BEST_PARAMS = {
    "learning_rate": 0.00019993575066590182,
    "weight_decay": 0.011365217816091013,
    "warmup_ratio": 0.15,
    "lr_scheduler_type": "cosine",
    "lora_r": 64,
    "lora_dropout": 0.0002000771333436424,
    "lora_alpha": 256  # Calculated as lora_r (64) * alpha_multiplier (4)
}

def seed_everything(seed=CFG.SEED):
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
    full_train_df['text_input'] = ("Rule: " + full_train_df['rule'] + "\n" + "Comment: " + full_train_df['body'])
    full_train_df.rename(columns={'rule_violation': 'labels'}, inplace=True)
    return full_train_df[['text_input', 'labels', 'rule']].reset_index(drop=True)

def train_model(train_dataset, val_dataset, tokenizer, val_df_for_metric, fold):
    output_dir = f"/kaggle/working/model_fold_{fold}/"
    os.makedirs(output_dir, exist_ok=True)
    
    lora_config = LoraConfig(
            r=BEST_PARAMS["lora_r"], lora_alpha=BEST_PARAMS["lora_alpha"],
            target_modules=["query_proj", "key_proj", "value_proj", "dense", "o_proj"], # DeBERTa style modules
            lora_dropout=BEST_PARAMS["lora_dropout"], bias="none", task_type=TaskType.SEQ_CLS,
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
        output_dir=output_dir,
        num_train_epochs=CFG.NUM_EPOCHS,
        learning_rate=BEST_PARAMS["learning_rate"],
        weight_decay=BEST_PARAMS["weight_decay"],
        lr_scheduler_type=BEST_PARAMS["lr_scheduler_type"],
        warmup_ratio=BEST_PARAMS["warmup_ratio"],
        per_device_train_batch_size=CFG.PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=CFG.PER_DEVICE_BATCH_SIZE * 2,
        gradient_accumulation_steps=CFG.GRAD_ACCUMULATION_STEPS,
        bf16=CFG.BF16, optim="adamw_8bit", logging_steps=100,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=False,
        ddp_find_unused_parameters=False,
        metric_for_best_model="averaged_auc", greater_is_better=True,
        report_to="none", seed=CFG.SEED, data_seed=CFG.SEED,
    )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=val_dataset, tokenizer=tokenizer,
        compute_metrics=compute_averaged_auc,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
    )
    
    trainer.train()
    
    final_log = [log for log in trainer.state.log_history if 'eval_averaged_auc' in log]
    if final_log:
        last_score = final_log[-1]['eval_averaged_auc']
        with open(f"/kaggle/working/score_fold_{fold}.txt", "w") as f:
            f.write(str(last_score))

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, help="Fold number for CV")
    args = parser.parse_args()
    
    seed_everything(CFG.SEED)
    full_df = prepare_training_data()
    
    skf = StratifiedKFold(n_splits=CFG.NUM_FOLDS, shuffle=True, random_state=CFG.SEED)
    train_idx, val_idx = list(skf.split(full_df, full_df['labels']))[args.fold]
    
    train_df = full_df.iloc[train_idx]
    val_df = full_df.iloc[val_idx]

    tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_ID)
    def tokenize_function(examples):
        return tokenizer(examples['text_input'], truncation=True, max_length=CFG.MAX_LENGTH)
    
    train_dataset = Dataset.from_pandas(train_df).map(tokenize_function, batched=True, remove_columns=['rule'])
    val_dataset = Dataset.from_pandas(val_df).map(tokenize_function, batched=True, remove_columns=['rule'])
    
    train_model(train_dataset, val_dataset, tokenizer, val_df, args.fold)


%%writefile inference_dbbase_cv_model.py

import os
import gc
import glob
import pandas as pd
import torch
import argparse
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator

class CFG:
    # <<< CHANGE: Updated the base model ID to match training >>>
    BASE_MODEL_ID = "/kaggle/input/huggingfacedebertav3variants/deberta-v3-base"
    DATA_PATH = "/kaggle/input/jigsaw-agile-community-rules/"
    MAX_LENGTH = 512
    BATCH_SIZE = 8

def find_best_checkpoint_path(fold): 
    base_path = f"/kaggle/working/model_fold_{fold}/"
    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"Model directory not found for fold {fold}: {base_path}.")
    
    checkpoints = [d for d in os.listdir(base_path) if d.startswith('checkpoint-') and os.path.isdir(os.path.join(base_path, d))]
    
    if not checkpoints:
        raise FileNotFoundError(f"Could not find any 'checkpoint-*' directory in {base_path}.")
        
    best_checkpoint_name = checkpoints[0]
    best_model_path = os.path.join(base_path, best_checkpoint_name)
    return best_model_path

def run_inference(fold):
    accelerator = Accelerator()
    
    best_model_path = find_best_checkpoint_path(fold)
    print(f"--- Using checkpoint for fold {fold}: {best_model_path} ---")
    
    tokenizer = AutoTokenizer.from_pretrained(best_model_path)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        CFG.BASE_MODEL_ID, num_labels=2, torch_dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(base_model, best_model_path)
    model = accelerator.prepare(model)
    model.eval()

    test_df = pd.read_csv(os.path.join(CFG.DATA_PATH, "test.csv"))
    full_dataset = Dataset.from_pandas(test_df)

    with accelerator.split_between_processes(full_dataset) as data_slice:
        if data_slice:
            slice_dataset = Dataset.from_dict(data_slice.to_dict())
            slice_dataset = slice_dataset.map(lambda ex: {'text_input': ("Rule: " + str(ex['rule']) + "\n" + "Comment: " + str(ex['body']))})            
            def tokenize_function(ex): return tokenizer(ex['text_input'], truncation=True, max_length=CFG.MAX_LENGTH)
            
            tokenized_data = slice_dataset.map(tokenize_function, batched=True, remove_columns=slice_dataset.column_names)
            tokenized_data.set_format('torch')
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
            loader = DataLoader(tokenized_data, batch_size=CFG.BATCH_SIZE, shuffle=False, collate_fn=data_collator)
            
            all_probs = []
            row_ids = slice_dataset['row_id']
            with torch.no_grad():
                for batch in tqdm(loader, desc=f"Proc {accelerator.process_index}", disable=not accelerator.is_main_process):
                    outputs = model(**batch)
                    probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
                    all_probs.extend(probs.cpu().float().numpy())
            
            result_df = pd.DataFrame({'row_id': row_ids, 'rule_violation': all_probs})
            result_df.to_csv(f"preds_part_{fold}_{accelerator.process_index}.csv", index=False)

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        part_files = glob.glob(f"preds_part_{fold}_*.csv")
        all_dfs = [pd.read_csv(f) for f in part_files]
        if all_dfs:
            final_preds = pd.concat(all_dfs).sort_values('row_id').reset_index(drop=True)
            final_preds.to_csv(f"preds_fold_{fold}.csv", index=False)
            for f in part_files: os.remove(f)
            
    del model, base_model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, help="Fold number for CV")
    args = parser.parse_args()
    run_inference(args.fold)


%%writefile single_trial_runner_gemma.py

# --- IMPORTS ---
import os
import gc
import json
import pandas as pd
import torch
import random
import numpy as np
import argparse
from functools import partial
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from transformers.utils import is_torch_bf16_gpu_available

# --- CONSTANTS & UTILS (Copied from your original implementation) ---
# It's good practice for the worker to be self-contained
BASE_MODEL_PATH = "/kaggle/input/qwen-3-embedding/transformers/0.6b/1"
DATA_PATH = "/kaggle/input/jigsaw-agile-community-rules/"
OUTPUT_DIR_BASE = "/kaggle/working/gemma-hpo-trials-distributed/" # HPO Base Dir
POSITIVE_ANSWER = "Yes"
NEGATIVE_ANSWER = "No"
COMPLETE_PHRASE = "Answer:"
BASE_PROMPT = """You are an expert Reddit moderator. Your task is to determine if a comment violates a given subreddit rule.
Analyze the rule and the comment carefully. Respond with only "Yes" if it violates the rule, or "No" if it does not."""
SEED = 1988
TEST_SIZE = 0.1

def set_seed(seed_value=SEED):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_dataframe_to_train(data_path):
    train_df_orig = pd.read_csv(f"{data_path}/train.csv")
    test_df_orig = pd.read_csv(f"{data_path}/test.csv")
    all_data = []
    train_main = train_df_orig[['rule', 'subreddit', 'body', 'rule_violation']].copy()
    all_data.append(train_main)
    for i in range(1, 3):
        for prefix in ['positive', 'negative']:
            label = 1 if prefix == 'positive' else 0
            df = train_df_orig[['rule', 'subreddit', f'{prefix}_example_{i}']].copy()
            df.rename(columns={f'{prefix}_example_{i}': 'body'}, inplace=True)
            df['rule_violation'] = label
            all_data.append(df)
            df = test_df_orig[['rule', 'subreddit', f'{prefix}_example_{i}']].copy()
            df.rename(columns={f'{prefix}_example_{i}': 'body'}, inplace=True)
            df['rule_violation'] = label
            all_data.append(df)
    full_train_df = pd.concat(all_data, ignore_index=True)
    full_train_df.dropna(subset=['body'], inplace=True)
    full_train_df['body'] = full_train_df['body'].astype(str)
    full_train_df.drop_duplicates(subset=['rule', 'body'], keep='first', inplace=True)
    return full_train_df

def build_prompt(row):
    return f"""{BASE_PROMPT}\n\nRule: {row["rule"]}\n\nComment: {row["body"]}\n\n{COMPLETE_PHRASE}"""

def build_dataset(dataframe, is_train=True):
    dataframe["prompt"] = dataframe.apply(build_prompt, axis=1)
    columns_to_keep = ["prompt", "rule"]
    if is_train and "rule_violation" in dataframe.columns:
        dataframe["completion"] = dataframe["rule_violation"].map({1: POSITIVE_ANSWER, 0: NEGATIVE_ANSWER})
        columns_to_keep.extend(["completion", "rule_violation"])
    dataset = Dataset.from_pandas(dataframe[columns_to_keep])
    return dataset

# --- METRIC FUNCTIONS (Unchanged from your original implementation) ---
def compute_averaged_auc(eval_pred, validation_df):
    logits, _ = eval_pred.predictions, eval_pred.label_ids
    probs = torch.softmax(torch.from_numpy(logits), dim=-1)
    positive_scores = probs[:, 0].numpy()
    results_df = pd.DataFrame({
        'labels': validation_df['rule_violation'].values,
        'preds': positive_scores,
        'rule': validation_df['rule'].values
    })
    auc_scores = [roc_auc_score(group['labels'], group['preds']) for _, group in results_df.groupby('rule') if len(group['labels'].unique()) > 1]
    return {"averaged_auc": np.mean(auc_scores) if auc_scores else 0.0}

def preprocess_logits_for_metrics(logits, labels, tokenizer):
    pos_token_id = tokenizer.convert_tokens_to_ids(POSITIVE_ANSWER)
    neg_token_id = tokenizer.convert_tokens_to_ids(NEGATIVE_ANSWER)
    mask = labels != -100
    first_completion_token_indices = torch.argmax(mask.to(torch.int), axis=1)
    prediction_indices = first_completion_token_indices - 1
    batch_size = logits.shape[0]
    pred_logits = logits[torch.arange(batch_size), prediction_indices]
    return pred_logits[:, [pos_token_id, neg_token_id]]

# --- MAIN TRIAL FUNCTION ---
def run_single_trial(params, trial_num, train_dataset, eval_dataset, val_df_for_metric):
    trial_output_dir = os.path.join(OUTPUT_DIR_BASE, f"trial_{trial_num}")
    
    lora_config = LoraConfig(
        r=params["lora_r"],
        lora_alpha=params["lora_alpha"],
        lora_dropout=params["lora_dropout"],
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir=trial_output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=2,
        optim="paged_adamw_8bit",
        learning_rate=params["learning_rate"],
        weight_decay=params["weight_decay"],
        max_grad_norm=1.0,
        lr_scheduler_type=params["lr_scheduler_type"],
        warmup_ratio=params["warmup_ratio"],
        bf16=is_torch_bf16_gpu_available(),
        fp16=not is_torch_bf16_gpu_available(),
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=False, # We find the best later
        metric_for_best_model="averaged_auc",
        greater_is_better=True,
        report_to="none",
        completion_only_loss=True,
        packing=False,
        remove_unused_columns=False,
        seed=SEED,
        data_seed=SEED
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    trainer = SFTTrainer(
        model=BASE_MODEL_PATH,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
        compute_metrics=partial(compute_averaged_auc, validation_df=val_df_for_metric),
        preprocess_logits_for_metrics=partial(preprocess_logits_for_metrics, tokenizer=tokenizer),
    )
    
    best_score = 0.0
    try:
        trainer.train()
        # Find the best score from the training history
        for log in trainer.state.log_history:
            if 'eval_averaged_auc' in log:
                score = log['eval_averaged_auc']
                if score > best_score:
                    best_score = score
    except Exception as e:
        print(f"TRIAL {trial_num} FAILED INSIDE TRAINER with error: {e}")
    finally:
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

    # <<< KEY CHANGE: Write score to file for the manager process >>>
    if os.environ.get("LOCAL_RANK", "0") == "0":
        score_file = os.path.join(OUTPUT_DIR_BASE, f"trial_{trial_num}_score.json")
        with open(score_file, 'w') as f:
            json.dump({"averaged_auc": best_score}, f)
        print(f"TRIAL {trial_num} SCORE: {best_score} written to {score_file}")

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params_json", type=str, required=True, help="JSON string of the trial parameters")
    parser.add_argument("--trial_num", type=int, required=True, help="The trial number")
    args = parser.parse_args()
    
    params = json.loads(args.params_json)
    
    set_seed(SEED)
    
    # Data preparation must happen inside this new process
    full_dataframe = get_dataframe_to_train(DATA_PATH)
    train_df, val_df = train_test_split(
        full_dataframe, test_size=TEST_SIZE, random_state=SEED, stratify=full_dataframe['rule_violation']
    )
    train_dataset = build_dataset(train_df, is_train=True)
    eval_dataset = build_dataset(val_df, is_train=True)
    
    run_single_trial(params, args.trial_num, train_dataset, eval_dataset, val_df)


%%writefile run_hpo_manager_gemma.py

# --- IMPORTS ---
import os
import json
import optuna
from optuna.samplers import TPESampler

# --- CONFIGURATION (Just what the manager needs) ---
class CFG:
    OUTPUT_DIR_BASE = "/kaggle/working/gemma-hpo-trials-distributed/"
    N_TRIALS = 5 # Adjust as needed
    SEED = 1988

def run_hpo_manager():
    os.makedirs(CFG.OUTPUT_DIR_BASE, exist_ok=True)
    
    sampler = TPESampler(seed=CFG.SEED) 
    study = optuna.create_study(direction="maximize", study_name="Gemma-LoRA-Jigsaw-Distributed", sampler=sampler)

    for i in range(CFG.N_TRIALS):
        trial = study.ask()
        
        # <<< KEY CHANGE: Parameter space defined for Gemma/LoRA fine-tuning >>>
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 9e-4, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 0.01, 0.2, log=True),
            "warmup_ratio": trial.suggest_categorical("warmup_ratio", [0.1, 0.15, 0.2]),
            "lr_scheduler_type": trial.suggest_categorical("lr_scheduler_type", ["linear", "cosine"]),
            "lora_r": trial.suggest_categorical("lora_r", [16, 32, 64]),
            "alpha_multiplier": trial.suggest_categorical("alpha_multiplier", [2, 4]),
            "lora_dropout": trial.suggest_float("lora_dropout", 1e-4, 0.05, log=True)
        }
        params["lora_alpha"] = params["lora_r"] * params["alpha_multiplier"]
        del params["alpha_multiplier"]
        
        trial_num = trial.number
        print(f"\n{'='*20} LAUNCHING GEMMA TRIAL {trial_num} {'='*20}")
        print(f"PARAMETERS: {json.dumps(params, indent=2)}")

        params_json_str = json.dumps(params)

        # <<< KEY CHANGE: Launch the Gemma worker script >>>
        command = (
            f'accelerate launch --multi_gpu --num_processes=2 --mixed_precision=bf16 single_trial_runner_gemma.py '
            f'--trial_num {trial_num} --params_json \'{params_json_str}\''
        )
        
        exit_code = os.system(command)

        score = 0.0
        if exit_code == 0:
            try:
                score_file = os.path.join(CFG.OUTPUT_DIR_BASE, f"trial_{trial_num}_score.json")
                with open(score_file, 'r') as f:
                    result = json.load(f)
                    score = result.get("averaged_auc", 0.0)
                print(f"TRIAL {trial_num} SUCCEEDED. Score: {score}")
            except Exception as e:
                print(f"TRIAL {trial_num} finished but could not read score file. Error: {e}")
        else:
            print(f"TRIAL {trial_num} FAILED with a non-zero exit code ({exit_code}).")
        
        study.tell(trial, score)
    
    print("\n--- Gemma HPO Complete ---")
    best_trial = study.best_trial
    print(f"Best Value (Averaged AUC): {best_trial.value:.5f}")
    print("Best Parameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
        
    results_path = os.path.join(CFG.OUTPUT_DIR_BASE, "hpo_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            "best_trial_number": best_trial.number,
            "best_value_auc": best_trial.value,
            "best_params": best_trial.params,
        }, f, indent=4)
    print(f"\nBest trial results saved to: {results_path}")

if __name__ == "__main__":
    run_hpo_manager()


%%writefile inference_accelerate_best_model_gemma.py

# --- IMPORTS ---
import os
import gc
import json
import glob
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator

# --- CONSTANTS & UTILS (Slightly adapted for HPO context) ---
class CFG:
    HPO_BASE_DIR = "/kaggle/working/gemma-hpo-trials-distributed/"
    RESULTS_FILE = os.path.join(HPO_BASE_DIR, "hpo_results.json") 
    BASE_MODEL_PATH = "/kaggle/input/qwen-3-embedding/transformers/0.6b/1"
    DATA_PATH = "/kaggle/input/jigsaw-agile-community-rules/"
    MAX_LENGTH = 2048
    BATCH_SIZE = 4
    POSITIVE_ANSWER = "Yes"
    NEGATIVE_ANSWER = "No"
    COMPLETE_PHRASE = "Answer:"
    BASE_PROMPT = """You are an expert Reddit moderator. Your task is to determine if a comment violates a given subreddit rule.
Analyze the rule and the comment carefully. Respond with only "Yes" if it violates the rule, or "No" if it does not."""

def build_prompt(row):
    return f"""{CFG.BASE_PROMPT}\n\nRule: {row["rule"]}\n\nComment: {row["body"]}\n\n{CFG.COMPLETE_PHRASE}"""

def find_subsequence(haystack, needle):
    h, n = haystack.size(0), needle.size(0)
    for i in range(h - n, -1, -1):
        if torch.equal(haystack[i:i+n], needle):
            return i + n - 1
    return -1

# In inference_accelerate_best_model_gemma.py

def find_best_checkpoint_path(trial_number):
    """Finds the path to the best model checkpoint from an HPO trial directory."""
    trial_path = os.path.join(CFG.HPO_BASE_DIR, f"trial_{trial_number}")
    if not os.path.isdir(trial_path):
        raise FileNotFoundError(f"Trial directory not found: {trial_path}.")

    checkpoints = [d for d in os.listdir(trial_path) if d.startswith('checkpoint-') and os.path.isdir(os.path.join(trial_path, d))]
    if not checkpoints:
        raise FileNotFoundError(f"Could not find any 'checkpoint-*' directory in {trial_path}.")

    # Find the final checkpoint to read its trainer_state.json
    # This state file contains the history for the entire run.
    latest_checkpoint_name = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
    latest_checkpoint_path = os.path.join(trial_path, latest_checkpoint_name)
    
    state_file = os.path.join(latest_checkpoint_path, "trainer_state.json")
    if not os.path.exists(state_file):
        raise FileNotFoundError(f"Could not find 'trainer_state.json' in {latest_checkpoint_path}.")

    with open(state_file, 'r') as f:
        state = json.load(f)

    # Now, parse the log history to find which step had the best score
    best_metric = -1.0
    best_step = -1
    for log_entry in state['log_history']:
        if 'eval_averaged_auc' in log_entry:
            current_metric = log_entry['eval_averaged_auc']
            if current_metric > best_metric:
                best_metric = current_metric
                best_step = log_entry['step']
    
    if best_step == -1:
        # Fallback if no eval metric is found (should not happen)
        print(f"Warning: Could not determine best step from logs. Using latest checkpoint: {latest_checkpoint_name}")
        best_model_path = latest_checkpoint_path
    else:
        # The actual best model is in the folder named after its step number
        best_model_path = os.path.join(trial_path, f"checkpoint-{best_step}")

    if not os.path.isdir(best_model_path):
         # This can happen if save_total_limit deleted the best checkpoint.
         # In that case, the latest one is the only one available.
        print(f"Warning: Best checkpoint '{best_model_path}' was not found (likely deleted by save_total_limit). Using latest available checkpoint: {latest_checkpoint_path}")
        best_model_path = latest_checkpoint_path

    print(f"--- Automatically retrieved best model path for trial {trial_number}: {best_model_path} ---")
    return best_model_path

# --- MAIN INFERENCE LOGIC ---
# In inference_accelerate_best_model_gemma.py

def run_inference():
    accelerator = Accelerator()
    
    print(f"Loading HPO results from: {CFG.RESULTS_FILE}")
    with open(CFG.RESULTS_FILE, 'r') as f:
        hpo_results = json.load(f)
    best_trial_num = hpo_results["best_trial_number"]
    best_model_path = find_best_checkpoint_path(best_trial_num)
    
    tokenizer = AutoTokenizer.from_pretrained(CFG.BASE_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    pos_token_id = tokenizer.convert_tokens_to_ids(CFG.POSITIVE_ANSWER)
    neg_token_id = tokenizer.convert_tokens_to_ids(CFG.NEGATIVE_ANSWER)
    
    # This is fine, it gets moved to the device correctly for each process
    prompt_end_phrase_ids = torch.tensor(tokenizer(CFG.COMPLETE_PHRASE, add_special_tokens=False).input_ids, device=accelerator.device)

    if accelerator.is_main_process: print(f"Loading base model and applying adapter from {best_model_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(CFG.BASE_MODEL_PATH, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base_model, best_model_path)
    model = model.merge_and_unload()
    
    # This prepares the model for the correct device
    model = accelerator.prepare(model)
    model.eval()

    test_df = pd.read_csv(os.path.join(CFG.DATA_PATH, "test.csv"))
    all_probs = []

    with accelerator.split_between_processes(test_df) as slice_df:
        slice_df['prompt'] = slice_df.apply(build_prompt, axis=1)
        tokenized_data = tokenizer(slice_df['prompt'].tolist(), truncation=True, max_length=CFG.MAX_LENGTH, padding='longest')
        slice_dataset = Dataset.from_dict(tokenized_data).with_format('torch')
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        loader = DataLoader(slice_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, collate_fn=data_collator)

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Inference Process {accelerator.process_index}", disable=not accelerator.is_main_process):
                # <<< THIS IS THE FIX >>>
                # Move the entire batch of tensors to the accelerator's device (e.g., cuda:0 or cuda:1)
                batch = {k: v.to(accelerator.device) for k, v in batch.items()}
                
                # Now, all tensors in `batch` are on the correct GPU
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                logits = outputs.logits
                
                # This will now work correctly as both tensors are on the same GPU
                prediction_indices = [find_subsequence(ids, prompt_end_phrase_ids) for ids in batch['input_ids']]
                
                pred_logits = logits[torch.arange(batch['input_ids'].shape[0]), prediction_indices]
                yes_no_logits = pred_logits[:, [pos_token_id, neg_token_id]]
                probs = torch.softmax(yes_no_logits, dim=-1)
                all_probs.extend(probs[:, 0].cpu().float().numpy())

    gathered_probs = accelerator.gather_for_metrics(all_probs)
    
    if accelerator.is_main_process:
        print("\nAggregating results...")
        result_df = pd.DataFrame({'row_id': test_df['row_id'], 'rule_violation': gathered_probs[:len(test_df)]})
        result_df.to_csv("submission_qwen.csv", index=False)
        print("\nInference complete. submission.csv created successfully.")
        print(result_df.head())

if __name__ == "__main__":
    run_inference()


import pandas as pd
import os
import shutil

# --- Configuration ---
COMP_DATA_PATH = "/kaggle/input/jigsaw-agile-community-rules/"
HPO_OUTPUT_DIR = "/kaggle/working/gemma-hpo-trials-distributed/"

# --- Main Logic ---
test_df = pd.read_csv(os.path.join(COMP_DATA_PATH, "test.csv"))

# This logic allows the notebook to commit quickly without running the full HPO
if len(test_df) <= 10:
    print("Toy test set detected. Creating a dummy submission file.")
    dummy_submission = pd.DataFrame({'row_id': test_df['row_id'], 'rule_violation': 0.5})
    dummy_submission.to_csv("submission_qwen.csv", index=False)
    print("Dummy submission.csv created successfully.")
else:
    print("Full test set detected. Proceeding with HPO, training, and inference.")

    print(f"Removing previous HPO state from: {HPO_OUTPUT_DIR}")
    if os.path.exists(HPO_OUTPUT_DIR):
        shutil.rmtree(HPO_OUTPUT_DIR)
    
    print("\n--- Starting HPO Training Manager for Gemma ---")
    os.system("python run_hpo_manager_gemma.py")

    print("\n--- Starting Inference on Best Gemma Model ---")
    os.system("accelerate launch inference_accelerate_best_model_gemma.py")

    print("\n--- Gemma Pipeline Complete ---")





import pandas as pd
import numpy as np

q = pd.read_csv('submission_deberta_large.csv')
l = pd.read_csv('submission_e5.csv')
m = pd.read_csv('submission_qwen.csv')


rq = q['rule_violation'].rank(method='average') / (len(q)+1)
rl = l['rule_violation'].rank(method='average') / (len(l)+1)
rm = m['rule_violation'].rank(method='average') / (len(m)+1)


blend = (rq + rl +rm )/3
q['rule_violation'] = blend
q.to_csv('/kaggle/working/submission.csv', index=False)
































































