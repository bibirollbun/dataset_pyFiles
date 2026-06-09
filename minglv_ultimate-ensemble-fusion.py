import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
from torch.utils.data import DataLoader   
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
from datasets import Dataset
from peft import PeftModel
import gc

print("Step 1: Libraries imported and environment set up.")


DATA_PATH = '/kaggle/input/map-charting-student-math-misunderstandings/'

train = pd.read_csv(f'{DATA_PATH}train.csv')
test = pd.read_csv(f'{DATA_PATH}test.csv')

print(f"Step 2: Data loaded - Train: {train.shape}, Test: {test.shape}")


le = LabelEncoder()
train.Misconception = train.Misconception.fillna('NA')
train['target'] = train.Category + ':' + train.Misconception
train['label'] = le.fit_transform(train['target'])

# Create correct answer mapping
idx = train.apply(lambda row: row.Category.split('_')[0], axis=1) == 'True'
correct = train.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId', 'MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c', ascending=False).drop_duplicates(['QuestionId'])[['QuestionId', 'MC_Answer']]
correct['is_correct'] = 1

test = test.merge(correct, on=['QuestionId', 'MC_Answer'], how='left')
test.is_correct = test.is_correct.fillna(0)

print(f"Step 3: Feature engineering completed - {len(le.classes_)} classes")


# def format_input(row):
#     correctness = "This answer is correct." if row['is_correct'] else "This answer is incorrect."
#     return f"Question: {row['QuestionText']}\nAnswer: {row['MC_Answer']}\n{correctness}\nStudent Explanation: {row['StudentExplanation']}"

# test['text'] = test.apply(format_input, axis=1)
# ds_test = Dataset.from_pandas(test)

# print("Step 4: Model input formatted.")


def format_input(row):
    correctness = "This answer is correct." if row['is_correct'] else "This answer is incorrect."
    return f"Question: {row['QuestionText']}\nAnswer: {row['MC_Answer']}\n{correctness}\nStudent Explanation: {row['StudentExplanation']}"
test['text'] = test.apply(format_input, axis=1)
ds_test = Dataset.from_pandas(test)
print("Step 4: Model input formatted.")


def get_predictions(model_path, ds_test, model_type="standard"):
    """Enhanced inference function for all model types"""
    print(f"Loading model from: {model_path}")
    
    # Clear memory
    torch.cuda.empty_cache()
    gc.collect()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model - try PEFT first, fallback to standard
    if model_type == "peft":
        try:
            print("Attempting PEFT model loading...")
            # Try different base model paths
            base_model_paths = [
                "/kaggle/input/gemma2-9b-it-bf16", 
                "google/gemma-2-9b-it",
                model_path  # Use the model path itself as base
            ]
            
            model = None
            for base_path in base_model_paths:
                try:
                    print(f"Trying base model: {base_path}")
                    base_model = AutoModelForSequenceClassification.from_pretrained(
                        base_path,
                        num_labels=65,
                        torch_dtype=torch.bfloat16,
                        device_map="auto"
                    )
                    model = PeftModel.from_pretrained(base_model, model_path)
                    print(f"PEFT loading successful with base: {base_path}")
                    break
                except Exception as e:
                    print(f"Failed with base {base_path}: {str(e)[:100]}")
                    continue
            
            if model is None:
                raise Exception("All PEFT attempts failed")
                
        except Exception as e:
            print(f"PEFT loading failed: {str(e)[:100]}")
            print("Falling back to standard model loading...")
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
    else:
        # For standard models (Mistral/Qwen)
        print("Loading standard model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
    
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Tokenization
    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)
    
    ds_test_tokenized = ds_test.map(tokenize, batched=True)
    
    # Setup trainer for inference
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./temp",
            do_predict=True,
            per_device_eval_batch_size=2,
            fp16=True,
            report_to='none'
        ),
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer)
    )
    
    print("Starting inference...")
    predictions = trainer.predict(ds_test_tokenized)
    logits = predictions.predictions
    
    # Cleanup
    del model, trainer, tokenizer, ds_test_tokenized
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"Inference completed. Shape: {logits.shape}")
    return logits

print("Step 5: Enhanced inference function defined.")


# ========================== CELL 6: Three-Model Ensemble Inference ==========================
# Check and recreate ds_test if needed
if 'ds_test' not in globals():
    print("ds_test not found, recreating...")
    def format_input(row):
        correctness = "This answer is correct." if row['is_correct'] else "This answer is incorrect."
        return f"Question: {row['QuestionText']}\nAnswer: {row['MC_Answer']}\n{correctness}\nStudent Explanation: {row['StudentExplanation']}"
    
    test['text'] = test.apply(format_input, axis=1)
    ds_test = Dataset.from_pandas(test)
    print("ds_test recreated successfully.")

# Model paths
mistral_model_path = '/kaggle/input/map-exp-14-full/MAP_EXP_14_FULL'
qwen_model_path = '/kaggle/input/qwen3-8b-map-competition/MAP_EXP_16_FULL'
deepseek_model_path = '/kaggle/input/deekseepmath-7b-map-competition/MAP_EXP_09_FULL'

print("Starting three-model ensemble inference...")

print("\n=== Model 1: Mistral-7B ===")
predictions_mistral = get_predictions(mistral_model_path, ds_test, "standard")

print("\n=== Model 2: Qwen-3.8B ===")
predictions_qwen = get_predictions(qwen_model_path, ds_test, "standard")

print("\n=== Model 3: DeepSeek-Math-7B ===")
predictions_deepseek = get_predictions(deepseek_model_path, ds_test, "standard")

print("Step 6: All three models completed successfully!")


# ========================== CELL 7: Generate Enhanced Submission ==========================
# Three-model ensemble: Mistral + Qwen + DeepSeek-Math
print("Creating enhanced three-model ensemble...")

# Weight configuration based on reference script and your successful approach
mistral_weight = 0.1    # Your proven weight
qwen_weight = 0.6       # Higher weight for your best performer
deepseek_weight = 0.3   # Solid weight for math-specialized model

print(f"Using weights: Mistral={mistral_weight}, Qwen={qwen_weight}, DeepSeek-Math={deepseek_weight}")

# Weighted ensemble combination
ensembled_predictions = (mistral_weight * predictions_mistral + 
                        qwen_weight * predictions_qwen + 
                        deepseek_weight * predictions_deepseek)

# Get Top 3 predictions
top3_indices = np.argsort(-ensembled_predictions, axis=1)[:, :3]

# Convert to labels
flat_top3 = top3_indices.flatten()
decoded_labels = le.inverse_transform(flat_top3)
top3_labels = decoded_labels.reshape(top3_indices.shape)

# Format output
joined_preds = [" ".join(preds) for preds in top3_labels]

# Create submission
submission_df = pd.DataFrame({
    "row_id": test.row_id.values,
    "Category:Misconception": joined_preds
})

submission_df.to_csv("submission.csv", index=False)

print("Step 7: Three-model ensemble submission created!")
print(f"Submission shape: {submission_df.shape}")
print(f"Applied weights: Mistral={mistral_weight}, Qwen={qwen_weight}, DeepSeek={deepseek_weight}")
print("Target score: 0.94 → 0.945+")
print("\nFirst few predictions:")
print(submission_df.head())

