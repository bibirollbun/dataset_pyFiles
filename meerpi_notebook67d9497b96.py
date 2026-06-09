# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import gc
import os

# --- 0. Setup ---
# Set device and enable optimizations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Enable mixed precision if available
use_fp16 = torch.cuda.is_available()
if use_fp16:
    print("Mixed precision training enabled")

# Memory optimization
torch.backends.cudnn.benchmark = True
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# --- 1. Load Data ---
file_path = '/kaggle/input/neurips-open-polymer-prediction-2025/train.csv'
try:
    # Use more efficient data types
    df = pd.read_csv(file_path, dtype={'SMILES': 'string'})
    print("Successfully loaded the dataset.")
except FileNotFoundError:
    print(f"Error: The file was not found at {file_path}")
    exit()

# --- 2. Data Cleaning and Preparation ---
# Drop rows where the target variable 'FFV' is NaN
df_clean = df.dropna(subset=['FFV']).copy()

# Keep longer SMILES for maximum VRAM utilization
initial_count = len(df_clean)
df_clean = df_clean[df_clean['SMILES'].str.len() <= 1000]  # Much higher limit
print(f"Filtered out {initial_count - len(df_clean)} sequences longer than 1000 characters")

# Normalize target values for better training stability
ffv_mean = df_clean['FFV'].mean()
ffv_std = df_clean['FFV'].std()
df_clean['FFV_normalized'] = (df_clean['FFV'] - ffv_mean) / ffv_std

df_model = df_clean[['SMILES', 'FFV_normalized']].rename(columns={'FFV_normalized': 'labels'})
print(f"Data prepared. Using {len(df_model)} rows for training and evaluation.")
print(f"Target normalization: mean={ffv_mean:.4f}, std={ffv_std:.4f}")

# Clear memory
del df, df_clean
gc.collect()

# --- 3. Create Hugging Face Dataset ---
dataset = Dataset.from_pandas(df_model)

# --- 4. Tokenization ---
model_name = "seyonec/ChemBERTa-zinc-base-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add padding token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    # Longer sequences to utilize more VRAM
    return tokenizer(
        examples["SMILES"], 
        truncation=True, 
        max_length=512,   # Back to full length for maximum utilization
        padding=False     # Dynamic padding with DataCollator
    )

# Use more workers for faster tokenization (reduced for compatibility)
tokenized_datasets = dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=['SMILES']  # Remove unnecessary columns
)

# Split the data with stratification for better evaluation
split_dataset = tokenized_datasets.train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

# --- 5. Model Setup ---
# Load model WITHOUT gradient checkpointing for max speed
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=1,
    problem_type="regression"  # Explicitly set for regression
).to(device)

# Don't enable gradient checkpointing - we want to use more VRAM for speed
# model.gradient_checkpointing_enable()  # Commented out

# --- 6. Custom Metrics ---
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.squeeze()
    
    # Denormalize predictions and labels for interpretable metrics
    pred_denorm = predictions * ffv_std + ffv_mean
    labels_denorm = labels * ffv_std + ffv_mean
    
    mse = mean_squared_error(labels_denorm, pred_denorm)
    mae = mean_absolute_error(labels_denorm, pred_denorm)
    r2 = r2_score(labels_denorm, pred_denorm)
    rmse = np.sqrt(mse)
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }

# --- 7.5. Dynamic Batch Size Finder (Optional) ---
def find_max_batch_size(model, train_dataset, tokenizer, data_collator, start_batch_size=128):
    """Automatically find the maximum batch size that fits in VRAM"""
    print("\nğŸ”� Finding maximum batch size...")
    
    batch_size = start_batch_size
    max_working_batch_size = 16  # Fallback
    
    for batch_size in [256, 192, 128, 96, 64, 48, 32, 16]:
        try:
            print(f"Testing batch size: {batch_size}")
            
            # Create temporary training args
            temp_args = TrainingArguments(
                output_dir="./temp",
                per_device_train_batch_size=batch_size,
                max_steps=1,  # Just one step for testing
                logging_steps=999,
                fp16=use_fp16,
                report_to=[],
            )
            
            # Create temporary trainer
            temp_trainer = Trainer(
                model=model,
                args=temp_args,
                train_dataset=train_dataset.select(range(min(batch_size * 2, len(train_dataset)))),
                tokenizer=tokenizer,
                data_collator=data_collator,
            )
            
            # Try one training step
            temp_trainer.train()
            max_working_batch_size = batch_size
            print(f"âœ… Batch size {batch_size} works!")
            break
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"â�Œ Batch size {batch_size} - Out of memory")
                torch.cuda.empty_cache()
                continue
            else:
                print(f"â�Œ Batch size {batch_size} - Other error: {e}")
                continue
        except Exception as e:
            print(f"â�Œ Batch size {batch_size} - Error: {e}")
            continue
    
    print(f"ğŸ�¯ Maximum working batch size: {max_working_batch_size}")
    return max_working_batch_size

# Uncomment to automatically find max batch size
# optimal_batch_size = find_max_batch_size(model, train_dataset, tokenizer, data_collator)
# print(f"Using optimal batch size: {optimal_batch_size}")

# --- 8. Training Setup ---
# Data collator for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# High VRAM utilization training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=128,  # Much larger batch size
    per_device_eval_batch_size=256,   # Even larger for eval
    gradient_accumulation_steps=1,    # No accumulation needed with large batches
    
    # Learning rate optimization for large batches
    learning_rate=5e-5,  # Higher LR for larger batches
    warmup_steps=100,
    weight_decay=0.01,
    
    # Efficiency settings
    fp16=use_fp16,
    
    # Evaluation and logging
    do_eval=True,
    logging_steps=25,    # More frequent logging
    save_steps=100,
    
    # Disable unnecessary features
    report_to=[],
)

# Create trainer with optimizations (simplified for older versions)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# --- 8. Training ---
print("\nStarting high-VRAM model fine-tuning...")
print(f"Training samples: {len(train_dataset)}")
print(f"Evaluation samples: {len(eval_dataset)}")
print(f"Effective batch size: {training_args.per_device_train_batch_size}")
print(f"Expected VRAM usage: HIGH (batch_size=128, max_length=512)")

# Don't clear cache - we want to use all available VRAM
# if torch.cuda.is_available():
#     torch.cuda.empty_cache()

trainer.train()
print("Fine-tuning finished.")

# --- 9. Final Evaluation ---
print("\nEvaluating the final model...")
final_metrics = trainer.evaluate()
print("\n--- Final Evaluation Results ---")
for key, value in final_metrics.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

# --- 10. Save Model ---
final_model_path = "./chemberta_ffv_predictor"
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)

# Save normalization parameters for inference
normalization_params = {
    'mean': float(ffv_mean),
    'std': float(ffv_std)
}
import json
with open(f"{final_model_path}/normalization_params.json", 'w') as f:
    json.dump(normalization_params, f)

print(f"\nâœ… Optimized fine-tuning complete!")
print(f"Model and tokenizer saved to '{final_model_path}'")
print(f"Normalization parameters saved for inference")

# --- 12.5. Individual SMILES Prediction Function ---
def predict_single_smiles(model_path, smiles_string, ffv_mean, ffv_std):
    """Predict FFV for a single SMILES string"""
    try:
        # Load model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.eval()
        
        # Tokenize
        inputs = tokenizer(
            smiles_string,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            prediction_normalized = outputs.logits.squeeze().cpu().numpy()
        
        # Denormalize
        prediction = prediction_normalized * ffv_std + ffv_mean
        
        return prediction.item() if hasattr(prediction, 'item') else prediction
        
    except Exception as e:
        print(f"Error predicting SMILES: {e}")
        return None

# Example usage after training:
print(f"\nğŸ”¬ Testing individual SMILES predictions:")
test_smiles = [
    "*C(=O)NNC(=O)c1ccc([Si](c2ccccc2)(c2ccccc2)c2ccc(C(=O)NNC(=O)c3ccc(*)nc3)cc2)cc1",
    "*C(=O)NNC(=O)c1ccc([Si](c2ccccc2)(c2ccccc2)c2ccc(C(=O)NNC(=O)c3cncc(*)c3)cc2)cc1"
]

for smiles in test_smiles:
    prediction = predict_single_smiles(final_model_path, smiles, ffv_mean, ffv_std)
    if prediction:
        print(f"SMILES: {smiles[:80]}...")
        print(f"Predicted FFV: {prediction:.6f}")
        print("-" * 40)

# --- 12. Test on External Dataset ---
def test_model_on_dataset(model_path, test_file_path, ffv_mean, ffv_std):
    """Test the fine-tuned model on external dataset and calculate accuracy metrics"""
    print(f"\nğŸ§ª Testing model on external dataset: {test_file_path}")
    
    try:
        # Load test dataset
        test_df = pd.read_csv(test_file_path)
        print(f"Loaded test dataset with {len(test_df)} samples")
        print("Sample data:")
        print(test_df.head())
        
        # Clean test data
        test_df_clean = test_df.dropna(subset=['FFV']).copy()
        print(f"After cleaning: {len(test_df_clean)} samples")
        
        # Load the fine-tuned model and tokenizer
        print("Loading fine-tuned model...")
        test_model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        test_tokenizer = AutoTokenizer.from_pretrained(model_path)
        test_model.eval()  # Set to evaluation mode
        
        # Prepare predictions
        predictions = []
        actual_values = test_df_clean['FFV'].values
        smiles_list = test_df_clean['SMILES'].tolist()
        
        print("Making predictions...")
        # Process in batches for efficiency
        batch_size = 32
        with torch.no_grad():
            for i in range(0, len(smiles_list), batch_size):
                batch_smiles = smiles_list[i:i + batch_size]
                
                # Tokenize batch
                inputs = test_tokenizer(
                    batch_smiles,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(device)
                
                # Get predictions
                outputs = test_model(**inputs)
                batch_predictions = outputs.logits.squeeze().cpu().numpy()
                
                # Handle single sample case
                if batch_predictions.ndim == 0:
                    batch_predictions = [batch_predictions.item()]
                elif len(batch_predictions.shape) == 0:
                    batch_predictions = [batch_predictions]
                
                predictions.extend(batch_predictions)
        
        predictions = np.array(predictions)
        
        # Denormalize predictions (convert back to original scale)
        predictions_denorm = predictions * ffv_std + ffv_mean
        actual_denorm = actual_values  # Actual values are already in original scale
        
        # Calculate metrics
        mse = mean_squared_error(actual_denorm, predictions_denorm)
        mae = mean_absolute_error(actual_denorm, predictions_denorm)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual_denorm, predictions_denorm)
        
        # Calculate percentage errors
        mape = np.mean(np.abs((actual_denorm - predictions_denorm) / actual_denorm)) * 100
        
        print(f"\nğŸ“Š Test Results on External Dataset:")
        print(f"{'='*50}")
        print(f"Number of test samples: {len(actual_denorm)}")
        print(f"Mean Squared Error (MSE): {mse:.6f}")
        print(f"Mean Absolute Error (MAE): {mae:.6f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
        print(f"RÂ² Score: {r2:.6f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        print(f"{'='*50}")
        
        # Show some example predictions
        print(f"\nğŸ”� Sample Predictions vs Actual:")
        print(f"{'Actual':<10} {'Predicted':<12} {'Error':<10} {'SMILES (first 50 chars)'}")
        print("-" * 80)
        
        for i in range(min(10, len(actual_denorm))):
            error = abs(actual_denorm[i] - predictions_denorm[i])
            smiles_short = smiles_list[i][:50] + "..." if len(smiles_list[i]) > 50 else smiles_list[i]
            print(f"{actual_denorm[i]:<10.4f} {predictions_denorm[i]:<12.4f} {error:<10.4f} {smiles_short}")
        
        # Create a simple scatter plot data for analysis
        print(f"\nğŸ“ˆ Prediction Analysis:")
        correlation = np.corrcoef(actual_denorm, predictions_denorm)[0, 1]
        print(f"Pearson Correlation: {correlation:.6f}")
        
        # Identify best and worst predictions
        errors = np.abs(actual_denorm - predictions_denorm)
        best_idx = np.argmin(errors)
        worst_idx = np.argmax(errors)
        
        print(f"\nBest prediction (lowest error):")
        print(f"  Actual: {actual_denorm[best_idx]:.4f}, Predicted: {predictions_denorm[best_idx]:.4f}, Error: {errors[best_idx]:.4f}")
        print(f"  SMILES: {smiles_list[best_idx][:100]}...")
        
        print(f"\nWorst prediction (highest error):")
        print(f"  Actual: {actual_denorm[worst_idx]:.4f}, Predicted: {predictions_denorm[worst_idx]:.4f}, Error: {errors[worst_idx]:.4f}")
        print(f"  SMILES: {smiles_list[worst_idx][:100]}...")
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'correlation': correlation,
            'predictions': predictions_denorm,
            'actual': actual_denorm
        }
        
    except FileNotFoundError:
        print(f"â�Œ Error: Test file not found at {test_file_path}")
        return None
    except Exception as e:
        print(f"â�Œ Error during testing: {str(e)}")
        return None

# Run the test after training is complete
test_file_path = '/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset4.csv'
test_results = test_model_on_dataset(final_model_path, test_file_path, ffv_mean, ffv_std)

if test_results:
    print(f"\nğŸ�¯ Final Model Performance Summary:")
    print(f"External Test RÂ² Score: {test_results['r2']:.4f}")
    print(f"External Test RMSE: {test_results['rmse']:.4f}")
    print(f"External Test MAPE: {test_results['mape']:.2f}%")
else:
    print("â�Œ Testing failed - check file path and format")

# --- 13. Memory Cleanup ---
del model, trainer
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("Memory cleanup completed.")




