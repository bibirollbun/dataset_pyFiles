%%writefile gemma2_inference.py

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import os
from IPython.display import display, Math, Latex
import torch
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from datasets import Dataset
import pandas as pd, numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from peft import PeftModel
from scipy.special import softmax
from tqdm import tqdm


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
lora_path = "/kaggle/input/gemma2-9b-it-cv945"
MAX_LEN = 256
# helpers
def format_input(row):
    x = "Yes"
    if not row['is_correct']:
        x = "No"
    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"Correct? {x}\n"
        f"Student Explanation: {row['StudentExplanation']}"
    )

# Tokenization function
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

le = LabelEncoder()

train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')

train.Misconception = train.Misconception.fillna('NA')
train['target'] = train.Category+":"+train.Misconception
train['label'] = le.fit_transform(train['target'])
target_classes = le.classes_
n_classes = len(target_classes)
print(f"Train shape: {train.shape} with {n_classes} target classes")
idx = train.apply(lambda row: row.Category.split('_')[0],axis=1)=='True'
correct = train.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c',ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct = correct[['QuestionId','MC_Answer']]
correct['is_correct'] = 1

# Prepare test data
test = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')
test = test.merge(correct, on=['QuestionId','MC_Answer'], how='left')
test.is_correct = test.is_correct.fillna(0)
test['text'] = test.apply(format_input, axis=1)


# load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(lora_path)
model = AutoModelForSequenceClassification.from_pretrained(
    "/kaggle/input/gemma2-9b-it-bf16",
    num_labels=n_classes,
    torch_dtype=torch.float16,
    device_map="auto",
)

model = PeftModel.from_pretrained(model, lora_path)
model.eval()

# Tokenize dataset
ds_test = Dataset.from_pandas(test[['text']])
ds_test = ds_test.map(tokenize, batched=True, remove_columns=['text'])

# Create data collator for efficient batching with padding
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    max_length=MAX_LEN,  
    return_tensors="pt")

dataloader = DataLoader(
    ds_test,
    batch_size=8,  
    shuffle=False,
    collate_fn=data_collator,
    pin_memory=True,  
    num_workers=2     
)

# Fast inference loop
all_logits = []
device = next(model.parameters()).device

with torch.no_grad():
    for batch in tqdm(dataloader, desc="Inference"):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        logits = outputs.logits
        
        # Convert bfloat16 to float32 then move to CPU and store
        all_logits.append(logits.float().cpu().numpy())

# Concatenate all logits
predictions = np.concatenate(all_logits, axis=0)

# Convert to probs
probs = softmax(predictions, axis=1)

# Get top predictions (all 65 classes ranked)
top_indices = np.argsort(-probs, axis=1)

# Decode to class names
flat_indices = top_indices.flatten()
decoded_labels = le.inverse_transform(flat_indices)
top_labels = decoded_labels.reshape(top_indices.shape)

# Create submission (top 3)
joined_preds = [" ".join(row[:3]) for row in top_labels]

sub = pd.DataFrame({
    "row_id": test.row_id.values,
    "Category:Misconception": joined_preds
})
sub.to_csv("submission_gemma.csv", index=False)

prob_data = []
for i in range(len(test)):
    prob_dict = {f"prob_{j}": probs[i, top_indices[i, j]] for j in range(25)}  # Top 25
    prob_dict['row_id'] = test.row_id.values[i]
    prob_dict['top_classes'] = " ".join(top_labels[i, :25])  # Top 25 class names
    prob_data.append(prob_dict)

prob_df = pd.DataFrame(prob_data)
prob_df.to_csv("submission_gemma_prob.csv", index=False)


%%writefile qwen3_deepseek_inference.py

# we do parallel inference, for deepseek and qwen3
import os
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import threading
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from scipy.special import softmax
from tqdm import tqdm
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"


train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
test  = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')

model_paths = [
    "/kaggle/input/deekseepmath-7b-map-competition/MAP_EXP_09_FULL",
   "/kaggle/input/qwen3-8b-map-competition/MAP_EXP_16_FULL"]

def format_input(row):
    x = "This answer is correct."
    if not row['is_correct']:
        x = "This is answer is incorrect."
    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"{x}\n"
        f"Student Explanation: {row['StudentExplanation']}")


le = LabelEncoder()
train.Misconception  = train.Misconception.fillna('NA')
train['target']   = train.Category + ':' +train.Misconception
train['label']    = le.fit_transform(train['target'])

n_classes = len(le.classes_)
print(f"Train shape: {train.shape} with {n_classes} target classes")
idx = train.apply(lambda row: row.Category.split('_')[0],axis=1)=='True'
correct = train.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c',ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct = correct[['QuestionId','MC_Answer']]
correct['is_correct'] = 1

test = test.merge(correct, on=['QuestionId','MC_Answer'], how='left')
test.is_correct = test.is_correct.fillna(0)
test['text'] = test.apply(format_input,axis=1)
ds_test = Dataset.from_pandas(test)


def run_inference_on_gpu(model_path, gpu_id, test_data, output_name):
    """Run inference for one model on one GPU"""
    
    device = f"cuda:{gpu_id}"
    print(f"Loading {output_name} on {device}...")
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, 
        device_map=device, 
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    
    # Tokenize function
    def tokenize(batch):
        return tokenizer(batch["text"], 
                        truncation=True,
                        max_length=256)
    
    ds_test = Dataset.from_pandas(test_data[['text']])
    ds_test = ds_test.map(tokenize, batched=True, remove_columns=['text'])
    
    # Data collator
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt"
    )
    
    # DataLoader
    dataloader = DataLoader(
        ds_test,
        batch_size=4,
        shuffle=False,
        collate_fn=data_collator,
        pin_memory=True,
        num_workers=0
    )
    
    # Inference
    all_logits = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"{output_name}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            all_logits.append(outputs.logits.float().cpu().numpy())
    
    predictions = np.concatenate(all_logits, axis=0)
    
    # Process results
    probs = softmax(predictions, axis=1)
    top_indices = np.argsort(-probs, axis=1)
    
    # Decode labels
    flat_indices = top_indices.flatten()
    decoded_labels = le.inverse_transform(flat_indices)
    top_labels = decoded_labels.reshape(top_indices.shape)
    
    # Save top-3 submission
    joined_preds = [" ".join(row[:3]) for row in top_labels]
    sub = pd.DataFrame({
        "row_id": test_data.row_id.values,
        "Category:Misconception": joined_preds
    })
    sub.to_csv(f"submission_{output_name}.csv", index=False)
    
    # Save probabilities for ensemble
    prob_data = []
    for i in range(len(predictions)):
        prob_dict = {f"prob_{j}": probs[i, top_indices[i, j]] for j in range(25)}
        prob_dict['row_id'] = test_data.row_id.values[i]
        prob_dict['top_classes'] = " ".join(top_labels[i, :25])
        prob_data.append(prob_dict)
    
    prob_df = pd.DataFrame(prob_data)
    prob_df.to_csv(f"submission_{output_name}_probabilities.csv", index=False)
    
    print(f" {output_name} completed - saved submission and probabilities")
    
    # Clean up GPU memory
    del model, tokenizer
    torch.cuda.empty_cache()

print(" Starting multi-GPU inference...")
start_time = time.time()

threads = []
gpu_assignments = [
    (model_paths[0], 0, "deepseek"),
    (model_paths[1], 1, "qwen3"),
]

# Start threads
for model_path, gpu_id, name in gpu_assignments:
    if gpu_id < torch.cuda.device_count():  
        thread = threading.Thread(
            target=run_inference_on_gpu,
            args=(model_path, gpu_id, test, name)
        )
        threads.append(thread)
        thread.start()
        time.sleep(10)  # Stagger starts to avoid memory issues

# Wait for completion
for thread in threads:
    thread.join()

end_time = time.time()
print(f" completed in {end_time - start_time:.2f} seconds!")


import time 
!python /kaggle/working/gemma2_inference.py
time.sleep(20)
!python /kaggle/working/qwen3_deepseek_inference.py


import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.special import softmax
def extract_class_probabilities(row, model_suffix='', top_k=25):
    """Extract class names and probabilities from a row"""
    # Get top classes
    classes_col = f'top_classes{model_suffix}'
    if classes_col in row:
        classes = row[classes_col].split(' ')[:top_k]
    else:
        return {}
    # Get probabilities
    class_probs = {}
    for i in range(min(top_k, len(classes))):
        prob_col = f'prob_{i}{model_suffix}'
        if prob_col in row:
            class_probs[classes[i]] = row[prob_col]
    return class_probs





def ensemble_with_disagreement_handling(prob_files, model_weights=None, top_k=3):
    n_models = len(prob_files)
    prob_dfs = []
    final_predictions = []
    
    for file_path in prob_files:
        df = pd.read_csv(file_path)
        prob_dfs.append(df)
    
    # Merge on row_id
    merged_df = prob_dfs[0]
    for i, df in enumerate(prob_dfs[1:], 1):
        merged_df = pd.merge(merged_df, df, on='row_id', suffixes=('', f'_model{i+1}'))
      
    for idx, row in merged_df.iterrows():
        
        # Extract probabilities from each model
        all_class_probs = []
        for i in range(n_models):
            suffix = f'_model{i+1}' if i > 0 else ''
            class_probs = extract_class_probabilities(row, suffix, top_k=25)
            all_class_probs.append(class_probs)
        
        # Get all unique classes
        all_classes = set()
        for class_probs in all_class_probs:
            all_classes.update(class_probs.keys())
        
        # Calculate agreement and disagreement
        class_votes = defaultdict(int)
        class_total_prob = defaultdict(float)
        class_max_prob = defaultdict(float)
        
        for i, class_probs in enumerate(all_class_probs):
            weight = model_weights[i]
            
            for class_name, prob in class_probs.items():
                class_votes[class_name] += 1
                class_total_prob[class_name] += prob * weight
                class_max_prob[class_name] = max(class_max_prob[class_name], prob * weight)
        
        final_scores = {}
        for class_name in all_classes:
            
            # Base score: weighted average probability
            base_score = class_total_prob[class_name]
            
            # Agreement : classes predicted by more models get boost
            agreement_bonus = class_votes[class_name] / n_models
            
            # Confidence bonus: classes with high max probability get boost
            confidence_bonus = class_max_prob[class_name]
            
            # Combined score
            final_scores[class_name] = (
                base_score * 0.6 +           # 60% base probs
                agreement_bonus * 0.3 +      # 30% agreement
                confidence_bonus * 0.1       # 10% confidence
            )
        
        # Sort and get top-k
        sorted_classes = sorted(final_scores.items(), key=lambda x: -x[1])
        top_classes = [class_name for class_name, _ in sorted_classes[:top_k]]
        
        final_predictions.append(' '.join(top_classes))
    
    return final_predictions


w1 = 1.2
w2 = 1.0
w3 = 0.8

prob_files = [
    '/kaggle/working/submission_deepseek_probabilities.csv',
    '/kaggle/working/submission_gemma_prob.csv',
        '/kaggle/working/submission_qwen3_probabilities.csv'

]

predictions = ensemble_with_disagreement_handling(
        prob_files, 
        model_weights=[w1, w2, w3],  
        top_k=3
    )
    
test_df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')

submission = pd.DataFrame({
    'row_id': test_df.row_id.values,
    'Category:Misconception': predictions
})

submission.to_csv('/kaggle/working/submission.csv', index=False)
submission

