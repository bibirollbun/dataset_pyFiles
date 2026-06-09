from IPython.display import Image, display
display(Image(filename="/kaggle/input/map-chart/MAP.png"))



!pip install -q transformers accelerate torch sentence-transformers



%%writefile gemma_qwen_deepseek_inference.py
# ===============================
# Imports & Environment Setup
# ===============================
import os, time, threading
import torch
import pandas as pd, numpy as np
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from peft import PeftModel
from scipy.special import softmax
from tqdm import tqdm
from collections import defaultdict

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# ===============================
# Paths & Constants
# ===============================
lora_path = "/kaggle/input/gemma2-9b-it-cv945"
gemma_model_path = "/kaggle/input/gemma2-9b-it-bf16"
deepseek_path = "/kaggle/input/deekseepmath-7b-map-competition/MAP_EXP_09_FULL"
qwen_path    = "/kaggle/input/qwen3-8b-map-competition/MAP_EXP_16_FULL"
MAX_LEN = 256
BATCH_SIZE = 8

# ===============================
# Load Train/Test Data
# ===============================
train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
test  = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')

# Fill missing & encode labels
train.Misconception = train.Misconception.fillna('NA')
train['target'] = train.Category + ":" + train.Misconception
le = LabelEncoder()
train['label'] = le.fit_transform(train['target'])
n_classes = len(le.classes_)

# Compute correct answers
idx = train.apply(lambda row: row.Category.split('_')[0] == 'True', axis=1)
correct = train.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c', ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct = correct[['QuestionId','MC_Answer']]
correct['is_correct'] = 1

# Merge with test
test = test.merge(correct, on=['QuestionId','MC_Answer'], how='left')
test.is_correct = test.is_correct.fillna(0)

# ===============================
# Helper Functions
# ===============================
def format_input(row):
    x = "This answer is correct." if row['is_correct'] else "This answer is incorrect."
    return f"Question: {row['QuestionText']}\nAnswer: {row['MC_Answer']}\n{x}\nStudent Explanation: {row['StudentExplanation']}"

test['text'] = test.apply(format_input, axis=1)

def tokenize(batch, tokenizer):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=MAX_LEN)

# ===============================
# Gemma 2-9B Inference
# ===============================
tokenizer_gemma = AutoTokenizer.from_pretrained(lora_path)
model_gemma = AutoModelForSequenceClassification.from_pretrained(
    gemma_model_path,
    num_labels=n_classes,
    torch_dtype=torch.float16,
    device_map="auto"
)
model_gemma = PeftModel.from_pretrained(model_gemma, lora_path)
model_gemma.eval()

ds_test_gemma = Dataset.from_pandas(test[['text']])
ds_test_gemma = ds_test_gemma.map(lambda x: tokenize(x, tokenizer_gemma), batched=True, remove_columns=['text'])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer_gemma, max_length=MAX_LEN, return_tensors="pt")
dataloader_gemma = DataLoader(ds_test_gemma, batch_size=BATCH_SIZE, shuffle=False, collate_fn=data_collator)

all_logits_gemma = []
device = next(model_gemma.parameters()).device
with torch.no_grad():
    for batch in tqdm(dataloader_gemma, desc="Gemma2-9B Inference"):
        batch = {k: v.to(device) for k,v in batch.items()}
        outputs = model_gemma(**batch)
        all_logits_gemma.append(outputs.logits.float().cpu().numpy())

preds_gemma = np.concatenate(all_logits_gemma, axis=0)
probs_gemma = softmax(preds_gemma, axis=1)
top_indices_gemma = np.argsort(-probs_gemma, axis=1)
decoded_labels_gemma = le.inverse_transform(top_indices_gemma.flatten()).reshape(top_indices_gemma.shape)

sub_gemma = pd.DataFrame({
    "row_id": test.row_id.values,
    "Category:Misconception": [" ".join(row[:3]) for row in decoded_labels_gemma]
})
sub_gemma.to_csv("submission_gemma.csv", index=False)

prob_data_gemma = []
for i in range(len(test)):
    prob_dict = {f"prob_{j}": probs_gemma[i, top_indices_gemma[i,j]] for j in range(25)}
    prob_dict['row_id'] = test.row_id.values[i]
    prob_dict['top_classes'] = " ".join(decoded_labels_gemma[i,:25])
    prob_data_gemma.append(prob_dict)
pd.DataFrame(prob_data_gemma).to_csv("submission_gemma_prob.csv", index=False)

# ===============================
# Parallel Inference: DeepSeek & Qwen3
# ===============================
model_paths = [deepseek_path, qwen_path]

def run_inference(model_path, gpu_id, name):
    device = f"cuda:{gpu_id}"
    print(f"Loading {name} on {device}...")
    model = AutoModelForSequenceClassification.from_pretrained(model_path, device_map=device, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    
    ds_test_model = Dataset.from_pandas(test[['text']])
    ds_test_model = ds_test_model.map(lambda x: tokenize(x, tokenizer), batched=True, remove_columns=['text'])
    dataloader = DataLoader(ds_test_model, batch_size=4, shuffle=False, collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt"))
    
    all_logits = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"{name}"):
            batch = {k: v.to(device) for k,v in batch.items()}
            outputs = model(**batch)
            all_logits.append(outputs.logits.float().cpu().numpy())
    
    predictions = np.concatenate(all_logits, axis=0)
    probs = softmax(predictions, axis=1)
    top_indices = np.argsort(-probs, axis=1)
    decoded_labels = le.inverse_transform(top_indices.flatten()).reshape(top_indices.shape)
    
    sub = pd.DataFrame({
        "row_id": test.row_id.values,
        "Category:Misconception": [" ".join(row[:3]) for row in decoded_labels]
    })
    sub.to_csv(f"submission_{name}.csv", index=False)
    
    prob_data = []
    for i in range(len(test)):
        prob_dict = {f"prob_{j}": probs[i, top_indices[i,j]] for j in range(25)}
        prob_dict['row_id'] = test.row_id.values[i]
        prob_dict['top_classes'] = " ".join(decoded_labels[i,:25])
        prob_data.append(prob_dict)
    pd.DataFrame(prob_data).to_csv(f"submission_{name}_probabilities.csv", index=False)
    
    print(f"{name} completed.")
    del model, tokenizer
    torch.cuda.empty_cache()

threads = []
gpu_assignments = [(model_paths[0], 0, "deepseek"), (model_paths[1], 1, "qwen3")]
for path, gpu, name in gpu_assignments:
    if gpu < torch.cuda.device_count():
        thread = threading.Thread(target=run_inference, args=(path, gpu, name))
        threads.append(thread)
        thread.start()
        time.sleep(5)
for thread in threads:
    thread.join()

# ===============================
# Ensemble Predictions
# ===============================
def extract_class_probs(row, suffix="", top_k=25):
    col_name = f"top_classes{suffix}"
    if col_name not in row: return {}
    classes = row[col_name].split(' ')[:top_k]
    return {classes[i]: row[f"prob_{i}{suffix}"] for i in range(len(classes))}

def ensemble_with_disagreement(prob_files, model_weights=[1,1,1], top_k=3):
    n_models = len(prob_files)
    dfs = [pd.read_csv(f) for f in prob_files]
    merged = dfs[0]
    for i, df in enumerate(dfs[1:],1):
        merged = pd.merge(merged, df, on='row_id', suffixes=('', f'_model{i+1}'))
    
    final_preds = []
    for _, row in merged.iterrows():
        all_probs = [extract_class_probs(row, f'_model{i+1}' if i>0 else '', top_k=25) for i in range(n_models)]
        all_classes = set(c for ap in all_probs for c in ap.keys())
        scores = {}
        for c in all_classes:
            base = sum(ap.get(c,0)*model_weights[i] for i, ap in enumerate(all_probs))
            agree = sum(1 for ap in all_probs if c in ap)/n_models
            confidence = max(ap.get(c,0)*model_weights[i] for i, ap in enumerate(all_probs))
            scores[c] = 0.6*base + 0.3*agree + 0.1*confidence
        top_classes = [c for c,_ in sorted(scores.items(), key=lambda x:-x[1])[:top_k]]
        final_preds.append(' '.join(top_classes))
    return final_preds

prob_files = ["submission_deepseek_probabilities.csv", "submission_gemma_prob.csv", "submission_qwen3_probabilities.csv"]
weights = [1.2, 0.8, 1.0]
predictions = ensemble_with_disagreement(prob_files, model_weights=weights, top_k=3)

submission = pd.DataFrame({'row_id': test.row_id.values, 'Category:Misconception': predictions})
submission.to_csv('submission.csv', index=False)
print("Ensemble submission saved to submission.csv")





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
time.sleep(10)
!python /kaggle/working/qwen3_deepseek_inference.py




import pandas as pd
from collections import defaultdict
import os

def extract_class_probabilities(row, model_suffix='', top_k=25):
    """Extract class names and probabilities from a row"""
    classes_col = f'top_classes{model_suffix}'
    if classes_col not in row:
        return {}
    
    classes = row[classes_col].split(' ')[:top_k]
    class_probs = {}
    for i in range(len(classes)):
        prob_col = f'prob_{i}{model_suffix}'
        if prob_col in row:
            class_probs[classes[i]] = row[prob_col]
    return class_probs


def ensemble_with_disagreement_handling(prob_files, model_weights=None, top_k=3):
    """Ensemble predictions from multiple models with weighted disagreement handling"""
    prob_dfs = []
    existing_files = []
    
    # Load only existing files
    for file_path in prob_files:
        if os.path.exists(file_path):
            prob_dfs.append(pd.read_csv(file_path))
            existing_files.append(file_path)
        else:
            print(f"Warning: {file_path} not found, skipping.")
    
    if len(prob_dfs) == 0:
        raise FileNotFoundError("No valid probability files found.")

    n_models = len(prob_dfs)
    if model_weights is None:
        model_weights = [1.0] * n_models
    else:
        # adjust weights if some files were missing
        if len(model_weights) != n_models:
            model_weights = model_weights[:n_models]

    # Merge on row_id
    merged_df = prob_dfs[0]
    for i, df in enumerate(prob_dfs[1:], 1):
        merged_df = pd.merge(merged_df, df, on='row_id', suffixes=('', f'_model{i+1}'))

    final_predictions = []

    for _, row in merged_df.iterrows():
        all_class_probs = []
        for i in range(n_models):
            suffix = f'_model{i+1}' if i > 0 else ''
            class_probs = extract_class_probabilities(row, suffix, top_k=25)
            all_class_probs.append(class_probs)
        
        all_classes = set()
        for class_probs in all_class_probs:
            all_classes.update(class_probs.keys())

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
            base_score = class_total_prob[class_name]
            agreement_bonus = class_votes[class_name] / n_models
            confidence_bonus = class_max_prob[class_name]
            final_scores[class_name] = base_score * 0.6 + agreement_bonus * 0.3 + confidence_bonus * 0.1

        # Get top-k predictions
        sorted_classes = sorted(final_scores.items(), key=lambda x: -x[1])
        top_classes = [class_name for class_name, _ in sorted_classes[:top_k]]
        final_predictions.append(' '.join(top_classes))  # concatenate with spaces

    return final_predictions


# ------------------------------
# Example usage (offline paths)
# ------------------------------
w1, w2, w3 = 1.2, 1.0, 0.8
prob_files = [
    'submission_deepseek_probabilities.csv',
    'submission_gemma_prob.csv',
    'submission_qwen3_probabilities.csv'
]

predictions = ensemble_with_disagreement_handling(
    prob_files,
    model_weights=[w1, w2, w3],
    top_k=3  # top-3 classes per row
)

# Load local test CSV
test_df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')

submission = pd.DataFrame({
    'row_id': test_df.row_id.values,
    'Category:Misconception': predictions
})

submission.to_csv('submission.csv', index=False)
print(submission.head())
print("Submission saved to submission.csv")


