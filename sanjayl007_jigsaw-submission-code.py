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


!uv pip install --system --no-index --find-links='/kaggle/input/packagesv1/whls' \
'torch==2.6.0' \
'transformers==4.53.3' \
'peft==0.16.0' \
'accelerate==1.9.0' \
'sentencepiece==0.2.0' \
'tqdm==4.67.1' \
'numpy==1.26.4' \
'pandas==2.2.3' \
'bitsandbytes==0.46.1'


import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import warnings
warnings.filterwarnings('ignore')

import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("="*60)
print("JIGSAW SUBMISSION - INTERNET OFF (FIXED)")
print("="*60)

# ===== PATHS =====
ADAPTER_PATH = "/kaggle/input/jigsaw-trained-model/kaggle_offline_model/RedditModerationClassifier"
BASE_MODEL_PATH = "/kaggle/input/jigsaw-trained-model/kaggle_offline_model/base_model"
MAX_SEQ_LENGTH = 512

# ===== LOAD MODEL =====
print("\nStep 1: Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_PATH,
    local_files_only=True,
    trust_remote_code=True
)
print("âœ“ Tokenizer loaded")

print("\nStep 2: Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16,
    local_files_only=True,
    trust_remote_code=True
)
print("âœ“ Base model loaded")

print("\nStep 3: Loading LoRA adapter...")
model = PeftModel.from_pretrained(
    model,
    ADAPTER_PATH,
    local_files_only=True
)
model.eval()
print("âœ“ Model ready!")


# ===== PROMPT FUNCTIONS (FIXED) =====
SYSTEM_INSTRUCTION = (
    "You are a moderation assistant that strictly determines whether a Reddit comment "
    "violates a specific community rule using only the evidence provided."
)

def build_instruction(row):
    """Build instruction - EXACT format"""
    return f"""Determine if the following Reddit comment violates this rule in subreddit r/{row['subreddit']}:
RULE: "{row['rule']}"
Positive rule-violation examples:
1) {row['positive_example_1']}
2) {row['positive_example_2']}

Negative (NOT violation) examples:
1) {row['negative_example_1']}
2) {row['negative_example_2']}

Respond with exactly one token from this set: VIOLATION or NO_VIOLATION.
Comment:
{row['body']}
"""

def build_inference_prompt(row):
    """âœ… FIX: Match training format EXACTLY"""
    instruction = build_instruction(row)
    # âœ… FIXED: Added << >> and newlines
    return f"<<SYS>>\n{SYSTEM_INSTRUCTION}\n<</SYS>>\n\n{instruction}"


# ===== PREDICTION FUNCTION (FIXED) =====
@torch.no_grad()
def predict_violation_probability(prompt):
    """Calculate probability using log-likelihood"""
    device = next(model.parameters()).device
    
    # âœ… FIXED: Use underscore in NO_VIOLATION (matches training)
    labels = ["VIOLATION", "NO_VIOLATION"]
    log_probabilities = []
    
    for label in labels:
        # âœ… FIXED: Add "Answer: " prefix (matches training)
        full_text = prompt + f"\nAnswer: {label}"
        
        # Tokenize
        tokens = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        )
        
        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        # Get model outputs
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1]
        targets = input_ids[:, 1:]
        
        # Calculate prompt length
        prompt_tokens = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        )
        prompt_length = prompt_tokens['input_ids'].shape[1]
        
        # Extract label portion
        label_length = input_ids.shape[1] - prompt_length
        
        if label_length > 0:
            label_targets = targets[:, -label_length:]
            label_logits = logits[:, -label_length:, :]
            
            # Calculate log probability
            log_probs = torch.log_softmax(label_logits, dim=-1)
            token_log_probs = log_probs.gather(-1, label_targets.unsqueeze(-1)).squeeze(-1)
            total_log_prob = token_log_probs.sum(dim=1).item()
        else:
            total_log_prob = -100.0  # âœ… FIXED: Use -100 for invalid, not 0.0
        
        log_probabilities.append(total_log_prob)
    
    # Normalize to probability
    violation_logp, no_violation_logp = log_probabilities
    
    # âœ… FIXED: Handle edge cases
    if violation_logp == -100 or no_violation_logp == -100:
        return 0.5  # Return uncertain if invalid
    
    max_logp = max(violation_logp, no_violation_logp)
    
    violation_prob = math.exp(violation_logp - max_logp)
    no_violation_prob = math.exp(no_violation_logp - max_logp)
    
    total = violation_prob + no_violation_prob
    return violation_prob / total if total > 0 else 0.5

# ===== LOAD TEST DATA =====
print("\n" + "="*60)
print("RUNNING INFERENCE")
print("="*60)

print("\nLoading test data...")
test_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')
print(f"âœ“ Loaded {len(test_df)} test samples\n")

# ===== GENERATE PREDICTIONS =====
print("Generating predictions...")
probabilities = []
errors = 0

for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Inference"):
    try:
        prompt = build_inference_prompt(row)
        prob = predict_violation_probability(prompt)
        probabilities.append(prob)
    except Exception as e:
        errors += 1
        if errors <= 5:
            print(f"\nError at row {idx}: {str(e)[:100]}")
        probabilities.append(0.5)

if errors > 0:
    print(f"\nâš ï¸� Total errors: {errors} (used 0.5 as fallback)")

# ===== CREATE SUBMISSION =====
submission_df = pd.DataFrame({
    'row_id': test_df['row_id'],
    'rule_violation': probabilities
})

submission_df.to_csv('submission.csv', index=False)

# ===== SUMMARY =====
print("\n" + "="*60)
print("âœ“ SUBMISSION COMPLETE (FIXED VERSION)")
print("="*60)
print(f"\nFile: submission.csv")
print(f"Shape: {submission_df.shape}")
print(f"Samples: {len(probabilities)}")
print(f"\nStatistics:")
print(f"  Mean:   {np.mean(probabilities):.4f}")
print(f"  Median: {np.median(probabilities):.4f}")
print(f"  Std:    {np.std(probabilities):.4f}")
print(f"  Min:    {np.min(probabilities):.4f}")
print(f"  Max:    {np.max(probabilities):.4f}")
print(f"\nDistribution:")
print(f"  < 0.3:   {sum(p < 0.3 for p in probabilities)} ({sum(p < 0.3 for p in probabilities)/len(probabilities)*100:.1f}%)")
print(f"  0.3-0.7: {sum(0.3 <= p < 0.7 for p in probabilities)} ({sum(0.3 <= p < 0.7 for p in probabilities)/len(probabilities)*100:.1f}%)")
print(f"  >= 0.7:  {sum(p >= 0.7 for p in probabilities)} ({sum(p >= 0.7 for p in probabilities)/len(probabilities)*100:.1f}%)")

print(f"\nFirst 5 predictions:")
print(submission_df.head())
print(f"\nLast 5 predictions:")
print(submission_df.tail())

print("\n" + "="*60)
print("âœ… BUGS FIXED - READY TO SUBMIT!")
print("="*60)
print("\nğŸ”§ Fixed Issues:")
print("   1. Prompt format: Added << >> and newlines")
print("   2. Label: Changed NOVIOLATION â†’ NO_VIOLATION")
print("   3. Prefix: Added 'Answer: ' before label")
print("\nğŸ“ˆ Expected improvement: +40-50% AUC")
print("   Previous: 0.46 AUC")
print("   Expected: 0.75-0.85 AUC")
print("="*60)

