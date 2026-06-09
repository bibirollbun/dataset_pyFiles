import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from wordcloud import WordCloud

# Load data
print("Loading data...")
train_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
test_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")

print("="*80)
print("DATASET OVERVIEW")
print("="*80)
print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
print(f"\nTrain columns: {train_df.columns.tolist()}")
print(f"Test columns: {test_df.columns.tolist()}")

# Class distribution
print("\n" + "="*80)
print("CLASS DISTRIBUTION")
print("="*80)
print(train_df['rule_violation'].value_counts())
print(f"\nViolation rate: {train_df['rule_violation'].mean()*100:.2f}%")
print(f"Balance ratio: {train_df['rule_violation'].value_counts()[1] / train_df['rule_violation'].value_counts()[0]:.2f}")

# Text length analysis
print("\n" + "="*80)
print("TEXT LENGTH ANALYSIS")
print("="*80)

train_df['text_length'] = train_df['body'].astype(str).str.len()
train_df['word_count'] = train_df['body'].astype(str).str.split().str.len()

for label in [0, 1]:
    subset = train_df[train_df['rule_violation'] == label]
    print(f"\nClass {label} ({'VIOLATION' if label == 1 else 'CLEAN'}):")
    print(f"  Avg text length: {subset['text_length'].mean():.1f}")
    print(f"  Avg word count: {subset['word_count'].mean():.1f}")
    print(f"  Median text length: {subset['text_length'].median():.1f}")
    print(f"  Max text length: {subset['text_length'].max()}")

# Rule distribution
print("\n" + "="*80)
print("RULE DISTRIBUTION")
print("="*80)
print("\nTop 5 rules in training data:")
print(train_df['rule'].value_counts().head())

print("\nRules in test data:")
print(test_df['rule'].value_counts())

# Subreddit distribution
print("\n" + "="*80)
print("SUBREDDIT DISTRIBUTION")
print("="*80)
print("\nTop 10 subreddits in training:")
print(train_df['subreddit'].value_counts().head(10))

print("\nSubreddits in test:")
print(test_df['subreddit'].value_counts())

# URL analysis
print("\n" + "="*80)
print("URL ANALYSIS")
print("="*80)

train_df['has_url'] = train_df['body'].astype(str).str.contains(r'http|www\.', case=False, na=False)
test_df['has_url'] = test_df['body'].astype(str).str.contains(r'http|www\.', case=False, na=False)

print(f"\nTrain - URLs by class:")
for label in [0, 1]:
    pct = train_df[train_df['rule_violation'] == label]['has_url'].mean() * 100
    print(f"  Class {label}: {pct:.1f}% have URLs")

print(f"\nTest - URLs: {test_df['has_url'].mean()*100:.1f}%")

# Special characters analysis
print("\n" + "="*80)
print("SPECIAL CHARACTERS ANALYSIS")
print("="*80)

# Define regex patterns outside f-strings
question_pattern = r'\?'
digit_pattern = r'\d'

for label in [0, 1]:
    subset = train_df[train_df['rule_violation'] == label]['body'].astype(str)
    exclaim_avg = subset.str.count('!').mean()
    question_avg = subset.str.count(question_pattern).mean()
    digit_avg = subset.str.count(digit_pattern).mean()
    caps_avg = subset.apply(lambda x: sum(c.isupper() for c in x) / max(len(x), 1)).mean()
    
    print(f"\nClass {label}:")
    print(f"  Avg exclamation marks: {exclaim_avg:.2f}")
    print(f"  Avg question marks: {question_avg:.2f}")
    print(f"  Avg numbers: {digit_avg:.2f}")
    print(f"  Avg caps ratio: {caps_avg:.3f}")

test_texts = test_df['body'].astype(str)
test_exclaim = test_texts.str.count('!').mean()
test_question = test_texts.str.count(question_pattern).mean()
test_caps = test_texts.apply(lambda x: sum(c.isupper() for c in x) / max(len(x), 1)).mean()

print(f"\nTest data:")
print(f"  Avg exclamation marks: {test_exclaim:.2f}")
print(f"  Avg question marks: {test_question:.2f}")
print(f"  Avg caps ratio: {test_caps:.3f}")

# Keyword analysis
print("\n" + "="*80)
print("SPAM KEYWORD ANALYSIS")
print("="*80)

spam_keywords = ['free', 'click', 'buy', 'discount', 'win', 'prize', 'earn', 'money', 'cash', 'bonus']

print("\nKeyword frequency in VIOLATIONS:")
for kw in spam_keywords:
    count = train_df[train_df['rule_violation'] == 1]['body'].astype(str).str.contains(kw, case=False).sum()
    pct = count / len(train_df[train_df['rule_violation'] == 1]) * 100
    print(f"  '{kw}': {count} ({pct:.1f}%)")

print("\nKeyword frequency in CLEAN:")
for kw in spam_keywords:
    count = train_df[train_df['rule_violation'] == 0]['body'].astype(str).str.contains(kw, case=False).sum()
    pct = count / len(train_df[train_df['rule_violation'] == 0]) * 100
    print(f"  '{kw}': {count} ({pct:.1f}%)")

print("\nKeyword frequency in TEST:")
for kw in spam_keywords:
    count = test_df['body'].astype(str).str.contains(kw, case=False).sum()
    print(f"  '{kw}': {count} / {len(test_df)}")

# Sample texts
print("\n" + "="*80)
print("SAMPLE VIOLATIONS (First 3)")
print("="*80)
for i, row in train_df[train_df['rule_violation'] == 1].head(3).iterrows():
    print(f"\n{i+1}. [{row['subreddit']}] {row['body'][:200]}...")

print("\n" + "="*80)
print("SAMPLE CLEAN (First 3)")
print("="*80)
for i, row in train_df[train_df['rule_violation'] == 0].head(3).iterrows():
    print(f"\n{i+1}. [{row['subreddit']}] {row['body'][:200]}...")

print("\n" + "="*80)
print("ALL TEST SAMPLES")
print("="*80)
for i, row in test_df.iterrows():
    print(f"\n{i+1}. [{row['subreddit']}] {row['body'][:150]}...")

# Visualization
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS...")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Class distribution
axes[0, 0].bar(['Clean', 'Violation'], train_df['rule_violation'].value_counts().sort_index())
axes[0, 0].set_title('Class Distribution', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Count')

# 2. Text length by class
train_df.boxplot(column='text_length', by='rule_violation', ax=axes[0, 1])
axes[0, 1].set_title('Text Length Distribution by Class', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Class (0=Clean, 1=Violation)')
axes[0, 1].set_ylabel('Text Length')

# 3. Word count by class
train_df.boxplot(column='word_count', by='rule_violation', ax=axes[0, 2])
axes[0, 2].set_title('Word Count Distribution by Class', fontsize=14, fontweight='bold')
axes[0, 2].set_xlabel('Class (0=Clean, 1=Violation)')
axes[0, 2].set_ylabel('Word Count')

# 4. URL presence
url_data = train_df.groupby('rule_violation')['has_url'].mean() * 100
axes[1, 0].bar(['Clean', 'Violation'], url_data)
axes[1, 0].set_title('URL Presence by Class (%)', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('Percentage')

# 5. Top subreddits
top_subs = train_df['subreddit'].value_counts().head(10)
axes[1, 1].barh(range(len(top_subs)), top_subs.values)
axes[1, 1].set_yticks(range(len(top_subs)))
axes[1, 1].set_yticklabels(top_subs.index)
axes[1, 1].set_title('Top 10 Subreddits', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Count')

# 6. Top rules
top_rules = train_df['rule'].value_counts().head(5)
axes[1, 2].barh(range(len(top_rules)), top_rules.values)
axes[1, 2].set_yticks(range(len(top_rules)))
axes[1, 2].set_yticklabels([r[:30] + '...' for r in top_rules.index])
axes[1, 2].set_title('Top 5 Rules', fontsize=14, fontweight='bold')
axes[1, 2].set_xlabel('Count')

plt.tight_layout()
plt.savefig('data_analysis.png', dpi=150, bbox_inches='tight')
print("Saved: data_analysis.png")

# Train vs Test comparison
print("\n" + "="*80)
print("TRAIN VS TEST DISTRIBUTION COMPARISON")
print("="*80)

print("\nFeature comparison:")
print(f"{'Feature':<20} {'Train (Violation)':<20} {'Train (Clean)':<20} {'Test':<20}")
print("-" * 80)

features = {
    'Avg text length': (
        train_df[train_df['rule_violation']==1]['text_length'].mean(),
        train_df[train_df['rule_violation']==0]['text_length'].mean(),
        test_df['body'].astype(str).str.len().mean()
    ),
    'Has URL (%)': (
        train_df[train_df['rule_violation']==1]['has_url'].mean()*100,
        train_df[train_df['rule_violation']==0]['has_url'].mean()*100,
        test_df['has_url'].mean()*100
    ),
    'Avg ! count': (
        train_df[train_df['rule_violation']==1]['body'].astype(str).str.count('!').mean(),
        train_df[train_df['rule_violation']==0]['body'].astype(str).str.count('!').mean(),
        test_df['body'].astype(str).str.count('!').mean()
    )
}

for feat_name, (viol, clean, test_val) in features.items():
    print(f"{feat_name:<20} {viol:<20.2f} {clean:<20.2f} {test_val:<20.2f}")

print("\n" + "="*80)
print("KEY INSIGHTS & RECOMMENDATIONS")
print("="*80)

# Calculate similarity to each class
test_avg_len = test_df['body'].astype(str).str.len().mean()
viol_avg_len = train_df[train_df['rule_violation']==1]['text_length'].mean()
clean_avg_len = train_df[train_df['rule_violation']==0]['text_length'].mean()

test_url_pct = test_df['has_url'].mean()
viol_url_pct = train_df[train_df['rule_violation']==1]['has_url'].mean()
clean_url_pct = train_df[train_df['rule_violation']==0]['has_url'].mean()

print("\n1. Distribution Analysis:")
print(f"   - Test data is closer to VIOLATION class in URL presence")
print(f"   - Test samples: {len(test_df)} (very small - high variance risk)")

print("\n2. Class Balance:")
print(f"   - Train is balanced ({train_df['rule_violation'].mean()*100:.1f}% violations)")
print(f"   - Good for training, no major class imbalance issues")

print("\n3. Recommended Strategy:")
print("   - Focus on spam-specific features (URLs, keywords, special chars)")
print("   - Use TF-IDF with character n-grams for spam patterns")
print("   - Classical ML may outperform deep learning (small, specific test set)")
print("   - Cross-validation is critical (test set too small for reliable validation)")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)


"""
JIGSAW UNIFIED MODEL - BEST OF BOTH WORLDS
Strategy: Combines best hyperparameters from both approaches
Target: 0.89+ through optimal training + Average TTA
"""

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# CONFIG - BEST HYPERPARAMETERS FROM BOTH MODELS
# =====================================================

class Config:
    MODEL_PATH = "/kaggle/input/qwen2-5-transformers-0-5b-v1-tar"
    OUTPUT_DIR = "./qwen_unified_best"
    
    # FROM BALANCED ULTRA: Optimal epochs
    N_EPOCHS = 2  # Sweet spot - not 3!
    
    # FROM HYPERPARAMETER OPTIMIZED: Better batch training
    BATCH_SIZE = 2  # More stable gradients
    GRADIENT_ACCUMULATION = 8  # Effective batch = 16
    
    # FROM HYPERPARAMETER OPTIMIZED: Better learning
    LEARNING_RATE = 1.5e-4  # Optimal learning rate
    WARMUP_STEPS = 75  # More warmup for stability
    
    # FROM BALANCED ULTRA: Better regularization
    LORA_R = 24  # Optimal capacity
    LORA_ALPHA = 48  # Strong adaptation
    LORA_DROPOUT = 0.05  # Balanced dropout
    
    # FROM BALANCED ULTRA: Average TTA (not max!)
    N_TTA = 2
    USE_AVERAGE_TTA = True  # Critical: Use mean not max
    
    MAX_LENGTH = 512
    SEED = 42
    torch.manual_seed(SEED)

print("=" * 70)
print("ğŸš€ JIGSAW UNIFIED MODEL - BEST OF BOTH WORLDS")
print("=" * 70)
print("Configuration:")
print(f"  ğŸ“Š Epochs: {Config.N_EPOCHS} (Balanced Ultra)")
print(f"  ğŸ“¦ Batch Size: {Config.BATCH_SIZE} (Hyperparameter Opt)")
print(f"  ğŸ�¯ Learning Rate: {Config.LEARNING_RATE} (Hyperparameter Opt)")
print(f"  ğŸ”¥ LoRA R/Alpha: {Config.LORA_R}/{Config.LORA_ALPHA} (Balanced Ultra)")
print(f"  ğŸ’§ Dropout: {Config.LORA_DROPOUT} (Balanced Ultra)")
print(f"  ğŸ�² TTA: Average of {Config.N_TTA} (Balanced Ultra)")
print("=" * 70)

# =====================================================
# DATA PREPARATION
# =====================================================

def prepare_training_data():
    print("\n" + "=" * 70)
    print("PREPARING TRAINING DATA")
    print("=" * 70)
    
    train_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
    test_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
    
    all_data = []
    
    # Training data
    for _, row in train_df.iterrows():
        pos_ex = row['positive_example_1'] if np.random.rand() < 0.5 else row['positive_example_2']
        neg_ex = row['negative_example_1'] if np.random.rand() < 0.5 else row['negative_example_2']
        
        all_data.append({
            'subreddit': row['subreddit'],
            'rule': row['rule'],
            'body': row['body'],
            'pos_ex': pos_ex,
            'neg_ex': neg_ex,
            'label': int(row['rule_violation'])
        })
    
    # Test data augmentation
    for _, row in test_df.iterrows():
        for i in [1, 2]:
            all_data.append({
                'subreddit': row['subreddit'],
                'rule': row['rule'],
                'body': row[f'positive_example_{i}'],
                'pos_ex': row[f'positive_example_{3-i}'],
                'neg_ex': row['negative_example_1'] if np.random.rand() < 0.5 else row['negative_example_2'],
                'label': 1
            })
        
        for i in [1, 2]:
            all_data.append({
                'subreddit': row['subreddit'],
                'rule': row['rule'],
                'body': row[f'negative_example_{i}'],
                'pos_ex': row['positive_example_1'] if np.random.rand() < 0.5 else row['positive_example_2'],
                'neg_ex': row[f'negative_example_{3-i}'],
                'label': 0
            })
    
    df = pd.DataFrame(all_data).drop_duplicates(subset=['body']).reset_index(drop=True)
    
    print(f"Total samples: {len(df)}")
    print(f"  Violations: {df['label'].sum()}")
    print(f"  Non-violations: {len(df) - df['label'].sum()}")
    
    return df

# =====================================================
# TOKENIZATION
# =====================================================

def create_prompt_and_tokenize(row, tokenizer):
    messages = [
        {'role': 'system', 'content': 'You are a content moderator. Answer Yes or No.'},
        {'role': 'user', 'content': f"""Rule: {row['rule']}
Violation: {row['pos_ex']}
Non-violation: {row['neg_ex']}

Comment: {row['body']}
Violates?"""},
        {'role': 'assistant', 'content': 'Yes' if row['label'] == 1 else 'No'}
    ]
    
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    full_tokens = tokenizer(full_text, truncation=True, max_length=Config.MAX_LENGTH, padding='max_length')
    
    assistant_response = 'Yes' if row['label'] == 1 else 'No'
    response_tokens = tokenizer.encode(assistant_response, add_special_tokens=False)
    
    labels = full_tokens['input_ids'].copy()
    response_len = len(response_tokens)
    
    for i in range(len(labels) - response_len + 1):
        if labels[i:i+response_len] == response_tokens:
            for j in range(i):
                labels[j] = -100
            for j in range(i+response_len, len(labels)):
                labels[j] = -100
            break
    else:
        labels = [-100] * len(labels)
    
    return {
        'input_ids': full_tokens['input_ids'],
        'attention_mask': full_tokens['attention_mask'],
        'labels': labels
    }

def prepare_dataset(df, tokenizer):
    print("\nTokenizing dataset...")
    data_list = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        data_list.append(create_prompt_and_tokenize(row, tokenizer))
    return Dataset.from_list(data_list)

# =====================================================
# TRAINING
# =====================================================

def train_model(model, train_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc="Training")
    
    for step, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        loss = outputs.loss / Config.GRADIENT_ACCUMULATION
        loss.backward()
        
        total_loss += loss.item() * Config.GRADIENT_ACCUMULATION
        
        if (step + 1) % Config.GRADIENT_ACCUMULATION == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            pbar.set_postfix({'loss': f"{total_loss / (step + 1):.4f}"})
    
    return total_loss / len(train_loader)

# =====================================================
# MAIN PIPELINE
# =====================================================

def main():
    import gc
    torch.cuda.empty_cache()
    gc.collect()
    
    # Step 1: Load model and tokenizer
    print("\n" + "=" * 70)
    print("LOADING MODEL")
    print("=" * 70)
    
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_PATH, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True
    )
    
    model.gradient_checkpointing_enable()
    
    # Step 2: Apply LoRA with BEST hyperparameters
    print("\nApplying UNIFIED LoRA configuration...")
    print(f"  R: {Config.LORA_R}")
    print(f"  Alpha: {Config.LORA_ALPHA}")
    print(f"  Dropout: {Config.LORA_DROPOUT}")
    
    lora_config = LoraConfig(
        r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        lora_dropout=Config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Step 3: Prepare data
    train_df = prepare_training_data()
    train_dataset = prepare_dataset(train_df, tokenizer)
    
    def collate_fn(batch):
        return {
            'input_ids': torch.tensor([item['input_ids'] for item in batch], dtype=torch.long),
            'attention_mask': torch.tensor([item['attention_mask'] for item in batch], dtype=torch.long),
            'labels': torch.tensor([item['labels'] for item in batch], dtype=torch.long)
        }
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=True)
    
    # Step 4: Setup optimizer with BEST hyperparameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=0.01)
    
    total_steps = len(train_loader) * Config.N_EPOCHS // Config.GRADIENT_ACCUMULATION
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=Config.WARMUP_STEPS, num_training_steps=total_steps)
    
    device = next(model.parameters()).device
    
    # Step 5: Train with optimal configuration
    print("\n" + "=" * 70)
    print(f"TRAINING - {Config.N_EPOCHS} EPOCHS (UNIFIED BEST)")
    print("=" * 70)
    
    best_loss = float('inf')
    for epoch in range(Config.N_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.N_EPOCHS}")
        avg_loss = train_model(model, train_loader, optimizer, scheduler, device)
        print(f"Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"âœ… Best: {best_loss:.4f}")
        
        if avg_loss < 0.4:
            print("âš ï¸�  Loss too low - potential overfitting")
        elif avg_loss < 0.7:
            print("ğŸ�¯ Excellent learning!")
        elif avg_loss < 1.0:
            print("âœ… Good learning")
    
    # Step 6: Save model
    print("\n" + "=" * 70)
    print("SAVING MODEL")
    print("=" * 70)
    
    model.save_pretrained(Config.OUTPUT_DIR)
    tokenizer.save_pretrained(Config.OUTPUT_DIR)
    
    print(f"âœ… Model saved to: {Config.OUTPUT_DIR}")
    print(f"Final best loss: {best_loss:.4f}")
    
    # Step 7: AVERAGE TTA Inference (CRITICAL!)
    print("\n" + "=" * 70)
    print("AVERAGE TTA INFERENCE (NOT MAX!)")
    print("=" * 70)
    
    model.eval()
    test_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
    
    all_predictions = []
    
    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]
    
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting"):
        
        tta_preds = []
        
        # TTA: 2 variations
        for use_alt in [False, True]:
            if use_alt:
                pos_ex = row['positive_example_2']
                neg_ex = row['negative_example_2']
            else:
                pos_ex = row['positive_example_1']
                neg_ex = row['negative_example_1']
            
            messages = [
                {'role': 'system', 'content': 'You are a content moderator. Answer Yes or No.'},
                {'role': 'user', 'content': f"""Rule: {row['rule']}
Violation: {pos_ex}
Non-violation: {neg_ex}

Comment: {row['body']}
Violates?"""}
            ]
            
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=Config.MAX_LENGTH).to(model.device)
            
            with torch.no_grad():
                logits = model(**inputs).logits[0, -1, :]
            
            yes_logit = logits[yes_id].cpu().item()
            no_logit = logits[no_id].cpu().item()
            
            prob = np.exp(yes_logit) / (np.exp(yes_logit) + np.exp(no_logit))
            tta_preds.append(prob)
        
        # CRITICAL: Use MEAN instead of MAX
        final_pred = np.mean(tta_preds)
        all_predictions.append(final_pred)
    
    predictions = np.array(all_predictions)
    
    # Step 8: Create submission
    submission = pd.DataFrame({
        'row_id': test_df['row_id'].values,
        'rule_violation': predictions
    })
    
    submission.to_csv('submission.csv', index=False)
    
    # Step 9: Display results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(submission.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("âœ… SUBMISSION READY: submission.csv")
    print("=" * 70)
    
    print(f"\nPrediction statistics:")
    print(f"  Min:    {predictions.min():.4f}")
    print(f"  Max:    {predictions.max():.4f}")
    print(f"  Mean:   {predictions.mean():.4f}")
    print(f"  Std:    {predictions.std():.4f}")
    
    print(f"\nExpected: 0.89-0.91+")
    print(f"  âœ… 2 epochs (optimal)")
    print(f"  âœ… Batch size 2 (stable)")
    print(f"  âœ… LR 1.5e-4 (best learning)")
    print(f"  âœ… LoRA 24/48 (optimal capacity)")
    print(f"  âœ… Average TTA (not max!)")
    
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    hist, _ = np.histogram(predictions, bins=bins)
    print("\nPrediction distribution:")
    for i in range(len(bins)-1):
        print(f"  [{bins[i]:.1f}-{bins[i+1]:.1f}]: {hist[i]:3d} ({hist[i]/len(predictions)*100:5.1f}%)")

if __name__ == "__main__":
    main()




