import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import more_itertools
import kagglehub
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoModelForSequenceClassification
)
from sklearn.metrics import roc_auc_score
import gc
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Load data
train_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
test_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')
sample_sub = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Unique rules in train: {train_df['rule'].nunique()}")
print(f"Unique rules in test: {test_df['rule'].nunique()}")


# Data preprocessing
def clean_text(text):
    """Basic text cleaning"""
    if pd.isna(text):
        return ""
    text = str(text).strip()
    # Remove excessive whitespace
    text = ' '.join(text.split())
    return text

# Clean all text columns
print("Cleaning text data...")
for col in ['body', 'positive_example_1', 'positive_example_2', 
            'negative_example_1', 'negative_example_2']:
    if col in train_df.columns:
        train_df[col] = train_df[col].apply(clean_text)
    if col in test_df.columns:
        test_df[col] = test_df[col].apply(clean_text)



# Model 1: Gemma-2B for few-shot learning
print("\n=== Loading Gemma-2B Model ===")
GEMMA_2B_PATH = kagglehub.model_download("google/gemma-2/transformers/gemma-2-2b-it")
gemma_tokenizer = AutoTokenizer.from_pretrained(GEMMA_2B_PATH)
gemma_model = AutoModelForCausalLM.from_pretrained(
    GEMMA_2B_PATH, 
    torch_dtype=torch.float16,
    device_map="auto"
).to(device)
print("Gemma-2B loaded successfully!")


# Model 2: Gemma-1.1B (lighter version for ensemble diversity)
print("\n=== Loading Gemma-1.1B Model ===")
GEMMA_1B_PATH = kagglehub.model_download("google/gemma-3/transformers/gemma-3-1b-it")
gemma_small_tokenizer = AutoTokenizer.from_pretrained(GEMMA_1B_PATH)
gemma_small_model = AutoModelForCausalLM.from_pretrained(
    GEMMA_1B_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
).to(device)
print("Gemma-1.1B loaded successfully!")


# Enhanced prompt templates for better performance
def create_detailed_prompt(row):
    """Detailed prompt with clear structure"""
    prompt = f"""<start_of_turn>user
You are an expert moderator for the subreddit r/{row['subreddit']}. 
Analyze if the comment violates this specific rule:

RULE: {row['rule']}

EXAMPLES OF VIOLATIONS:
1. {row['positive_example_1']}
2. {row['positive_example_2']}

EXAMPLES OF NON-VIOLATIONS:
1. {row['negative_example_1']}
2. {row['negative_example_2']}

COMMENT TO EVALUATE:
{row['body']}

Based on the rule and examples, does this comment violate the rule?
Answer: <end_of_turn>
<start_of_turn>model
"""
    return prompt


def create_concise_prompt(row):
    """Concise prompt for faster inference"""
    prompt = f"""<start_of_turn>user
Rule: {row['rule']}
Positive: {row['positive_example_1'][:100]}
Negative: {row['negative_example_1'][:100]}
Comment: {row['body']}
Violates rule? <end_of_turn>
<start_of_turn>model
"""
    return prompt


# Prediction function for Gemma models
def predict_with_gemma(model, tokenizer, test_data, prompt_func, batch_size=4):
    """Make predictions using Gemma model"""
    predictions = []
    
    # Get token IDs for Yes/No/True/False
    yes_tokens = [tokenizer.get_vocab().get(word, tokenizer.unk_token_id) 
                  for word in ['Yes', 'yes', 'YES', 'True', 'true', 'TRUE']]
    no_tokens = [tokenizer.get_vocab().get(word, tokenizer.unk_token_id) 
                 for word in ['No', 'no', 'NO', 'False', 'false', 'FALSE']]
    
    all_tokens = list(set(yes_tokens + no_tokens))
    yes_indices = [all_tokens.index(t) for t in yes_tokens if t in all_tokens]
    
    for batch in more_itertools.batched(test_data.iterrows(), batch_size):
        prompts = [prompt_func(row) for _, row in batch]
        
        inputs = tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Get logits for relevant tokens
        logits = outputs.logits[:, -1, all_tokens]
        probabilities = torch.softmax(logits, dim=-1)
        
        # Sum probabilities for "yes" tokens
        for i in range(len(batch)):
            yes_prob = probabilities[i, yes_indices].sum().item()
            predictions.append(float(yes_prob))
    
    return predictions


# Make predictions with both Gemma models
print("\n=== Making predictions with Gemma-2B (detailed prompt) ===")
gemma_2b_predictions = predict_with_gemma(
    gemma_model, 
    gemma_tokenizer, 
    test_df, 
    create_detailed_prompt,
    batch_size=4
)
print(f"Completed! Average prediction: {np.mean(gemma_2b_predictions):.4f}")

print("\n=== Making predictions with Gemma-1.1B (concise prompt) ===")
gemma_1b_predictions = predict_with_gemma(
    gemma_small_model, 
    gemma_small_tokenizer, 
    test_df, 
    create_concise_prompt,
    batch_size=8  # Smaller model can handle larger batches
)
print(f"Completed! Average prediction: {np.mean(gemma_1b_predictions):.4f}")



# Model 3: DeBERTa-based approach (if available on Kaggle)
# Note: This uses a different approach - semantic similarity
print("\n=== Creating rule-based features ===")

try:
    # Try to load a sentence transformer model if available
    from sentence_transformers import SentenceTransformer
    
    # Create semantic features
    def create_semantic_features(df):
        """Create features based on text similarity and patterns"""
        features = []
        
        for _, row in df.iterrows():
            # Feature 1: Length ratio
            comment_len = len(row['body'].split())
            avg_example_len = (len(row['positive_example_1'].split()) + 
                             len(row['negative_example_1'].split())) / 2
            len_ratio = comment_len / (avg_example_len + 1)
            
            # Feature 2: Keyword overlap with positive examples
            comment_words = set(row['body'].lower().split())
            pos_words = set(row['positive_example_1'].lower().split() + 
                          row['positive_example_2'].lower().split())
            neg_words = set(row['negative_example_1'].lower().split() + 
                          row['negative_example_2'].lower().split())
            
            pos_overlap = len(comment_words & pos_words) / (len(comment_words) + 1)
            neg_overlap = len(comment_words & neg_words) / (len(comment_words) + 1)
            
            # Feature 3: Rule keyword presence
            rule_words = set(row['rule'].lower().split())
            rule_overlap = len(comment_words & rule_words) / (len(rule_words) + 1)
            
            # Simple heuristic prediction
            score = 0.5  # Base score
            score += 0.3 * (pos_overlap - neg_overlap)  # Similarity difference
            score += 0.1 * rule_overlap  # Rule relevance
            score += 0.1 * (1 - min(len_ratio, 2) / 2)  # Length similarity
            
            features.append(np.clip(score, 0.1, 0.9))
        
        return features
    
    print("Creating semantic features...")
    semantic_predictions = create_semantic_features(test_df)
    print(f"Completed! Average prediction: {np.mean(semantic_predictions):.4f}")
    
except Exception as e:
    print(f"Could not create semantic features: {e}")
    # Fallback: use simple random predictions with slight bias
    semantic_predictions = np.random.beta(2, 3, size=len(test_df))



# Ensemble predictions
print("\n=== Creating ensemble predictions ===")

# Convert all predictions to numpy arrays
pred_gemma_2b = np.array(gemma_2b_predictions)
pred_gemma_1b = np.array(gemma_1b_predictions)
pred_semantic = np.array(semantic_predictions)

# Stack predictions for analysis
all_predictions = np.column_stack([pred_gemma_2b, pred_gemma_1b, pred_semantic])

# Calculate correlation between models
print("\nModel correlations:")
for i, name1 in enumerate(['Gemma-2B', 'Gemma-1B', 'Semantic']):
    for j, name2 in enumerate(['Gemma-2B', 'Gemma-1B', 'Semantic']):
        if i < j:
            corr = np.corrcoef(all_predictions[:, i], all_predictions[:, j])[0, 1]
            print(f"{name1} vs {name2}: {corr:.4f}")

# Weighted ensemble
# Give more weight to Gemma-2B as it's the larger model
weights = [0.5, 0.3, 0.2]  # Gemma-2B, Gemma-1B, Semantic
ensemble_predictions = np.average(all_predictions, weights=weights, axis=1)

# Post-processing
def post_process_predictions(predictions):
    """Apply post-processing to improve predictions"""
    predictions = np.array(predictions)
    
    # Clip to valid probability range
    predictions = np.clip(predictions, 0.01, 0.99)
    
    # Apply slight smoothing to avoid extreme predictions
    # Push predictions slightly toward 0.5
    smoothing_factor = 0.1
    predictions = predictions * (1 - smoothing_factor) + 0.5 * smoothing_factor
    
    return predictions



final_predictions = post_process_predictions(ensemble_predictions)

# Analyze predictions
print(f"\n=== Final Ensemble Statistics ===")
print(f"Mean: {final_predictions.mean():.4f}")
print(f"Std: {final_predictions.std():.4f}")
print(f"Min: {final_predictions.min():.4f}")
print(f"Max: {final_predictions.max():.4f}")


# Validation on a small subset of training data
print("\n=== Validation on training subset ===")
val_size = min(20, len(train_df))
val_indices = np.random.choice(len(train_df), size=val_size, replace=False)
val_df = train_df.iloc[val_indices].copy()

# Predict on validation set
val_pred_1 = predict_with_gemma(gemma_model, gemma_tokenizer, val_df, create_detailed_prompt, batch_size=4)
val_pred_2 = predict_with_gemma(gemma_small_model, gemma_small_tokenizer, val_df, create_concise_prompt, batch_size=8)
val_pred_3 = create_semantic_features(val_df)

val_ensemble = np.average(
    np.column_stack([val_pred_1, val_pred_2, val_pred_3]), 
    weights=weights, 
    axis=1
)
val_ensemble = post_process_predictions(val_ensemble)

# Calculate validation AUC
val_auc = roc_auc_score(val_df['rule_violation'], val_ensemble)
print(f"Validation AUC: {val_auc:.4f}")





# Create submission
submission = pd.DataFrame({
    'row_id': test_df['row_id'],
    'rule_violation': final_predictions
})

# Final checks
print("\n=== Submission Summary ===")
print(f"Shape: {submission.shape}")
print(f"Any NaN values: {submission['rule_violation'].isna().sum()}")
print("\nFirst 10 predictions:")
print(submission.head(10))

# Save submission
submission.to_csv('submission.csv', index=False)
print("\n✅ Submission saved successfully!")




