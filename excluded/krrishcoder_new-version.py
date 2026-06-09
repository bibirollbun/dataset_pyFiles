


# Kaggle notebook: Bi-Encoder (Sentence-Transformers) for Reddit Rule Violation
# Ready-to-run. Designed for Kaggle environment.
# Approach: Use a bi-encoder (sentence-transformers) to embed rules and comments,
# incorporate rule examples into rule text, fine-tune with MultipleNegativesRankingLoss,
# then calibrate similarity to probabilities with logistic regression. Produces submission.csv

# ======== 0. Install dependencies (run this cell first) ========
# In Kaggle, use pip install for any missing packages.


try:
    import sentence_transformers
except Exception:
    !pip install -q sentence-transformers==2.2.2
    !pip install -q transformers
    !pip install -q scikit-learn
    !pip install -q matplotlib
    !pip install -q pandas
    !pip install -q tqdm





# ======== 1. Imports ========
import os
import gc
import random
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader

import torch

print('PyTorch:', torch.__version__)
# print('Sentence-Transformers:', SentenceTransformer.__version__)

# ======== 2. Load data ========
# Update paths as appropriate on Kaggle. Typical layout: /kaggle/input/<dataset-folder>/train.csv

INPUT_DIR = '/kaggle/input/jigsaw-agile-community-rules/'
# If your dataset directory name differs, change it here. On Kaggle you can find it in the dataset panel.
# We'll attempt to locate train.csv and test.csv automatically inside /kaggle/input

def find_file(filename):
    for root, dirs, files in os.walk(INPUT_DIR):
        if filename in files:
            return os.path.join(root, filename)
    return None

train_path = find_file('train.csv')
test_path = find_file('test.csv')

if train_path is None or test_path is None:
    raise FileNotFoundError('Could not find train.csv or test.csv under /kaggle/input. Please upload dataset or adjust paths.')

print('train_path:', train_path)
print('test_path:', test_path)

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print('Train shape:', train.shape)
print('Test shape:', test.shape)

# Quick peek
print(train.columns.tolist())
train.head(3)








# ======== 3. Basic EDA & cleaning ========
# Ensure expected columns exist
expected_cols = {'id','body','rule','subreddit','positive_example_1','positive_example_2','negative_example_1','negative_example_2','rule_violation'}
print('Expected columns subset present:', expected_cols.intersection(set(train.columns)))

# Count unique rules
unique_rules = train['rule'].nunique()
print('Unique rules in training data:', unique_rules)
print(train['rule'].value_counts())

# Fill NAs where appropriate
for c in ['body','positive_example_1','positive_example_2','negative_example_1','negative_example_2','rule']:
    if c in train.columns:
        train[c] = train[c].fillna('').astype(str)
    if c in test.columns:
        test[c] = test[c].fillna('').astype(str)


# ======== 4. Build 'rule_text' that includes examples ========
# We'll build a compact representation of the rule that includes a short list of positive/negative examples

def build_rule_text(row):
    parts = []
    rule = str(row.get('rule','')).strip()
    if rule:
        parts.append(f"Rule: {rule}")
    pos = []
    neg = []
    for c in ['positive_example_1','positive_example_2']:
        if c in row and row[c] and str(row[c]).strip():
            pos.append(str(row[c]).strip())
    for c in ['negative_example_1','negative_example_2']:
        if c in row and row[c] and str(row[c]).strip():
            neg.append(str(row[c]).strip())
    if pos:
        parts.append('Positive examples: ' + ' || '.join(pos[:2]))
    if neg:
        parts.append('Negative examples: ' + ' || '.join(neg[:2]))
    return ' \n '.join(parts)

# Apply to train and test
train['rule_text'] = train.apply(build_rule_text, axis=1)
# For test, we may not have example fields for unseen rules, but we'll build similarly when available
for c in ['positive_example_1','positive_example_2','negative_example_1','negative_example_2']:
    if c not in test.columns:
        test[c] = ''

test['rule_text'] = test.apply(build_rule_text, axis=1)

# Also create comment_text column (shorten extreme long bodies if necessary)
train['comment_text'] = train['body'].astype(str)
# Optionally truncate very long comments to save compute
MAX_COMMENT_CHARS = 1000
train['comment_text'] = train['comment_text'].apply(lambda x: x if len(x)<=MAX_COMMENT_CHARS else x[:MAX_COMMENT_CHARS])

test['comment_text'] = test['body'].astype(str)
if 'body' in test.columns:
    test['comment_text'] = test['comment_text'].apply(lambda x: x if len(x)<=MAX_COMMENT_CHARS else x[:MAX_COMMENT_CHARS])

print('Sample rule_text and comment_text:')
for i in range(2):
    print('---')
    print('Rule text:', train.loc[i,'rule_text'])
    print('Comment:', train.loc[i,'comment_text'][:200])




train


























# ======== 5. Prepare training examples for bi-encoder ========
# NEW: Use both train and test examples to create additional pairs.

pos_rows = train[train['rule_violation']==1].copy()
print('Number of positive pairs (train only):', pos_rows.shape[0])

examples = []


# -------------------------------------------------------
# 1) MAIN POSITIVE PAIRS FROM TRAIN (label = 1)
# -------------------------------------------------------
for _, r in pos_rows.iterrows():
    a = str(r['rule_text'])
    p = str(r['comment_text'])
    if a and p:
        examples.append(InputExample(texts=[a, p]))



# -------------------------------------------------------
# 2) EXTRA POSITIVES USING TRAIN positive_example columns
# -------------------------------------------------------
for _, r in train.iterrows():
    rule_t = str(r['rule_text']).strip()
    for c in ['positive_example_1','positive_example_2']:
        if c in r and isinstance(r[c], str) and r[c].strip():
            examples.append(InputExample(texts=[rule_t, r[c].strip()]))

# -------------------------------------------------------
# 3) NEW SECTION: USE TEST DATA AS EXTRA POSITIVE PAIRS
# -------------------------------------------------------
# The logic: rule_text in test → positive_example_1/2 → treat comment_text as another example
# We assume that test comments are valid examples for their rule.
# This will help the bi-encoder generalize.

print("Adding test pairs...")

for _, r in test.iterrows():
    rule_t = str(r['rule_text']).strip()
    comment = str(r['comment_text']).strip()

    # test bodies often follow the rule → treat as positive pair
    if rule_t and comment:
        examples.append(InputExample(texts=[rule_t, comment]))

    # also use test's example fields if available
    for c in ['positive_example_1','positive_example_2']:
        if c in r and isinstance(r[c], str) and r[c].strip():
            examples.append(InputExample(texts=[rule_t, r[c].strip()]))

print('Total InputExample pairs prepared for training:', len(examples))



examples[0]


ex = examples[0]
print("GUID:", ex.guid)
print("Text 1:", ex.texts[0])
print("Text 2:", ex.texts[1])
print("Label:", ex.label)






# ============================================================
# Kaggle-Safe Fix: Disable chat templates / telemetry (no internet needed)
# ============================================================

import os, sys

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_NO_ADDITIONAL_CHAT_TEMPLATES"] = "1"

# Clear cached modules from memory to avoid template lookup
for m in list(sys.modules.keys()):
    if m.startswith("transformers") or m.startswith("sentence_transformers") or m.startswith("huggingface_hub"):
        del sys.modules[m]

print("✅ Environment flags set. No internet or pip installs required on Kaggle.")



from sentence_transformers import SentenceTransformer










model = SentenceTransformer("/kaggle/input/sentencetransformersallmpnetbasev2/all-mpnet-base-v2")





# ======== 6. Train bi-encoder (Sentence-Transformers) ========
import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"

# Optionally patch wandb to no-op (hard disable)
try:
    import wandb
    wandb.init = lambda *args, **kwargs: None
except:
    pass

MODEL_NAME = "/kaggle/input/sentencetransformersallmpnetbasev2/all-mpnet-base-v2"
OUTPUT_DIR = '/kaggle/working/rule_comment_biencoder'

model = SentenceTransformer(MODEL_NAME)

# Create DataLoader
train_batch_size = 8
train_dataloader = DataLoader(examples, shuffle=True, batch_size=train_batch_size)

# Loss
train_loss = losses.MultipleNegativesRankingLoss(model)

# Training settings
num_epochs = 3
warmup_steps = max(100, int(len(train_dataloader) * num_epochs * 0.1))

print('Training settings:', 'batch_size', train_batch_size, 'epochs', num_epochs, 'warmup', warmup_steps)

# Train (now W&B won't run!)
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    show_progress_bar=True,
    output_path=OUTPUT_DIR
)



gc.collect()


























# ======== 7. Encode train/validation for calibration ========
# We will compute embeddings for each row and then train a small LogisticRegression to map similarity (and other features) -> probability

# Create a cross-validation split for calibration. Use GroupKFold by subreddit so that model sees different subreddits in validation.
# Note: Training data has only 2 rules; grouping by rule wouldn't be useful. Grouping by subreddit helps reduce leakage.

if 'subreddit' in train.columns:
    groups = train['subreddit'].fillna('unknown').values
else:
    groups = np.arange(len(train))

gkf = GroupKFold(n_splits=5)
# We'll use first fold as validation for calibrator training here. You may loop folds for robust calibration.

train_idx, val_idx = next(gkf.split(train, train['rule_violation'], groups))
train_cal = train.iloc[train_idx].reset_index(drop=True)
val_cal = train.iloc[val_idx].reset_index(drop=True)

print('Calibrator train size:', train_cal.shape, 'val size:', val_cal.shape)

# Encode rule_text and comment_text for both sets
# For rules, to better represent a rule we will average embeddings of (rule_text + its positive examples + negative examples)

def compute_rule_representation(df, model):
    # returns numpy array of shape (len(df), emb_dim)
    texts_for_rule = df['rule_text'].fillna('').tolist()
    # encode directly
    emb_rules = model.encode(texts_for_rule, convert_to_numpy=True, show_progress_bar=True)
    return emb_rules

print('Encoding calibrator sets...')
train_rule_emb = compute_rule_representation(train_cal, model)
train_comment_emb = model.encode(train_cal['comment_text'].tolist(), convert_to_numpy=True, show_progress_bar=True)

val_rule_emb = compute_rule_representation(val_cal, model)
val_comment_emb = model.encode(val_cal['comment_text'].tolist(), convert_to_numpy=True, show_progress_bar=True)



# Build features: cosine similarity, absolute diff, elementwise product, lengths
from sklearn.preprocessing import StandardScaler

def build_features(rule_emb, comment_emb, df):
    # rule_emb, comment_emb: numpy arrays (n, d)
    assert rule_emb.shape[0] == comment_emb.shape[0]
    sims = np.sum(rule_emb * comment_emb, axis=1) / (
        np.linalg.norm(rule_emb, axis=1) * np.linalg.norm(comment_emb, axis=1) + 1e-9)
    absdiff = np.abs(rule_emb - comment_emb)
    prod = rule_emb * comment_emb
    # We'll use low-d features to keep logistic regression manageable: (similarity, mean absdiff, mean prod)
    feat = np.vstack([
        sims,
        np.mean(absdiff, axis=1),
        np.mean(prod, axis=1),
        df['comment_text'].str.len().fillna(0).values,
        df['rule_text'].str.len().fillna(0).values
    ]).T
    return feat, sims

X_tr, sims_tr = build_features(train_rule_emb, train_comment_emb, train_cal)
X_val, sims_val = build_features(val_rule_emb, val_comment_emb, val_cal)




# Scale features
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_val_scaled = scaler.transform(X_val)

# Train logistic regression calibrator
clf = LogisticRegression(max_iter=2000)
clf.fit(X_tr_scaled, train_cal['rule_violation'].values)

val_probs = clf.predict_proba(X_val_scaled)[:,1]
val_auc = roc_auc_score(val_cal['rule_violation'].values, val_probs)
print('Validation AUC (calibrator):', val_auc)

# Also try using raw cosine similarity with sigmoid for a quick baseline
from scipy.special import expit
scale = 4.0
val_probs_baseline = expit(scale * sims_val)
print('Validation AUC (scaled-cosine baseline):', roc_auc_score(val_cal['rule_violation'].values, val_probs_baseline))




# ======== 8. Full dataset embeddings (for inference) ========
# Compute embeddings for all training rows (useful for additional calibrator training or ensembling)

all_rule_emb = compute_rule_representation(train, model)
all_comment_emb = model.encode(train['comment_text'].tolist(), convert_to_numpy=True, show_progress_bar=True)

# Optionally retrain calibrator on all data (train+val) to use full signal for final submission
X_all, sims_all = build_features(all_rule_emb, all_comment_emb, train)
X_all_scaled = scaler.fit_transform(X_all)
clf_full = LogisticRegression(max_iter=2000)
clf_full.fit(X_all_scaled, train['rule_violation'].values)




# ======== 9. Inference on test set ========
# Build rule_text for test (already done above). Compute embeddings and predict

print('Encoding test set...')
# For test, there might be many unseen rules. Our rule_text includes any positive/negative examples present in the test file (if provided).

test_rule_emb = model.encode(test['rule_text'].tolist(), convert_to_numpy=True, show_progress_bar=True)
test_comment_emb = model.encode(test['comment_text'].tolist(), convert_to_numpy=True, show_progress_bar=True)

X_test, sims_test = build_features(test_rule_emb, test_comment_emb, test)
X_test_scaled = scaler.transform(X_test)

test_probs = clf_full.predict_proba(X_test_scaled)[:,1]
# Also compute scaled-cosine baseline
test_probs_baseline = expit(scale * sims_test)

# Choose final probability as average of both (simple ensembling)
final_probs = 0.6 * test_probs + 0.4 * test_probs_baseline




# ======== 10. Create submission ========
# The test file must contain an id column. If not, use index.
if 'row_id' in test.columns:
    out_ids = test['row_id']
else:
    out_ids = test.index

submission = pd.DataFrame({'row_id': out_ids, 'rule_violation': final_probs})
submission_path = '/kaggle/working/submission.csv'
submission.to_csv(submission_path, index=False)
print('Saved submission to', submission_path)




submission.head()














# # ======== 11. Save models/artifacts ========
# # Save sentence-transformers model (already saved to OUTPUT_DIR by model.fit)
# # Save calibrator and scaler using joblib

# import joblib
# joblib.dump(clf_full, '/kaggle/working/calibrator_lr.joblib')
# joblib.dump(scaler, '/kaggle/working/scaler.joblib')

# print('Saved calibrator and scaler to /kaggle/working')

# # ======== 12. Additional suggestions & next steps (manual) ========
# # - If you have more GPU/time: increase num_epochs, increase batch_size, try a larger backbone (all-mpnet-base-v2 -> paraphrase-mpnet-base-v2)
# # - Add hard negative mining: sample negative comments that are semantically close to positives and re-train.
# # - Add ensembling across different sentence-transformers backbones and average probabilities.
# # - Consider fine-tuning a cross-encoder on the small dataset with extreme caution (overfitting risk). Use NLI-pretrained cross-encoders if you try.
# # - If the test set provides rule examples for unseen rules, include them in the rule_text to improve results.

# print('\nNotebook run complete. Review submission.csv and outputs in /kaggle/working.')

