"""
JIGSAW AGILE â€” FINAL ENSEMBLE SUBMISSION (0.95+ TARGET)
âœ“ 3-seed ensemble for maximum robustness
âœ“ Hard negative mining (70/20/10 split)
âœ“ Adaptive score weighting
âœ“ Automatic ensemble averaging
"""
import os, sys, random, re, math
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from tqdm.auto import tqdm
from collections import Counter

try:
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader
    ST_AVAILABLE = True
except:
    ST_AVAILABLE = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist

os.environ["WANDB_DISABLED"] = "true"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# ========== ENSEMBLE CONFIGURATION ==========
ENSEMBLE_SEEDS = [42, 123, 999]  # 3 seeds for robust ensemble
RUN_ENSEMBLE = True  # Set to False to run single seed only

CONFIG = {
    'input_csv': '/kaggle/input/jigsaw-agile-community-rules/test.csv',
    'output_submission': 'submission.csv',
    'local_models_dir': '/kaggle/input/all-mpnet-base-v2/transformers/all-mpnet-base-v2/1',
    'do_finetune': True,
    'epochs': 2,
    'batch_size': 16,
    'lr': 2e-5,
    'triplet_margin': 0.4,
    'augmentation_factor': 4,
    'hard_negative_ratio': 0.7,
    'cross_rule_ratio': 0.2,
    'hard_negative_top_k': 5,
    'cross_rule_top_k': 3,
    'tfidf_max_features': 8192,
}

# -------------------- Enhanced Cleaning --------------------
def enhanced_cleaner(text):
    if text is None or pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'\b(spam|advertisement|ad|promo|referral|link|buy|sell|offer|discount|deal)\b', r' \1 \1 ', text)
    
    url_pattern = r"https?://[^\s<>\"{}|\\^`\\[\]]+"
    def rep(m):
        try:
            parsed = urlparse(m.group(0))
            domain = parsed.netloc.lower().replace("www.", "")
            parts = [p for p in parsed.path.split("/") if p]
            return f" urllink {domain} {parts[0] if parts else ''} "
        except:
            return " urllink "
    
    text = re.sub(url_pattern, rep, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -------------------- Hard Negative Mining --------------------
def mine_hard_negatives(df, text_to_emb_map, top_k=5):
    """Mine non-violations that are confusingly similar to violations"""
    hard_negative_map = {}
    
    for rule in df['rule'].unique():
        rule_df = df[df['rule'] == rule]
        
        violation_embs = []
        for _, r in rule_df.iterrows():
            for c in ['negative_example_1', 'negative_example_2']:
                v = r.get(c)
                if pd.notna(v):
                    t = enhanced_cleaner(v)
                    if t in text_to_emb_map:
                        violation_embs.append(text_to_emb_map[t])
        
        if not violation_embs:
            continue
        
        non_violation_pool = []
        for _, r in rule_df.iterrows():
            for c in ['positive_example_1', 'positive_example_2']:
                v = r.get(c)
                if pd.notna(v):
                    t = enhanced_cleaner(v)
                    if t in text_to_emb_map:
                        non_violation_pool.append((t, text_to_emb_map[t]))
        
        if not non_violation_pool or len(non_violation_pool) < 2:
            continue
        
        violation_arr = np.vstack(violation_embs)
        non_violation_texts, non_violation_embs = zip(*non_violation_pool)
        non_violation_arr = np.vstack(non_violation_embs)
        
        distances = cdist(non_violation_arr, violation_arr, 'cosine').min(axis=1)
        hardest_indices = np.argsort(distances)[:min(top_k, len(distances))]
        
        hard_negative_map[rule] = [non_violation_texts[i] for i in hardest_indices]
    
    return hard_negative_map

def mine_cross_rule_hard_negatives(df, text_to_emb_map, top_k=3):
    """Mine confusing non-violations from OTHER rules"""
    cross_rule_map = {}
    all_rules = df['rule'].unique()
    
    for rule in all_rules:
        rule_df = df[df['rule'] == rule]
        
        violation_pool = []
        for _, r in rule_df.iterrows():
            for c in ['negative_example_1', 'negative_example_2']:
                v = r.get(c)
                if pd.notna(v):
                    t = enhanced_cleaner(v)
                    if t in text_to_emb_map:
                        violation_pool.append(text_to_emb_map[t])
        
        if not violation_pool:
            continue
        
        violation_arr = np.vstack(violation_pool)
        
        cross_non_violations = []
        for other_rule in all_rules:
            if other_rule == rule:
                continue
            
            other_df = df[df['rule'] == other_rule]
            for _, r in other_df.iterrows():
                for c in ['positive_example_1', 'positive_example_2']:
                    v = r.get(c)
                    if pd.notna(v):
                        t = enhanced_cleaner(v)
                        if t in text_to_emb_map:
                            cross_non_violations.append((t, text_to_emb_map[t]))
        
        if not cross_non_violations:
            continue
        
        texts, embs = zip(*cross_non_violations)
        embs = np.vstack(embs)
        
        distances = cdist(embs, violation_arr, 'cosine').min(axis=1)
        hardest_indices = np.argsort(distances)[:min(top_k, len(distances))]
        
        cross_rule_map[rule] = [texts[i] for i in hardest_indices]
    
    return cross_rule_map

# -------------------- Advanced Triplet Creation --------------------
def create_advanced_triplets(df, hard_map, cross_map, aug_factor=4, hard_ratio=0.7, cross_ratio=0.2):
    """Creates triplets with 70/20/10 hard negative split"""
    anchors, positives, negatives = [], [], []
    
    grouped = df.groupby('rule')
    for rule, group in grouped:
        rule_clean = enhanced_cleaner(rule)
        
        violation_pool = list({enhanced_cleaner(x) for x in group[['negative_example_1','negative_example_2']].values.flatten() 
                              if isinstance(x, str)})
        non_violation_pool = list({enhanced_cleaner(x) for x in group[['positive_example_1','positive_example_2']].values.flatten() 
                                  if isinstance(x, str)})
        
        if not violation_pool or not non_violation_pool:
            continue
        
        hard_negatives = hard_map.get(rule, [])
        cross_negatives = cross_map.get(rule, [])
        
        n_aug = min(len(violation_pool) * len(non_violation_pool) * aug_factor, 250)
        n_hard = int(n_aug * hard_ratio)
        n_cross = int(n_aug * cross_ratio)
        n_random = n_aug - n_hard - n_cross
        
        if hard_negatives:
            for _ in range(n_hard):
                anchors.append(rule_clean)
                positives.append(random.choice(violation_pool))
                negatives.append(random.choice(hard_negatives))
        
        if cross_negatives:
            for _ in range(n_cross):
                anchors.append(rule_clean)
                positives.append(random.choice(violation_pool))
                negatives.append(random.choice(cross_negatives))
        
        for _ in range(n_random):
            anchors.append(rule_clean)
            positives.append(random.choice(violation_pool))
            negatives.append(random.choice(non_violation_pool))
    
    combined = list(zip(anchors, positives, negatives))
    random.shuffle(combined)
    
    return [InputExample(texts=[a,p,n]) for a,p,n in combined] if ST_AVAILABLE else combined

# -------------------- Centroid Creation --------------------
def create_centroids(df, text_to_emb_map):
    """Creates violation/non-violation centroids"""
    rule_centroids = {}
    
    for rule in df['rule'].unique():
        rule_df = df[df['rule'] == rule]
        violation_embs, non_violation_embs = [], []
        
        for _, r in rule_df.iterrows():
            for c in ['negative_example_1','negative_example_2']:
                v = r.get(c)
                if pd.notna(v):
                    t = enhanced_cleaner(v)
                    if t in text_to_emb_map:
                        violation_embs.append(text_to_emb_map[t])
            
            for c in ['positive_example_1','positive_example_2']:
                v = r.get(c)
                if pd.notna(v):
                    t = enhanced_cleaner(v)
                    if t in text_to_emb_map:
                        non_violation_embs.append(text_to_emb_map[t])
        
        if violation_embs and non_violation_embs:
            violation_arr = np.vstack(violation_embs)
            non_violation_arr = np.vstack(non_violation_embs)
            
            rule_centroids[rule] = {
                'violation_mean': normalize(violation_arr.mean(axis=0).reshape(1,-1))[0],
                'violation_median': normalize(np.median(violation_arr, axis=0).reshape(1,-1))[0],
                'violation_boundary': normalize(violation_arr[np.argmin(cdist(violation_arr, non_violation_arr, 'cosine').min(axis=1))].reshape(1,-1))[0],
                'non_violation_mean': normalize(non_violation_arr.mean(axis=0).reshape(1,-1))[0],
                'non_violation_median': normalize(np.median(non_violation_arr, axis=0).reshape(1,-1))[0],
                'non_violation_boundary': normalize(non_violation_arr[np.argmin(cdist(non_violation_arr, violation_arr, 'cosine').min(axis=1))].reshape(1,-1))[0],
                'violation_all': violation_arr,
                'non_violation_all': non_violation_arr,
                'violation_count': len(violation_embs),
                'non_violation_count': len(non_violation_embs)
            }
    
    return rule_centroids

# -------------------- Adaptive Score Weighting --------------------
def get_adaptive_weights(violation_count, non_violation_count):
    """Sample-size aware weighting"""
    min_count = min(violation_count, non_violation_count)
    
    if min_count < 5:
        return {'mean': 0.05, 'median': 0.15, 'boundary': 0.45, 'min_dist': 0.30, 'rule_sim': 0.05}
    elif min_count < 10:
        return {'mean': 0.20, 'median': 0.30, 'boundary': 0.30, 'min_dist': 0.15, 'rule_sim': 0.05}
    else:
        return {'mean': 0.40, 'median': 0.30, 'boundary': 0.15, 'min_dist': 0.10, 'rule_sim': 0.05}

# -------------------- Enhanced Scoring --------------------
def compute_scores(body_emb, centroids, rule_emb=None):
    """Score = HIGH for violations, LOW for non-violations"""
    scores = {}
    
    scores['mean'] = float(
        np.dot(body_emb, centroids['violation_mean']) - 
        np.dot(body_emb, centroids['non_violation_mean'])
    )
    
    scores['median'] = float(
        np.dot(body_emb, centroids['violation_median']) - 
        np.dot(body_emb, centroids['non_violation_median'])
    )
    
    scores['boundary'] = float(
        np.dot(body_emb, centroids['violation_boundary']) - 
        np.dot(body_emb, centroids['non_violation_boundary'])
    )
    
    min_dist_to_violations = cdist(body_emb.reshape(1,-1), centroids['violation_all'], 'cosine').min()
    min_dist_to_non_violations = cdist(body_emb.reshape(1,-1), centroids['non_violation_all'], 'cosine').min()
    scores['min_dist'] = float(min_dist_to_non_violations - min_dist_to_violations)
    
    scores['rule_sim'] = float(np.dot(body_emb, rule_emb)) if rule_emb is not None else 0.0
    
    return scores

def keyword_boost(body, rule):
    """Enhanced heuristic bonus"""
    b = body.lower()
    rule_l = rule.lower()
    bonus = 0.0
    
    if any(k in b for k in ["buy","sell","offer","referral","promo","discount","ad","advertisement","deal"]):
        if "advertising" in rule_l or "spam" in rule_l or "promotion" in rule_l:
            bonus += 0.30
    
    if any(k in b for k in ["legal","lawyer","attorney","sue","court","lawsuit"]):
        if "legal" in rule_l:
            bonus += 0.30
    
    if any(k in b for k in ["hate","kill","violence","racist","attack","threat"]):
        if "hate" in rule_l or "violence" in rule_l or "threat" in rule_l:
            bonus += 0.30
    
    if any(k in b for k in ["offtopic","unrelated","irrelevant"]):
        if "off-topic" in rule_l or "topic" in rule_l:
            bonus += 0.25
    
    if "urllink" in b:
        if "link" in rule_l or "spam" in rule_l or "advertising" in rule_l:
            bonus += 0.20
    
    return bonus

# -------------------- Model Finding --------------------
def find_local_model(base_dir):
    if not base_dir or not os.path.exists(base_dir):
        return None
    for root, dirs, files in os.walk(base_dir):
        if any(f in [fn.lower() for fn in files] for f in ['config.json','pytorch_model.bin','model.safetensors']):
            return root
    subs = [os.path.join(base_dir, s) for s in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir,s))]
    return subs[0] if subs else None

# ========== MAIN PIPELINE FUNCTION ==========
def run_single_seed(seed_value, test_df, all_texts, local_model_path):
    """Run complete pipeline for a single seed"""
    print(f"\n{'='*60}")
    print(f"ğŸ�² RUNNING WITH SEED = {seed_value}")
    print(f"{'='*60}")
    
    # Set seed
    random.seed(seed_value)
    np.random.seed(seed_value)
    
    # Load model
    USE_TFIDF_FALLBACK = False
    model = None
    
    if local_model_path and ST_AVAILABLE:
        try:
            model = SentenceTransformer(local_model_path)
            print(f"âœ“ Loaded SentenceTransformer")
        except Exception as e:
            print(f"â�Œ Failed: {e}")
    
    if model is None or not ST_AVAILABLE:
        USE_TFIDF_FALLBACK = True
        print("âš ï¸� Using TF-IDF fallback")
    
    # Generate base embeddings
    text_to_base_emb = {}
    if USE_TFIDF_FALLBACK:
        vectorizer = TfidfVectorizer(max_features=CONFIG['tfidf_max_features'], ngram_range=(1,3), stop_words='english')
        X = vectorizer.fit_transform(all_texts)
        for i, t in enumerate(all_texts):
            v = X[i].toarray().reshape(-1)
            v /= np.linalg.norm(v) + 1e-9
            text_to_base_emb[t] = v
    else:
        print("ğŸ”„ Generating base embeddings...")
        base_embs = model.encode(all_texts, batch_size=CONFIG['batch_size'], normalize_embeddings=True, 
                                convert_to_tensor=False, show_progress_bar=False)
        for t, e in zip(all_texts, base_embs):
            text_to_base_emb[t] = np.array(e)
    
    # Mine hard negatives
    hard_negative_map = mine_hard_negatives(test_df, text_to_base_emb, CONFIG['hard_negative_top_k'])
    cross_rule_map = mine_cross_rule_hard_negatives(test_df, text_to_base_emb, CONFIG['cross_rule_top_k'])
    print(f"â›�ï¸� Mined: {len(hard_negative_map)} within-rule, {len(cross_rule_map)} cross-rule")
    
    # Create triplets
    triplets = create_advanced_triplets(test_df, hard_negative_map, cross_rule_map, 
                                       CONFIG['augmentation_factor'], 
                                       CONFIG['hard_negative_ratio'], 
                                       CONFIG['cross_rule_ratio'])
    print(f"âœ“ Created {len(triplets)} triplets")
    
    # Finetune
    if not USE_TFIDF_FALLBACK and model is not None and CONFIG['do_finetune'] and len(triplets) > 0:
        train_dataloader = DataLoader(triplets, shuffle=True, batch_size=CONFIG['batch_size'])
        loss_fn = losses.TripletLoss(model=model, triplet_margin=CONFIG['triplet_margin'])
        
        print(f"ğŸ”¥ Fine-tuning for {CONFIG['epochs']} epochs...")
        model.fit(
            train_objectives=[(train_dataloader, loss_fn)], 
            epochs=CONFIG['epochs'], 
            warmup_steps=int(0.1*len(train_dataloader)),
            optimizer_params={'lr': CONFIG['lr']},
            show_progress_bar=False
        )
        print("âœ“ Fine-tuning complete")
    
    # Generate finetuned embeddings
    text_to_emb = {}
    if USE_TFIDF_FALLBACK:
        text_to_emb = text_to_base_emb.copy()
    else:
        print("ğŸ”„ Generating finetuned embeddings...")
        finetuned_embs = model.encode(all_texts, batch_size=CONFIG['batch_size'], 
                                      normalize_embeddings=True, convert_to_tensor=False,
                                      show_progress_bar=False)
        for t, e in zip(all_texts, finetuned_embs):
            text_to_emb[t] = np.array(e)
    
    # Create centroids
    rule_centroids = create_centroids(test_df, text_to_emb)
    print(f"âœ“ Centroids for {len(rule_centroids)} rules")
    
    # Predict
    predictions = []
    for rule in test_df['rule'].unique():
        if rule not in rule_centroids:
            continue
        
        rule_rows = test_df[test_df['rule'] == rule]
        centroids = rule_centroids[rule]
        rule_emb = text_to_emb.get(enhanced_cleaner(rule))
        
        w = get_adaptive_weights(centroids['violation_count'], centroids['non_violation_count'])
        
        for _, r in rule_rows.iterrows():
            rid = r['row_id']
            body = enhanced_cleaner(r.get('body',''))
            
            if not body:
                predictions.append({'row_id': rid, 'rule_violation': 0.0, 'body': '', 'rule': rule})
                continue
            
            body_emb = text_to_emb.get(body)
            if body_emb is None:
                if USE_TFIDF_FALLBACK:
                    v = vectorizer.transform([body])
                    arr = v.toarray().reshape(-1)
                    arr /= np.linalg.norm(arr) + 1e-9
                    body_emb = arr
                else:
                    body_emb = model.encode(body, normalize_embeddings=True)
            
            multi = compute_scores(np.array(body_emb), centroids, rule_emb)
            
            score = (w['mean']*multi['mean'] + w['median']*multi['median'] + 
                     w['boundary']*multi['boundary'] + w['min_dist']*multi['min_dist'] + 
                     w['rule_sim']*multi['rule_sim'])
            
            predictions.append({'row_id': rid, 'rule_violation': score, 'body': body, 'rule': rule})
    
    # Calibrate
    sub_df = pd.DataFrame(predictions)
    
    raw_scores = sub_df['rule_violation'].values
    p1, p99 = np.percentile(raw_scores, [1, 99])
    sub_df['rule_violation'] = np.clip(raw_scores, p1, p99)
    
    for idx, row in sub_df.iterrows():
        boost = keyword_boost(row['body'], row['rule'])
        sub_df.at[idx, 'rule_violation'] += boost
    
    def gentle_sigmoid(x, scale=0.8):
        return 1 / (1 + np.exp(-scale * x))
    
    sub_df['rule_violation'] = sub_df['rule_violation'].apply(lambda x: gentle_sigmoid(x, scale=0.8))
    
    for rule in sub_df['rule'].unique():
        mask = sub_df['rule'] == rule
        vals = sub_df.loc[mask, 'rule_violation']
        if len(vals) > 1:
            min_val, max_val = vals.min(), vals.max()
            if max_val > min_val:
                sub_df.loc[mask, 'rule_violation'] = 0.05 + 0.90 * (vals - min_val) / (max_val - min_val)
    
    final_sub = test_df[['row_id']].merge(sub_df[['row_id', 'rule_violation']], on='row_id', how='left')
    final_sub['rule_violation'] = final_sub['rule_violation'].fillna(0.5)
    final_sub['rule_violation'] = np.clip(final_sub['rule_violation'], 0.0, 1.0)
    
    s = final_sub['rule_violation'].describe()
    print(f"ğŸ“Š Stats â†’ Min:{s['min']:.4f} Max:{s['max']:.4f} Mean:{s['mean']:.4f} Std:{s['std']:.4f}")
    
    return final_sub

# ========== MAIN EXECUTION ==========
print("="*60)
print("ğŸ�¯ JIGSAW AGILE â€” FINAL ENSEMBLE SUBMISSION")
print("="*60)

# Load data
test_df = pd.read_csv(CONFIG['input_csv'])
print(f"ğŸ“Š Loaded {len(test_df)} rows; {test_df['rule'].nunique()} unique rules")

# Collect all texts
all_texts = set()
for _, row in test_df.iterrows():
    for c in ['rule','body','positive_example_1','positive_example_2','negative_example_1','negative_example_2']:
        v = row.get(c)
        if pd.notna(v):
            all_texts.add(enhanced_cleaner(v))
all_texts = list(all_texts)
print(f"ğŸ“� Unique texts: {len(all_texts)}")

# Find model
local_model_path = find_local_model(CONFIG['local_models_dir'])
print(f"ğŸ“� Model: {local_model_path}")

# Run ensemble
if RUN_ENSEMBLE and len(ENSEMBLE_SEEDS) > 1:
    ensemble_submissions = []
    
    for seed in ENSEMBLE_SEEDS:
        sub = run_single_seed(seed, test_df, all_texts, local_model_path)
        ensemble_submissions.append(sub)
    
    # Ensemble averaging
    print(f"\n{'='*60}")
    print(f"ğŸ”€ CREATING ENSEMBLE FROM {len(ENSEMBLE_SEEDS)} SEEDS")
    print(f"{'='*60}")
    
    final_sub = ensemble_submissions[0][['row_id']].copy()
    
    # Simple arithmetic mean
    scores_sum = np.zeros(len(final_sub))
    for sub in ensemble_submissions:
        scores_sum += sub['rule_violation'].values
    
    final_sub['rule_violation'] = scores_sum / len(ENSEMBLE_SEEDS)
    final_sub['rule_violation'] = np.clip(final_sub['rule_violation'], 0.0, 1.0)
    
    # Verify diversity
    print(f"\nâœ“ Ensemble diversity check:")
    for i in range(len(ENSEMBLE_SEEDS)-1):
        corr = ensemble_submissions[i]['rule_violation'].corr(ensemble_submissions[i+1]['rule_violation'])
        print(f"  Seed {ENSEMBLE_SEEDS[i]} vs {ENSEMBLE_SEEDS[i+1]}: correlation = {corr:.4f}")
    
    disagreement = np.abs(ensemble_submissions[0]['rule_violation'].values - ensemble_submissions[1]['rule_violation'].values)
    print(f"  Mean absolute difference: {disagreement.mean():.4f}")
    print(f"  Max difference: {disagreement.max():.4f}")
    
else:
    # Single seed run
    final_sub = run_single_seed(ENSEMBLE_SEEDS[0], test_df, all_texts, local_model_path)

# Save
final_sub.to_csv(CONFIG['output_submission'], index=False)

s = final_sub['rule_violation'].describe()
print(f"\n{'='*60}")
print(f"âœ… FINAL SUBMISSION SAVED: {CONFIG['output_submission']}")
print(f"{'='*60}")
print(f"ğŸ“Š Final Stats:")
print(f"  Min: {s['min']:.4f}")
print(f"  Max: {s['max']:.4f}")
print(f"  Mean: {s['mean']:.4f}")
print(f"  Std: {s['std']:.4f}")
print(f"\nğŸš€ 3-SEED ENSEMBLE READY FOR 0.95+ LB!")
print(f"{'='*60}")

