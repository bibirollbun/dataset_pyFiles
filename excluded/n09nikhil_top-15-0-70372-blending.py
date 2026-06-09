import os, re
import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import spearmanr


ROOT_DIR = '/kaggle/input/blending-ps-s5e12'
OUT_DIR = '/kaggle/working/'
os.makedirs(OUT_DIR, exist_ok=True)


def detect_pred_col(df):
    for c in ['diagnosed_diabetes', 'target', 'prediction', 'pred']:
        if c in df.columns: return c
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    nonid = [c for c in numeric if c.lower() not in ('id', 'ids', 'patient_id')]
    return nonid[0] if nonid else (numeric[0] if numeric else None)

def extract_score(name):
    m = re.search(r"([0-9]+\.[0-9]{4,6})", os.path.basename(name))
    return float(m.group(1)) if m else 0.0


files = sorted([f for f in os.listdir(ROOT_DIR) if f.lower().endswith('.csv')])
models = []

for fn in files:
    try:
        df = pd.read_csv(os.path.join(ROOT_DIR, fn))
        pred_col = detect_pred_col(df)
        if pred_col is None: continue
        
        score = extract_score(fn)
        if score < 0.69: continue
        
        models.append({
            'name': fn,
            'score': score,
            'preds': df[pred_col].astype(float).values,
            'id': df['id'].values if 'id' in df.columns else None
        })
    except:
        continue


lengths = [len(m['preds']) for m in models]
target_len = Counter(lengths).most_common(1)[0][0]
models = [m for m in models if len(m['preds']) == target_len]
models = sorted(models, key=lambda x: x['score'], reverse=True)

ids = models[0]['id']

print(f"Loaded {len(models)} models")
print(f"Top model score: {models[0]['score']:.5f}")


best_single = models[0]['preds']

preds_top20 = np.column_stack([models[i]['preds'] for i in range(20)])
correlations = []
for i in range(1, 20):
    corr, _ = spearmanr(best_single, models[i]['preds'])
    correlations.append((i, corr, models[i]['score']))

correlations.sort(key=lambda x: (x[2], -x[1]), reverse=True)
diverse_idx = correlations[0][0]
diverse_model = models[diverse_idx]['preds']


top5_preds = np.column_stack([models[i]['preds'] for i in range(5)])
top5_scores = np.array([models[i]['score'] for i in range(5)])
top5_weights = np.power(top5_scores, 10)
top5_weights = top5_weights / top5_weights.sum()
strat2a = top5_preds @ top5_weights


meta3 = best_single * 0.85 + strat2a * 0.15
final_preds = np.clip(meta3, 0, 1)


df = pd.DataFrame({'id': ids, 'diagnosed_diabetes': final_preds})
df.to_csv(os.path.join(OUT_DIR, 'submission.csv'), index=False)

print(f"\nâœ… Submission saved!")
print(f"Mean: {final_preds.mean():.6f}")
print(f"Std: {final_preds.std():.6f}")




