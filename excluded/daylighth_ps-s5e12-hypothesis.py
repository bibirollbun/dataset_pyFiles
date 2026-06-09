%%writefile utils.py
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

TRAIN_PATH = "/kaggle/input/playground-series-s5e12/train.csv"
TEST_PATH = "/kaggle/input/playground-series-s5e12/test.csv"
ORIG_PATH = "/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv"
OUTPUT_DIR = "outputs"
N_SPLITS = 10
RANDOM_STATE = 42
CUTOFF_ID = 678260

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    orig = pd.read_csv(ORIG_PATH)
    
    BASE = [col for col in train.columns if col not in ['id', 'diagnosed_diabetes']]
    
    print("ğŸ”§ Adding Original Data Features...")
    
    # --- Original Data Mean/Count Encoding ---
    for col in BASE:
        if col in orig.columns:
            mean_map = orig.groupby(col)['diagnosed_diabetes'].mean()
            new_col_mean = f'orig_mean_{col}'
            
            train[new_col_mean] = train[col].map(mean_map)
            test[new_col_mean] = test[col].map(mean_map)
            
            global_mean = orig['diagnosed_diabetes'].mean()
            train[new_col_mean] = train[new_col_mean].fillna(global_mean)
            test[new_col_mean] = test[new_col_mean].fillna(global_mean)
            
            count_map = orig.groupby(col).size()
            new_col_count = f'orig_count_{col}'
            
            train[new_col_count] = train[col].map(count_map)
            test[new_col_count] = test[col].map(count_map)
            
            train[new_col_count] = train[new_col_count].fillna(0)
            test[new_col_count] = test[new_col_count].fillna(0)
    
    print(f"âœ… Added {len(BASE) * 2} Original features")
    
    X = train.drop("diagnosed_diabetes", axis=1)
    y = train["diagnosed_diabetes"]
    
    return X, y, test 

def add_interaction_features(X, X_test):
    print("ğŸ”§ Generating Interaction Features...")
    
    n_train = len(X)
    df_all = pd.concat([X, X_test], axis=0).reset_index(drop=True)
    
    if 'systolic_bp' in df_all.columns and 'diastolic_bp' in df_all.columns:
        df_all['pulse_pressure'] = df_all['systolic_bp'] - df_all['diastolic_bp']
        df_all['map_bp'] = df_all['diastolic_bp'] + (df_all['pulse_pressure'] / 3)
    
    if 'age' in df_all.columns:
        df_all['age_bin'] = pd.qcut(df_all['age'], q=10, labels=False, duplicates='drop').astype(str)
    
    if 'bmi' in df_all.columns:
        df_all['bmi_bin'] = pd.qcut(df_all['bmi'], q=10, labels=False, duplicates='drop').astype(str)
        
    if 'glucose_fasting' in df_all.columns:
        df_all['glucose_bin'] = pd.qcut(df_all['glucose_fasting'], q=5, labels=False, duplicates='drop').astype(str)

    if 'age_bin' in df_all.columns and 'gender' in df_all.columns:
        df_all['gender_age_inter'] = df_all['gender'].astype(str) + '_' + df_all['age_bin']
        
    if 'ethnicity' in df_all.columns and 'bmi_bin' in df_all.columns:
        df_all['ethnicity_bmi_inter'] = df_all['ethnicity'].astype(str) + '_' + df_all['bmi_bin']
        
    if 'education_level' in df_all.columns and 'smoking_status' in df_all.columns:
        df_all['edu_smoke_inter'] = df_all['education_level'].astype(str) + '_' + df_all['smoking_status'].astype(str)
    
    if 'family_history_diabetes' in df_all.columns and 'age_bin' in df_all.columns:
        df_all['family_age_inter'] = df_all['family_history_diabetes'].astype(str) + '_' + df_all['age_bin']

    X_new = df_all.iloc[:n_train].copy()
    X_test_new = df_all.iloc[n_train:].copy()
    
    X_new.index = X.index
    X_test_new.index = X_test.index
    
    print(f"âœ… Interaction Features added. Total cols: {X_new.shape[1]}")
    return X_new, X_test_new

def encode_categorical(X, X_test):
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    
    n_train = len(X)
    df_all = pd.concat([X, X_test], axis=0)
    
    for col in cat_cols:
        df_all[col] = df_all[col].astype("category").cat.codes
        
    X_enc = df_all.iloc[:n_train]
    X_test_enc = df_all.iloc[n_train:]
    
    return X_enc, X_test_enc

def get_folds(X, y):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    return list(skf.split(X, y))

def save_predictions(oof, pred, test_ids, name):
    pd.DataFrame({'oof': oof}).to_csv(f"{OUTPUT_DIR}/oof_{name}.csv", index=False)
    pd.DataFrame({'id': test_ids, 'diagnosed_diabetes': pred}).to_csv(f"{OUTPUT_DIR}/submission_{name}.csv", index=False)
    
    train = pd.read_csv(TRAIN_PATH)
    y_true = train['diagnosed_diabetes']
    cutoff_mask = train['id'].values >= CUTOFF_ID
    
    full_auc = roc_auc_score(y_true, oof)
    cutoff_auc = roc_auc_score(y_true[cutoff_mask], oof[cutoff_mask])
    
    print(f"\n{'='*50}")
    print(f"âœ… {name.upper()} saved")
    print(f"ğŸ“Š Full OOF AUC:   {full_auc:.5f}")
    print(f"â­� Cutoff AUC:     {cutoff_auc:.5f} (id >= {CUTOFF_ID}, {cutoff_mask.sum()} samples)")
    print(f"{'='*50}")


import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

TRAIN_PATH = "/kaggle/input/playground-series-s5e12/train.csv"
TEST_PATH = "/kaggle/input/playground-series-s5e12/test.csv"
ORIG_PATH = "/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv"

SCAN_START_ID = 600000 
WINDOW_SIZE = 20000    
STEP_SIZE = 1000       

print("1. Loading Data & Preprocessing...")
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
df_orig = pd.read_csv(ORIG_PATH)

features = [c for c in train.columns if c not in ['id', 'diagnosed_diabetes'] and c in test.columns]
X_train_raw = train[features].copy()
X_test_raw = test[features].copy()

X_train_raw['_is_train'] = 1
X_test_raw['_is_train'] = 0
df_all = pd.concat([X_train_raw, X_test_raw], axis=0).reset_index(drop=True)

cat_cols = df_all.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    le = LabelEncoder()
    df_all[col] = le.fit_transform(df_all[col].astype(str))

X_train_full = df_all[df_all['_is_train'] == 1].drop(columns=['_is_train']).reset_index(drop=True)
X_test_full = df_all[df_all['_is_train'] == 0].drop(columns=['_is_train']).reset_index(drop=True)
y_test = np.ones(len(X_test_full))
train_ids = train['id'].values

start_index = train[train['id'] >= SCAN_START_ID].index[0]

print(f"2. Starting MICRO-SCAN from ID {SCAN_START_ID} (Index {start_index})...")
print(f"   Using Window={WINDOW_SIZE}, Step={STEP_SIZE}")

results = []
centers = []

params = {
    'objective': 'binary', 'metric': 'auc', 'n_estimators': 100,
    'learning_rate': 0.1, 'max_depth': 4, 'random_state': 42, 'verbose': -1, 'n_jobs': -1
}

for start_idx in range(start_index, len(X_train_full) - WINDOW_SIZE + 1, STEP_SIZE):
    end_idx = start_idx + WINDOW_SIZE
    
    X_subset = X_train_full.iloc[start_idx:end_idx]
    y_subset = np.zeros(len(X_subset))
    
    X_adv = pd.concat([X_subset, X_test_full], axis=0)
    y_adv = np.concatenate([y_subset, y_test])
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    aucs = []
    for t_idx, v_idx in skf.split(X_adv, y_adv):
        X_t, y_t = X_adv.iloc[t_idx], y_adv[t_idx]
        X_v, y_v = X_adv.iloc[v_idx], y_adv[v_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_t, y_t)
        preds = model.predict_proba(X_v)[:, 1]
        aucs.append(roc_auc_score(y_v, preds))
    
    avg_auc = np.mean(aucs)
    
    window_start_id = train_ids[start_idx]
    
    results.append(avg_auc)
    centers.append(window_start_id)
    
    print(f"   Window Start: {window_start_id:<8} | AUC: {avg_auc:.4f}")

plt.figure(figsize=(14, 8))
plt.plot(centers, results, 'o-', color='red', markersize=4, linewidth=1, label='AUC (Window Start vs Test)')

plt.axhline(0.5, color='green', linestyle='--', label='Perfect Match (0.5)')
plt.axvline(678260, color='blue', linestyle=':', label='Masaya ID (678260)')

plt.title("Micro-Scan of the Tail Transition (Finding the Exact Cutoff)", fontsize=16)
plt.xlabel("Window Start ID", fontsize=12)
plt.ylabel("Adversarial AUC", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

candidates = [centers[i] for i, auc in enumerate(results) if auc < 0.52]
if candidates:
    print(f"\nâœ… First 'Clean' ID (AUC < 0.52): {candidates[0]}")
    min_auc = min(results)
    best_id = centers[results.index(min_auc)]
    print(f"ğŸ�† Best ID (Lowest AUC {min_auc:.4f}): {best_id}")
else:
    print("\nâš ï¸� No clean point found. Check range or metric.")


from scipy.stats import wasserstein_distance
from sklearn.preprocessing import MinMaxScaler

df_head = train[train['id'] < best_id].copy()
df_tail = train[train['id'] >= best_id].copy()
df_test = test.copy()

common_features = [c for c in features if c in df_orig.columns]

numeric_features = train[common_features].select_dtypes(include=['int64', 'float64']).columns.tolist()

scaler = MinMaxScaler()
all_data_for_scale = pd.concat([train[numeric_features], test[numeric_features], df_orig[numeric_features]], axis=0)
scaler.fit(all_data_for_scale)

head_scaled = pd.DataFrame(scaler.transform(df_head[numeric_features]), columns=numeric_features)
tail_scaled = pd.DataFrame(scaler.transform(df_tail[numeric_features]), columns=numeric_features)
test_scaled = pd.DataFrame(scaler.transform(df_test[numeric_features]), columns=numeric_features)
orig_scaled = pd.DataFrame(scaler.transform(df_orig[numeric_features]), columns=numeric_features)

metrics = []

for col in numeric_features:
    d_head = wasserstein_distance(head_scaled[col], test_scaled[col])
    d_tail = wasserstein_distance(tail_scaled[col], test_scaled[col])
    d_orig = wasserstein_distance(orig_scaled[col], test_scaled[col])
    
    metrics.append({
        'Feature': col,
        'Dist(Head, Test)': d_head,
        'Dist(Tail, Test)': d_tail,
        'Dist(Orig, Test)': d_orig,
        'Delta_Tail': d_tail - d_head
    })

df_metrics = pd.DataFrame(metrics)
df_sorted = df_metrics.sort_values('Dist(Head, Test)', ascending=True)

plt.figure(figsize=(14, len(numeric_features) * 0.6))

y = np.arange(len(df_sorted))
height = 0.25

plt.barh(y - height, df_sorted['Dist(Head, Test)'], height, label='Head (Dirty)', color='gray', alpha=0.4)
plt.barh(y, df_sorted['Dist(Tail, Test)'], height, label='Tail (Cleaned)', color='blue')
plt.barh(y + height, df_sorted['Dist(Orig, Test)'], height, label='Original (Ground Truth)', color='green')

plt.yticks(y, df_sorted['Feature'])
plt.xlabel("Wasserstein Distance (Normalized)")
plt.title("Distribution Distance: Head vs Tail vs Original (Benchmark)")
plt.legend()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

print("Wasserstein Distance Metrics (Sorted by Head Gap):")
print("-" * 105)
print(f"{'Feature':<35} | {'Dist(Head)':<10} | {'Dist(Tail)':<10} | {'Dist(Orig)':<10} | {'Tail vs Orig'}")
print("-" * 105)

for _, row in df_metrics.sort_values('Dist(Head, Test)', ascending=False).iterrows():
    winner = "âœ… Tail Better" if row['Dist(Tail, Test)'] < row['Dist(Orig, Test)'] else "Original Better"
    print(f"{row['Feature']:<35} | {row['Dist(Head, Test)']:<10.4f} | {row['Dist(Tail, Test)']:<10.4f} | {row['Dist(Orig, Test)']:<10.4f} | {winner}")


cat_features = [c for c in train[common_features].select_dtypes(include=['object']).columns 
               if c in test.columns]

metrics_cat = []

for col in cat_features:
    p_head = df_head[col].value_counts(normalize=True)
    p_tail = df_tail[col].value_counts(normalize=True)
    p_orig = df_orig[col].value_counts(normalize=True)
    p_test = df_test[col].value_counts(normalize=True)
    
    all_categories = set(p_head.index) | set(p_tail.index) | set(p_test.index) | set(p_orig.index)
    
    p_head = p_head.reindex(all_categories, fill_value=0)
    p_tail = p_tail.reindex(all_categories, fill_value=0)
    p_orig = p_orig.reindex(all_categories, fill_value=0)
    p_test = p_test.reindex(all_categories, fill_value=0)
    
    d_head = 0.5 * np.sum(np.abs(p_head - p_test))
    d_tail = 0.5 * np.sum(np.abs(p_tail - p_test))
    d_orig = 0.5 * np.sum(np.abs(p_orig - p_test))
    
    metrics_cat.append({
        'Feature': col,
        'Dist(Head, Test)': d_head,
        'Dist(Tail, Test)': d_tail,
        'Dist(Orig, Test)': d_orig
    })

df_cat_metrics = pd.DataFrame(metrics_cat)

if not df_cat_metrics.empty:
    df_sorted_cat = df_cat_metrics.sort_values('Dist(Head, Test)', ascending=True)

    plt.figure(figsize=(14, len(cat_features) * 0.8))
    y = np.arange(len(df_sorted_cat))
    height = 0.25

    plt.barh(y - height, df_sorted_cat['Dist(Head, Test)'], height, label='Head', color='gray', alpha=0.4)
    plt.barh(y, df_sorted_cat['Dist(Tail, Test)'], height, label='Tail', color='orange')
    plt.barh(y + height, df_sorted_cat['Dist(Orig, Test)'], height, label='Original', color='green')

    plt.yticks(y, df_sorted_cat['Feature'])
    plt.xlabel("Total Variation Distance")
    plt.title("Categorical Distribution Distance Comparison")
    plt.legend()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\nCategorical Distance Metrics (TVD):")
    print("-" * 105)
    print(f"{'Feature':<35} | {'Dist(Head)':<10} | {'Dist(Tail)':<10} | {'Dist(Orig)':<10}")
    print("-" * 105)
    for _, row in df_cat_metrics.sort_values('Dist(Head, Test)', ascending=False).iterrows():
        print(f"{row['Feature']:<35} | {row['Dist(Head, Test)']:<10.4f} | {row['Dist(Tail, Test)']:<10.4f} | {row['Dist(Orig, Test)']:<10.4f}")
else:
    print("No common categorical features found.")


std_head = df_head[numeric_features].std()
std_tail = df_tail[numeric_features].std()
std_test = df_test[numeric_features].std()

ratio_head = std_head / std_test
ratio_tail = std_tail / std_test

variance_df = pd.DataFrame({
    'Feature': numeric_features,
    'Head_Ratio': ratio_head,
    'Tail_Ratio': ratio_tail
}).sort_values('Head_Ratio', key=lambda x: abs(x - 1), ascending=False)

plt.figure(figsize=(14, len(numeric_features) * 0.5))
y = np.arange(len(variance_df))
height = 0.35

plt.barh(y - height/2, variance_df['Head_Ratio'], height, label='Head / Test Std', color='gray', alpha=0.5)
plt.barh(y + height/2, variance_df['Tail_Ratio'], height, label='Tail / Test Std', color='blue')

plt.axvline(1.0, color='green', linestyle='--', linewidth=2, label='Perfect Variance Match (1.0)')
plt.yticks(y, variance_df['Feature'])
plt.xlabel("Standard Deviation Ratio (vs Test)")
plt.title("Variance Consistency Check: Is the data spread the same?")
plt.legend()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()


import math
import seaborn as sns

n_iterations = 1000
sample_size = 5000 

n_cols = 3
n_rows = math.ceil(len(numeric_features) / n_cols)
plt.figure(figsize=(18, n_rows * 3.5))

for i, col in enumerate(numeric_features):
    means_head = []
    means_tail = []
    means_test = []
    
    for _ in range(n_iterations):
        means_head.append(df_head[col].sample(sample_size, replace=True).mean())
        means_tail.append(df_tail[col].sample(sample_size, replace=True).mean())
        means_test.append(df_test[col].sample(sample_size, replace=True).mean())
    
    plt.subplot(n_rows, n_cols, i+1)
    
    sns.kdeplot(means_head, color='gray', fill=True, alpha=0.3, label='Head' if i==0 else "")
    sns.kdeplot(means_tail, color='blue', fill=True, alpha=0.3, label='Tail' if i==0 else "")
    sns.kdeplot(means_test, color='red', linestyle='--', linewidth=2, label='Test' if i==0 else "")
    
    plt.title(f"{col}", fontsize=10)
    plt.xlabel("")
    plt.yticks([])
    
    if i == 0: 
        plt.legend(loc='upper right')

plt.tight_layout()
plt.suptitle("Bootstrap Mean Distribution: Do they overlap with Test?", y=1.02, fontsize=16)
plt.show()

print("Variance Ratio Metrics (Sorted by Head Deviation):")
print("-" * 65)
print(f"{'Feature':<35} | {'Head Ratio':<12} | {'Tail Ratio':<12}")
print("-" * 65)
for _, row in variance_df.iterrows():
    print(f"{row['Feature']:<35} | {row['Head_Ratio']:<12.4f} | {row['Tail_Ratio']:<12.4f}")


from sklearn.model_selection import train_test_split

X_encoded = X_train_full.copy() 
y_full = train['diagnosed_diabetes'] 
ids = train['id'].values        

mask_tail = ids >= best_id

X_tail = X_encoded[mask_tail]
y_tail = y_full[mask_tail]

X_head_all = X_encoded[~mask_tail]
y_head_all = y_full[~mask_tail]

X_train_proxy, X_valid_head, y_train_proxy, y_valid_head = train_test_split(
    X_head_all, y_head_all, test_size=0.2, random_state=42, stratify=y_head_all
)

print(f"1. Training Proxy Model on {len(X_train_proxy)} Head samples...")
model_proxy = lgb.LGBMClassifier(
    objective='binary', metric='auc', n_estimators=100, 
    learning_rate=0.1, random_state=42, verbose=-1, n_jobs=-1
)
model_proxy.fit(X_train_proxy, y_train_proxy)

print("2. Generating Predictions...")
preds_tail = model_proxy.predict_proba(X_tail)[:, 1]
preds_head = model_proxy.predict_proba(X_valid_head)[:, 1]

print("3. Running Bootstrap AUC Stability Test (1000 iterations)...")

def bootstrap_auc_std(y_true, y_pred, n_iterations=1000, name="Set"):
    aucs = []
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n_size = len(y_true)
    
    for i in range(n_iterations):
        indices = np.random.choice(n_size, n_size, replace=True)
        score = roc_auc_score(y_true[indices], y_pred[indices])
        aucs.append(score)
    
    std = np.std(aucs)
    mean = np.mean(aucs)
    print(f"   >> {name:<15} | Sample Size: {n_size:<6} | Mean AUC: {mean:.4f} | AUC Std Dev (Noise): {std:.5f}")
    return aucs

# A. Tail (20k)
aucs_tail = bootstrap_auc_std(y_tail, preds_tail, name="Tail (Small)")

# B. Head Validation (Large)
aucs_head = bootstrap_auc_std(y_valid_head, preds_head, name="Head (Large)")

# C. Head Downsampled (20k) 
indices_20k = np.random.choice(len(y_valid_head), len(y_tail), replace=False)
aucs_head_small = bootstrap_auc_std(y_valid_head.iloc[indices_20k], preds_head[indices_20k], name="Head (Small)")

plt.figure(figsize=(10, 6))
sns.kdeplot(aucs_tail, fill=True, label='Tail (20k) - The Candidate', color='blue')
sns.kdeplot(aucs_head, fill=True, label='Head (120k) - The "Big Data"', color='gray')
sns.kdeplot(aucs_head_small, linestyle='--', label='Head (20k) - Control Group', color='black')

plt.title("Sampling Variance Check: AUC Stability Distribution")
plt.xlabel("AUC Score")
plt.yticks([])
plt.legend()
plt.grid(axis='x', alpha=0.3)
plt.show()


CUTOFF_ID = best_id

cat_cols = train.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))

train_ids = train['id'].values
cutoff_mask = train_ids >= CUTOFF_ID

X_tail = train.loc[cutoff_mask].drop(columns=['id', 'diagnosed_diabetes'])
y_adversarial_tail = pd.Series(1, index=X_tail.index)

X_head = train.loc[~cutoff_mask].drop(columns=['id', 'diagnosed_diabetes'])
y_adversarial_head = pd.Series(0, index=X_head.index)

X_adv_train = pd.concat([X_head, X_tail], axis=0)
y_adv_train = pd.concat([y_adversarial_head, y_adversarial_tail], axis=0)

print(f"1. Training Discriminator: Head({len(X_head)}) vs Tail({len(X_tail)})...")
params = {
    'objective': 'binary',
    'metric': 'auc',
    'n_estimators': 500, 
    'learning_rate': 0.05,
    'num_leaves': 31,
    'n_jobs': -1,
    'random_state': 42,
    'verbose': -1
}

model = lgb.LGBMClassifier(**params)
model.fit(X_adv_train, y_adv_train)

print("2. Evaluating Head samples...")
# åˆ†æ•°è¶Šæ�¥è¿‘ 1ï¼Œè¯´æ˜�è¯¥ Head æ ·æœ¬çš„ç‰¹å¾�åˆ†å¸ƒè¶Šåƒ� Tail
head_similarity_score = model.predict_proba(X_head)[:, 1]

head_true_labels = train.loc[~cutoff_mask, 'diagnosed_diabetes'].values

df_eval = pd.DataFrame({
    'tail_similarity': head_similarity_score,
    'true_label': head_true_labels
})

# æŒ‰ç›¸ä¼¼åº¦åˆ†æˆ� 10 ç»„ (Binning)
df_eval['bin'] = pd.qcut(df_eval['tail_similarity'], q=10, labels=False, duplicates='drop')

agg = df_eval.groupby('bin', observed=False).agg({
    'true_label': ['count', 'mean'],
    'tail_similarity': 'mean'
})
agg.columns = ['count', 'target_mean', 'similarity_mean']

# è®¡ç®—çœŸæ­£çš„ Tail æ ‡ç­¾å�‡å€¼
true_tail_mean = train.loc[cutoff_mask, 'diagnosed_diabetes'].mean()

print(agg)
print(f"\nTail Set True Mean: {true_tail_mean:.4f}")

# ç»˜å›¾
plt.figure(figsize=(12, 6))
bars = sns.barplot(x=agg.index, y=agg['target_mean'], color='skyblue', alpha=0.8)

# æ·»åŠ æ•°å€¼æ ‡ç­¾
for i, v in enumerate(agg['target_mean']):
    plt.text(i, v + 0.005, f"{v:.3f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

# ç”»Tail çš„çœŸå®�å�‡å€¼
plt.axhline(true_tail_mean, color='red', linestyle='--', linewidth=2, label=f'True Tail Mean ({true_tail_mean:.3f})')

plt.title(" Do 'Tail-like' Head samples have correct labels?", fontsize=14)
plt.xlabel("Similarity to Tail (0=Unlikely, 9=Indistinguishable from Tail)", fontsize=12)
plt.ylabel("Diabetes Rate (Target Mean)", fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()


print("1. Loading Data...")
train = pd.read_csv(TRAIN_PATH)
orig = pd.read_csv(ORIG_PATH)

orig = orig.dropna()

if 'Diabetes_binary' in orig.columns:
    orig = orig.rename(columns={'Diabetes_binary': 'diagnosed_diabetes'})

common_cols = [c for c in train.columns if c in orig.columns and c != 'id' and c != 'diagnosed_diabetes']
X_tail = train.loc[train['id'] >= CUTOFF_ID, common_cols]
y_tail = train.loc[train['id'] >= CUTOFF_ID, 'diagnosed_diabetes']

X_orig = orig[common_cols]
y_orig = orig['diagnosed_diabetes']

print(f"   Tail Shape: {X_tail.shape}, Orig Shape: {X_orig.shape}")

for col in X_tail.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    full_col = pd.concat([X_tail[col], X_orig[col]], axis=0).astype(str)
    le.fit(full_col)
    X_tail[col] = le.transform(X_tail[col].astype(str))
    X_orig[col] = le.transform(X_orig[col].astype(str))

X_adv = pd.concat([X_orig, X_tail], axis=0)
y_adv = np.concatenate([np.zeros(len(X_orig)), np.ones(len(X_tail))])

print("2. Training Discriminator: Orig vs Tail...")
params = {
    'objective': 'binary', 'metric': 'auc',
    'n_estimators': 500, 'learning_rate': 0.05, 'num_leaves': 31,
    'n_jobs': -1, 'random_state': 42, 'verbose': -1
}

model = lgb.LGBMClassifier(**params)
model.fit(X_adv, y_adv)

print("3. Evaluating Orig samples...")
orig_similarity_score = model.predict_proba(X_orig)[:, 1]

df_eval = pd.DataFrame({
    'tail_similarity': orig_similarity_score,
    'true_label': y_orig.values
})

df_eval['bin'] = pd.qcut(df_eval['tail_similarity'], q=10, labels=False, duplicates='drop')

agg = df_eval.groupby('bin', observed=False).agg({
    'true_label': ['count', 'mean'],
    'tail_similarity': 'mean'
})
agg.columns = ['count', 'target_mean', 'similarity_mean']

true_tail_mean = y_tail.mean()

print(agg)
print(f"\nTail Set True Mean: {true_tail_mean:.4f}")

plt.figure(figsize=(12, 6))
bars = sns.barplot(x=agg.index, y=agg['target_mean'], color='lightgreen', alpha=0.8)

for i, v in enumerate(agg['target_mean']):
    plt.text(i, v + 0.005, f"{v:.3f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.axhline(true_tail_mean, color='red', linestyle='--', linewidth=2, label=f'True Tail Mean ({true_tail_mean:.3f})')

plt.title("Is Original Data 'Honest'? (Orig Labels vs Tail Similarity)", fontsize=14)
plt.xlabel("Similarity to Tail (0=Distinct, 9=Look-alike)", fontsize=12)
plt.ylabel("Diabetes Rate (Target Mean)", fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()


from utils import load_data, TRAIN_PATH

features = [c for c in train.columns if c not in ['id', 'diagnosed_diabetes']]
target = 'diagnosed_diabetes'

print("ğŸ”§ Pre-processing: Encoding Categorical Features for Correlation Scan...")

for col in features:
    if train[col].dtype == 'object' or train[col].dtype.name == 'category':
        train[col] = train[col].fillna("MISSING").astype(str)
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col])

print(f"ğŸš€ Running Rolling Correlation Analysis (Window=20000)...")

WINDOW_SIZE = 20000 
STEP = 5000

rolling_corrs = {f: [] for f in features}
ids = []

train = train.sort_values('id').reset_index(drop=True)
max_id = train.shape[0]

for start in range(0, max_id - WINDOW_SIZE, STEP):
    end = start + WINDOW_SIZE
    subset = train.iloc[start:end]
    
    mean_id = subset['id'].mean()
    ids.append(mean_id)
    
    for f in features:
        if subset[f].nunique() > 1:
            corr = subset[[f, target]].corr().iloc[0, 1]
        else:
            corr = 0
        rolling_corrs[f].append(corr)

n_cols = 3
n_rows = (len(features) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
axes = axes.flatten()

for i, f in enumerate(features):
    ax = axes[i]
    y_vals = rolling_corrs[f]
    
    ax.plot(ids, y_vals, label='Rolling Corr', color='blue', alpha=0.7)
    tail_subset = train.iloc[-int(len(train)*0.1):]
    if tail_subset[f].nunique() > 1:
        tail_corr = tail_subset[[f, target]].corr().iloc[0, 1]
    else:
        tail_corr = 0
        
    ax.axhline(tail_corr, color='red', linestyle='--', label='Tail Corr')
    
    ax.axvline(678260, color='green', linestyle=':', label='Cutoff ID')
    
    ax.set_title(f"{f}")
    ax.set_ylim(-0.3, 0.3)
    ax.grid(True, alpha=0.3)
    
    if i == 0:
        ax.legend()

for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()


print("ğŸš€ Starting Comparative Scan: Tail vs Orig vs Head...")

train = pd.read_csv(TRAIN_PATH)
df_tail = train[train['id'] >= CUTOFF_ID].copy()
df_head = train[train['id'] < CUTOFF_ID].copy()

df_orig = pd.read_csv(ORIG_PATH)
if 'Diabetes_binary' in df_orig.columns:
    df_orig = df_orig.rename(columns={'Diabetes_binary': 'diagnosed_diabetes'})

common_cols = [c for c in df_tail.columns if c in df_orig.columns and c not in ['id', 'diagnosed_diabetes']]

print("ğŸ”§ Aligning Data Types...")
for col in common_cols:
    if pd.api.types.is_numeric_dtype(df_tail[col]):
        df_orig[col] = pd.to_numeric(df_orig[col], errors='coerce')
        df_head[col] = pd.to_numeric(df_head[col], errors='coerce')

    is_cat = False
    if df_tail[col].dtype == 'object' or df_orig[col].dtype == 'object':
        is_cat = True
    
    if is_cat:
        le = LabelEncoder()
        # å¡«å……ç¼ºå¤±å€¼è½¬å­—ç¬¦ä¸²
        s1 = df_tail[col].fillna("MISSING").astype(str)
        s2 = df_head[col].fillna("MISSING").astype(str)
        s3 = df_orig[col].fillna("MISSING").astype(str)
        
        full_s = pd.concat([s1, s2, s3])
        le.fit(full_s)
        
        df_tail[col] = le.transform(s1)
        df_head[col] = le.transform(s2)
        df_orig[col] = le.transform(s3)

def analyze_feature(df, feat, bins=20):
    """è®¡ç®— Log-Odds å’Œ R2"""
    try:
        is_numeric = (df[feat].nunique() > 15)
        
        if is_numeric:
            df['temp_bin'] = pd.qcut(df[feat], q=bins, duplicates='drop')
            agg = df.groupby('temp_bin', observed=True).agg({
                'diagnosed_diabetes': 'mean',
                feat: 'mean'
            }).rename(columns={feat: 'x_val'})
        else:
            agg = df.groupby(feat, observed=True).agg({
                'diagnosed_diabetes': 'mean'
            })
            agg['x_val'] = agg.index
            
        agg = agg.rename(columns={'diagnosed_diabetes': 'prob'})
        agg = agg[(agg['prob'] > 0.001) & (agg['prob'] < 0.999)]
        
        if len(agg) < 3: return None, 0
        
        agg['log_odds'] = np.log(agg['prob'] / (1 - agg['prob']))
        agg['x_val'] = agg['x_val'].astype(float) # ä¿®å¤�æŠ¥é”™çš„å…³é”®
        
        # è®¡ç®— R2
        r2 = np.corrcoef(agg['x_val'], agg['log_odds'])[0, 1]**2
        return agg[['x_val', 'log_odds']], r2
    except:
        return None, 0

# å¼€å§‹åˆ†æ��
results = []
print(f"\n{'Feature':<30} | {'Tail R2':<8} | {'Orig R2':<8} | {'Head R2':<8} | {'Status'}")
print("-" * 85)

for feat in common_cols:
    data_tail, r2_tail = analyze_feature(df_tail.copy(), feat)
    data_orig, r2_orig = analyze_feature(df_orig.copy(), feat)
    data_head, r2_head = analyze_feature(df_head.copy(), feat)
    
    # ç®€å�•çš„çŠ¶æ€�åˆ¤æ–­
    status = ""
    if r2_tail > 0.8 and r2_orig > 0.8: status = "âœ… Linear Match"
    elif abs(r2_tail - r2_orig) > 0.5:  status = "âš ï¸� Mismatch"
    else: status = "âšª Weak/Complex"
    
    print(f"{feat:<30} | {r2_tail:.4f}   | {r2_orig:.4f}   | {r2_head:.4f}   | {status}")
    
    results.append({
        'feature': feat,
        'tail': (data_tail, r2_tail),
        'orig': (data_orig, r2_orig),
        'head': (data_head, r2_head)
    })

# æ�’åº�ä»¥ä¾¿ç»˜å›¾ (æŒ‰ Tail R2)
results.sort(key=lambda x: x['tail'][1], reverse=True)

# ç»˜å›¾
n_features = len(results)
n_cols = 4
n_rows = math.ceil(n_features / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
axes = axes.flatten()

for i, item in enumerate(results):
    ax = axes[i]
    feat = item['feature']
    
    # 1. Tail (Red)
    dt, r2t = item['tail']
    if dt is not None:
        ax.scatter(dt['x_val'], dt['log_odds'], color='red', s=15, alpha=0.6, label=f'Tail (R2={r2t:.2f})')
        z = np.polyfit(dt['x_val'], dt['log_odds'], 1)
        ax.plot(dt['x_val'], np.poly1d(z)(dt['x_val']), 'r-', lw=1.5)

    # 2. Orig (Blue)
    do, r2o = item['orig']
    if do is not None:
        ax.scatter(do['x_val'], do['log_odds'], color='blue', s=15, alpha=0.3, label=f'Orig (R2={r2o:.2f})')
        try:
            z = np.polyfit(do['x_val'], do['log_odds'], 1)
            ax.plot(do['x_val'], np.poly1d(z)(do['x_val']), 'b--', lw=1.5)
        except: pass

    # 3. Head (Grey)
    dh, r2h = item['head']
    if dh is not None:
        # Head ç”»æ·¡ä¸€ç‚¹ï¼Œä½œä¸ºèƒŒæ™¯å�‚è€ƒ
        ax.scatter(dh['x_val'], dh['log_odds'], color='gray', s=10, alpha=0.1)
        try:
            z = np.polyfit(dh['x_val'], dh['log_odds'], 1)
            ax.plot(dh['x_val'], np.poly1d(z)(dh['x_val']), 'k:', lw=1)
        except: pass

    ax.set_title(f"{feat}", fontweight='bold')
    if i == 0: ax.legend(fontsize='small')

for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()




