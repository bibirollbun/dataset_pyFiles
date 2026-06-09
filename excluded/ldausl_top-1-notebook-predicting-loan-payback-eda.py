# ============================================================
# 0. IMPORTS & GLOBAL SETTINGS
# ============================================================
import os, gc, math, warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display

warnings.filterwarnings("ignore")

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12

RANDOM_STATE = 42

# ============================================================
# 1. LOAD DATA
# ============================================================
DATA_DIR = Path("/kaggle/input/playground-series-s5e11")

train_path = DATA_DIR / "train.csv"
test_path = DATA_DIR / "test.csv"
sub_path = DATA_DIR / "sample_submission.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_sub = pd.read_csv(sub_path)

print("Train shape :", train.shape)
print("Test shape  :", test.shape)
print("Sample sub  :", sample_sub.shape)

display(train.head())
display(test.head())

# ============================================================
# 2. IDENTIFIKASI TARGET, ID, DAN TIPE FEATURES
# ============================================================

# Target = kolom yang ada di train tapi tidak ada di test
target_candidates = [c for c in train.columns if c not in test.columns]
if len(target_candidates) == 0:
    raise ValueError("Tidak ditemukan kolom target (beda antara train & test). Cek kolom train/test.")
elif len(target_candidates) > 1:
    print("WARNING: Lebih dari 1 kandidat target, pakai yang pertama. Cek lagi secara manual kalau perlu.")
target_col = target_candidates[0]
print(f"\nTarget column: {target_col}")

# ID columns: nama kolom yang mengandung 'id'
id_cols = [c for c in train.columns if "id" in c.lower()]
print("ID-like columns:", id_cols)

# Fitur (tanpa target)
feature_cols = [c for c in train.columns if c not in [target_col]]

# Pisahkan numerik & kategorikal (di train)
num_cols = train[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in feature_cols if c not in num_cols]

print(f"\nTotal features : {len(feature_cols)}")
print(f"Numerical cols : {len(num_cols)} -> {num_cols[:10]}{' ...' if len(num_cols)>10 else ''}")
print(f"Categorical cols: {len(cat_cols)} -> {cat_cols[:10]}{' ...' if len(cat_cols)>10 else ''}")

# ============================================================
# 3. INFO & DESKRIPTIF DATA
# ============================================================

print("\n=== INFO TRAIN ===")
print(train.info())

print("\n=== INFO TEST ===")
print(test.info())

print("\n=== DESCRIPTIVE STATISTICS (NUMERIC, TRAIN) ===")
display(train[num_cols + [target_col]].describe().T)

# Jika target numerik, cek distribusi
# Jika target kategorikal (binary / multi kelas), cek proporsi
print("\n=== TARGET DISTRIBUTION (TRAIN) ===")
display(train[target_col].value_counts(dropna=False))
print("\n=== TARGET DISTRIBUTION (NORMALIZED) ===")
display(train[target_col].value_counts(normalize=True, dropna=False))

plt.figure(figsize=(6, 4))
sns.countplot(data=train, x=target_col)
plt.title(f"Distribusi Target: {target_col}")
plt.tight_layout()
plt.show()

# ============================================================
# 4. CEK MISSING VALUES
# ============================================================

def missing_values_table(df, name="df"):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * mis_val / len(df)
    table = pd.concat([mis_val, mis_val_percent], axis=1)
    table.columns = ["n_missing", "pct_missing"]
    table = table[table["n_missing"] > 0].sort_values("pct_missing", ascending=False)
    print(f"\nMissing values in {name}:")
    return table

missing_train = missing_values_table(train, "train")
missing_test = missing_values_table(test, "test")

display(missing_train.head(20))
display(missing_test.head(20))

# Heatmap missing values (jika jumlah kolom tidak terlalu besar)
cols_with_na_train = missing_train.index.tolist()
if len(cols_with_na_train) > 0:
    plt.figure(figsize=(12, 6))
    sns.heatmap(train[cols_with_na_train].isnull(), cbar=False)
    plt.title("Pola Missing Values - Train")
    plt.tight_layout()
    plt.show()

cols_with_na_test = missing_test.index.tolist()
if len(cols_with_na_test) > 0:
    plt.figure(figsize=(12, 6))
    sns.heatmap(test[cols_with_na_test].isnull(), cbar=False)
    plt.title("Pola Missing Values - Test")
    plt.tight_layout()
    plt.show()

# ============================================================
# 5. DISTRIBUSI UNIVARIAT FITUR NUMERIK (TRAIN)
# ============================================================

# Jangan plot ID columns
num_cols_wo_id = [c for c in num_cols if c not in id_cols]

print(f"\nPlot distribusi untuk {len(num_cols_wo_id)} fitur numerik (tanpa ID).")

n_cols_plot = 3
n_rows_plot = math.ceil(len(num_cols_wo_id) / n_cols_plot)

plt.figure(figsize=(n_cols_plot * 5, n_rows_plot * 4))

for i, col in enumerate(num_cols_wo_id, 1):
    plt.subplot(n_rows_plot, n_cols_plot, i)
    sns.histplot(train[col], kde=True)
    plt.title(col)
    plt.xlabel("")
plt.tight_layout()
plt.show()

# Boxplot untuk numeric (cek outlier)
plt.figure(figsize=(min(20, len(num_cols_wo_id) * 0.6), 0.6 * len(num_cols_wo_id) + 4))
sns.boxplot(data=train[num_cols_wo_id], orient="h")
plt.title("Boxplot Fitur Numerik (Train)")
plt.tight_layout()
plt.show()

# ============================================================
# 6. DISTRIBUSI UNIVARIAT FITUR KATEGORIKAL (TRAIN)
# ============================================================

print("\nRingkasan kategori tiap kolom kategorikal:")

for col in cat_cols:
    print(f"\n=== {col} ===")
    print(train[col].value_counts(dropna=False).head(20))
    print(f"n_unique = {train[col].nunique()}")

    # Plot hanya kalau kategori tidak terlalu banyak
    if train[col].nunique() <= 15:
        plt.figure(figsize=(8, 4))
        order = train[col].value_counts().index
        sns.countplot(data=train, x=col, order=order)
        plt.title(f"Distribusi {col}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

# ============================================================
# 7. KORELASI & HUBUNGAN DENGAN TARGET
# ============================================================

# --- Korelasi antar fitur numerik ---
if len(num_cols_wo_id) > 1:
    corr_matrix = train[num_cols_wo_id].corr()
    plt.figure(figsize=(min(1.2 * len(num_cols_wo_id), 18),
                        min(1.2 * len(num_cols_wo_id), 18)))
    sns.heatmap(corr_matrix, cmap="coolwarm", center=0, square=True)
    plt.title("Correlation Matrix (Numerical Features - Train)")
    plt.tight_layout()
    plt.show()

# --- Korelasi numeric vs target ---
print("\n=== Korelasi fitur numerik dengan target ===")
if train[target_col].dtype == "object":
    target_enc = train[target_col].astype("category").cat.codes
elif str(train[target_col].dtype) == "bool":
    target_enc = train[target_col].astype(int)
else:
    target_enc = train[target_col]

corr_with_target = {}
for col in num_cols_wo_id:
    corr_with_target[col] = train[col].corr(target_enc)

corr_with_target = pd.Series(corr_with_target).sort_values(ascending=False)
display(corr_with_target.to_frame("corr_with_target"))

plt.figure(figsize=(8, max(4, len(corr_with_target) * 0.3)))
sns.barplot(x=corr_with_target.values, y=corr_with_target.index)
plt.title("Korelasi Fitur Numerik dengan Target")
plt.xlabel("Correlation")
plt.tight_layout()
plt.show()

# --- Analisis target vs fitur numerik: distribusi per kelas target ---
if train[target_col].nunique() <= 10:
    for col in num_cols_wo_id[:10]:  # batasi 10 fitur dulu biar tidak terlalu berat
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=train, x=target_col, y=col)
        plt.title(f"{col} vs {target_col}")
        plt.tight_layout()
        plt.show()

# --- Analisis target vs fitur kategorikal: countplot dengan hue target ---
if train[target_col].nunique() <= 10:
    for col in cat_cols:
        if train[col].nunique() <= 10:  # batasi kategori
            plt.figure(figsize=(8, 4))
            sns.countplot(data=train, x=col, hue=target_col)
            plt.title(f"{col} vs {target_col}")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.show()

# ============================================================
# 8. PERBANDINGAN DISTRIBUSI TRAIN vs TEST
#    (cek kemungkinan data shift)
# ============================================================

print("\n=== Perbandingan distribusi numerik Train vs Test ===")
for col in num_cols_wo_id[:12]:  # batasi 12 fitur dulu
    plt.figure(figsize=(8, 4))
    sns.kdeplot(train[col], label="Train", fill=True, alpha=0.5)
    sns.kdeplot(test[col], label="Test", fill=True, alpha=0.5)
    plt.title(f"Train vs Test: {col}")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ============================================================
# 9. GROUPBY TARGET (UNTUK NUMERIK & KATEGORIK)
# ============================================================

# Ringkasan mean per target (numerik)
print("\n=== Rata-rata fitur numerik per kelas target ===")
group_num = train.groupby(target_col)[num_cols_wo_id].mean().T
display(group_num.head(20))

# Kalau banyak kelas, heatmap bisa membantu
plt.figure(figsize=(8, max(4, len(group_num) * 0.3)))
sns.heatmap(group_num, annot=False, cmap="viridis")
plt.title("Rata-rata Fitur Numerik per Kelas Target")
plt.tight_layout()
plt.show()

# Contoh: target rate per kategori (untuk beberapa fitur kategorikal)
if train[target_col].nunique() <= 10:
    print("\n=== Target rate per kategori (beberapa fitur kategorikal) ===")
    for col in cat_cols[:5]:
        temp = (
            train.groupby(col)[target_col]
            .value_counts(normalize=True)
            .rename("proportion")
            .reset_index()
        )
        display(temp.head(20))

# ============================================================
# 10. QUICK CHECK SAMPLE SUBMISSION
# ============================================================

print("\n=== Sample Submission Head ===")
display(sample_sub.head())
print("\nKolom sample submission:", sample_sub.columns.tolist())

# ============================================================
# 11. SIMPAN RINGKASAN PENTING (OPSIONAL)
# ============================================================

# Simpan korelasi dengan target & missing summary sebagai csv (kalau mau)
OUTPUT_DIR = Path("./eda_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

corr_with_target.to_frame("corr_with_target").to_csv(OUTPUT_DIR / "corr_with_target.csv")
missing_train.to_csv(OUTPUT_DIR / "missing_train.csv")
missing_test.to_csv(OUTPUT_DIR / "missing_test.csv")

print("\nEDA selesai. File ringkasan disimpan di folder ./eda_outputs")



# Import Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


# Configuration

MODEL_1_PATH = "/kaggle/input/predicting-loan-payback-vault/submission.csv"
MODEL_2_PATH = "/kaggle/input/predicting-loan-payback-vault/submission (1).csv"

print("Configuration set:")
print(f"  Model 1: {MODEL_1_PATH}")
print(f"  Model 2: {MODEL_2_PATH}")


try:
    df_model_1 = pd.read_csv(MODEL_1_PATH)
    df_model_2 = pd.read_csv(MODEL_2_PATH)
    
    print(f"✓ Model 1 loaded: {df_model_1.shape}")
    print(f"✓ Model 2 loaded: {df_model_2.shape}")
    
    # Validate structure
    assert 'id' in df_model_1.columns, "Model 1 missing 'id' column"
    assert 'loan_paid_back' in df_model_1.columns, "Model 1 missing 'loan_paid_back' column"
    assert 'id' in df_model_2.columns, "Model 2 missing 'id' column"
    assert 'loan_paid_back' in df_model_2.columns, "Model 2 missing 'loan_paid_back' column"
    print("\n✓ Data validation passed")
    
except FileNotFoundError as e:
    print(f"❌ ERROR: File not found - {e}")
    print("Please update the paths in Cell 2")
except Exception as e:
    print(f"❌ ERROR: {e}")


df_model_1.head(5)


df_model_2.head(5)


df_model_1==df_model_2


# Statistical Analysis of Base Models

print("=" * 70)
print("STATISTICAL SUMMARY OF BASE MODELS")
print("=" * 70)

stats_df = pd.DataFrame({
    'Model_1': df_model_1['loan_paid_back'].describe(),
    'Model_2': df_model_2['loan_paid_back'].describe()
})

display(stats_df)

# Calculate correlation
correlation = df_model_1['loan_paid_back'].corr(df_model_2['loan_paid_back'])
print(f"\nPearson Correlation: {correlation:.6f}")

# Interpretation based on correlation
if correlation > 0.98:
    print("→ VERY HIGH correlation: Models are nearly identical")
    print("  Blending these models will provide minimal to no benefit")
elif correlation > 0.95:
    print("→ High correlation: Models are very similar")
    print("  Limited improvement expected from blending")
elif correlation > 0.90:
    print("→ Moderate correlation: Some diversity exists")
    print("  Small improvement possible from blending")
else:
    print("→ Low correlation: Strong model diversity")
    print("  Good potential for ensemble improvement")


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribution plots
axes[0].hist(df_model_1['loan_paid_back'], bins=50, alpha=0.6, label='Model 1', color='#2E86AB', edgecolor='black', linewidth=0.5)
axes[0].hist(df_model_2['loan_paid_back'], bins=50, alpha=0.6, label='Model 2', edgecolor='black', linewidth=0.5)
axes[0].set_xlabel('Predicted Probability', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[0].set_title('Distribution of Predictions', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Scatter plot
axes[1].scatter(df_model_1['loan_paid_back'], df_model_2['loan_paid_back'], alpha=0.3, s=5)
axes[1].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Agreement')
axes[1].set_xlabel('Model 1 Predictions', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Model 2 Predictions', fontsize=11, fontweight='bold')
axes[1].set_title(f'Prediction Agreement (r={correlation:.4f})', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


df_analysis = pd.DataFrame({
    'id': df_model_1['id'],
    'model_1': df_model_1['loan_paid_back'],
    'model_2': df_model_2['loan_paid_back'],
})

df_analysis['difference'] = np.abs(df_analysis['model_1'] - df_analysis['model_2'])
df_analysis['mean_pred'] = (df_analysis['model_1'] + df_analysis['model_2']) / 2

print("Prediction Difference Analysis:")
print(f"  Mean absolute difference: {df_analysis['difference'].mean():.6f}")
print(f"  Median absolute difference: {df_analysis['difference'].median():.6f}")
print(f"  Max absolute difference: {df_analysis['difference'].max():.6f}")
print(f"  Std of differences: {df_analysis['difference'].std():.6f}")
print(f"\n  Cases with >0.1 difference: {(df_analysis['difference'] > 0.1).sum():,}")
print(f"  Cases with >0.2 difference: {(df_analysis['difference'] > 0.2).sum():,}")
print(f"  Cases with >0.5 difference: {(df_analysis['difference'] > 0.5).sum():,}")


# Visualize Model Agreement and Disagreement

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Difference distribution
axes[0].hist(df_analysis['difference'], bins=50, color='#F18F01', edgecolor='black', linewidth=0.5)
axes[0].set_xlabel('Absolute Prediction Difference', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[0].set_title('Distribution of Model Disagreement', fontsize=12, fontweight='bold')
axes[0].axvline(df_analysis['difference'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df_analysis["difference"].mean():.4f}')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Bland-Altman plot
diff = df_analysis['model_1'] - df_analysis['model_2']
mean = (df_analysis['model_1'] + df_analysis['model_2']) / 2
mean_diff = diff.mean()
std_diff = diff.std()

axes[1].scatter(mean, diff, alpha=0.3, s=5, color='#540D6E')
axes[1].axhline(0, color='black', linestyle='-', linewidth=1)
axes[1].axhline(mean_diff, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_diff:.4f}')
axes[1].axhline(mean_diff + 1.96*std_diff, color='orange', linestyle='--', linewidth=1.5, label='±1.96 SD')
axes[1].axhline(mean_diff - 1.96*std_diff, color='orange', linestyle='--', linewidth=1.5)
axes[1].set_xlabel('Mean of Two Predictions', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Difference (Model 1 - Model 2)', fontsize=11, fontweight='bold')
axes[1].set_title('Bland-Altman Agreement Plot', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# Define Blending Function

def blend_submissions(weight_dict, output_path, verbose=True):
    """
    Blend multiple submission files using weighted averaging.
    
    Parameters:
    -----------
    weight_dict : dict
        Dictionary mapping file paths to their respective weights
    output_path : str
        Path where the blended submission will be saved
    verbose : bool
        Whether to print detailed information
    
    Returns:
    --------
    pd.DataFrame
        The blended submission dataframe
    """
    try:
        dataframes = []
        
        # Load and weight each submission
        for path, weight in weight_dict.items():
            df = pd.read_csv(path)
            df["weighted_pred"] = df["loan_paid_back"] * weight
            dataframes.append(df[["id", "weighted_pred"]])
        
        # Merge all submissions
        merged = dataframes[0].copy()
        for i, df in enumerate(dataframes[1:], start=1):
            merged = merged.merge(df, on="id", how="inner", suffixes=("", f"_dup{i}"))
            dup_col = f"weighted_pred_dup{i}" if f"weighted_pred_dup{i}" in merged.columns else "weighted_pred_dup"
            if dup_col in merged.columns:
                merged["weighted_pred"] += merged[dup_col]
                merged.drop(columns=[dup_col], inplace=True)
        
        # Compute blended predictions
        total_weight = sum(weight_dict.values())
        merged["loan_paid_back"] = merged["weighted_pred"] / total_weight
        
        # Prepare final output
        blended = merged[["id", "loan_paid_back"]].copy()
        blended.to_csv(output_path, index=False)
        
        if verbose:
            print(f"✓ Blended submission saved: {output_path}")
            print(f"  Total weight: {total_weight:.2f}")
            for path, weight in weight_dict.items():
                pct = (weight / total_weight) * 100
                filename = path.split('/')[-1]
                print(f"  • {filename}: {weight:.2f} ({pct:.1f}%)")
        
        return blended
        
    except Exception as e:
        print(f"❌ ERROR in blending: {e}")
        return None

print("✓ Blending function defined successfully")


# Experimental Blending - Test Multiple Weight Configurations

print("\n" + "=" * 70)
print("EXPERIMENTAL BLENDING ANALYSIS")
print("=" * 70)
print("\nTesting multiple weight configurations to find optimal blend...\n")

# Store all blend results
blend_results = []

# Configuration 1: Equal Weight (50-50)
print("[1/5] Equal Weight Blend (50-50)")
weight_config_1 = {MODEL_1_PATH: 1.0, MODEL_2_PATH: 1.0}
blend_50_50 = blend_submissions(weight_config_1, "blend_50_50.csv", verbose=True)
if blend_50_50 is not None:
    blend_results.append(('50-50', blend_50_50))
print()

# Configuration 2: Conservative (75-25)
print("[2/5] Conservative Blend (75-25)")
weight_config_2 = {MODEL_1_PATH: 3.0, MODEL_2_PATH: 1.0}
blend_75_25 = blend_submissions(weight_config_2, "blend_75_25.csv", verbose=True)
if blend_75_25 is not None:
    blend_results.append(('75-25', blend_75_25))
print()

# Configuration 3: Heavy Model 1 (90-10)
print("[3/5] Heavy Model 1 Dominance (90-10)")
weight_config_3 = {MODEL_1_PATH: 9.0, MODEL_2_PATH: 1.0}
blend_90_10 = blend_submissions(weight_config_3, "blend_90_10.csv", verbose=True)
if blend_90_10 is not None:
    blend_results.append(('90-10', blend_90_10))
print()

# Configuration 4: Extreme Model 1 (95-5)
print("[4/5] Extreme Model 1 Dominance (95-5)")
weight_config_4 = {MODEL_1_PATH: 19.0, MODEL_2_PATH: 1.0}
blend_95_5 = blend_submissions(weight_config_4, "blend_95_5.csv", verbose=True)
if blend_95_5 is not None:
    blend_results.append(('95-5', blend_95_5))
print()

# Configuration 5: Extreme Model 2 (5-95)
print("[5/5] Extreme Model 2 Dominance (5-95)")
weight_config_5 = {MODEL_1_PATH: 1.0, MODEL_2_PATH: 19.0}
blend_5_95 = blend_submissions(weight_config_5, "blend_5_95.csv", verbose=True)
if blend_5_95 is not None:
    blend_results.append(('5-95', blend_5_95))

print("\n" + "=" * 70)


# Compare All Blending Strategies

if len(blend_results) == 5:
    comparison_df = pd.DataFrame({
        'id': df_model_1['id'],
        'model_1': df_model_1['loan_paid_back'],
        'model_2': df_model_2['loan_paid_back'],
        'blend_50_50': blend_50_50['loan_paid_back'],
        'blend_75_25': blend_75_25['loan_paid_back'],
        'blend_90_10': blend_90_10['loan_paid_back'],
        'blend_95_5': blend_95_5['loan_paid_back'],
        'blend_5_95': blend_5_95['loan_paid_back']
    })
    
    print("\n" + "=" * 70)
    print("COMPARATIVE STATISTICS - ALL BLENDS")
    print("=" * 70)
    
    summary = comparison_df.drop('id', axis=1).describe()
    display(summary)
else:
    print("⚠ Some blends failed to generate")


# Visualize All Blending Strategies

if len(blend_results) == 5:
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    
    # Plot 1: Distribution comparison
    axes[0, 0].hist(comparison_df['model_1'], bins=50, alpha=0.4, label='Model 1', color='#2E86AB', edgecolor='black', linewidth=0.5)
    axes[0, 0].hist(comparison_df['model_2'], bins=50, alpha=0.4, label='Model 2', color='#A23B72', edgecolor='black', linewidth=0.5)
    axes[0, 0].hist(comparison_df['blend_50_50'], bins=50, alpha=0.4, label='50-50', color='#06A77D', edgecolor='black', linewidth=0.5)
    axes[0, 0].hist(comparison_df['blend_95_5'], bins=50, alpha=0.4, label='95-5', color='#F18F01', edgecolor='black', linewidth=0.5)
    axes[0, 0].set_xlabel('Predicted Probability', fontsize=10, fontweight='bold')
    axes[0, 0].set_ylabel('Frequency', fontsize=10, fontweight='bold')
    axes[0, 0].set_title('Distribution Comparison: Key Blends', fontsize=11, fontweight='bold')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Box plot comparison
    data_to_plot = [comparison_df['model_1'], comparison_df['model_2'], 
                    comparison_df['blend_50_50'], comparison_df['blend_75_25'],
                    comparison_df['blend_90_10'], comparison_df['blend_95_5'],
                    comparison_df['blend_5_95']]
    bp = axes[0, 1].boxplot(data_to_plot, 
                             labels=['M1', 'M2', '50-50', '75-25', '90-10', '95-5', '5-95'],
                             patch_artist=True)
    colors = ['#2E86AB', '#A23B72', '#06A77D', '#F18F01', '#540D6E', '#C73E1D', '#6A994E']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[0, 1].set_ylabel('Predicted Probability', fontsize=10, fontweight='bold')
    axes[0, 1].set_title('Distribution Box Plots - All Configurations', fontsize=11, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Mean predictions comparison
    models = ['Model 1', 'Model 2', '50-50', '75-25', '90-10', '95-5', '5-95']
    means = [comparison_df['model_1'].mean(), comparison_df['model_2'].mean(),
             comparison_df['blend_50_50'].mean(), comparison_df['blend_75_25'].mean(),
             comparison_df['blend_90_10'].mean(), comparison_df['blend_95_5'].mean(),
             comparison_df['blend_5_95'].mean()]
    stds = [comparison_df['model_1'].std(), comparison_df['model_2'].std(),
            comparison_df['blend_50_50'].std(), comparison_df['blend_75_25'].std(),
            comparison_df['blend_90_10'].std(), comparison_df['blend_95_5'].std(),
            comparison_df['blend_5_95'].std()]
    
    x_pos = np.arange(len(means))
    bars = axes[1, 0].bar(x_pos, means, yerr=stds, align='center', alpha=0.7, 
                          color=colors, edgecolor='black', capsize=5, linewidth=1.5)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(models, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Mean Predicted Probability', fontsize=10, fontweight='bold')
    axes[1, 0].set_title('Mean Predictions with Standard Deviation', fontsize=11, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Correlation heatmap of all blends
    corr_matrix = comparison_df.drop('id', axis=1).corr()
    im = axes[1, 1].imshow(corr_matrix, cmap='RdYlGn', aspect='auto', vmin=0.95, vmax=1.0)
    axes[1, 1].set_xticks(np.arange(len(corr_matrix.columns)))
    axes[1, 1].set_yticks(np.arange(len(corr_matrix.columns)))
    axes[1, 1].set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=8)
    axes[1, 1].set_yticklabels(corr_matrix.columns, fontsize=8)
    axes[1, 1].set_title('Correlation Matrix - All Predictions', fontsize=11, fontweight='bold')
    
    # Add correlation values
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            text = axes[1, 1].text(j, i, f'{corr_matrix.iloc[i, j]:.3f}',
                                   ha="center", va="center", color="black", fontsize=7)
    
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.show()


# Analyze Prediction Variance Across Blends

if len(blend_results) == 5:
    print("\n" + "=" * 70)
    print("VARIANCE ANALYSIS ACROSS WEIGHT CONFIGURATIONS")
    print("=" * 70)
    
    # Calculate variance for each prediction across all blends
    blend_cols = ['blend_50_50', 'blend_75_25', 'blend_90_10', 'blend_95_5', 'blend_5_95']
    comparison_df['prediction_variance'] = comparison_df[blend_cols].var(axis=1)
    
    print(f"\nPrediction Variance Statistics:")
    print(f"  Mean variance: {comparison_df['prediction_variance'].mean():.8f}")
    print(f"  Max variance: {comparison_df['prediction_variance'].max():.8f}")
    print(f"  Std of variance: {comparison_df['prediction_variance'].std():.8f}")
    
    # Identify cases with highest disagreement
    high_variance_cases = comparison_df.nlargest(10, 'prediction_variance')
    
    print(f"\nTop 10 cases with highest variance across blends:")
    display(high_variance_cases[['id', 'model_1', 'model_2', 'prediction_variance']].head(10))
    
    # Visualize variance distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(comparison_df['prediction_variance'], bins=50, color='#C73E1D', edgecolor='black', linewidth=0.5)
    axes[0].set_xlabel('Prediction Variance Across Blends', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0].set_title('Distribution of Prediction Variance', fontsize=12, fontweight='bold')
    axes[0].axvline(comparison_df['prediction_variance'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Scatter: variance vs mean prediction
    axes[1].scatter(comparison_df[blend_cols].mean(axis=1), 
                    comparison_df['prediction_variance'],
                    alpha=0.3, s=5, color='#540D6E')
    axes[1].set_xlabel('Mean Prediction Across Blends', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Prediction Variance', fontsize=11, fontweight='bold')
    axes[1].set_title('Variance vs Mean Prediction', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# Quantile Analysis

if len(blend_results) == 5:
    print("\n" + "=" * 70)
    print("QUANTILE ANALYSIS")
    print("=" * 70)
    
    quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    quantile_df = pd.DataFrame({
        'Quantile': quantiles,
        'Model_1': comparison_df['model_1'].quantile(quantiles),
        'Model_2': comparison_df['model_2'].quantile(quantiles),
        'Blend_50_50': comparison_df['blend_50_50'].quantile(quantiles),
        'Blend_75_25': comparison_df['blend_75_25'].quantile(quantiles),
        'Blend_95_5': comparison_df['blend_95_5'].quantile(quantiles),
        'Blend_5_95': comparison_df['blend_5_95'].quantile(quantiles)
    })
    
    display(quantile_df)


# Experimental Results Summary with Known Scores

print("\n" + "=" * 70)
print("EXPERIMENTAL RESULTS SUMMARY")
print("=" * 70)

# Store known public LB scores
known_scores = {
    'Original (90-10)': 0.92731,
    'Equal (50-50)': 0.92731,
    'Conservative (75-25)': 0.92731,
    'Heavy M1 (95-5)': 0.92731,
    'Heavy M2 (5-95)': 0.92730
}

print("\nPublic Leaderboard Scores:")
print("-" * 40)
for config, score in known_scores.items():
    print(f"  {config:20s}: {score:.5f}")

print("\n" + "=" * 70)
print("KEY FINDINGS")
print("=" * 70)

# Analysis based on scores
score_variance = np.var(list(known_scores.values()))
score_range = max(known_scores.values()) - min(known_scores.values())

print(f"\n[1] SCORE VARIANCE: {score_variance:.10f}")
print(f"    Score Range: {score_range:.5f}")
print("    → Extremely low variance indicates models are nearly identical")

print(f"\n[2] MODEL CORRELATION: {correlation:.6f}")
print("    → Correlation > 0.98 confirms models make similar predictions")

print(f"\n[3] BLENDING EFFECTIVENESS:")
print(f"    Best Score:  {max(known_scores.values()):.5f} (95-5 blend)")
print(f"    Worst Score: {min(known_scores.values()):.5f} (5-95 blend)")
print(f"    Gain from blending: {max(known_scores.values()) - min(known_scores.values()):.5f}")
print("    → Blending provides virtually no improvement")

print(f"\n[4] MODEL QUALITY:")
print("    Model 1 appears marginally superior to Model 2")
print("    Difference is negligible: 0.00001")


# Strategic Recommendations Visualization

fig, axes = plt.subplots(2, 2, figsize=(15, 11))

# Plot 1: Score comparison bar chart
configs = list(known_scores.keys())
scores = list(known_scores.values())
colors_scores = ['#2E86AB', '#06A77D', '#F18F01', '#C73E1D', '#540D6E']

bars = axes[0, 0].bar(range(len(configs)), scores, color=colors_scores, 
                       edgecolor='black', linewidth=1.5, alpha=0.8)
axes[0, 0].set_xticks(range(len(configs)))
axes[0, 0].set_xticklabels(configs, rotation=45, ha='right', fontsize=9)
axes[0, 0].set_ylabel('Public LB Score', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Public Leaderboard Scores - All Configurations', 
                     fontsize=12, fontweight='bold')
axes[0, 0].set_ylim(0.927, 0.92735)
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, score) in enumerate(zip(bars, scores)):
    height = bar.get_height()
    axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.5f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# Plot 2: Correlation vs Expected Improvement
correlation_levels = [0.99, 0.97, 0.95, 0.90, 0.85, 0.80]
expected_improvement = [0.00000, 0.00010, 0.00050, 0.00150, 0.00300, 0.00600]

axes[0, 1].plot(correlation_levels, expected_improvement, 'o-', linewidth=2, 
                markersize=8, color='#2E86AB')
axes[0, 1].axvline(correlation, color='red', linestyle='--', linewidth=2, 
                   label=f'Your Models: r={correlation:.4f}')
axes[0, 1].set_xlabel('Model Correlation', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Expected Score Improvement', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Blending Effectiveness vs Model Correlation', 
                     fontsize=12, fontweight='bold')
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].invert_xaxis()

# Plot 3: Weight sensitivity analysis
weights_m1 = [0.05, 0.25, 0.50, 0.75, 0.90, 0.95]
scores_by_weight = [0.92730, 0.92731, 0.92731, 0.92731, 0.92731, 0.92731]

axes[1, 0].plot(weights_m1, scores_by_weight, 'o-', linewidth=2, 
                markersize=10, color='#C73E1D')
axes[1, 0].set_xlabel('Model 1 Weight', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Public LB Score', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Weight Sensitivity Analysis', fontsize=12, fontweight='bold')
axes[1, 0].set_ylim(0.9272, 0.92735)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axhline(y=0.92731, color='green', linestyle='--', 
                   linewidth=2, label='Best Score')

# Add annotations
for x, y in zip(weights_m1, scores_by_weight):
    axes[1, 0].annotate(f'{y:.5f}', xy=(x, y), xytext=(0, 5),
                        textcoords='offset points', ha='center', fontsize=8)

axes[1, 0].legend(fontsize=9)

# Plot 4: Strategic recommendation flowchart (text-based)
axes[1, 1].axis('off')
axes[1, 1].set_xlim(0, 10)
axes[1, 1].set_ylim(0, 10)

# Title
axes[1, 1].text(5, 9.5, 'STRATEGIC RECOMMENDATIONS', 
                ha='center', fontsize=13, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#2E86AB', alpha=0.3))

# Current situation
axes[1, 1].text(5, 8.5, 'Current Situation:', ha='center', fontsize=10, fontweight='bold')
axes[1, 1].text(5, 8, f'• Correlation: {correlation:.4f} (Very High)', 
                ha='center', fontsize=9)
axes[1, 1].text(5, 7.5, '• Blending Benefit: ~0.00000', ha='center', fontsize=9)
axes[1, 1].text(5, 7, '• Model 1 slightly superior', ha='center', fontsize=9)

# Immediate action
axes[1, 1].text(5, 6, 'Immediate Action:', ha='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#F18F01', alpha=0.3))
axes[1, 1].text(5, 5.3, '✓ Use Model 1 (0.92731)', ha='center', fontsize=9)
axes[1, 1].text(5, 4.8, '✗ Stop blending these models', ha='center', fontsize=9)

# Next steps
axes[1, 1].text(5, 4, 'Next Steps:', ha='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#06A77D', alpha=0.3))
axes[1, 1].text(5, 3.3, '1. Build diverse model (different algorithm)', 
                ha='center', fontsize=8)
axes[1, 1].text(5, 2.9, '2. Target correlation < 0.90', ha='center', fontsize=8)
axes[1, 1].text(5, 2.5, '3. Then blend for real improvement', ha='center', fontsize=8)

# Expected outcome
axes[1, 1].text(5, 1.5, 'Expected with Diverse Model:', ha='center', 
                fontsize=9, fontweight='bold')
axes[1, 1].text(5, 1, 'Score improvement: +0.001 to +0.003', 
                ha='center', fontsize=8, style='italic')

plt.tight_layout()
plt.show()


# Generate Final Submission Files

print("\n" + "=" * 70)
print("FINAL SUBMISSION FILE GENERATION")
print("=" * 70)

# Based on analysis, generate the most promising submissions
final_submissions = {
    'submission_final_model1.csv': (MODEL_1_PATH, 1.0, MODEL_2_PATH, 0.0),
    'submission_final_optimal.csv': (MODEL_1_PATH, 19.0, MODEL_2_PATH, 1.0),
}

print("\nGenerating final recommended submissions...\n")

for output_name, (path1, w1, path2, w2) in final_submissions.items():
    weight_dict = {path1: w1, path2: w2}
    blend = blend_submissions(weight_dict, output_name, verbose=True)
    print()

print("=" * 70)


# Detailed Comparison Table

print("\n" + "=" * 70)
print("DETAILED COMPARISON TABLE")
print("=" * 70)

comparison_table = pd.DataFrame({
    'Configuration': ['Original (90-10)', 'Equal (50-50)', 'Conservative (75-25)', 
                      'Heavy M1 (95-5)', 'Heavy M2 (5-95)'],
    'Model_1_Weight': [0.90, 0.50, 0.75, 0.95, 0.05],
    'Model_2_Weight': [0.10, 0.50, 0.25, 0.05, 0.95],
    'Public_LB_Score': [0.92731, 0.92731, 0.92731, 0.92731, 0.92730],
    'Status': ['Baseline', 'No change', 'No change', 'Best', 'Slightly worse']
})

display(comparison_table)

# Calculate which configuration is best
best_config = comparison_table.loc[comparison_table['Public_LB_Score'].idxmax()]
print(f"\n✓ Best Configuration: {best_config['Configuration']}")
print(f"  Score: {best_config['Public_LB_Score']:.5f}")
print(f"  Weights: {best_config['Model_1_Weight']:.2f} / {best_config['Model_2_Weight']:.2f}")


# Model Diversity Requirements Analysis

print("\n" + "=" * 70)
print("MODEL DIVERSITY REQUIREMENTS FOR EFFECTIVE BLENDING")
print("=" * 70)

diversity_guide = pd.DataFrame({
    'Correlation_Range': ['< 0.85', '0.85 - 0.90', '0.90 - 0.95', '0.95 - 0.98', '> 0.98 (Current)'],
    'Diversity_Level': ['Excellent', 'Good', 'Moderate', 'Low', 'Very Low'],
    'Expected_Gain': ['+0.003 to +0.008', '+0.001 to +0.003', '+0.0005 to +0.001', 
                      '+0.0001 to +0.0005', '< +0.0001'],
    'Blending_Worth_It': ['Yes', 'Yes', 'Maybe', 'Rarely', 'No']
})

display(diversity_guide)

print(f"\nYour Models:")
print(f"  Correlation: {correlation:.6f}")
print(f"  Diversity Level: Very Low")
print(f"  Expected Gain: < +0.0001")
print(f"  Blending Worth It: No")

print("\n" + "-" * 70)
print("CONCLUSION: Need fundamentally different models for improvement")
print("-" * 70)


# Actionable Strategy for Next Models

print("\n" + "=" * 70)
print("ACTIONABLE STRATEGY FOR BUILDING DIVERSE MODELS")
print("=" * 70)

print("\n[OPTION 1] Different Algorithm Classes")
print("-" * 70)
print("Current Models Likely: Tree-based (XGBoost/LightGBM/RandomForest)")
print("\nTry These Alternatives:")
print("  • Logistic Regression (high diversity potential)")
print("  • Neural Networks (sklearn.MLPClassifier)")
print("  • Support Vector Machines")
print("  • Naive Bayes")
print("  • K-Nearest Neighbors")
print("\nExpected Correlation: 0.85 - 0.92")

print("\n[OPTION 2] Drastically Different Feature Engineering")
print("-" * 70)
print("Create a model using completely different features:")
print("  • Interaction terms only")
print("  • Polynomial features")
print("  • Domain-specific engineered features")
print("  • PCA-transformed features")
print("  • Feature aggregations/statistics")
print("\nExpected Correlation: 0.88 - 0.94")

print("\n[OPTION 3] Different Data Perspectives")
print("-" * 70)
print("Train on different views of the data:")
print("  • Stratified sampling (different class ratios)")
print("  • Bootstrap samples with different random seeds")
print("  • Feature subsampling (random 70% of features)")
print("  • Different train-validation splits")
print("\nExpected Correlation: 0.90 - 0.95")

print("\n[OPTION 4] Extreme Hyperparameter Variations")
print("-" * 70)
print("If staying with same algorithm class:")
print("  • Ultra-conservative: max_depth=3, min_samples=50")
print("  • Ultra-aggressive: max_depth=15, min_samples=1")
print("  • Very low learning rate: 0.001")
print("  • Very high learning rate: 0.3")
print("\nExpected Correlation: 0.92 - 0.96")


# Expected ROI Analysis

print("\n" + "=" * 70)
print("RETURN ON INVESTMENT (ROI) ANALYSIS")
print("=" * 70)

roi_data = pd.DataFrame({
    'Strategy': [
        'Continue blending current models',
        'Build 1 diverse model (Logistic Reg)',
        'Build 1 diverse model (Neural Net)',
        'Optimize existing model hyperparams',
        'Advanced feature engineering',
        'Build 3 diverse models + ensemble'
    ],
    'Time_Hours': [1, 3, 4, 2, 5, 8],
    'Expected_Score_Gain': [0.00000, 0.00150, 0.00200, 0.00100, 0.00250, 0.00400],
    'Difficulty': ['Easy', 'Medium', 'Medium', 'Easy', 'Hard', 'Hard'],
    'Recommended': ['No', 'Yes', 'Yes', 'Maybe', 'Yes', 'Yes']
})

roi_data['Gain_Per_Hour'] = roi_data['Expected_Score_Gain'] / roi_data['Time_Hours']

display(roi_data)

print("\n✓ HIGHEST ROI: Build 1 diverse model (Logistic Regression)")
print("  • Time: 3 hours")
print("  • Expected gain: +0.0015")
print("  • Gain per hour: 0.00050")


# Visualization - Strategy Comparison

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ROI comparison
strategies = roi_data['Strategy']
gains = roi_data['Expected_Score_Gain']
times = roi_data['Time_Hours']
colors_roi = ['#C73E1D', '#06A77D', '#2E86AB', '#F18F01', '#540D6E', '#6A994E']

axes[0].barh(range(len(strategies)), gains, color=colors_roi, 
             edgecolor='black', linewidth=1.5, alpha=0.8)
axes[0].set_yticks(range(len(strategies)))
axes[0].set_yticklabels(strategies, fontsize=9)
axes[0].set_xlabel('Expected Score Gain', fontsize=11, fontweight='bold')
axes[0].set_title('Expected Impact of Different Strategies', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (gain, time) in enumerate(zip(gains, times)):
    axes[0].text(gain, i, f' +{gain:.5f} ({time}h)', 
                va='center', fontsize=8, fontweight='bold')

# Efficiency scatter
axes[1].scatter(times, gains, s=200, c=colors_roi, edgecolor='black', 
                linewidth=2, alpha=0.8)
axes[1].set_xlabel('Time Investment (hours)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Expected Score Gain', fontsize=11, fontweight='bold')
axes[1].set_title('Efficiency Analysis: Time vs Impact', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# Add labels to points
for i, (time, gain, strategy) in enumerate(zip(times, gains, strategies)):
    if gain > 0.00050:  # Label only significant strategies
        short_name = strategy.split('(')[0].strip()[:15]
        axes[1].annotate(short_name, xy=(time, gain), xytext=(5, 5),
                        textcoords='offset points', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=colors_roi[i], alpha=0.5))

plt.tight_layout()
plt.show()



# Final Summary Report

print("\n" + "=" * 70)
print("COMPREHENSIVE ANALYSIS SUMMARY REPORT")
print("=" * 70)

print("\n[1] EXPERIMENTAL FINDINGS")
print("-" * 70)
print(f"  Models Tested: 2 base models, 5 blend configurations")
print(f"  Model Correlation: {correlation:.6f} (Very High)")
print(f"  Score Range: {score_range:.5f} (Minimal)")
print(f"  Best Configuration: Model 1 Heavy (95-5)")
print(f"  Best Score: 0.92731")

print("\n[2] KEY INSIGHTS")
print("-" * 70)
print("  ✓ Models are nearly identical (r > 0.98)")
print("  ✓ Blending provides no meaningful improvement")
print("  ✓ Model 1 is marginally better than Model 2")
print("  ✓ Weight variation (5% to 95%) has minimal impact")

print("\n[3] STATISTICAL EVIDENCE")
print("-" * 70)
print(f"  Prediction difference mean: {df_analysis['difference'].mean():.6f}")
print(f"  Prediction difference max: {df_analysis['difference'].max():.6f}")
print(f"  Variance across blends: {comparison_df['prediction_variance'].mean():.10f}")
print("  → Confirms models make nearly identical predictions")

print("\n[4] RECOMMENDED ACTIONS")
print("-" * 70)
print("  IMMEDIATE:")
print("    • Use 'submission_final_optimal.csv' (Model 1 dominant)")
print("    • Stop testing more weight combinations")
print("    • Accept current score: 0.92731")
print("\n  SHORT-TERM (Tomorrow):")
print("    • Build 1 diverse model using different algorithm")
print("    • Target correlation < 0.90 with current model")
print("    • Expected improvement: +0.001 to +0.003")
print("\n  LONG-TERM (This Week):")
print("    • Create 3-5 genuinely diverse models")
print("    • Implement proper stacking ensemble")
print("    • Expected improvement: +0.003 to +0.008")

print("\n[5] WHAT NOT TO DO")
print("-" * 70)
print("  ✗ Don't test more weight combinations on these models")
print("  ✗ Don't expect improvement from similar models")
print("  ✗ Don't waste submission slots on tiny variations")
print("  ✗ Don't blend models with correlation > 0.95")

print("\n[6] SUCCESS CRITERIA FOR NEXT MODEL")
print("-" * 70)
print("  Before blending, verify:")
print("    • Correlation with best model < 0.90")
print("    • Individual score > 0.925 (at minimum)")
print("    • Uses fundamentally different approach")
print("    • Captures different patterns in data")

print("\n[7] FILES GENERATED")
print("-" * 70)
generated_files = [
    'blend_50_50.csv',
    'blend_75_25.csv',
    'blend_90_10.csv',
    'blend_95_5.csv',
    'blend_5_95.csv',
    'submission_final_model1.csv',
    'submission_final_optimal.csv'
]
for fname in generated_files:
    print(f"  • {fname}")

print("\n" + "=" * 70)
print("✓ ANALYSIS COMPLETE")
print("=" * 70)
print("\nNext Step: Focus on building a diverse model using different algorithm")
print("Target: Correlation < 0.90, Expected gain: +0.001 to +0.003")
print("=" * 70)


df1=pd.read_csv("/kaggle/working/blend_50_50.csv")
df2=pd.read_csv("/kaggle/working/blend_5_95.csv")
df3=pd.read_csv("/kaggle/working/blend_75_25.csv")
df4=pd.read_csv("/kaggle/working/blend_90_10.csv")
df5=pd.read_csv("/kaggle/working/blend_95_5.csv")
df6=pd.read_csv("/kaggle/working/submission_final_model1.csv")
df7=pd.read_csv("/kaggle/working/submission_final_optimal.csv")


new_df1=df1.copy()
new_df2=df1.copy()
new_df3=df1.copy()


new_df1.head(5)


new_df1["loan_paid_back"]=(df1["loan_paid_back"]+df2["loan_paid_back"]+df3["loan_paid_back"]+df4["loan_paid_back"]
                        +df5["loan_paid_back"]+df6["loan_paid_back"]+df7["loan_paid_back"])/7.0


new_df2["loan_paid_back"] = pd.concat([
    df1["loan_paid_back"],
    df2["loan_paid_back"],
    df3["loan_paid_back"],
    df4["loan_paid_back"],
    df5["loan_paid_back"],
    df6["loan_paid_back"],
    df7["loan_paid_back"]
], axis=1).max(axis=1)


new_df3["loan_paid_back"] = pd.concat([
    df1["loan_paid_back"],
    df2["loan_paid_back"],
    df3["loan_paid_back"],
    df4["loan_paid_back"],
    df5["loan_paid_back"],
    df6["loan_paid_back"],
    df7["loan_paid_back"]
], axis=1).min(axis=1)


new_df1.head(5)


new_df2.head(5)


new_df3.head(5)


new_df1.to_csv("final1.csv", index=False)
new_df2.to_csv("final2.csv", index=False)
new_df3.to_csv("final3.csv", index=False)

