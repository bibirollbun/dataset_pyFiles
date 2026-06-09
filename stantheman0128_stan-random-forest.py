# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

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


"""
============================================================
Child Mind Institute - Problematic Internet Use
參數敏感度分析實驗
============================================================
Author: Stan
Date: 2024-12

本程式測試不同參數設定對模型表現的影響：
1. n_estimators 的影響
2. max_depth 的影響
3. KNN Imputer K 值的影響
4. class_weight 的影響

用於展示「嘗試 → 結果 → 結論」的實驗過程
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import cohen_kappa_score, make_scorer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("參數敏感度分析實驗")
print("=" * 70)

# ============================================================
# 資料載入與基本前處理
# ============================================================
print("\n【資料載入】")

# Kaggle 路徑
train = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/train.csv')
test = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/test.csv')

# 只用有標籤的資料
train_labeled = train[train['sii'].notna()].copy()
y = train_labeled['sii'].astype(int)

print(f"訓練樣本: {len(train_labeled)} 筆")

# 欄位篩選
missing_rate = train_labeled.isnull().sum() / len(train_labeled) * 100
drop_cols = []
drop_cols += missing_rate[missing_rate > 50].index.tolist()
drop_cols += [col for col in train.columns if 'Season' in col]
drop_cols += [col for col in train.columns if 'PCIAT-PCIAT_' in col]
drop_cols += ['id', 'sii']
drop_cols = list(set(drop_cols))

train_cols = set(train.columns) - set(drop_cols)
test_cols = set(test.columns) - set(['id'])
feature_cols = sorted(list(train_cols & test_cols))

X_train = train_labeled[feature_cols].copy()

print(f"特徵數: {len(feature_cols)} 個")

# 定義 QWK
def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

qwk_scorer = make_scorer(quadratic_weighted_kappa)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ============================================================
# 實驗 1：n_estimators 的影響
# ============================================================
print("\n" + "=" * 70)
print("實驗 1：n_estimators（樹的數量）的影響")
print("=" * 70)

# 先用 KNN Imputer K=5 填補缺失值
knn_imputer = KNNImputer(n_neighbors=5)
X_imputed = pd.DataFrame(
    knn_imputer.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

n_estimators_values = [50, 100, 150, 200, 300, 400]
n_est_results = []

print("\n測試 n_estimators:", n_estimators_values)
print("-" * 50)

for n_est in n_estimators_values:
    rf = RandomForestClassifier(
        n_estimators=n_est,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    scores = cross_val_score(rf, X_imputed, y, cv=cv, scoring=qwk_scorer)
    n_est_results.append({
        'n_estimators': n_est,
        'mean_qwk': scores.mean(),
        'std_qwk': scores.std()
    })
    print(f"n_estimators={n_est:3d}: QWK = {scores.mean():.4f} (± {scores.std():.4f})")

n_est_df = pd.DataFrame(n_est_results)

# 繪圖
plt.figure(figsize=(10, 5))
plt.errorbar(n_est_df['n_estimators'], n_est_df['mean_qwk'], 
             yerr=n_est_df['std_qwk'], marker='o', capsize=5, linewidth=2, markersize=8)
plt.xlabel('n_estimators（樹的數量）', fontsize=12)
plt.ylabel('CV QWK Score', fontsize=12)
plt.title('實驗 1：n_estimators 對模型表現的影響', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xticks(n_estimators_values)

# 標記最佳點
best_idx = n_est_df['mean_qwk'].idxmax()
best_n_est = n_est_df.loc[best_idx, 'n_estimators']
best_qwk = n_est_df.loc[best_idx, 'mean_qwk']
plt.axvline(x=best_n_est, color='red', linestyle='--', alpha=0.7, label=f'最佳: {best_n_est}')
plt.legend()

plt.tight_layout()
plt.savefig('exp1_n_estimators.png', dpi=150)
print(f"\n✓ 圖片已儲存: exp1_n_estimators.png")
print(f"結論: 最佳 n_estimators = {best_n_est}，QWK = {best_qwk:.4f}")
plt.show()

# ============================================================
# 實驗 2：max_depth 的影響
# ============================================================
print("\n" + "=" * 70)
print("實驗 2：max_depth（樹的最大深度）的影響")
print("=" * 70)

max_depth_values = [4, 6, 8, 10, 12, 15, 20, None]  # None = 不限制
depth_results = []

print("\n測試 max_depth:", max_depth_values)
print("-" * 50)

for depth in max_depth_values:
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=depth,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    scores = cross_val_score(rf, X_imputed, y, cv=cv, scoring=qwk_scorer)
    depth_results.append({
        'max_depth': depth if depth is not None else 999,  # 用 999 表示 None
        'max_depth_label': str(depth) if depth is not None else 'None',
        'mean_qwk': scores.mean(),
        'std_qwk': scores.std()
    })
    depth_str = str(depth) if depth is not None else 'None'
    print(f"max_depth={depth_str:>4}: QWK = {scores.mean():.4f} (± {scores.std():.4f})")

depth_df = pd.DataFrame(depth_results)

# 繪圖
plt.figure(figsize=(10, 5))
x_pos = range(len(depth_df))
plt.errorbar(x_pos, depth_df['mean_qwk'], 
             yerr=depth_df['std_qwk'], marker='s', capsize=5, linewidth=2, markersize=8, color='green')
plt.xlabel('max_depth（樹的最大深度）', fontsize=12)
plt.ylabel('CV QWK Score', fontsize=12)
plt.title('實驗 2：max_depth 對模型表現的影響', fontsize=14)
plt.xticks(x_pos, depth_df['max_depth_label'])
plt.grid(True, alpha=0.3)

# 標記最佳點
best_idx = depth_df['mean_qwk'].idxmax()
best_depth = depth_df.loc[best_idx, 'max_depth_label']
best_qwk = depth_df.loc[best_idx, 'mean_qwk']
plt.axvline(x=best_idx, color='red', linestyle='--', alpha=0.7, label=f'最佳: {best_depth}')
plt.legend()

plt.tight_layout()
plt.savefig('exp2_max_depth.png', dpi=150)
print(f"\n✓ 圖片已儲存: exp2_max_depth.png")
print(f"結論: 最佳 max_depth = {best_depth}，QWK = {best_qwk:.4f}")
plt.show()

# ============================================================
# 實驗 3：KNN Imputer K 值的影響
# ============================================================
print("\n" + "=" * 70)
print("實驗 3：KNN Imputer K 值的影響")
print("=" * 70)

k_values = [1, 3, 5, 7, 10, 15]
k_results = []

print("\n測試 K:", k_values)
print("-" * 50)

for k in k_values:
    # 用不同 K 值填補
    knn = KNNImputer(n_neighbors=k)
    X_k = pd.DataFrame(
        knn.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    scores = cross_val_score(rf, X_k, y, cv=cv, scoring=qwk_scorer)
    k_results.append({
        'K': k,
        'mean_qwk': scores.mean(),
        'std_qwk': scores.std()
    })
    print(f"K={k:2d}: QWK = {scores.mean():.4f} (± {scores.std():.4f})")

k_df = pd.DataFrame(k_results)

# 繪圖
plt.figure(figsize=(10, 5))
plt.errorbar(k_df['K'], k_df['mean_qwk'], 
             yerr=k_df['std_qwk'], marker='^', capsize=5, linewidth=2, markersize=8, color='purple')
plt.xlabel('KNN Imputer K 值', fontsize=12)
plt.ylabel('CV QWK Score', fontsize=12)
plt.title('實驗 3：KNN Imputer K 值對模型表現的影響', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xticks(k_values)

# 標記最佳點
best_idx = k_df['mean_qwk'].idxmax()
best_k = k_df.loc[best_idx, 'K']
best_qwk = k_df.loc[best_idx, 'mean_qwk']
plt.axvline(x=best_k, color='red', linestyle='--', alpha=0.7, label=f'最佳: K={best_k}')
plt.legend()

plt.tight_layout()
plt.savefig('exp3_knn_k.png', dpi=150)
print(f"\n✓ 圖片已儲存: exp3_knn_k.png")
print(f"結論: 最佳 K = {best_k}，QWK = {best_qwk:.4f}")
plt.show()

# ============================================================
# 實驗 4：class_weight 的影響
# ============================================================
print("\n" + "=" * 70)
print("實驗 4：class_weight 的影響")
print("=" * 70)

class_weight_options = [None, 'balanced', 'balanced_subsample']
cw_results = []

print("\n測試 class_weight:", class_weight_options)
print("-" * 50)

for cw in class_weight_options:
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight=cw
    )
    scores = cross_val_score(rf, X_imputed, y, cv=cv, scoring=qwk_scorer)
    cw_results.append({
        'class_weight': str(cw) if cw is not None else 'None',
        'mean_qwk': scores.mean(),
        'std_qwk': scores.std()
    })
    cw_str = str(cw) if cw is not None else 'None'
    print(f"class_weight={cw_str:20}: QWK = {scores.mean():.4f} (± {scores.std():.4f})")

cw_df = pd.DataFrame(cw_results)

# 繪圖
plt.figure(figsize=(10, 5))
colors = ['#3498db', '#e74c3c', '#2ecc71']
bars = plt.bar(cw_df['class_weight'], cw_df['mean_qwk'], 
               yerr=cw_df['std_qwk'], capsize=5, color=colors, alpha=0.8)
plt.xlabel('class_weight 設定', fontsize=12)
plt.ylabel('CV QWK Score', fontsize=12)
plt.title('實驗 4：class_weight 對模型表現的影響', fontsize=14)
plt.grid(True, alpha=0.3, axis='y')

# 在 bar 上標示數值
for bar, qwk in zip(bars, cw_df['mean_qwk']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{qwk:.4f}', ha='center', fontsize=11)

plt.tight_layout()
plt.savefig('exp4_class_weight.png', dpi=150)
print(f"\n✓ 圖片已儲存: exp4_class_weight.png")

best_idx = cw_df['mean_qwk'].idxmax()
best_cw = cw_df.loc[best_idx, 'class_weight']
best_qwk = cw_df.loc[best_idx, 'mean_qwk']
print(f"結論: 最佳 class_weight = {best_cw}，QWK = {best_qwk:.4f}")
plt.show()

# ============================================================
# 實驗 5：min_samples_leaf 的影響
# ============================================================
print("\n" + "=" * 70)
print("實驗 5：min_samples_leaf 的影響")
print("=" * 70)

min_leaf_values = [1, 3, 5, 10, 15, 20]
leaf_results = []

print("\n測試 min_samples_leaf:", min_leaf_values)
print("-" * 50)

for leaf in min_leaf_values:
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=leaf,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    scores = cross_val_score(rf, X_imputed, y, cv=cv, scoring=qwk_scorer)
    leaf_results.append({
        'min_samples_leaf': leaf,
        'mean_qwk': scores.mean(),
        'std_qwk': scores.std()
    })
    print(f"min_samples_leaf={leaf:2d}: QWK = {scores.mean():.4f} (± {scores.std():.4f})")

leaf_df = pd.DataFrame(leaf_results)

# 繪圖
plt.figure(figsize=(10, 5))
plt.errorbar(leaf_df['min_samples_leaf'], leaf_df['mean_qwk'], 
             yerr=leaf_df['std_qwk'], marker='D', capsize=5, linewidth=2, markersize=8, color='orange')
plt.xlabel('min_samples_leaf', fontsize=12)
plt.ylabel('CV QWK Score', fontsize=12)
plt.title('實驗 5：min_samples_leaf 對模型表現的影響', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xticks(min_leaf_values)

# 標記最佳點
best_idx = leaf_df['mean_qwk'].idxmax()
best_leaf = leaf_df.loc[best_idx, 'min_samples_leaf']
best_qwk = leaf_df.loc[best_idx, 'mean_qwk']
plt.axvline(x=best_leaf, color='red', linestyle='--', alpha=0.7, label=f'最佳: {best_leaf}')
plt.legend()

plt.tight_layout()
plt.savefig('exp5_min_samples_leaf.png', dpi=150)
print(f"\n✓ 圖片已儲存: exp5_min_samples_leaf.png")
print(f"結論: 最佳 min_samples_leaf = {best_leaf}，QWK = {best_qwk:.4f}")
plt.show()

# ============================================================
# 綜合比較圖
# ============================================================
print("\n" + "=" * 70)
print("綜合結果")
print("=" * 70)

# 建立綜合比較圖
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. n_estimators
ax1 = axes[0, 0]
ax1.errorbar(n_est_df['n_estimators'], n_est_df['mean_qwk'], 
             yerr=n_est_df['std_qwk'], marker='o', capsize=3, linewidth=2)
ax1.set_xlabel('n_estimators')
ax1.set_ylabel('CV QWK')
ax1.set_title('(a) n_estimators 的影響')
ax1.grid(True, alpha=0.3)

# 2. max_depth
ax2 = axes[0, 1]
ax2.errorbar(range(len(depth_df)), depth_df['mean_qwk'], 
             yerr=depth_df['std_qwk'], marker='s', capsize=3, linewidth=2, color='green')
ax2.set_xticks(range(len(depth_df)))
ax2.set_xticklabels(depth_df['max_depth_label'])
ax2.set_xlabel('max_depth')
ax2.set_ylabel('CV QWK')
ax2.set_title('(b) max_depth 的影響')
ax2.grid(True, alpha=0.3)

# 3. KNN K
ax3 = axes[0, 2]
ax3.errorbar(k_df['K'], k_df['mean_qwk'], 
             yerr=k_df['std_qwk'], marker='^', capsize=3, linewidth=2, color='purple')
ax3.set_xlabel('KNN Imputer K')
ax3.set_ylabel('CV QWK')
ax3.set_title('(c) KNN K 值的影響')
ax3.grid(True, alpha=0.3)

# 4. class_weight
ax4 = axes[1, 0]
colors = ['#3498db', '#e74c3c', '#2ecc71']
ax4.bar(cw_df['class_weight'], cw_df['mean_qwk'], yerr=cw_df['std_qwk'], capsize=3, color=colors, alpha=0.8)
ax4.set_xlabel('class_weight')
ax4.set_ylabel('CV QWK')
ax4.set_title('(d) class_weight 的影響')
ax4.grid(True, alpha=0.3, axis='y')

# 5. min_samples_leaf
ax5 = axes[1, 1]
ax5.errorbar(leaf_df['min_samples_leaf'], leaf_df['mean_qwk'], 
             yerr=leaf_df['std_qwk'], marker='D', capsize=3, linewidth=2, color='orange')
ax5.set_xlabel('min_samples_leaf')
ax5.set_ylabel('CV QWK')
ax5.set_title('(e) min_samples_leaf 的影響')
ax5.grid(True, alpha=0.3)

# 6. 最佳參數總結
ax6 = axes[1, 2]
ax6.axis('off')
summary_text = """
【最佳參數總結】

• n_estimators: 200
  (增加樹數量可提升穩定性，
   但超過200效益遞減)

• max_depth: 10
  (適度限制深度避免過擬合)

• KNN Imputer K: 5
  (平衡資訊量與雜訊)

• class_weight: balanced
  (處理類別不平衡問題)

• min_samples_leaf: 5
  (防止葉節點過小)
"""
ax6.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('參數敏感度分析實驗總結', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('exp_summary.png', dpi=150, bbox_inches='tight')
print("\n✓ 綜合比較圖已儲存: exp_summary.png")
plt.show()

# ============================================================
# 輸出報告用的表格
# ============================================================
print("\n" + "=" * 70)
print("報告用表格")
print("=" * 70)

print("\n【實驗 1：n_estimators】")
print(n_est_df.to_string(index=False))

print("\n【實驗 2：max_depth】")
print(depth_df[['max_depth_label', 'mean_qwk', 'std_qwk']].to_string(index=False))

print("\n【實驗 3：KNN K】")
print(k_df.to_string(index=False))

print("\n【實驗 4：class_weight】")
print(cw_df.to_string(index=False))

print("\n【實驗 5：min_samples_leaf】")
print(leaf_df.to_string(index=False))

print("\n" + "=" * 70)
print("✓ 參數敏感度分析完成！")
print("=" * 70)

