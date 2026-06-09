!pip install RDkit


import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from rdkit import Chem
from rdkit.Chem import Draw
from scipy.stats import pearsonr
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore", message="use_inf_as_na option is deprecated")


train=pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")
test=pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")
dataset1=pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset1.csv")
dataset2=pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset2.csv")
dataset3=pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset3.csv")
dataset4=pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset4.csv")
dataset1 = dataset1.rename(columns={'TC_mean': 'Tc'})
train = pd.concat([train, dataset1[['SMILES', 'Tc']]], axis=0, ignore_index=True)
train = pd.concat([train, dataset3[['SMILES', 'Tg']]], axis=0, ignore_index=True)
train = pd.concat([train, dataset4[['SMILES', 'FFV']]], axis=0, ignore_index=True)


target=['Tg','Density','Rg','Tc','FFV']



train.shape



null_ratio=train[target].isnull().mean()*100
null_ratio=null_ratio[null_ratio>0].sort_values(ascending=False)
null_ratio


plt.figure(figsize=(12,6))
sns.barplot(x=null_ratio.index,y=null_ratio.values,palette='viridis')
plt.xticks(rotation=45,ha='right')
plt.ylabel('Null Ratio(%)')
plt.tight_layout()
plt.show()


null_matrix=train[target].isnull()
plt.figure(figsize=(10,12))
sns.heatmap(
    null_matrix,cmap=['white','black'],
    cbar=False,
    xticklabels=True,
    yticklabels=False
    )
plt.xlabel('Target Columns')
plt.ylabel('Sample Index')
plt.title('Null Value Heatmap')
plt.show()



for _target in target:
    plt.figure(figsize=(8,6))
    plt.hist(train[_target].dropna(), bins=30, edgecolor='black')
    plt.title(f'Histogram of {_target}')
    plt.xlabel(_target)
    plt.ylabel('Frequency')
    plt.show()



for _target in target:
    plt.figure(figsize=(6, 4))
    sns.boxplot(y=train[_target])
    plt.title(f'Box plot of {_target}')
    plt.show()


for _target in target:
    print(f"\nğŸ”� Visualizing molecules for target: {_target}")
    
    # Drop rows with missing values in target or SMILES
    valid_df = train[['SMILES', _target]].dropna()

    # Calculate IQR
    Q1 = valid_df[_target].quantile(0.25)
    Q3 = valid_df[_target].quantile(0.75)
    IQR = Q3 - Q1

    # Define outliers (1.5*IQR rule)
    outliers = valid_df[(valid_df[_target] < Q1 - 1.5 * IQR) | (valid_df[_target] > Q3 + 1.5 * IQR)]

    # Define normal points (within IQR)
    normal = valid_df[(valid_df[_target] >= Q1) & (valid_df[_target] <= Q3)]

    # Sample SMILES
    outlier_smiles = outliers['SMILES'].unique()[:12]  # Limit to 12 molecules
    normal_smiles = normal['SMILES'].sample(n=20, random_state=42)

    # Convert to RDKit Mol objects
    outlier_mols = [Chem.MolFromSmiles(smi) for smi in outlier_smiles if Chem.MolFromSmiles(smi)]
    normal_mols = [Chem.MolFromSmiles(smi) for smi in normal_smiles if Chem.MolFromSmiles(smi)]

    # Draw molecules
    print("ğŸ§ª Outlier Molecules")
    if outlier_mols:
        img = Draw.MolsToGridImage(outlier_mols, molsPerRow=4, subImgSize=(200, 200), legends=["Outlier"]*len(outlier_mols))
        display(img)
    else:
        print("No outliers found.")

    print("âœ… Normal Molecules")
    img = Draw.MolsToGridImage(normal_mols, molsPerRow=5, subImgSize=(200, 200), legends=["Normal"]*len(normal_mols))
    display(img)


train['SMILES']



all_smiles = ''.join(train['SMILES'].dropna().tolist())
char_counts = Counter(all_smiles)
total_chars = len(all_smiles)

print("Character | Count | Percentage")
print("------------------------------")

# í�¼ì„¼íŠ¸ ê³„ì‚° í›„ ë‚´ë¦¼ì°¨ìˆœ ì •ë ¬
sorted_chars = sorted(char_counts.items(), key=lambda x: x[1] / total_chars, reverse=True)

for char, count in sorted_chars:
    percentage = count / total_chars * 100
    print(f"   '{char}'   |  {count}  |  {percentage:.2f}%")


# SMILES ì „ì²´ ë¬¸ì��ì—´ ê²°í•© í›„ ë¬¸ì�� ìˆ˜ ì„¸ê¸°
all_smiles = ''.join(train['SMILES'].dropna().tolist())
char_counts = Counter(all_smiles)

# ë”•ì…”ë„ˆë¦¬ë¥¼ DataFrameìœ¼ë¡œ ë³€í™˜
char_df = pd.DataFrame(char_counts.items(), columns=['Character', 'Count']).sort_values(by='Count', ascending=False)

# ì‹œê°�í™”
plt.figure(figsize=(12, 6))
sns.barplot(x='Character', y='Count', data=char_df, palette='viridis')

plt.title('Character Frequency in SMILES Strings')
plt.xlabel('Character')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


target_char_ratios = {}

for t in target:
    # ê²°ì¸¡ì¹˜ ì—†ëŠ” SMILES ì¶”ì¶œ
    smiles_list = train[train[t].notnull()]['SMILES'].dropna().tolist()
    
    # ë¬¸ì�� ì—°ê²° ë°� ë¹ˆë�„ìˆ˜ ê³„ì‚°
    all_chars = ''.join(smiles_list)
    total_chars = len(all_chars)
    char_count = Counter(all_chars)

    # ë¹„ìœ¨ ê³„ì‚°
    char_ratio = {char: (count / total_chars) * 100 for char, count in char_count.items()}
    target_char_ratios[t] = char_ratio

# ê°� íƒ€ê²Ÿë³„ ìƒ�ìœ„ 10ê°œ ë¬¸ì�� ë¹„ìœ¨ ì¶œë ¥
for t in target:
    print(f'\nâ–¶ Target: {t}')
    sorted_chars = sorted(target_char_ratios[t].items(), key=lambda x: x[1], reverse=True)
    for char, ratio in sorted_chars[:10]:
        print(f"'{char}': {ratio:.2f}%")




# íƒ€ê²Ÿë³„ ë¬¸ì�� ë¹„ìœ¨ì�„ ëª¨ë‘� ëª¨ì�„ ë¹ˆ ë�°ì�´í„°í”„ë ˆì�„
all_data = []

for t in target:
    smiles_list = train[train[t].notnull()]['SMILES'].dropna().tolist()
    all_chars = ''.join(smiles_list)
    total_chars = len(all_chars)

    char_count = Counter(all_chars)
    char_ratio = {k: v / total_chars * 100 for k, v in char_count.items()}
    
    # ë�°ì�´í„°í”„ë ˆì�„ ìƒ�ì„± ë°� íƒ€ê²Ÿ ì�´ë¦„ ì»¬ëŸ¼ ì¶”ê°€
    df = pd.DataFrame(char_ratio.items(), columns=['Character', 'Percentage'])
    df['Target'] = t
    all_data.append(df)

# ëª¨ë‘� í•©ì¹˜ê¸°
combined_df = pd.concat(all_data)

# ìƒ�ìœ„ nê°œ ë¬¸ì��ë§Œ ë¹„êµ�í•˜ê³  ì‹¶ìœ¼ë©´ ì „ì²´ ë¬¸ì�� ë¹ˆë�„ í•©ì‚° í›„ ìƒ�ìœ„ nê°œ ì¶”ì¶œ
top_chars = combined_df.groupby('Character')['Percentage'].sum().sort_values(ascending=False).head(30).index
filtered_df = combined_df[combined_df['Character'].isin(top_chars)]

plt.figure(figsize=(15, 8))
sns.barplot(data=filtered_df, x='Character', y='Percentage', hue='Target', palette='viridis')
plt.title("SMILES Character Ratio by Target (Top 30 Characters)")
plt.xlabel("Character")
plt.ylabel("Frequency (%)")
plt.legend(title='Target')
plt.tight_layout()
plt.show()


chars_to_test = ['C', 'c', 'O', 'N', '2', '3', '4']
all_chars = set(''.join(chars_to_test))

print(f"ì´� ë¬¸ì�� ê°œìˆ˜: {len(chars_to_test)}")

# ë„ˆë¬´ ë§�ìœ¼ë©´ ìƒ�ìœ„ Nê°œ ë¬¸ì��ë§Œ ì„ íƒ� (ì˜ˆ: 20ê°œ)
char_total_counts = Counter(''.join(chars_to_test))
top_chars = [ch for ch, _ in char_total_counts.most_common(20)]

colors = plt.cm.tab20(np.linspace(0, 1, len(top_chars)))

for t in target:
    data_list = []
    for idx, row in train.iterrows():
        smiles = row['SMILES']
        target_val = row[t]
        if pd.isna(smiles) or pd.isna(target_val):
            continue
        
        char_count = Counter(smiles)
        for ch in top_chars:
            count = char_count.get(ch, 0)
            data_list.append({'TargetValue': target_val, 'Character': ch, 'Count': count})
    
    df_plot = pd.DataFrame(data_list)
    
    plt.figure(figsize=(14, 8))
    for i, ch in enumerate(top_chars):
        sub_df = df_plot[df_plot['Character'] == ch]
        plt.scatter(sub_df['TargetValue'], sub_df['Count'], label=ch, color=colors[i], alpha=0.6, s=30)
    
    plt.xlabel(t)
    plt.ylabel('Character Count in SMILES')
    plt.title(f'Scatter Plot of Top {len(top_chars)} Character Counts vs {t}')
    plt.legend(title='Character', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


chars_to_test = ['C', 'c', 'O', 'N', '2', '3', '4']

for t in target:
    print(f"\nğŸ“Š pearson (target: {t})")
    
    valid_rows = train[train[t].notnull()][['SMILES', t]]

    count_data = []
    for _, row in valid_rows.iterrows():
        smiles = row['SMILES']
        target_val = row[t]
        char_count = Counter(smiles)
        char_vector = [char_count.get(ch, 0) for ch in chars_to_test]
        char_vector.append(target_val)
        count_data.append(char_vector)

    df_counts = pd.DataFrame(count_data, columns=chars_to_test + [t])
    corr_matrix = df_counts.corr()

    plt.figure(figsize=(9, 7))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'Character Frequency + Target Correlation ( Target: {t})')

    # â–¶ Yì¶• ë’¤ì§‘ê¸° â†’ ì˜¤ë¥¸ìª½ ìœ„ì—�ì„œ ì™¼ìª½ ì•„ë�˜ë¡œ ëŒ€ê°�ì„ 
    plt.gca().invert_yaxis()

    plt.show()




# íƒ€ê²Ÿë“¤
target_list = ['Tg','Density','Rg','Tc','FFV']

# ê°� rowë§ˆë‹¤ C, c ê°œìˆ˜ ê³„ì‚°
C_counts = []
c_counts = []

for idx, row in train.iterrows():
    smiles = row['SMILES']
    if pd.isna(smiles):
        C_counts.append(np.nan)
        c_counts.append(np.nan)
        continue
    
    char_count = Counter(smiles)
    C_counts.append(char_count.get('C', 0))
    c_counts.append(char_count.get('c', 0))

# ì§€í‘œ ê³„ì‚°
df_features = pd.DataFrame({
    'C': C_counts,
    'c': c_counts,
})

df_features['C_div_c'] = df_features['C'] / (df_features['c'] + 1e-6)  # 0 ë‚˜ëˆ„ê¸° ë°©ì§€
df_features['C_minus_c'] = df_features['C'] - df_features['c']

# ê²°ê³¼ ì €ì�¥ìš© ë¦¬ìŠ¤íŠ¸
results = []

for t in target_list:
    target_vals = train[t]
    
    for feature in ['C', 'c', 'C_div_c', 'C_minus_c']:
        # ê²°ì¸¡ì¹˜ ì�ˆëŠ” í–‰ ì œì™¸
        valid_idx = (~df_features[feature].isna()) & (~target_vals.isna())
        if valid_idx.sum() == 0:
            corr = np.nan
            pval = np.nan
        else:
            corr, pval = pearsonr(df_features.loc[valid_idx, feature], target_vals[valid_idx])
        
        results.append({'Target': t, 'Feature': feature, 'Pearson_r': corr, 'p_value': pval})

# ê²°ê³¼ DataFrame
df_corr = pd.DataFrame(results)

# ìƒ�ê´€ê³„ìˆ˜ ì ˆëŒ€ê°’ ê¸°ì¤€ ë‚´ë¦¼ì°¨ìˆœ ì •ë ¬ (ë³´í†µ ê´€ê³„ ê°•ë�„ ë³´ê¸° í�¸í•¨)
df_corr = df_corr.sort_values(by='Pearson_r', key=lambda x: x.abs(), ascending=False).reset_index(drop=True)

print(df_corr)


# í”¼ë²— í…Œì�´ë¸” ìƒ�ì„±: í–‰ = Target, ì—´ = Feature, ê°’ = Pearson_r
corr_pivot = df_corr.pivot(index='Target', columns='Feature', values='Pearson_r')

plt.figure(figsize=(8, 6))
sns.heatmap(corr_pivot, annot=True, fmt=".2f", cmap='coolwarm', center=0,
            cbar_kws={'label': 'Pearson Correlation'})

plt.title('Pearson Correlation Heatmap between Targets and Features')
plt.ylabel('Target')
plt.xlabel('Feature')
plt.tight_layout()
plt.show()



target_list = ['FFV']  

for t in target_list:
    y_values = []
    formula_y1 = []
    formula_y2 = []
    
    for idx, row in train.iterrows():
        smiles = row['SMILES']
        target_val = row[t]
        if pd.isna(smiles) or pd.isna(target_val):
            continue
        
        char_count = Counter(smiles)
        c_count = char_count.get('C', 0)
        small_c_count = char_count.get('c', 0)
        total_C = c_count + small_c_count
        
        if total_C == 0:
            continue
        
        y_values.append(target_val)
        
        # Formula 1: FFV = 0.4 + 0.1 / sqrt(C + c + 1)
        val1 = 0.4 + 0.1 / np.sqrt(total_C + 1)
        formula_y1.append(val1)
        
        # Formula 2: FFV = 0.4 + 0.1 * exp(-0.2 * (C + c))
        val2 = 0.4 + 0.1 * np.exp(-0.2 * total_C)
        formula_y2.append(val2)


    # ìƒ�ê´€ê³„ìˆ˜ ê³„ì‚°
    corr_1, p1 = pearsonr(y_values[:len(formula_y1)], formula_y1)
    corr_2, p2 = pearsonr(y_values[:len(formula_y2)], formula_y2)
    
    
    print(f"== Target: {t} ==")
    print(f"Formula 1 (0.4 + 0.1/sqrt(C+c+1)) Pearson r: {corr_1:.4f}, p-value: {p1:.4e}")
    print(f"Formula 2 (0.4 + 0.1*exp(-0.2*(C+c))) Pearson r: {corr_2:.4f}, p-value: {p2:.4e}")

    # ì‹œê°�í™”
    all_vals = y_values[:len(formula_y2)] + formula_y2
    min_val =0.4
    max_val =0.5 

    plt.figure(figsize=(21, 5))
    
    # Formula 1
    plt.subplot(1, 3, 1)
    plt.scatter(y_values[:len(formula_y1)], formula_y1, alpha=0.6, color='blue', label='Data points', s=20)
    plt.plot([min_val, max_val], [min_val, max_val], '--', color='gray', label='y = x')
    plt.xlabel('Actual FFV')
    plt.ylabel('Predicted FFV: 0.4 + 0.1 / sqrt(C + c + 1)')
    plt.title(f'Formula 1 vs FFV\nPearson r = {corr_1:.4f}')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    
    # Formula 2
    plt.subplot(1, 3, 2)
    plt.scatter(y_values[:len(formula_y2)], formula_y2, alpha=0.6, color='green', label='Data points', s=20)
    plt.plot([min_val, max_val], [min_val, max_val], '--', color='gray', label='y = x')
    plt.xlabel('Actual FFV')
    plt.ylabel('Predicted FFV: 0.4 + 0.1 * exp(-0.2 * (C + c))')
    plt.title(f'Formula 2 vs FFV\nPearson r = {corr_2:.4f}')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()


    plt.tight_layout()
    plt.show()


import re
from sklearn.preprocessing import StandardScaler


def plot_scaled_targets_vs_atom_features_with_mean(train_df, features, targets, n_cols=2):
    train_df = train_df.reset_index(drop=True)

    scaler_target = StandardScaler()
    scaled_targets = scaler_target.fit_transform(train_df[targets])
    scaled_target_df = pd.DataFrame(scaled_targets, columns=targets).reset_index(drop=True)

    # inf ê°’ â†’ NaN ë³€í™˜
    train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    scaled_target_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    n_rows = int(np.ceil(len(features) / n_cols))
    
    plt.figure(figsize=(10 * n_cols, 6 * n_rows))

    for i, feat in enumerate(features):
        plt.subplot(n_rows, n_cols, i + 1)

        for target in targets:
            valid_idx = train_df[feat].notna() & scaled_target_df[target].notna()

            # ì¶”ê°€ë¡œ ë¬´í•œëŒ€ ì²´í�¬ë�„ ë„£ì–´ì¤Œ (ì•ˆì •ì„± ê°•í™”)
            valid_idx &= np.isfinite(train_df[feat])
            valid_idx &= np.isfinite(scaled_target_df[target])

            sns.scatterplot(
                x=train_df.loc[valid_idx, feat],
                y=scaled_target_df.loc[valid_idx, target],
                alpha=0.5,
                label=f"{target} points",
                s=20
            )

            x_vals = train_df.loc[valid_idx, feat]
            y_vals = scaled_target_df.loc[valid_idx, target]
            bins = np.linspace(x_vals.min(), x_vals.max(), 30)
            bin_indices = np.digitize(x_vals, bins)

            bin_means = []
            bin_centers = []
            for b in range(1, len(bins)):
                bin_y = y_vals[bin_indices == b]
                if len(bin_y) > 0:
                    bin_means.append(bin_y.mean())
                    bin_centers.append((bins[b-1] + bins[b]) / 2)

            sns.lineplot(
                x=bin_centers,
                y=bin_means,
                label=f"{target} mean",
                linewidth=2.5
            )

        plt.title(f"{feat} vs Scaled Targets with Mean Lines", fontsize=16)
        plt.xlabel(feat, fontsize=12)
        plt.ylabel("Scaled Target", fontsize=12)
        plt.legend()

    plt.tight_layout()
    plt.show()

def longest_run(smi, char):
    max_run = 0
    current_run = 0
    for c in smi:
        if c == char:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run

def compute_atom_features(df):
    atom_counts = {
        'count_C': [], 'count_c': [],
        'C_over_c': [], 'C_minus_c': [],
        'len_SMILES': [],
        'C_ratio': [], 'c_ratio': [],
        'count_F': [], 'count_N': [], 'count_O': [], 'count_Cl': [],
        'F_ratio': [], 'N_ratio': [], 'O_ratio': [], 'Cl_ratio': [],
        'FNOCl_ratio': [],
        'max_C_run': [], 'max_c_run': [],
        'sum_of_digits': [],'sum_of_digits_ratio': []
    }
    eps = 1e-3

    for smi in df['SMILES']:
        count_C = smi.count('C')
        count_c = smi.count('c')
        count_F = smi.count('F')
        count_N = smi.count('N')
        count_O = smi.count('O')
        count_Cl = smi.count('Cl')
        len_smi = len(smi)

        digits = re.findall(r'\d', smi)
        sum_of_digits = sum(map(int, digits)) if digits else 0
        sum_of_digits_ratio=(sum_of_digits+1)/(len_smi+1)
        C_over_c = (count_C+1) / (count_c + 1)
        C_minus_c = count_C - count_c
        C_ratio = (count_C+1) / (len_smi + 1)
        c_ratio = (count_c+1) / (len_smi + 1)
        FNOCl_sum = count_F + count_N + count_O + count_Cl
        FNOCl_ratio = (FNOCl_sum+1) / (len_smi + 1)

        max_C_run = longest_run(smi, 'C')
        max_c_run = longest_run(smi, 'c')

        atom_counts['count_C'].append(count_C)
        atom_counts['count_c'].append(count_c)
        atom_counts['C_over_c'].append(C_over_c)
        atom_counts['C_minus_c'].append(C_minus_c)
        atom_counts['len_SMILES'].append(len_smi)
        atom_counts['C_ratio'].append(C_ratio)
        atom_counts['c_ratio'].append(c_ratio)
        atom_counts['count_F'].append(count_F)
        atom_counts['count_N'].append(count_N)
        atom_counts['count_O'].append(count_O)
        atom_counts['count_Cl'].append(count_Cl)
        atom_counts['F_ratio'].append((count_F+1) / (len_smi + 1))
        atom_counts['N_ratio'].append((count_N+1) / (len_smi + 1))
        atom_counts['O_ratio'].append((count_O+1) / (len_smi + 1))
        atom_counts['Cl_ratio'].append((count_Cl+1) / (len_smi + 1))
        atom_counts['FNOCl_ratio'].append(FNOCl_ratio)
        atom_counts['max_C_run'].append(max_C_run)
        atom_counts['max_c_run'].append(max_c_run)
        atom_counts['sum_of_digits'].append(sum_of_digits)
        atom_counts['sum_of_digits_ratio'].append(sum_of_digits_ratio)

    return pd.DataFrame(atom_counts)
    



atom_features_df = compute_atom_features(train)
train = pd.concat([train.reset_index(drop=True), atom_features_df], axis=1)



# ì˜ˆì‹œ ì‹¤í–‰
atom_features = [
    'count_C', 'count_c',
    'C_over_c', 'C_minus_c',
    'len_SMILES',
    'C_ratio', 'c_ratio',
    'count_F', 'count_N', 'count_O', 'count_Cl',
    'F_ratio', 'N_ratio', 'O_ratio', 'Cl_ratio',
    'FNOCl_ratio',
    'max_C_run', 'max_c_run',
    'sum_of_digits','sum_of_digits_ratio'
]

targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

plot_scaled_targets_vs_atom_features_with_mean(train, atom_features, targets)




def mol_from_smiles(smiles):
    try:
        return Chem.MolFromSmiles(smiles)
    except:
        return None

train['Mol'] = train['SMILES'].apply(mol_from_smiles)

descriptor_names = [desc_name[0] for desc_name in Descriptors._descList]

# ê°� Descriptor ê²°ê³¼ë¥¼ ë‹´ì�„ ë”•ì…”ë„ˆë¦¬ ìƒ�ì„±
desc_data = {}

for desc_name in descriptor_names:
    func = getattr(Descriptors, desc_name)
    desc_data[desc_name] = train['Mol'].apply(lambda x: func(x) if x else None)

# DataFrameìœ¼ë¡œ ë³€í™˜
desc_df = pd.DataFrame(desc_data)

# ì›�ë³¸ train ë�°ì�´í„°í”„ë ˆì�„ê³¼ í•œë²ˆì—� í•©ì¹˜ê¸°
train = pd.concat([train, desc_df], axis=1)

# Mol ì»¬ëŸ¼ ì‚­ì œ
train.drop(columns=['Mol'], inplace=True)


train.shape


null_counts=train.iloc[:,7:].isnull().sum()
cols_with_null=null_counts[null_counts!=0]
print(cols_with_null)


cols_of_interest = ['MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge', 'MinAbsPartialCharge']

for t in target:
    # íƒ€ê²Ÿì�´ null ì•„ë‹Œ í–‰ ì�¸ë�±ìŠ¤
    target_notnull_idx = set(train[train[t].notnull()].index)
    
    # 4ê°œ ì»¬ëŸ¼ ì¤‘ í•˜ë‚˜ë�¼ë�„ null ì•„ë‹Œ í–‰ ì�¸ë�±ìŠ¤
    charge_notnull_idx = set().union(*[set(train[train[col].notnull()].index) for col in cols_of_interest])
    
    # êµ�ì§‘í•©
    overlap_idx = target_notnull_idx.intersection(charge_notnull_idx)
    
    print(f"Target: {t} - ê²¹ì¹˜ëŠ” í–‰ ê°œìˆ˜: {len(overlap_idx)}")


for t in target:
    target_notnull_idx = set(train[train[t].notnull()].index)
    charge_notnull_idx = set().union(*[set(train[train[col].notnull()].index) for col in cols_of_interest])
    overlap_idx = target_notnull_idx.intersection(charge_notnull_idx)
    
    subset = train.loc[list(overlap_idx), cols_of_interest + [t]].dropna()
    
    corr = subset.corr()
    
    print(f"\nTarget: {t} - ìƒ�ê´€ê´€ê³„ (ê²¹ì¹˜ëŠ” í–‰ ê¸°ì¤€)")
    print(corr[t])
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(6,5))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'Correlation Matrix (Target: {t}) - Overlap Rows')
    plt.show()


# Get the number of unique values per column
unique_counts = train.nunique()

# Filter columns where unique count is exactly 1
single_value_columns = unique_counts[unique_counts == 1].index.tolist()

print("Columns with a single unique value:", single_value_columns)



# subset: train.iloc[:, 7:]  # í•„ìš”í•œ ê²½ìš° subset ì •ì�˜

subset = train.iloc[:, 7:]
corr_matrix = subset.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

high_corr_pairs = [(col1, col2) 
                   for col1 in upper.columns 
                   for col2 in upper.index 
                   if pd.notna(upper.loc[col1, col2]) and upper.loc[col1, col2] >= 0.95]

# ê²¹ì¹˜ëŠ” ë³€ìˆ˜ëª… ì§‘í•© ìƒ�ì„±
overlap_vars = set()
for col1, col2 in high_corr_pairs:
    overlap_vars.add(col1)
    overlap_vars.add(col2)

# ë¦¬ìŠ¤íŠ¸ë¡œ ë³€í™˜
overlap_vars_list = list(overlap_vars)

print("Variables involved in correlations >= 0.95:")
print(overlap_vars_list)


remove_vars = [
    'fr_phenol', 'fr_nitro_arom', 'Chi3v', 'Chi3n', 'MaxEStateIndex', 'VSA_EState6', 'fr_phos_acid', 'SMR_VSA2', 'NumAmideBonds',
    'MolMR', 'Chi4n', 'FpDensityMorgan1', 'fr_azide', 'MinAbsPartialCharge', 'BertzCT', 'NumAromaticRings', 'fr_Nhpyrrole', 'Chi1',
    'fr_halogen', 'fr_phos_ester', 'MinPartialCharge', 'fr_C_O_noCOO', 'Phi', 'Chi0v', 'MaxAbsEStateIndex', 'Kappa2', 'NOCount',
    'fr_alkyl_halide', 'fr_COO2', 'LabuteASA', 'FpDensityMorgan2', 'NumHDonors', 'NumAromaticCarbocycles', 'Chi0n', 'MinPartialCharge',
    'fr_diazo', 'fr_C_O', 'Chi0', 'fr_COO', 'Chi2n', 'Chi1v', 'NumValenceElectrons', 'SMR_VSA7', 'fr_amide', 'HeavyAtomMolWt', 'TPSA',
    'fr_Ar_OH', 'Chi2v', 'HeavyAtomCount', 'Chi1n', 'HallKierAlpha', 'Chi4v', 'fr_phenol_noOrthoHbond', 'fr_benzene', 'ExactMolWt',
    'Kappa1', 'SlogP_VSA6', 'fr_nitro', 'fr_Ar_NH', 'Kappa3', 'fr_nitrile', 'MolWt', 'VSA_EState2', 'NumRotatableBonds', 'NHOHCount',
    'FpDensityMorgan3', 'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW',
    'BCUT2D_MRHI', 'BCUT2D_MRLOW', 'NumRadicalElectrons', 'SMR_VSA8', 'SlogP_VSA9', 'fr_barbitur', 'fr_benzodiazepine',
    'fr_dihydropyridine', 'fr_isothiocyan', 'fr_lactam', 'fr_nitroso', 'fr_prisulfonamd', 'fr_thiocyan','MinAbsPartialCharge'
]
train = train.drop(columns=remove_vars, errors='ignore')


train['MaxAbsPartialCharge'] = train['MaxAbsPartialCharge'].fillna(0)
train['MaxPartialCharge'] = train['MaxPartialCharge'].fillna(0)


train.shape


# Select columns from the 7th onward (0-based index)
subset = train.iloc[:, 7:]

# Calculate correlation matrix for numerical columns only
corr_matrix = subset.corr()

print(corr_matrix)

# Optionally, visualize the correlation matrix as a heatmap
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=False, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix (Columns from 7th onward)')
plt.tight_layout()
plt.show()


targets = target  # your target list

for t in targets:
    print(f"\nTop 50 correlations with target: {t}")
    
    subset = train[train[t].notnull()]
    cols = train.columns[7:]
    if t not in cols:
        cols = cols.append(pd.Index([t]))
    subset_cols = subset.loc[:, cols]
    corr_matrix = subset_cols.corr()
    target_corr = corr_matrix[t].drop(labels=[t]).abs()
    top_50 = target_corr.sort_values(ascending=False).head(50)
    
    print(top_50)
    
    # Plotting
    plt.figure(figsize=(10,6))
    sns.barplot(x=top_50.values, y=top_50.index, palette="viridis")
    plt.title(f"Top 50 correlations with target: {t}")
    plt.xlabel("Absolute Correlation")
    plt.ylabel("Features")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.show()



targets = target  # your target list
top_features_all = []

for t in targets:
    subset = train[train[t].notnull()]
    cols = train.columns[7:]
    if t not in cols:
        cols = cols.append(pd.Index([t]))
    subset_cols = subset.loc[:, cols]
    corr_matrix = subset_cols.corr()
    target_corr = corr_matrix[t].drop(labels=[t]).abs()
    top_20 = target_corr.sort_values(ascending=False).head(20)
    
    top_features_all.extend(top_20.index.tolist())

feature_counts = Counter(top_features_all)

# Filter features appearing at least 3 times
filtered_features = [(f, c) for f, c in feature_counts.items() if c >= 3]
filtered_features.sort(key=lambda x: x[1], reverse=True)

print("Features appearing 3 times or more in top 20 correlations across targets:")
for feature, count in filtered_features:
    print(f"{feature}: {count} times")




features = ['PEOE_VSA14', 'fr_unbrch_alkane', 'VSA_EState7', 'EState_VSA5']
targets = target  # your list of target column names

# Initialize scaler
scaler = MinMaxScaler()

# Create a copy of the dataframe
train_scaled = train.copy()

# Scale only the target columns
train_scaled[targets] = scaler.fit_transform(train[targets])

for feature in features:
    # Calculate correlation with each target (using original data for feature and target)
    correlations = {}
    for t in targets:
        valid_idx = train[[feature, t]].dropna().index
        if len(valid_idx) == 0:
            corr = 0
        else:
            corr = train.loc[valid_idx, feature].corr(train.loc[valid_idx, t])
        correlations[t] = corr

    # Filter targets with abs(corr) > 0.4
    filtered_targets = [t for t, corr in correlations.items() if abs(corr) > 0.4]

    # Skip plotting if no targets passed threshold
    if not filtered_targets:
        print(f"No targets with correlation > 0.4 for feature {feature}, skipping plot.")
        continue

    # Create color palette only for filtered targets
    palette = sns.color_palette('tab10', n_colors=len(filtered_targets))
    color_map = dict(zip(filtered_targets, palette))

    plt.figure(figsize=(10, 6))
    for t in filtered_targets:
        sns.scatterplot(
            data=train_scaled,
            x=feature,        # original feature values, no scaling
            y=t,              # scaled target values
            label=f"{t} (corr={correlations[t]:.2f})",
            color=color_map[t],
            alpha=0.7
        )
    plt.title(f'{feature} vs Targets (Targets Min-Max Scaled)')
    plt.xlabel(feature)
    plt.ylabel('Target Values (scaled)')
    plt.legend(title='Targets')
    plt.show()


features = [
    'SMR_VSA10', 'SlogP_VSA1', 'SMR_VSA5', 'AvgIpc', 
    'SlogP_VSA7', 'SlogP_VSA5', 'VSA_EState8', 
    'fr_NH1', 'fr_ester', 'MolLogP'
]

targets = target  # your list of target column names

# Initialize scaler
scaler = MinMaxScaler()

# Copy dataframe and scale only target columns
train_scaled = train.copy()
train_scaled[targets] = scaler.fit_transform(train[targets])

for feature in features:
    # Calculate correlation for each target with current feature using original (unscaled) data
    correlations = {}
    for t in targets:
        valid_idx = train[[feature, t]].dropna().index
        if len(valid_idx) == 0:
            corr = 0
        else:
            corr = train.loc[valid_idx, feature].corr(train.loc[valid_idx, t])
        correlations[t] = corr

    # Filter targets where |corr| > 0.4
    filtered_targets = [t for t, corr in correlations.items() if abs(corr) > 0.4]

    if not filtered_targets:
        print(f"No targets with correlation > 0.4 for feature {feature}, skipping plot.")
        continue

    # Prepare color palette only for filtered targets
    palette = sns.color_palette('tab10', n_colors=len(filtered_targets))
    color_map = dict(zip(filtered_targets, palette))

    plt.figure(figsize=(10, 6))
    for t in filtered_targets:
        sns.scatterplot(
            data=train_scaled,
            x=feature,        # original feature values, no scaling
            y=t,              # scaled target values
            label=f"{t} (corr={correlations[t]:.2f})",
            color=color_map[t],
            alpha=0.7
        )
    plt.title(f'{feature} vs Targets (Targets Min-Max Scaled)')
    plt.xlabel(feature)
    plt.ylabel('Target Values (scaled)')
    plt.legend(title='Targets')
    plt.show()



# íƒ€ê²Ÿ ëª©ë¡�
targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

# fr, Num ë³€ìˆ˜ ì¶”ì¶œ
fr_cols = [col for col in train.columns if col.startswith('fr')]
num_cols = [col for col in train.columns if col.startswith('Num')]
all_cols = fr_cols + num_cols

# ìƒ�ê´€ê³„ìˆ˜ ì €ì�¥
corr_df = pd.DataFrame()

for target in targets:
    corr = train[all_cols + [target]].corr()[target].drop(target)
    corr_df[target] = corr

# ì ˆëŒ€ê°’ ê¸°ì¤€ ì •ë ¬í•´ì„œ ë³´ê¸° ì¢‹ê²Œ
corr_df_abs = corr_df.abs().sort_values(by=targets[0], ascending=False)  # ì²« ë²ˆì§¸ íƒ€ê²Ÿ ê¸°ì¤€
display(corr_df_abs)


threshold_corr = 0.2   # ìƒ�ê´€ê³„ìˆ˜ ì ˆëŒ€ê°’ ê¸°ì¤€
threshold_nan = 0.8     # NaN ë¹„ìœ¨ ê¸°ì¤€ (ì˜ˆ: 80%)

# fr, Num ì»¬ëŸ¼ ë¦¬ìŠ¤íŠ¸
fr_cols = [col for col in train.columns if col.startswith('fr')]
num_cols = [col for col in train.columns if col.startswith('Num')]
all_cols = fr_cols + num_cols

# ìƒ�ê´€ê³„ìˆ˜ ê³„ì‚°
corr_df = pd.DataFrame()
for target in targets:
    corr = train[all_cols + [target]].corr()[target].drop(target)
    corr_df[target] = corr

corr_df_abs = corr_df.abs()

# ì“¸ëª¨ì—†ëŠ” ì»¬ëŸ¼ íƒ�ìƒ‰
useless_cols = []

for col in corr_df_abs.index:
    low_corr = (corr_df_abs.loc[col] < threshold_corr).all()
    nan_corr = corr_df_abs.loc[col].isna().any()  # í•˜ë‚˜ë�¼ë�„ NaN ìƒ�ê´€ê³„ìˆ˜
    high_nan = (train[col].isna().mean() > threshold_nan)
    
    if low_corr or nan_corr or high_nan:
        useless_cols.append(col)

print(f"ğŸ“Œ ì´� {len(useless_cols)}ê°œì�˜ ë¶ˆí•„ìš”í•œ ë³€ìˆ˜ ë°œê²¬!")
print(useless_cols)





