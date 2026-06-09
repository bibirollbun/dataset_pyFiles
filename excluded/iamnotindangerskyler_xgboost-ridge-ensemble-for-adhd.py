import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Retro Synthwave paleti ve global stil ayarları
def set_synthwave_palette(style="whitegrid", context="notebook", font_family="sans-serif"):
    palette = ['#f72585', '#b5179e', '#7209b7', '#560bad', '#480ca8',
               '#3a0ca3', '#3f37c9', '#4361ee', '#4895ef', '#4cc9f0']

    sns.set_palette(palette)
    sns.set_style(style)
    sns.set_context(context)

    plt.rcParams.update({
        'axes.titlepad': 20,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'font.family': font_family,
        'figure.autolayout': True,
        'axes.edgecolor': '#3a0ca3',  # Koyu mor çerçeve
        'axes.facecolor': '#ffffff',  # Beyaz arka plan
        'figure.facecolor': '#ffffff',  # Beyaz figür arka planı
        'axes.labelcolor': '#3a0ca3',  # Koyu mor etiketler
        'axes.titlecolor': '#3a0ca3',  # Koyu mor başlık
        'xtick.color': '#3a0ca3',  # Koyu mor tick etiketleri
        'ytick.color': '#3a0ca3',
        'grid.color': '#4cc9f0',  # Açık mavi grid
        'grid.alpha': 0.5
    })

    return palette
    
palette = set_synthwave_palette()


from matplotlib.gridspec import GridSpec

plt.figure(figsize=(16, 8), facecolor='#0f0f23')

# Create grid layout
gs = GridSpec(2, 2, figure=plt.gcf(), hspace=0.4, wspace=0.3)

# Title box
ax_title = plt.subplot(gs[0, :])
ax_title.set_facecolor('#0f0f23')
ax_title.text(0.5, 0.7, 'EXPLORATORY DATA ANALYSIS', 
             fontsize=28, ha='center', va='center', 
             color='#f72585', fontweight='bold',
             fontfamily='monospace')
ax_title.text(0.5, 0.4, 'Hello and welcome to my WIDS25 Datathon EDA!', 
             fontsize=18, ha='center', va='center',
             color='#4cc9f0', fontfamily='monospace')
ax_title.text(0.5, 0.2, 'Dataset: [WIDS25 Datathon]', 
             fontsize=12, ha='center', va='center',
             color='white')
ax_title.axis('off')

# Sample distribution plot
ax1 = plt.subplot(gs[1, 0])
sns.histplot(np.random.normal(0, 1, 1000), 
             kde=True, color='#f72585', ax=ax1)
ax1.set_title('Target Distribution', 
              pad=15, color='white', fontsize=12)
ax1.set_facecolor('#0f0f23')
ax1.grid(color='#4cc9f0', alpha=0.2)

# Missing values placeholder
ax2 = plt.subplot(gs[1, 1])
missing_bars = ax2.barh(['Feature A', 'Feature B', 'Feature C'], 
                        [0.12, 0.25, 0.08], 
                        color=['#f72585', '#b5179e', '#7209b7'])
ax2.set_title('Missing Values Overview', 
              pad=15, color='white', fontsize=12)
ax2.set_facecolor('#0f0f23')
ax2.grid(color='#4cc9f0', alpha=0.2, axis='x')

# Styling for all axes
for ax in [ax1, ax2]:
    for spine in ax.spines.values():
        spine.set_edgecolor('#4cc9f0')
    ax.tick_params(colors='white', which='both')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')

plt.suptitle('', fontsize=1)
plt.tight_layout()
plt.show()


import numpy as np
import pandas as pd 

#train
solutions_metrics_data = pd.read_excel('/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAINING_SOLUTIONS.xlsx')
categorical_metrics_data = pd.read_excel('/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_CATEGORICAL_METADATA_new.xlsx')
metrics_data = pd.read_csv('/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv')
quantitative_metrics_data = pd.read_excel('/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_QUANTITATIVE_METADATA_new.xlsx')

#test
categorical_metrics_test_data = pd.read_excel('/kaggle/input/widsdatathon2025/TEST/TEST_CATEGORICAL.xlsx')
metrics_test_data = pd.read_csv('/kaggle/input/widsdatathon2025/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv')
quantitative_metrics_test_data = pd.read_excel('/kaggle/input/widsdatathon2025/TEST/TEST_QUANTITATIVE_METADATA.xlsx')


print(solutions_metrics_data.dtypes)
print(categorical_metrics_data.dtypes)
print(quantitative_metrics_data.dtypes)
print(categorical_metrics_test_data.dtypes)
print(quantitative_metrics_test_data.dtypes)


def create_missing_values_df(merged_df, df_names):
    # Eksik değer analizi yap
    null_counts = merged_df.isnull().sum()
    total_rows = len(merged_df)
    null_percent = (null_counts / total_rows) * 100
    
    # Hangi veri setinden geldiğini belirle
    origin_mapping = {}
    for name, df in zip(df_names, dfs_to_merge):
        for col in df.columns:
            if col != 'participant_id':
                origin_mapping[col] = name
    
    # Sonuç DataFrame'ini oluştur
    result_df = pd.DataFrame({
        'Column_Name': null_counts.index,
        'Dataset_Origin': [origin_mapping.get(col, 'Merged') for col in null_counts.index],
        'Missing_Count': null_counts.values,
        'Missing_Percentage': null_percent.round(2).values
    })
    
    return result_df

# Train verilerini birleştir
dfs_to_merge = [solutions_metrics_data, categorical_metrics_data, quantitative_metrics_data]
train_names = ["Solutions Metrics", "Categorical Metrics", "Quantitative Metrics"]

# participant_id'e göre birleştirme
merged_train = dfs_to_merge[0]
for df in dfs_to_merge[1:]:
    merged_train = pd.merge(merged_train, df, on='participant_id', how='outer')

missing_values_train_df = create_missing_values_df(merged_train, train_names)

# Test verilerini birleştir
test_dfs = [categorical_metrics_test_data, quantitative_metrics_test_data]
test_names = ["Categorical Metrics Test", "Quantitative Metrics Test"]

merged_test = test_dfs[0]
for df in test_dfs[1:]:
    merged_test = pd.merge(merged_test, df, on='participant_id', how='outer')

missing_values_test_df = create_missing_values_df(merged_test, test_names)

# Sonuçları göster
print("=== TRAIN VERİLERİNDEKİ EKSİK DEĞERLER ===")
print(missing_values_train_df)
print("\n\n=== TEST VERİLERİNDEKİ EKSİK DEĞERLER ===")
print(missing_values_test_df)


def visualize_missing_data(missing_df, title):
    missing_df = missing_df.sort_values('Missing_Percentage', ascending=False)
    
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='Missing_Percentage', y='Column_Name', hue='Dataset_Origin', 
                    data=missing_df, palette=palette, dodge=False)
    
    plt.title(f'Missing Value Analysis - {title}', fontsize=16, pad=20)
    plt.xlabel('Missing Value Percentage (%)', fontsize=12)
    plt.ylabel('Columns', fontsize=12)
    plt.axvline(x=50, color='#f72585', linestyle='--', alpha=0.7)
    
    for p in ax.patches:
        width = p.get_width()
        if width > 0:
            ax.text(width + 1, p.get_y() + p.get_height()/2., 
                   f'{width:.1f}%', 
                   ha='left', va='center', fontsize=10)
    
    plt.legend(title='Dataset Origin', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

visualize_missing_data(missing_values_train_df, "Training Data")
visualize_missing_data(missing_values_test_df, "Test Data")

numerical_cols = [col for col in merged_train.select_dtypes(include=np.number).columns 
                 if col != 'participant_id']

if len(numerical_cols) > 0:
    plt.figure(figsize=(14, 6))
    for i, col in enumerate(numerical_cols[:10]):  # Limit to first 10 for clarity
        plt.subplot(2, 5, i+1)
        sns.histplot(merged_train[col], kde=True, color=palette[i%len(palette)])
        plt.title(col, fontsize=10)
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

categorical_cols = [col for col in merged_train.select_dtypes(exclude=np.number).columns 
                   if col != 'participant_id']

if len(categorical_cols) > 0:
    plt.figure(figsize=(14, 6))
    for i, col in enumerate(categorical_cols[:5]): 
        plt.subplot(1, 5, i+1)
        sns.countplot(y=col, data=merged_train, 
                     order=merged_train[col].value_counts().index,
                     color=palette[i%len(palette)])
        plt.title(col, fontsize=10)
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if len(numerical_cols) > 1:
    plt.figure(figsize=(12, 10))
    corr_matrix = merged_train[numerical_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", 
                cmap='coolwarm', center=0, linewidths=.5,
                cbar_kws={"shrink": .8})
    
    plt.title('Numerical Variables Correlation Matrix', fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

if len(numerical_cols) > 1:
    top_corr_vars = corr_matrix.abs().unstack().sort_values(ascending=False)
    top_corr_vars = top_corr_vars[top_corr_vars < 1].index[:5]
    selected_cols = list(set([x[0] for x in top_corr_vars] + [x[1] for x in top_corr_vars]))
    
    sns.pairplot(merged_train[selected_cols], 
                plot_kws={'alpha': 0.6, 'color': palette[0]},
                diag_kws={'color': palette[2]})
    plt.suptitle('Pairplot of Top Correlated Variables', y=1.02)
    plt.show()


# Kategorik ve numerik sütun listeleri
categorical_cols = [
    'Basic_Demos_Enroll_Year', 'Basic_Demos_Study_Site', 
    'PreInt_Demos_Fam_Child_Ethnicity', 'PreInt_Demos_Fam_Child_Race',
    'MRI_Track_Scan_Location', 'Barratt_Barratt_P1_Edu', 
    'Barratt_Barratt_P1_Occ', 'Barratt_Barratt_P2_Edu',
    'Barratt_Barratt_P2_Occ'
]

numerical_cols = [
    'EHQ_EHQ_Total', 'ColorVision_CV_Score', 'APQ_P_APQ_P_CP',
    'APQ_P_APQ_P_ID', 'APQ_P_APQ_P_INV', 'APQ_P_APQ_P_OPD',
    'APQ_P_APQ_P_PM', 'APQ_P_APQ_P_PP', 'SDQ_SDQ_Difficulties_Total',
    'SDQ_SDQ_Conduct_Problems', 'SDQ_SDQ_Emotional_Problems',
    'SDQ_SDQ_Externalizing', 'SDQ_SDQ_Generating_Impact',
    'SDQ_SDQ_Hyperactivity', 'SDQ_SDQ_Internalizing',
    'SDQ_SDQ_Peer_Problems', 'SDQ_SDQ_Prosocial', 'MRI_Track_Age_at_Scan'
]

def preprocess_data(train_df, test_df, categorical_cols, numerical_cols):

    train_processed = train_df.copy()
    test_processed = test_df.copy()
    
    # Kategorik sütunlar için eksik değer doldurma
    for col in categorical_cols:
        if col in train_processed.columns:

            train_processed[col] = train_processed[col].astype(str).replace('nan', 'MISSING')
            
            # Test'de train'de olmayan kategorileri 'MISSING' yap
            if col in test_processed.columns:
                test_processed[col] = test_processed[col].astype(str).replace('nan', 'MISSING')
                unique_train_cats = set(train_df[col].dropna().astype(str).unique())
                test_processed[col] = test_processed[col].apply(
                    lambda x: x if x in unique_train_cats else 'MISSING')
    
    # Numerik sütunlar için eksik değer doldurma
    for col in numerical_cols:
        if col in train_processed.columns:
            min_val = train_processed[col].min()
            
            if min_val < -500:
                print(f"Uyarı: {col} sütununda -500'den küçük değerler var ({min_val}). "
                      f"Bu sütun için eksik değerler doldurulamadı.")
            else:
                fill_value = -999
                train_processed[col] = train_processed[col].fillna(fill_value)
                
                if col in test_processed.columns:
                    test_processed[col] = test_processed[col].fillna(fill_value)
    
    return train_processed, test_processed

train_processed, test_processed = preprocess_data(merged_train, merged_test, categorical_cols, numerical_cols)


def check_missing_values(train_df, test_df, categorical_cols, numerical_cols):
    print("=== TRAIN VERİSİ EKSİK DEĞER KONTROLÜ ===")
    # Kategorik sütunlar
    print("\nKategorik Sütunlar:")
    cat_missing_train = train_df[categorical_cols].isnull().sum()
    print(cat_missing_train[cat_missing_train > 0])
    
    # Numerik sütunlar
    print("\nNumerik Sütunlar:")
    num_missing_train = train_df[numerical_cols].isnull().sum()
    print(num_missing_train[num_missing_train > 0])
    
    print("\n\n=== TEST VERİSİ EKSİK DEĞER KONTROLÜ ===")
    # Kategorik sütunlar
    print("\nKategorik Sütunlar:")
    cat_missing_test = test_df[categorical_cols].isnull().sum()
    print(cat_missing_test[cat_missing_test > 0])
    
    # Numerik sütunlar
    print("\nNumerik Sütunlar:")
    num_missing_test = test_df[numerical_cols].isnull().sum()
    print(num_missing_test[num_missing_test > 0])
    
    # Özet rapor
    print("\n\n=== ÖZET RAPOR ===")
    print(f"Train verisinde toplam eksik değer sayısı: {train_df.isnull().sum().sum()}")
    print(f"Test verisinde toplam eksik değer sayısı: {test_df.isnull().sum().sum()}")

check_missing_values(train_processed, test_processed, categorical_cols, numerical_cols)


import matplotlib.pyplot as plt

# Genel parametreler
GRID_SIZE = 28  # Karelerin boyutu
MATRIX_SIZE = 196  # Matrisin boyutu
UPPER_TRIANGLE_COLOR = (1.0, 0.6, 0.6, 0.6)  # Üst üçgen rengi (hafif kırmızı)
LOWER_TRIANGLE_COLOR = (0.6, 0.6, 1.0, 0.6)  # Alt üçgen rengi (hafif mavi)
EXCLUDE_TRIANGLE_INDICES = {1, 8, 14, 19, 23, 26, 28}  # Çıkarılacak üçgen numaraları
THRESHOLDS = {
    "positive": 0.44, #eşik değerleri
    "negative": -0.34 
}

# İstatistik hesaplama fonksiyonu
def calculate_triangle_statistics(triangle_values, triangle_index, region):
    total_values = len(triangle_values)
    pos_values = np.sum(triangle_values > THRESHOLDS["positive"])
    neg_values = np.sum(triangle_values < THRESHOLDS["negative"])

    pos_ratio = pos_values / total_values if total_values > 0 else np.nan
    neg_ratio = neg_values / total_values if total_values > 0 else np.nan
    pos_neg_ratio = pos_values / neg_values if neg_values > 0 else np.nan

    stats = {
        "Üçgen No": triangle_index,
        "Bölge": region,
        "Pozitif Sayısı": pos_values,
        "Negatif Sayısı": neg_values,
        "Pozitif Oran": pos_ratio,
        "Negatif Oran": neg_ratio,
        "Pozitif/Negatif Oran": pos_neg_ratio
    }

    return stats
    
###############################################################################
# İlk kullanıcının verisini alalım örnek bir kullanıcı üzerinden görselleştirme

first_user_metrics_data = metrics_data.iloc[0, 1:].values  # İlk kullanıcı verisi (ilk sütun hariç)

# Üçgen matrisin boyutlarını belirleyelim (196x196)
matrix = np.full((MATRIX_SIZE, MATRIX_SIZE), np.nan)  # Matrisin tamamını NaN ile dolduruyoruz (görselleştirme amacıyla)

start_index = 0
for i in range(MATRIX_SIZE):
    num_columns = MATRIX_SIZE - i
    if num_columns > 0:
        matrix[i, i:i + num_columns] = first_user_metrics_data[start_index:start_index + num_columns]
    start_index += num_columns

plt.figure(figsize=(12, 12))
plt.imshow(matrix, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.title("User Data Triangular Matrix Visualization (196x196)")
plt.xlabel("Column Index")
plt.ylabel("Throw Index")

# Karelere bölme ve her karenin üst üçgenini boyama
triangle_statistics = []
triangle_index = 1

# Karelere bölerek işlem yapma
for i in range(0, MATRIX_SIZE - GRID_SIZE + 1, GRID_SIZE):
    for j in range(i, MATRIX_SIZE - GRID_SIZE + 1, GRID_SIZE):
        if j >= i:
            square = matrix[i:i + GRID_SIZE, j:j + GRID_SIZE]
            
            upper_triangle = np.triu(square)
            lower_triangle = np.tril(square)
            
            upper_triangle_values = upper_triangle[np.isfinite(upper_triangle)]
            lower_triangle_values = lower_triangle[np.isfinite(lower_triangle)]

            # Köşegende olup olmadığını kontrol et
            is_diagonal = (i == j)
            
            # Alt üçgen istatistikleri
            # Köşegende veya normal durumda çıkarılmamış üçgenler için alt üçgen istatistiklerini hesapla
            if is_diagonal or (triangle_index not in EXCLUDE_TRIANGLE_INDICES):
                stats_lower = calculate_triangle_statistics(lower_triangle_values, triangle_index, "Alt")
                triangle_statistics.append(stats_lower)

                plt.gca().add_patch(
                    plt.Polygon([[j, i], [j + GRID_SIZE, i], [j + GRID_SIZE, i + GRID_SIZE]], color=LOWER_TRIANGLE_COLOR)
                )
            
            # Üst üçgen istatistikleri
            # Sadece köşegende olmayan üçgenler için üst üçgen istatistiklerini hesapla
            if not is_diagonal:
                stats_upper = calculate_triangle_statistics(upper_triangle_values, triangle_index, "Üst")
                triangle_statistics.append(stats_upper)

                plt.gca().add_patch(
                    plt.Polygon([[j, i], [j, i + GRID_SIZE], [j + GRID_SIZE, i + GRID_SIZE]], color=UPPER_TRIANGLE_COLOR)
                )

            triangle_index += 1

plt.show()

# İstatistikleri yazdırma
all_statistics = []
for stats in triangle_statistics:
    stat_line = (f"Üçgen {stats['Üçgen No']} - Bölge: {stats['Bölge']} | "
                 f"Pozitif: {stats['Pozitif Sayısı']} | Negatif: {stats['Negatif Sayısı']} | "
                 f"Pozitif Oran: {stats['Pozitif Oran']:.2f} | Negatif Oran: {stats['Negatif Oran']:.2f} | "
                 f"Pozitif/Negatif Oran: {stats['Pozitif/Negatif Oran']:.2f}")
    all_statistics.append(stat_line)

print("\n".join(all_statistics))

df_statistics = pd.DataFrame(triangle_statistics)

columns = []
for i in range(1, 29):
    if i in EXCLUDE_TRIANGLE_INDICES and i != 1:
        columns.append(f"Üçgen {i} - Üst")
    else:
        columns.append(f"Üçgen {i} - Alt")
        if i != 1:
            columns.append(f"Üçgen {i} - Üst")

df_final = pd.DataFrame(index=df_statistics.columns[2:], columns=columns)

# Her bir hesaplama için satırları ekleme
for col in columns:
    triangle_no, region = col.split(" - ")
    triangle_no = int(triangle_no.split()[1])
    region_stats = df_statistics[(df_statistics["Üçgen No"] == triangle_no) & (df_statistics["Bölge"] == region)].drop(columns=["Üçgen No", "Bölge"])
    if not region_stats.empty:
        df_final[col] = region_stats.values.flatten()


import numpy as np
import pandas as pd
from tqdm import tqdm

# Genel parametreler
MATRIX_SIZE = 196  # Matrisin boyutu
GRID_SIZE = 28  # Karelerin boyutu
EXCLUDE_TRIANGLE_INDICES = {1, 8, 14, 19, 23, 26, 28}  # Çıkarılacak üçgen numaraları
THRESHOLDS = {
    "positive": 0.44,
    "negative": -0.34 
}

# İstatistik hesaplama fonksiyonu
def calculate_triangle_statistics(triangle_values, triangle_index, region, participant_id):
    total_values = len(triangle_values)
    pos_values = np.sum(triangle_values > THRESHOLDS["positive"])
    neg_values = np.sum(triangle_values < THRESHOLDS["negative"])

    pos_ratio = pos_values / total_values if total_values > 0 else np.nan
    neg_ratio = neg_values / total_values if total_values > 0 else np.nan
    pos_neg_ratio = pos_values / neg_values if neg_values > 0 else np.nan

    stats = {
        "Kullanıcı ID": participant_id,
        "Üçgen No - Bölge": f"{triangle_index} - {region}",
        "Pozitif Sayısı": pos_values,
        "Negatif Sayısı": neg_values,
        "Pozitif Oran": pos_ratio,
        "Negatif Oran": neg_ratio,
        "Pozitif/Negatif Oran": pos_neg_ratio
    }

    return stats

# Tüm kullanıcılar için istatistikleri tutacak liste
all_users_statistics = []

# Her kullanıcı için işlemleri gerçekleştirelim
for user_id in tqdm(range(len(metrics_data)), desc="Processing users"):
    user_metrics_data = metrics_data.iloc[user_id, 1:].values  # Kullanıcı verisi (ilk sütun hariç)
    participant_id = categorical_metrics_data['participant_id'].iloc[user_id]
    
    user_metrics_data = pd.to_numeric(user_metrics_data, errors='coerce')
    
    # Matrisin tamamını NaN ile dolduruyoruz
    matrix = np.full((MATRIX_SIZE, MATRIX_SIZE), np.nan)
    
    start_index = 0
    for i in range(MATRIX_SIZE):
        num_columns = MATRIX_SIZE - i
        if num_columns > 0:
            matrix[i, i:i + num_columns] = user_metrics_data[start_index:start_index + num_columns]
        start_index += num_columns
    
    # Karelere bölerek işlem yapma
    triangle_statistics = []
    triangle_index = 1
    for i in range(0, MATRIX_SIZE - GRID_SIZE + 1, GRID_SIZE):
        for j in range(i, MATRIX_SIZE - GRID_SIZE + 1, GRID_SIZE):
            if j >= i:
                square = matrix[i:i + GRID_SIZE, j:j + GRID_SIZE]
                
                upper_triangle = np.triu(square)
                lower_triangle = np.tril(square)
                
                upper_triangle_values = upper_triangle[np.isfinite(upper_triangle)]
                lower_triangle_values = lower_triangle[np.isfinite(lower_triangle)]

                # Alt üçgen için oranlar
                if triangle_index not in EXCLUDE_TRIANGLE_INDICES:
                    stats_lower = calculate_triangle_statistics(lower_triangle_values, triangle_index, "Alt", participant_id)
                    triangle_statistics.append(stats_lower)
                
                # Üst üçgen için oranlar
                stats_upper = calculate_triangle_statistics(upper_triangle_values, triangle_index, "Üst", participant_id)
                triangle_statistics.append(stats_upper)
                
                triangle_index += 1
    
    all_users_statistics.extend(triangle_statistics)

df_all_statistics = pd.DataFrame(all_users_statistics)

# Her bir kullanıcı için verileri ekleme
all_users_combined_stats = []

for user_id in range(len(metrics_data)):
    participant_id = categorical_metrics_data['participant_id'].iloc[user_id]  # Gerçek kullanıcı ID'si
    user_stats = df_all_statistics[df_all_statistics['Kullanıcı ID'] == participant_id].copy()
    user_stats.set_index('Üçgen No - Bölge', inplace=True)
    
    combined_stats = user_stats.T
    combined_stats['Kullanıcı ID'] = participant_id
    
    all_users_combined_stats.append(combined_stats)

df_final_combined_stats = pd.concat(all_users_combined_stats)

# Kullanıcı ID satırlarını kaldırma ve yeni sütun ekleme
df_final_combined_stats.reset_index(inplace=True)
df_final_combined_stats.rename(columns={'index': 'Özellik'}, inplace=True)

# Yeni sütunu ekleme ve hücreleri birleştirme
df_final_combined_stats['Kullanıcı ID'] = df_final_combined_stats.apply(lambda row: row['Kullanıcı ID'] if row['Özellik'] != 'A' else row['Kullanıcı ID'], axis=1)

# Kullanıcı ID sütununu ilk sütun yapma ve Özellik sütununda Kullanıcı ID adında bir satır olmamasını sağlama
df_final_combined_stats = df_final_combined_stats[df_final_combined_stats['Özellik'] != 'Kullanıcı ID']
cols = df_final_combined_stats.columns.tolist()
cols.insert(0, cols.pop(cols.index('Kullanıcı ID')))
df_final_combined_stats = df_final_combined_stats[cols]


# Genel parametreler
GRID_SIZE = 28  # Karelerin boyutu
MATRIX_SIZE = 196  # Matrisin boyutu
UPPER_TRIANGLE_COLOR = (1.0, 0.6, 0.6, 0.6)  # Üst üçgen rengi (hafif kırmızı)
LOWER_TRIANGLE_COLOR = (0.6, 0.6, 1.0, 0.6)  # Alt üçgen rengi (hafif mavi)
EXCLUDE_TRIANGLE_INDICES = {1, 8, 14, 19, 23, 26, 28}  # Çıkarılacak üçgen numaraları
THRESHOLDS = {
    "positive": 0.46,  # Pozitif eşik değeri
    "negative": -0.34  # Negatif eşik değeri
}

# İstatistik hesaplama fonksiyonu
def calculate_triangle_statistics(triangle_values, triangle_index, region):
    total_values = len(triangle_values)
    pos_values = np.sum(triangle_values > THRESHOLDS["positive"])
    neg_values = np.sum(triangle_values < THRESHOLDS["negative"])

    pos_ratio = pos_values / total_values if total_values > 0 else np.nan
    neg_ratio = neg_values / total_values if total_values > 0 else np.nan
    pos_neg_ratio = pos_values / neg_values if neg_values > 0 else np.nan

    stats = {
        "Üçgen No": triangle_index,
        "Bölge": region,
        "Pozitif Sayısı": pos_values,
        "Negatif Sayısı": neg_values,
        "Pozitif Oran": pos_ratio,
        "Negatif Oran": neg_ratio,
        "Pozitif/Negatif Oran": pos_neg_ratio
    }

    return stats

# İlk kullanıcının verisini alalım
first_user_metrics_test_data = metrics_test_data.iloc[0, 1:].values 

# Üçgen matrisin boyutlarını belirleyelim (196x196)
matrix = np.full((MATRIX_SIZE, MATRIX_SIZE), np.nan)  # Matrisin tamamını NaN ile dolduruyoruz (görselleştirme amacıyla)

start_index = 0
for i in range(MATRIX_SIZE):
    num_columns = MATRIX_SIZE - i
    if num_columns > 0:
        matrix[i, i:i + num_columns] = first_user_metrics_test_data[start_index:start_index + num_columns]
    start_index += num_columns

plt.figure(figsize=(12, 12))
plt.imshow(matrix, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.title("Test User Data Triangular Matrix Visualization (196x196)")
plt.xlabel("Column Index")
plt.ylabel("Throw Index")

triangle_statistics = []
triangle_index = 1

# Karelere bölerek işlem yapma
for i in range(0, MATRIX_SIZE - GRID_SIZE + 1, GRID_SIZE):
    for j in range(i, MATRIX_SIZE - GRID_SIZE + 1, GRID_SIZE):
        if j >= i:
            square = matrix[i:i + GRID_SIZE, j:j + GRID_SIZE]
            
            upper_triangle = np.triu(square)
            lower_triangle = np.tril(square)
            
            upper_triangle_values = upper_triangle[np.isfinite(upper_triangle)]
            lower_triangle_values = lower_triangle[np.isfinite(lower_triangle)]

            # Köşegende olup olmadığını kontrol et
            is_diagonal = (i == j)
            
            # Alt üçgen için oranlar
            # Köşegende veya normal durumda çıkarılmamış üçgenler için alt üçgen istatistiklerini hesapla
            if is_diagonal or (triangle_index not in EXCLUDE_TRIANGLE_INDICES):
                stats_lower = calculate_triangle_statistics(lower_triangle_values, triangle_index, "Alt")
                triangle_statistics.append(stats_lower)

                plt.gca().add_patch(
                    plt.Polygon([[j, i], [j + GRID_SIZE, i], [j + GRID_SIZE, i + GRID_SIZE]], color=LOWER_TRIANGLE_COLOR)
                )
                
            # Üst üçgen için oranlar
            # Sadece köşegende olmayan üçgenler için üst üçgen istatistiklerini hesapla
            if not is_diagonal:
                stats_upper = calculate_triangle_statistics(upper_triangle_values, triangle_index, "Üst")
                triangle_statistics.append(stats_upper)

                plt.gca().add_patch(
                    plt.Polygon([[j, i], [j, i + GRID_SIZE], [j + GRID_SIZE, i + GRID_SIZE]], color=UPPER_TRIANGLE_COLOR)
                )

            triangle_index += 1

plt.show()

# İstatistikleri yazdırma
all_statistics = []
for stats in triangle_statistics:
    stat_line = (f"Üçgen {stats['Üçgen No']} - Bölge: {stats['Bölge']} | "
                 f">0.62: {stats['Pozitif Sayısı']} | Negatif: {stats['Negatif Sayısı']} | "
                 f"Pozitif Oran: {stats['Pozitif Oran']:.2f} | Negatif Oran: {stats['Negatif Oran']:.2f} | "
                 f"Pozitif/Negatif Oran: {stats['Pozitif/Negatif Oran']:.2f}")
    all_statistics.append(stat_line)

print("\n".join(all_statistics))

columns = []
for i in range(1, 29):
    if i in EXCLUDE_TRIANGLE_INDICES and i != 1:  # Üçgen 1 artık köşegen olduğu için mavi olacak
        columns.append(f"Üçgen {i} - Üst")
    else:
        columns.append(f"Üçgen {i} - Alt")
        if i != 1:  # Köşegen üçgenlerde sadece Alt üçgen var
            columns.append(f"Üçgen {i} - Üst")

df_statistics = pd.DataFrame(triangle_statistics)

df_final = pd.DataFrame(index=df_statistics.columns[2:], columns=columns)

# Her bir hesaplama için satırları ekleme
for col in columns:
    triangle_no, region = col.split(" - ")
    triangle_no = int(triangle_no.split()[1])
    region_stats = df_statistics[(df_statistics["Üçgen No"] == triangle_no) & (df_statistics["Bölge"] == region)].drop(columns=["Üçgen No", "Bölge"])
    if not region_stats.empty:
        df_final[col] = region_stats.values.flatten()


from tqdm import tqdm

# Genel parametreler
MATRIX_SIZE = 196  # Matrisin boyutu
GRID_SIZE = 28  # Karelerin boyutu
EXCLUDE_TRIANGLE_INDICES = {1, 8, 14, 19, 23, 26, 28}  # Çıkarılacak üçgen numaraları
THRESHOLDS = {
    "positive": 0.46,
    "negative": -0.34
}

# İstatistik hesaplama fonksiyonu
def calculate_triangle_statistics(triangle_values, triangle_index, region, participant_id):
    total_values = len(triangle_values)
    pos_values = np.sum(triangle_values > THRESHOLDS["positive"])
    neg_values = np.sum(triangle_values < THRESHOLDS["negative"])

    pos_ratio = pos_values / total_values if total_values > 0 else np.nan
    neg_ratio = neg_values / total_values if total_values > 0 else np.nan
    pos_neg_ratio = pos_values / neg_values if neg_values > 0 else np.nan

    stats = {
        "Kullanıcı ID": participant_id,
        "Üçgen No - Bölge": f"{triangle_index} - {region}",
        "Pozitif Sayısı": pos_values,
        "Negatif Sayısı": neg_values,
        "Pozitif Oran": pos_ratio,
        "Negatif Oran": neg_ratio,
        "Pozitif/Negatif Oran": pos_neg_ratio
    }

    return stats

# Tüm kullanıcılar için istatistikleri tutacak liste
all_users_statistics = []

# Her kullanıcı için işlemleri gerçekleştirelim
for user_id in tqdm(range(len(metrics_test_data)), desc="Processing users"):
    user_metrics_data = metrics_test_data.iloc[user_id, 1:].values
    participant_id = categorical_metrics_test_data['participant_id'].iloc[user_id]
    
    user_metrics_data = pd.to_numeric(user_metrics_data, errors='coerce')
    
    matrix = np.full((MATRIX_SIZE, MATRIX_SIZE), np.nan)
    
    start_index = 0
    for i in range(MATRIX_SIZE):
        num_columns = MATRIX_SIZE - i
        if num_columns > 0:
            matrix[i, i:i + num_columns] = user_metrics_data[start_index:start_index + num_columns]
        start_index += num_columns
    
    # Karelere bölerek işlem yapma
    triangle_statistics = []
    triangle_index = 1
    for i in range(0, MATRIX_SIZE - GRID_SIZE + 1, GRID_SIZE):
        for j in range(i, MATRIX_SIZE - GRID_SIZE + 1, GRID_SIZE):
            if j >= i:
                square = matrix[i:i + GRID_SIZE, j:j + GRID_SIZE]
                
                upper_triangle = np.triu(square)
                lower_triangle = np.tril(square)
                
                upper_triangle_values = upper_triangle[np.isfinite(upper_triangle)]
                lower_triangle_values = lower_triangle[np.isfinite(lower_triangle)]

                # Alt üçgen için oranlar
                if triangle_index not in EXCLUDE_TRIANGLE_INDICES:
                    stats_lower = calculate_triangle_statistics(lower_triangle_values, triangle_index, "Alt", participant_id)
                    triangle_statistics.append(stats_lower)
                
                # Üst üçgen için oranlar
                stats_upper = calculate_triangle_statistics(upper_triangle_values, triangle_index, "Üst", participant_id)
                triangle_statistics.append(stats_upper)
                
                triangle_index += 1
    
    all_users_statistics.extend(triangle_statistics)

df_all_statistics = pd.DataFrame(all_users_statistics)

all_users_combined_stats = []

for user_id in range(len(metrics_test_data)):
    participant_id = categorical_metrics_test_data['participant_id'].iloc[user_id]  # Gerçek kullanıcı ID'si
    user_stats = df_all_statistics[df_all_statistics['Kullanıcı ID'] == participant_id].copy()
    user_stats.set_index('Üçgen No - Bölge', inplace=True)
    
    combined_stats = user_stats.T
    combined_stats['Kullanıcı ID'] = participant_id
    
    all_users_combined_stats.append(combined_stats)

df_final_combined_stats_test = pd.concat(all_users_combined_stats)

# Kullanıcı ID satırlarını kaldırma ve yeni sütun ekleme
df_final_combined_stats_test.reset_index(inplace=True)
df_final_combined_stats_test.rename(columns={'index': 'Özellik'}, inplace=True)

# Yeni sütunu ekleme ve hücreleri birleştirme
df_final_combined_stats_test['Kullanıcı ID'] = df_final_combined_stats_test.apply(lambda row: row['Kullanıcı ID'] if row['Özellik'] != 'A' else row['Kullanıcı ID'], axis=1)

# Kullanıcı ID sütununu ilk sütun yapma ve Özellik sütununda Kullanıcı ID adında bir satır olmamasını sağlama
df_final_combined_stats_test = df_final_combined_stats_test[df_final_combined_stats_test['Özellik'] != 'Kullanıcı ID']
cols = df_final_combined_stats_test.columns.tolist()
cols.insert(0, cols.pop(cols.index('Kullanıcı ID')))
df_final_combined_stats_test = df_final_combined_stats_test[cols]


# Veri setlerini birleştirme
merged_data = categorical_metrics_data.merge(
    quantitative_metrics_data, on="participant_id", how="inner"
).merge(
    solutions_metrics_data, on="participant_id", how="inner"
)

# İstenmeyen sütunları silme
merged_data.drop(columns=['ADHD_Outcome', 'Sex_F'], inplace=True)


merged_test_data = categorical_metrics_test_data.merge(
    quantitative_metrics_test_data, on="participant_id", how="inner"
)


from xgboost import XGBRegressor, DMatrix, train
from xgboost.callback import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Numerik ve kategorik sütunlar
numeric_columns = [
    "EHQ_EHQ_Total", "ColorVision_CV_Score", "APQ_P_APQ_P_CP", "APQ_P_APQ_P_ID", 
    "APQ_P_APQ_P_INV", "APQ_P_APQ_P_OPD", "APQ_P_APQ_P_PM", "APQ_P_APQ_P_PP", 
    "SDQ_SDQ_Conduct_Problems", "SDQ_SDQ_Difficulties_Total", "SDQ_SDQ_Emotional_Problems", 
    "SDQ_SDQ_Externalizing", "SDQ_SDQ_Generating_Impact", "SDQ_SDQ_Hyperactivity", 
    "SDQ_SDQ_Internalizing", "SDQ_SDQ_Peer_Problems", "SDQ_SDQ_Prosocial"
]

categorical_columns = [col for col in merged_data.columns if col not in numeric_columns + ["MRI_Track_Age_at_Scan", "participant_id"]]

# Kategorik sütunları encode et
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    merged_data[col] = le.fit_transform(merged_data[col].astype(str))
    label_encoders[col] = le

# Numerik verilerde NaN kontrolü
merged_data[numeric_columns] = merged_data[numeric_columns].fillna(merged_data[numeric_columns].mean())

# MRI_Track_Age_at_Scan kolonundaki NaN olan verileri belirlemek
train_data = merged_data[~merged_data["MRI_Track_Age_at_Scan"].isna()]
test_data = merged_data[merged_data["MRI_Track_Age_at_Scan"].isna()]

# X ve y belirle
X_train = train_data.drop(columns=["MRI_Track_Age_at_Scan", "participant_id"])
y_train = train_data["MRI_Track_Age_at_Scan"]
X_test = test_data.drop(columns=["MRI_Track_Age_at_Scan", "participant_id"])

# Veriyi eğitim ve validasyon setlerine ayır
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# XGBoost için DMatrix formatına çevirme
dtrain = DMatrix(X_train, label=y_train)
dval = DMatrix(X_val, label=y_val)
dtest = DMatrix(X_test)

# XGBoost model parametreleri
params = {
    "objective": "reg:squarederror",
    "random_state": 42,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "learning_rate": 0.0025,
    "max_depth": 6,
    "n_estimators": 2000
}

# Modeli eğit (erken durdurma dahil)
print("XGBoost modeli eğitiliyor...")
xgboost_model = train(
    params,
    dtrain,
    num_boost_round=2000,
    evals=[(dval, "validation")],
    early_stopping_rounds=70,
    verbose_eval=0
)

# Eksik değerlere tahmin yapılıyor
print("Eksik değerlere tahmin yapılıyor...")
xgboost_preds = xgboost_model.predict(dtest)

# Performans ölçütleri
xgboost_train_preds = xgboost_model.predict(dtrain)
xgboost_train_mse = mean_squared_error(y_train, xgboost_train_preds)
xgboost_train_r2 = r2_score(y_train, xgboost_train_preds)

print(f"XGBoost - Eğitim MSE: {xgboost_train_mse:.4f}, Eğitim R2: {xgboost_train_r2:.4f}")

# Test verisini doldur
test_data.loc[:, "MRI_Track_Age_at_Scan"] = xgboost_preds

# participant_id sütununu koruyarak veriyi birleştir
filled_data = pd.concat([train_data, test_data]).sort_index()



filled_data


# Pivot işlemi
pivot_df = df_final_combined_stats.pivot(index='Kullanıcı ID', columns='Özellik')

# Sütun isimlerini düzeltme
pivot_df.columns = [f"{col[0]}_{col[1]}" for col in pivot_df.columns]

pivot_df.reset_index(inplace=True)

# Veri setlerini birleştirme
combined_df = pd.merge(filled_data, pivot_df, left_on='participant_id', right_on='Kullanıcı ID')

combined_df.drop(columns=['Kullanıcı ID'], inplace=True)


# Pivot işlemi
pivot_test_df = df_final_combined_stats_test.pivot(index='Kullanıcı ID', columns='Özellik')

# Sütun isimlerini düzeltme
pivot_test_df.columns = [f"{col[0]}_{col[1]}" for col in pivot_test_df.columns]

pivot_test_df.reset_index(inplace=True)

# Veri setlerini birleştirme
combined_test_df = pd.merge(merged_test_data, pivot_test_df, left_on='participant_id', right_on='Kullanıcı ID')

combined_test_df.drop(columns=['Kullanıcı ID'], inplace=True)

combined_test_df.fillna(0, inplace=True)


# Veri setlerini birleştirme
final_combined_df = pd.merge(combined_df, solutions_metrics_data, on='participant_id')

final_combined_df.fillna(0, inplace=True)


missing_combined_test = combined_test_df.isnull().sum().sum()
print(f"combined_test_df veri setindeki toplam boş hücre sayısı: {missing_combined_test}")

missing_final_combined = final_combined_df.isnull().sum().sum()
print(f"final_combined_df veri setindeki toplam boş hücre sayısı: {missing_final_combined}")


# Gerekli sütun isimlerini içeren anahtar kelimeler
column_keywords = ['Pozitif Sayısı', 'Negatif Sayısı', 'Pozitif Oran', 'Negatif Oran']
column_keywords2= ['Pozitif Sayısı', 'Negatif Sayısı', 'Pozitif Oran', 'Negatif Oran']

# Ortalama hesaplama fonksiyonu
def calculate_group_means(df, keywords):
    means = {}
    for keyword in keywords:
        # Anahtar kelimeyi içeren sütunları seç
        relevant_columns = [col for col in df.columns if keyword in col]
        if relevant_columns:
            group_mean = df[relevant_columns].mean().mean()
            means[keyword] = group_mean
        else:
            means[keyword] = None
    return means

# Ortalama hesaplama fonksiyonu
def calculate_group_means(df, keywords2):
    means = {}
    for keyword2 in keywords2:
        # Anahtar kelimeyi içeren sütunları seç
        relevant_columns = [col for col in df.columns if keyword2 in col]
        if relevant_columns:
            # Bu sütunların ortalamalarını al ve ardından bu ortalamaların ortalamasını hesapla
            group_mean = df[relevant_columns].mean().mean()
            means[keyword2] = group_mean
        else:
            # Eğer anahtar kelimeye uygun sütun bulunmazsa 0 ya da NaN dön
            means[keyword2] = None
    return means

# İstatistikleri hesapla
combined_means = calculate_group_means(combined_test_df, column_keywords)
final_means = calculate_group_means(final_combined_df, column_keywords2)

print("Combined Test DF Ortalama:")
for keyword, mean in combined_means.items():
    print(f"{keyword}: {mean}")

print("\nFinal Combined DF Ortalama:")
for keyword, mean in final_means.items():
    print(f"{keyword}: {mean}")


test_df = combined_test_df
train_df = final_combined_df


# Train ve Test DataFrame'lerinden 'basic_demos_study_site' sütununu sil
train_df = train_df.drop(columns=['basic_demos_study_site'], errors='ignore')
test_df = test_df.drop(columns=['basic_demos_study_site'], errors='ignore')


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def preprocess_data(df):
    # 1. Kombinasyonları oluşturma
    combinations = {
        'Basic_Demos_Study_Site-PreInt_Demos_Fam_Child_Ethnicity': df['Basic_Demos_Study_Site'].astype(str) + '-' + df['PreInt_Demos_Fam_Child_Ethnicity'].astype(str),
        'PreInt_Demos_Fam_Child_Ethnicity-MRI_Track_Scan_Location': df['PreInt_Demos_Fam_Child_Ethnicity'].astype(str) + '-' + df['MRI_Track_Scan_Location'].astype(str),
        'Basic_Demos_Study_Site-MRI_Track_Scan_Location': df['Basic_Demos_Study_Site'].astype(str) + '-' + df['MRI_Track_Scan_Location'].astype(str)
    }
    
    # Kombinasyon sütunlarını DataFrame'e ekleme
    df = pd.concat([df, pd.DataFrame(combinations)], axis=1)

    # 2. Yeni kombinasyon sütunlarını encode etme
    label_encoders = {}
    combination_columns = list(combinations.keys())
    
    for col in combination_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])  # Sütunu encode et
        label_encoders[col] = le  # Encoder'ı sakla (isteğe bağlı)

    # 3. Kategorik sütun oluşturma
    conditions = [
        df['EHQ_EHQ_Total'] == -100,  # 10th left
        (df['EHQ_EHQ_Total'] >= -28) & (df['EHQ_EHQ_Total'] < 48),  # middle
        df['EHQ_EHQ_Total'] == 100,  # 10th right
        df['EHQ_EHQ_Total'] > 48,  # 48'den büyük
        df['EHQ_EHQ_Total'] < -28  # -28'den küçük
    ]
    choices = [
        'left',  # 10th left
        'middle',  # middle
        'right',  # 10th right
        'greater_than_48',  # 48'den büyük
        'less_than_minus_28'  # -28'den küçük
    ]
    df['EHQ_Category'] = np.select(conditions, choices, default='unknown') 

    # 4. EHQ_Category sütununu encode etme
    le_ehq = LabelEncoder()
    df['EHQ_Category_Encoded'] = le_ehq.fit_transform(df['EHQ_Category'])
    label_encoders['EHQ_Category'] = le_ehq 

    columns_to_scale = []
    for i in range(1, 29): 
        columns_to_scale.extend([
            f'{i} - Üst_Negatif Sayısı',
            f'{i} - Üst_Pozitif Sayısı',
            f'{i} - Üst_Pozitif Oran',
            f'{i} - Üst_Pozitif/Negatif Oran'
        ])

    columns_to_scale = [col for col in columns_to_scale if col in df.columns]

    scaler = MinMaxScaler(feature_range=(-1, 1))

    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    return df, label_encoders

train_df, label_encoders = preprocess_data(train_df)
test_df, _ = preprocess_data(test_df) 
train_df.drop(columns=["EHQ_Category"], inplace=True)


train_df.info()


import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# Existing helper functions (unchanged)
def mcc_eval(y_true, y_pred):
    y_pred_labels = (y_pred > 0.5).astype(int)
    mcc = matthews_corrcoef(y_true, y_pred_labels)
    return 'mcc', mcc, True

def log_metrics(y_true, y_pred, y_probs, fold_results):
    fold_results['Accuracy'].append(accuracy_score(y_true, y_pred))
    fold_results['F1'].append(f1_score(y_true, y_pred))
    fold_results['AUC_ROC'].append(roc_auc_score(y_true, y_probs))
    fold_results['MCC'].append(matthews_corrcoef(y_true, y_pred))

def calculate_class_weights(y):
    unique_classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y)
    return dict(zip(unique_classes, weights))

def prepare_data(train_df, test_df):
    categorical_columns = [
        "Basic_Demos_Enroll_Year", "Basic_Demos_Study_Site", "PreInt_Demos_Fam_Child_Ethnicity",
        "PreInt_Demos_Fam_Child_Race", "MRI_Track_Scan_Location", "Barratt_Barratt_P1_Occ",
        "Barratt_Barratt_P2_Edu", "Barratt_Barratt_P2_Occ",
        "Basic_Demos_Study_Site-PreInt_Demos_Fam_Child_Ethnicity",
        "PreInt_Demos_Fam_Child_Ethnicity-MRI_Track_Scan_Location",
        "Basic_Demos_Study_Site-MRI_Track_Scan_Location", "EHQ_Category_Encoded"
    ]
    
    numeric_columns = [col for col in train_df.columns if col not in categorical_columns + ["ADHD_Outcome", "Sex_F", "participant_id"]]
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_columns)
    ])
    
    X = preprocessor.fit_transform(train_df)
    X_test = preprocessor.transform(test_df)
    
    feature_names = numeric_columns + preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_columns).tolist()
    
    return X, X_test, preprocessor, feature_names

def create_target(train_df):
    return train_df["ADHD_Outcome"].values

def train_and_evaluate(X, y, X_test, test_df, feature_names, params):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = defaultdict(list)
    combined_predictions_all = []
    feature_importance_dict = defaultdict(float)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n{'='*40}")
        print(f"Fold {fold}/5 Training...")
        print(f"{'='*40}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        class_weights = calculate_class_weights(y_train)
        sample_weights = np.array([class_weights[yi] for yi in y_train])
        
        model = xgb.XGBClassifier(
            **params,
            n_estimators=1500,
            early_stopping_rounds=200,
            enable_categorical=False
        )
        
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        importance = model.feature_importances_
        for idx, score in enumerate(importance):
            feature_name = feature_names[idx]
            feature_importance_dict[feature_name] += score
        
        val_probs = model.predict_proba(X_val)[:, 1]
        val_preds = (val_probs > 0.5).astype(int)
        
        log_metrics(y_val, val_preds, val_probs, fold_results)
        
        print(f"\nFold {fold} Results:")
        print(f"Accuracy: {fold_results['Accuracy'][-1]:.4f}")
        print(f"F1: {fold_results['F1'][-1]:.4f}")
        print(f"AUC-ROC: {fold_results['AUC_ROC'][-1]:.4f}")
        print(f"MCC: {fold_results['MCC'][-1]:.4f}")
        
        test_probs = model.predict_proba(X_test)[:, 1]
        combined_predictions_all.append(test_probs)
    
    final_threshold = 0.5
    feature_importance_avg = {k: v / 5 for k, v in feature_importance_dict.items()}
    
    return fold_results, combined_predictions_all, final_threshold, feature_importance_avg

def print_ensemble_results(results):
    print("\n\n" + "="*50)
    print("Final Ensemble Results (5-Fold Average ± Std):")
    print("="*50)
    metrics = ['Accuracy', 'F1', 'AUC_ROC', 'MCC']
    for metric in metrics:
        mean = np.mean(results[metric])
        std = np.std(results[metric])
        print(f"{metric}: {mean:.4f} ± {std:.4f}")

def save_all_predictions(test_df, predictions_list, threshold):
    submission_df = pd.DataFrame({'participant_id': test_df['participant_id']})
    for i, preds in enumerate(predictions_list, 1):
        submission_df[f'ADHD_Outcome_Model_{i}'] = (preds >= threshold).astype(int)
    
    sample_submission_df = pd.read_excel('/kaggle/input/widsdatathon2025/SAMPLE_SUBMISSION.xlsx')
    submission_df_sorted = sample_submission_df[['participant_id']].merge(
        submission_df, on='participant_id', how='left'
    )
    
    if submission_df_sorted.isnull().any().any():
        print("Warning: Some participant_ids in sample_submission_df were not found in submission_df.")
        submission_df_sorted.fillna(0, inplace=True)
    
    submission_df_sorted.to_excel('/kaggle/working/submission_multiple_models_xgb.xlsx', index=False)
    print("submission_multiple_models_xgb.xlsx saved.")
    
    submission_df_sorted['ADHD_Outcome_Model_1'].values 
def main(train_df, test_df):
    X, X_test, preprocessor, feature_names = prepare_data(train_df, test_df)
    y = create_target(train_df)
    
    # Define fixed hyperparameters for XGBoost
    model_params = {
        'learning_rate': 0.05600623106477138,
        'max_depth': 10,
        'min_child_weight': 0.001753139198397266,
        'subsample': 0.8556823379046696,
        'colsample_bytree': 0.7104719442915508,
        'reg_lambda': 3.6831497086151286,
        'reg_alpha': 0.16347733288433106,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': 42,
        'tree_method': 'hist',
        'device': 'cuda'
    }
    
    print("Training XGBoost model with fixed hyperparameters...")
    fold_results, combined_predictions_all, final_threshold, feature_importance_avg = train_and_evaluate(
        X, y, X_test, test_df, feature_names, model_params
    )
    
    print_ensemble_results(fold_results)
    
    combined_ensemble_pred = np.mean(combined_predictions_all, axis=0)
    all_predictions = [combined_ensemble_pred]
    
    # Save feature importance
    feature_importance_df = pd.DataFrame({
        'Feature': list(feature_importance_avg.keys()),
        'Importance': list(feature_importance_avg.values())
    }).sort_values(by='Importance', ascending=False)
    feature_importance_df.to_excel('/kaggle/working/feature_importance_xgb_f1.xlsx', index=False)
    
    save_all_predictions(test_df, all_predictions, final_threshold)

# Assuming train_df and test_df are provided
main(train_df, test_df)


import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.multioutput import MultiOutputClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

# Verileri yükle
def load_data():
    # Eğitim verileri
    train_quant = pd.read_excel("/kaggle/input/widsdatathon2025/TRAIN_OLD/TRAIN_QUANTITATIVE_METADATA.xlsx")
    train_cate = pd.read_excel("/kaggle/input/widsdatathon2025/TRAIN_OLD/TRAIN_CATEGORICAL_METADATA.xlsx")
    train_func = pd.read_csv("/kaggle/input/widsdatathon2025/TRAIN_OLD/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv")
    train_sol = pd.read_excel("/kaggle/input/widsdatathon2025/TRAIN_OLD/TRAINING_SOLUTIONS.xlsx")
    
    # Test verileri
    test_quant = pd.read_excel("/kaggle/input/widsdatathon2025/TEST/TEST_QUANTITATIVE_METADATA.xlsx")
    test_cate = pd.read_excel("/kaggle/input/widsdatathon2025/TEST/TEST_CATEGORICAL.xlsx")
    test_func = pd.read_csv("/kaggle/input/widsdatathon2025/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv")
    
    # Verileri birleştir
    train = train_quant.merge(train_cate, on='participant_id', how='left')
    train = train.merge(train_func, on='participant_id', how='left')
    train = train.merge(train_sol, on='participant_id', how='left').set_index('participant_id')
    
    test = test_quant.merge(test_cate, on='participant_id', how='left')
    test = test.merge(test_func, on='participant_id', how='left').set_index('participant_id')
    
    return train, test

# Verileri yükle
train, test = load_data()
sub = pd.read_excel('/kaggle/input/widsdatathon2025/SAMPLE_SUBMISSION.xlsx')

# Hedefler ve özellikler
targets = ['ADHD_Outcome', 'Sex_F']
features = test.columns

# Log dönüşümü uygulanacak özellikler
log_features = [f for f in features if (train[f] >= 0).all() and scipy.stats.skew(train[f]) > 0]

# Model pipeline'ı
model = MultiOutputClassifier(make_pipeline(
    ColumnTransformer([('imputer', SimpleImputer(), features)],
                      remainder='passthrough',
                      verbose_feature_names_out=False).set_output(transform='pandas'),
    ColumnTransformer([('log', FunctionTransformer(np.log1p), log_features)],
                      remainder='passthrough'),
    MinMaxScaler(),
    RidgeClassifier(alpha=12.65)
))

# 5 katlı çapraz doğrulama
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_scores = {t: [] for t in targets}

X = train.drop(targets, axis=1)
y = train[targets]

# Her kat için eğitim ve değerlendirme
print("5-Fold Cross-Validation Results:")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y['ADHD_Outcome'])):  # ADHD_Outcome için stratify
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Modeli eğit
    model.fit(X_train, y_train)
    
    # Tahmin yap
    y_pred = model.predict(X_val)
    
    # Her hedef için F1 skoru hesapla
    for i, target in enumerate(targets):
        f1 = f1_score(y_val[target], y_pred[:, i], average='micro')
        f1_scores[target].append(f1)
        print(f"Fold {fold+1}, {target} F1 Score: {f1:.4f}")

# Ortalama ve standart sapma
print("\nSummary of 5-Fold CV:")
for target in targets:
    mean_f1 = np.mean(f1_scores[target])
    std_f1 = np.std(f1_scores[target])
    print(f"{target} - Average F1 Score: {mean_f1:.4f} ± {std_f1:.4f}")

# PCA ile model (test tahmini için)
model_with_pca = MultiOutputClassifier(make_pipeline(
    ColumnTransformer([('imputer', SimpleImputer(), features)],
                      remainder='passthrough',
                      verbose_feature_names_out=False).set_output(transform='pandas'),
    ColumnTransformer([('log', FunctionTransformer(np.log1p), log_features)],
                      remainder='passthrough'),
    MinMaxScaler(),
    PCA(n_components=1213),
        RidgeClassifier(alpha=12.65)
))

# Modeli tüm eğitim verisiyle eğit
model_with_pca.fit(train.drop(targets, axis=1), train[targets])

# Test verisi üzerinde tahmin yap
y_pred = model_with_pca.predict(test)

# Submission dosyasını hazırla
sub['ADHD_Outcome'] = y_pred[:, 0]
sub['Sex_F'] = y_pred[:, 1]


submission_df = pd.read_excel('/kaggle/working/submission_multiple_models_xgb.xlsx')

sub_adhd = {}
sub_adhd['ADHD_Outcome'] = submission_df['ADHD_Outcome_Model_1']

sub['ADHD_Outcome'] = sub_adhd['ADHD_Outcome'].values 
# CSV'ye YAZDIRMA (index OLMADAN)
sub.to_csv('submission.csv', index=False)

