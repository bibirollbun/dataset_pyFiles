# 1. Install Libraries
!pip install -U sentence-transformers rank_bm25 faiss-cpu textstat

# 2. Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import re
import faiss
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# 3. Configs
import warnings
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 11, 'figure.dpi': 120})

# 4. Load Spacy
try:
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
except:
    !python -m spacy download en_core_web_sm
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])


# 1. Load Data
df = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/articles.csv', dtype=str)
df.head()



df.info()



print("TÃŒNH TRáº NG Dá»® LIá»†U KHUYáº¾T")
missing_desc = df['detail_desc'].isna() | (df['detail_desc'] == '')
missing_count = missing_desc.sum()
missing_pct = (missing_count / len(df)) * 100

# Váº½ biá»ƒu Ä‘á»“ trÃ²n thá»ƒ hiá»‡n tá»· lá»‡ Missing
plt.figure(figsize=(6, 6))
plt.pie([missing_count, len(df)-missing_count], 
        labels=[f'Missing ({missing_pct:.2f}%)', 'Available'], 
        colors=['#E74C3C', '#2ECC71'], explode=(0.1, 0), autopct='%1.1f%%', startangle=90)
plt.title(f"Tá»· lá»‡ sáº£n pháº©m bá»‹ thiáº¿u mÃ´ táº£ (detail_desc)")
plt.show()

print("---  Váº¼ 0.2: GAP ANALYSIS (KHOáº¢NG TRá»�NG THÃ”NG TIN) ---")

def check_category_in_desc(row):
    # Kiá»ƒm tra xem tÃªn loáº¡i sáº£n pháº©m (VD: Shoes) cÃ³ náº±m trong mÃ´ táº£ khÃ´ng?
    if pd.isna(row['detail_desc']) or pd.isna(row['product_type_name']):
        return False
    return str(row['product_type_name']).lower() in str(row['detail_desc']).lower()

df['has_category_in_desc'] = df.apply(check_category_in_desc, axis=1)
missing_keyword_count = (~df['has_category_in_desc']).sum()

# Váº½ Barplot so sÃ¡nh
plt.figure(figsize=(8, 5))
sns.countplot(x='has_category_in_desc', data=df, palette='viridis')
plt.title("Sáº£n pháº©m cÃ³ chá»©a TÃªn Category trong MÃ´ táº£ khÃ´ng?")
plt.xticks([0, 1], [f'NO (Cáº§n Enrichment)\n{missing_keyword_count} sp', 'YES (Ä�á»§ thÃ´ng tin)'])
plt.ylabel("Sá»‘ lÆ°á»£ng sáº£n pháº©m")
plt.show()



# Cáº¥u hÃ¬nh Ä‘á»ƒ hiá»ƒn thá»‹ full ná»™i dung mÃ´ táº£ (khÃ´ng bá»‹ cáº¯t bá»›t ...)
pd.set_option('display.max_colwidth', None)

# 1. Lá»�c ra cÃ¡c dÃ²ng "cÃ³ váº¥n Ä‘á»�" (Problematic Rows)
# LÃ  nhá»¯ng dÃ²ng mÃ  has_category_in_desc == False (tá»« bÆ°á»›c trÆ°á»›c)
problem_rows = df[df['has_category_in_desc'] == False]

# 2. Chá»�n lá»�c cÃ¡c cá»™t cáº§n hiá»ƒn thá»‹ Ä‘á»ƒ so sÃ¡nh
cols_evidence = ['article_id', 'prod_name','colour_group_name', 'product_type_name', 'detail_desc']

print(f"--- Báº°NG CHá»¨NG THá»°C Táº¾: {len(problem_rows)} sáº£n pháº©m thiáº¿u tá»« khÃ³a phÃ¢n loáº¡i ---")

# 3. Láº¥y máº«u ngáº«u nhiÃªn 5 dÃ²ng Ä‘á»ƒ kiá»ƒm chá»©ng
# random_state=42 Ä‘á»ƒ Ä‘áº£m báº£o láº§n nÃ o cháº¡y cÅ©ng ra káº¿t quáº£ giá»‘ng nhau (dá»… viáº¿t bÃ¡o cÃ¡o)
sample_evidence = problem_rows[cols_evidence].sample(5, random_state=42)
display(sample_evidence)

print("\n--- KIá»‚M TRA Cá»¤ THá»‚ NHÃ“M GIÃ€Y (SHOES) ---")
shoes_issues = problem_rows[
    problem_rows['product_type_name'].astype(str).str.contains('Shoe|Sneaker', case=False)
]

if not shoes_issues.empty:
    display(shoes_issues[cols_evidence].head(3))
else:
    print("NhÃ³m giÃ y dá»¯ liá»‡u tá»‘t, khÃ´ng tÃ¬m tháº¥y lá»—i.")


# 1. Kiá»ƒm tra trÃ¹ng láº·p hoÃ n toÃ n (Giá»‘ng nhau y há»‡t á»Ÿ Táº¤T Cáº¢ cÃ¡c cá»™t)
num_full_duplicates = df.duplicated().sum()
print(f"Tá»•ng sá»‘ dÃ²ng trÃ¹ng láº·p 100% (Full Duplicates): {num_full_duplicates}")

if num_full_duplicates > 0:
    print("Máº«u cÃ¡c dÃ²ng trÃ¹ng láº·p:")
    # keep=False Ä‘á»ƒ hiá»‡n táº¥t cáº£ cÃ¡c báº£n sao ra
    display(df[df.duplicated(keep=False)].sort_values(by=['article_id']).head())
else:
    print("=> Dá»¯ liá»‡u sáº¡ch, khÃ´ng cÃ³ dÃ²ng nÃ o trÃ¹ng láº·p hoÃ n toÃ n.")

print("-" * 30)

# 2. Kiá»ƒm tra trÃ¹ng láº·p Article ID (Quan trá»�ng hÆ¡n: 1 ID cÃ³ bá»‹ láº·p láº¡i 2 láº§n khÃ´ng?)
num_id_duplicates = df.duplicated(subset=['article_id']).sum()
print(f"Tá»•ng sá»‘ ID bá»‹ trÃ¹ng (Duplicate Primary Key): {num_id_duplicates}")

if num_id_duplicates > 0:
    print("Cáº£nh bÃ¡o: CÃ³ ID xuáº¥t hiá»‡n nhiá»�u láº§n trong dataset!")
    display(df[df.duplicated(subset=['article_id'], keep=False)].sort_values(by=['article_id']).head())
else:
    print("=> ID lÃ  duy nháº¥t (Unique), cáº¥u trÃºc chuáº©n.")


# --- Váº¼ 1: Text Length Distribution ---
print("ğŸ“Š [PLOT 1/7] Text Length Distribution")
raw_lens = df['detail_desc'].apply(lambda x: len(str(x).split()))

plt.figure(figsize=(10, 4))
sns.histplot(raw_lens, bins=40, kde=True, color='#2E86C1')
plt.axvline(raw_lens.mean(), color='red', linestyle='--', label=f'Mean: {raw_lens.mean():.1f}')
plt.title("PhÃ¢n bá»‘ Ä‘á»™ dÃ i mÃ´ táº£ (Tokens)")
plt.xlabel("Sá»‘ tá»«"); plt.ylabel("Sá»‘ lÆ°á»£ng sáº£n pháº©m")
plt.legend()
plt.show()
print("ğŸ“Š [PLOT 2/7] POS Distribution (Raw)")

sample_docs = (
    df['detail_desc']
    .dropna()                 #
    .astype(str)              
    .sample(n=min(2000, df['detail_desc'].notna().sum()), random_state=42)
    .tolist()
)

pos_counter = Counter()
for doc in nlp.pipe(sample_docs, batch_size=64):
    pos_counter.update([t.pos_ for t in doc])

pos_df = (pd.DataFrame.from_dict(pos_counter, orient='index', columns=['Count'])
          .sort_values('Count', ascending=False)
          .head(10))

plt.figure(figsize=(10, 4))
sns.barplot(x=pos_df['Count'], y=pos_df.index, palette='magma')
plt.title("Top 10 Tá»« loáº¡i (POS Tags)")
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

# 1. Lá»�c ra cÃ¡c sáº£n pháº©m thuá»™c cÃ¡c nhÃ³m há»�a tiáº¿t quan trá»�ng
target_patterns = ['Stripe', 'Check', 'Denim']
df_pattern = df[df['graphical_appearance_name'].isin(target_patterns)].copy()

# 2. Táº¡o cá»™t kiá»ƒm tra: Tá»« khÃ³a há»�a tiáº¿t cÃ³ náº±m trong TÃªn hoáº·c MÃ´ táº£ khÃ´ng?
# Logic: Náº¿u 'Stripe' xuáº¥t hiá»‡n trong 'prod_name' hoáº·c 'detail_desc' -> True (DÆ° thá»«a)
def check_redundancy(row):
    text = (str(row['prod_name']) + " " + str(row['detail_desc'])).lower()
    pattern = str(row['graphical_appearance_name']).lower()
    return pattern in text

df_pattern['is_redundant'] = df_pattern.apply(check_redundancy, axis=1)

# 3. Váº¼ BIá»‚U Ä�á»’ (VISUALIZATION)
plt.figure(figsize=(10, 6))
# Váº½ biá»ƒu Ä‘á»“ cá»™t chá»“ng hoáº·c nhÃ³m Ä‘á»ƒ so sÃ¡nh
ax = sns.countplot(x='graphical_appearance_name', hue='is_redundant', data=df_pattern, palette='viridis')

plt.title('Kiá»ƒm tra Ä‘á»™ dÆ° thá»«a thÃ´ng tin: Há»�a tiáº¿t cÃ³ sáºµn trong MÃ´ táº£ khÃ´ng?')
plt.xlabel('Loáº¡i há»�a tiáº¿t')
plt.ylabel('Sá»‘ lÆ°á»£ng sáº£n pháº©m')
plt.legend(title='TrÃ¹ng láº·p thÃ´ng tin', labels=['False (ChÆ°a cÃ³ - Cáº§n giá»¯)', 'True (Ä�Ã£ cÃ³ - DÆ° thá»«a)'])

# ThÃªm sá»‘ liá»‡u lÃªn Ä‘áº§u cá»™t cho ngáº§u
for container in ax.containers:
    ax.bar_label(container)

plt.show()

# 4. IN RA Báº°NG CHá»¨NG (EVIDENCE ROWS)
cols_show = ['article_id', 'prod_name', 'graphical_appearance_name', 'detail_desc']

print("\n=== CASE 1: DÆ¯ THá»ªA (REDUNDANT) - CÃ³ thá»ƒ loáº¡i bá»� cá»™t Há»�a tiáº¿t ===")
print("VÃ­ dá»¥ cÃ¡c dÃ²ng mÃ  tá»« khÃ³a há»�a tiáº¿t Ä�Ãƒ XUáº¤T HIá»†N trong mÃ´ táº£:")
display(df_pattern[df_pattern['is_redundant'] == True][cols_show].head(3))

print("\n=== CASE 2: KHÃ”NG TRÃ™NG (UNIQUE) - Máº¥t thÃ´ng tin náº¿u loáº¡i bá»� ===")
print("VÃ­ dá»¥ cÃ¡c dÃ²ng mÃ  mÃ´ táº£ KHÃ”NG Há»€ NHáº®C Ä�áº¾N há»�a tiáº¿t:")
display(df_pattern[df_pattern['is_redundant'] == False][cols_show].head(3))


import matplotlib.pyplot as plt
import seaborn as sns

# 1. Thá»‘ng kÃª táº§n suáº¥t
graphical_counts = df['graphical_appearance_name'].value_counts().reset_index()
graphical_counts.columns = ['Pattern', 'Count']

# 2. PhÃ¢n loáº¡i "Cháº¥t lÆ°á»£ng tá»« khÃ³a" (Manual Labeling for Analysis)
# Nhá»¯ng tá»« nÃ y quÃ¡ chung chung, ngÆ°á»�i dÃ¹ng Ã­t khi search chÃ­nh xÃ¡c tá»« nÃ y
generic_terms = ['Solid', 'All over pattern', 'Melange', 'Transparent', 'Treatment', 'Other structure']

def classify_term(term):
    if term in generic_terms:
        return 'Generic/Noise (Chung chung)'
    return 'Specific (Cá»¥ thá»ƒ)'

graphical_counts['Type'] = graphical_counts['Pattern'].apply(classify_term)

# TÃ­nh tá»•ng tá»· lá»‡
total_rows = len(df)
noise_rows = graphical_counts[graphical_counts['Type'] == 'Generic/Noise (Chung chung)']['Count'].sum()
noise_pct = (noise_rows / total_rows) * 100

print(f"Tá»•ng sá»‘ dÃ²ng: {total_rows}")
print(f"Sá»‘ dÃ²ng chá»©a tá»« khÃ³a chung chung (Noise): {noise_rows}")
print(f"Tá»· lá»‡ Nhiá»…u thÃ´ng tin: {noise_pct:.2f}%")

# 3. Váº½ biá»ƒu Ä‘á»“ Top 10
plt.figure(figsize=(12, 6))
# Láº¥y top 10 Ä‘á»ƒ váº½
top_10 = graphical_counts.head(10)
colors = ['#E74C3C' if x == 'Generic/Noise (Chung chung)' else '#2ECC71' for x in top_10['Type']]

barplot = sns.barplot(x='Count', y='Pattern', data=top_10, palette=colors)

# ThÃªm chÃº thÃ­ch
plt.title(f'PhÃ¢n tÃ­ch cháº¥t lÆ°á»£ng tá»« khÃ³a cá»™t Há»�a tiáº¿t (Noise Rate: {noise_pct:.1f}%)')
plt.xlabel('Sá»‘ lÆ°á»£ng sáº£n pháº©m')
plt.axvline(x=len(df)*0.5, color='gray', linestyle='--', label='50% Dá»¯ liá»‡u')

# Táº¡o legend thá»§ cÃ´ng
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#E74C3C', label='Noise (Tá»« chung chung - NÃªn bá»�)'),
                   Patch(facecolor='#2ECC71', label='Specific (Tá»« cá»¥ thá»ƒ - CÃ³ giÃ¡ trá»‹)')]
plt.legend(handles=legend_elements)

plt.show()



def construct_smart_text(row):
    color = str(row['colour_group_name']).strip()
    p_type = str(row['product_type_name']).strip()
    name = str(row['prod_name']).strip()
    desc = str(row['detail_desc']).strip()
    
    # A. Khá»­ trÃ¹ng láº·p (Deduplication)
    # Náº¿u TÃªn Ä‘Ã£ chá»©a Loáº¡i (VD: "Vito Derby Shoe" chá»©a "Shoe") -> áº¨n Loáº¡i Ä‘i
    show_type = True
    if p_type.lower() in name.lower(): show_type = False
    elif p_type.lower().rstrip('s') in name.lower().split(): show_type = False
    display_type = p_type if show_type else ""

    # B. LÃ m sáº¡ch mÃ´ táº£ (Safe Description Cutting)
    # Cáº¯t bá»� pháº§n Ä‘áº§u mÃ´ táº£ náº¿u nÃ³ láº·p láº¡i tÃªn sáº£n pháº©m, NHÆ¯NG chá»‰ cáº¯t náº¿u pháº§n cÃ²n láº¡i Ä‘á»§ dÃ i
    if desc.lower().startswith(name.lower()):
        potential_desc = desc[len(name):].strip().lstrip('.,- ')
        if len(potential_desc) > 15: desc = potential_desc
    elif desc.lower().startswith(p_type.lower()):
        potential_desc = desc[len(p_type):].strip().lstrip('.,- ')
        if len(potential_desc) > 15: desc = potential_desc
    
    desc = desc.lstrip('.,- ')
    
    # C. GhÃ©p chuá»—i (Æ¯u tiÃªn SBERT Ä‘á»�c tá»« trÃ¡i sang pháº£i)
    components = [color, display_type, name, desc]
    # Lá»�c bá»� rá»—ng vÃ  ghÃ©p láº¡i
    clean_text = " ".join([c for c in components if c and c.lower() != 'nan'])
    return re.sub(r'\s+', ' ', clean_text).strip()

print(" Ä�ang xÃ¢y dá»±ng Smart Text (Enrichment)...")
df['rich_source'] = df.apply(construct_smart_text, axis=1)

print(" Ä�Ã£ xá»­ lÃ½ xong! VÃ­ dá»¥ máº«u:")
print(f"Gá»‘c: {df.iloc[0]['prod_name']} | {df.iloc[0]['detail_desc']}")
print(f"Smart: {df.iloc[0]['rich_source']}")


class TextPreprocessor:
    def __init__(self, nlp_model):
        self.nlp = nlp_model
        
    def process_lexical(self, texts):
        """Pipeline 1: Cho BM25 (Giá»¯ Exact Term, No Lemma, Fix Regex %)"""
        # 1. Regex Clean: Giá»¯ chá»¯, sá»‘, -, %
        cleaned_text = [re.sub(r"[^a-z0-9\s\-\%]", " ", str(t).lower()) for t in texts]
        cleaned_text = [re.sub(r"\s+", " ", t).strip() for t in cleaned_text]
        
        results = []
        
        # LOGIC CHá»ˆ GIá»® VERB Dáº NG TÃ�NH Tá»ª (VBG/VBN)
        allowed_pos = {'NOUN', 'ADJ', 'PROPN', 'NUM'} 
        allowed_verb_tags = {'VBG', 'VBN'} 
        
        for doc in self.nlp.pipe(cleaned_text, batch_size=2000):
            tokens = []
            for t in doc:
                # Bá»� Stopword/Punctuation
                if t.is_stop or t.is_punct or len(t.text) < 2: continue
                
                # Logic lá»�c: Giá»¯ náº¿u lÃ  POS cho phÃ©p HOáº¶C lÃ  Verb Ä‘áº·c biá»‡t
                is_valid_pos = t.pos_ in allowed_pos
                is_special_verb = (t.pos_ == 'VERB' and t.tag_ in allowed_verb_tags)
                
                if is_valid_pos or is_special_verb:
                    tokens.append(t.text) # DÃ¹ng .text Ä‘á»ƒ giá»¯ nguyÃªn dáº¡ng tá»«
                    
            results.append(" ".join(tokens))
        return results
        
    def process_semantic(self, df):
        return df['rich_source'].tolist() 

print(" Ä�ang cháº¡y Preprocessing...")
prep = TextPreprocessor(nlp)

# Pipeline 1: Lexical cháº¡y trÃªn RICH SOURCE
df['clean_lexical'] = prep.process_lexical(df['rich_source'])

print(" Preprocessing hoÃ n táº¥t!")
print(f"ğŸ”¹ Lexical Sample: {df['clean_lexical'].iloc[0]}")


from collections import Counter

# --- Váº¼ 3: Vocabulary Size ---
print("ğŸ“Š [PLOT 3/7] Vocabulary Size Comparison")

# --- FIX Lá»–I Táº I Ä�Ã‚Y: ThÃªm .astype(str) Ä‘á»ƒ Ä‘áº£m báº£o khÃ´ng bá»‹ lá»—i float ---
vocab_raw = set(" ".join(df['detail_desc'].astype(str)).lower().split())
vocab_clean = set(" ".join(df['clean_lexical'].astype(str)).split())

plt.figure(figsize=(6, 4))
sns.barplot(x=['Raw Vocab', 'Clean Lexical Vocab'], y=[len(vocab_raw), len(vocab_clean)], palette='viridis')
plt.title(f"Vocab size trÆ°á»›c vÃ  sau: {len(vocab_raw)} -> {len(vocab_clean)}")
plt.ylabel("Sá»‘ lÆ°á»£ng tá»« vá»±ng (Unique)")
for i, v in enumerate([len(vocab_raw), len(vocab_clean)]):
    plt.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
plt.show()

# --- Váº¼ 4: Zipf's Law ---
print("ğŸ“Š [PLOT 4/7] Token Frequency (Zipf's Curve)")
# CÅ©ng thÃªm .astype(str) cho cháº¯c Äƒn
all_tokens = " ".join(df['clean_lexical'].astype(str)).split()
counts = Counter(all_tokens).most_common()
freqs = [x[1] for x in counts]
ranks = range(1, len(freqs)+1)

plt.figure(figsize=(8, 5))
plt.loglog(ranks, freqs, marker='.', linestyle='none', alpha=0.3, color='purple')
plt.title("Zipf's Law Check (Log-Log Scale)")
plt.xlabel("Rank"); plt.ylabel("Frequency")
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.show()


# 1. Train BM25
print(" [1/3] Training BM25 Index...")
tokenized_lexical = [t.split() for t in df['clean_lexical']]
bm25 = BM25Okapi(tokenized_lexical)

# 2. Encode SBERT
print(" [2/3] Encoding SBERT Embeddings...")
# DÃ¹ng Smart Text Ä‘á»ƒ encode
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = sbert_model.encode(df['rich_source'].tolist(), batch_size=64, show_progress_bar=True)
embeddings = embeddings.astype('float32')

# 3. Clustering (KMeans)
print(" [3/3] Clustering (For Visualization)...")
kmeans = KMeans(n_clusters=8, random_state=42).fit(embeddings)
df['cluster'] = kmeans.labels_


print("â�³ Running PCA...")
pca = PCA(n_components=2)
reduced_vecs = pca.fit_transform(embeddings)

# --- Váº¼ 5: Embedding Projection ---
print("ğŸ“Š [PLOT 5/7] Semantic Structure")
plt.figure(figsize=(10, 6))
plt.scatter(reduced_vecs[:,0], reduced_vecs[:,1], s=1, alpha=0.3, c='gray')
plt.title("KhÃ´ng gian ngá»¯ nghÄ©a SBERT (PCA Projection)")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.show()

# --- Váº¼ 6: Cluster Visualization ---
print("ğŸ“Š [PLOT 6/7] Pattern Discovery (Clustering)")
plt.figure(figsize=(10, 6))
scatter = plt.scatter(reduced_vecs[:,0], reduced_vecs[:,1], c=df['cluster'], cmap='tab10', s=3, alpha=0.7)
plt.colorbar(scatter, label='Cluster ID')
plt.title("PhÃ¢n cá»¥m sáº£n pháº©m dá»±a trÃªn ngá»¯ nghÄ©a")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.show()


class HybridSearchEngine:
    def __init__(self, df, bm25_model, sbert_model, embeddings):
        self.df = df.reset_index(drop=True)
        self.bm25 = bm25_model
        self.sbert_model = sbert_model
        self.embeddings = embeddings
        
        # FAISS Index
        faiss.normalize_L2(self.embeddings)
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)
        
        # --- PHRASE-BASED SYNONYMS (OPTIMIZED DOMAIN KNOWLEDGE) ---
        # Ä�Ã£ loáº¡i bá»� tá»« "shoes" chung chung Ä‘á»ƒ trÃ¡nh boost nháº§m giÃ y tÃ¢y (Derby)
        self.phrase_synonyms = {
            'running shoes': ['trainers', 'sneakers', 'runners', 'athletic footwear'],
            'running shoe': ['trainers', 'sneakers', 'runners'],
            'gym shoes': ['trainers', 'sneakers'],
            'joggers': ['sweatpants', 'track pants'], 
            'denim jeans': ['blue jeans', 'denim'],
            'hoodie': ['sweatshirt', 'hooded'],
            'summer dress': ['sundress', 'floral dress']
        }
        print("âœ… Engine Ready: Smart Text + Optimized Phrase Expansion.")

    def _min_max_normalize(self, scores):
        min_s, max_s = np.min(scores), np.max(scores)
        if max_s - min_s == 0: return np.zeros_like(scores)
        return (scores - min_s) / (max_s - min_s)
    
    def _expand_query_phrase(self, query):
        """Má»Ÿ rá»™ng query dá»±a trÃªn cá»¥m tá»«, trÃ¡nh nhiá»…u"""
        query_lower = str(query).lower()
        expansion_terms = []
        for phrase, synonyms in self.phrase_synonyms.items():
            if phrase in query_lower:
                expansion_terms.extend(synonyms)
        if expansion_terms:
            # Chá»‰ thÃªm tá»« Ä‘áº·c thÃ¹ (trainers, sneakers), khÃ´ng thÃªm tá»« gá»‘c trÃ¡nh láº·p
            return query_lower + " " + " ".join(list(set(expansion_terms)))
        return query_lower
    
    def search(self, query, top_k=10, alpha=0.5):
        # 1. Expand Query
        expanded_q = self._expand_query_phrase(query)
        
        # 2. Lexical Search (DÃ¹ng Expanded Query Ä‘á»ƒ báº¯t keyword)
        q_lexical = re.sub(r"[^a-z0-9\s\-\%]", " ", expanded_q).split()
        bm25_raw = self.bm25.get_scores(q_lexical)
        bm25_norm = self._min_max_normalize(bm25_raw)
        
        # 3. Semantic Search (DÃ¹ng Original Query Ä‘á»ƒ giá»¯ ngá»¯ cáº£nh chÃ­nh xÃ¡c)
        q_vec = self.sbert_model.encode([query]).astype('float32')
        faiss.normalize_L2(q_vec)
        D, I = self.index.search(q_vec, len(self.df))
        
        sbert_raw = np.zeros(len(self.df))
        sbert_raw[I[0]] = D[0]
        sbert_norm = self._min_max_normalize(sbert_raw)
        
        # 4. Fusion
        final_scores = (alpha * bm25_norm) + ((1 - alpha) * sbert_norm)
        
        # 5. Result
        top_indices = np.argsort(final_scores)[::-1][:top_k]
        results = self.df.iloc[top_indices][['prod_name', 'product_type_name', 'colour_group_name', 'rich_source']].copy()
        
        results['score'] = final_scores[top_indices]
        results['expanded_query'] = expanded_q
        results['bm25'] = bm25_norm[top_indices]
        results['sbert'] = sbert_norm[top_indices]
        
        return results

engine = HybridSearchEngine(df, bm25, sbert_model, embeddings)


# --- TEST SEARCH ---
query = "Black running shoes"
print(f"ğŸ”� TEST QUERY: '{query}'")
print("\n--- Alpha = 0.5 (Hybrid) ---")
display(engine.search(query, top_k=5, alpha=0.5))

# --- Váº¼ 7: Query Visualization ---
print("\nğŸ“Š [PLOT 7/7] Query vs Results Visualization")
def plot_query(query_text):
    results = engine.search(query_text, top_k=5, alpha=0.5)
    res_indices = results.index.tolist()
    
    q_vec = sbert_model.encode([query_text])
    res_vecs = embeddings[res_indices]
    
    combined = np.vstack([q_vec, res_vecs])
    pca_local = PCA(n_components=2).fit_transform(combined)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_local[0,0], pca_local[0,1], c='red', s=300, marker='*', label='Query')
    plt.text(pca_local[0,0], pca_local[0,1]+0.02, "QUERY", color='red', fontweight='bold')
    plt.scatter(pca_local[1:,0], pca_local[1:,1], c='blue', s=100, label='Results')
    
    for i in range(1, len(pca_local)):
        plt.plot([pca_local[0,0], pca_local[i,0]], [pca_local[0,1], pca_local[i,1]], 'k--', alpha=0.3)
        plt.text(pca_local[i,0], pca_local[i,1], f"#{i}", fontsize=9)
        
    plt.title(f"Query: '{query_text}'")
    plt.legend(); plt.show()

plot_query(query)

def get_related_products(self, article_id, top_k=5):
        """Gá»£i Ã½ sáº£n pháº©m tÆ°Æ¡ng tá»± dá»±a trÃªn vector"""
        try:
            # TÃ¬m index cá»§a sáº£n pháº©m trong dataframe
            idx = self.df[self.df['article_id'].astype(str) == str(article_id)].index[0]
            
            # Láº¥y vector cá»§a nÃ³
            target_vec = self.embeddings[idx].reshape(1, -1).astype('float32')
            faiss.normalize_L2(target_vec)
            
            # Search (Láº¥y top_k + 1 vÃ¬ káº¿t quáº£ Ä‘áº§u tiÃªn lÃ  chÃ­nh nÃ³)
            D, I = self.index.search(target_vec, top_k + 1)
            
            # Bá»� qua káº¿t quáº£ Ä‘áº§u tiÃªn (chÃ­nh nÃ³)
            related_indices = I[0][1:]
            related_products = self.df.iloc[related_indices].copy()
            related_products['score'] = D[0][1:]
            
            return related_products
        except:
            return None






import pickle
import os
import shutil

# 1. Táº¡o thÆ° má»¥c chá»©a model
MODEL_DIR = 'models_best'
if os.path.exists(MODEL_DIR):
    shutil.rmtree(MODEL_DIR) # XÃ³a cÅ© náº¿u cÃ³ Ä‘á»ƒ ghi má»›i
os.makedirs(MODEL_DIR)

print(f"ğŸ’¾ Ä�ang lÆ°u Artifacts vÃ o thÆ° má»¥c '{MODEL_DIR}'...")

# 2. LÆ°u DataFrame (Chá»©a Smart Text 'rich_source' cá»±c quan trá»�ng)
# Chá»‰ lÆ°u cÃ¡c cá»™t cáº§n thiáº¿t Ä‘á»ƒ nháº¹ file
cols_to_keep = ['article_id', 'prod_name', 'product_type_name', 'colour_group_name', 'detail_desc', 'rich_source']
df_export = df[cols_to_keep].copy()

with open(f'{MODEL_DIR}/df_products.pkl', 'wb') as f:
    pickle.dump(df_export, f)
print("âœ… Ä�Ã£ lÆ°u DataFrame (df_products.pkl)")

# 3. LÆ°u BM25 Model
with open(f'{MODEL_DIR}/bm25_model.pkl', 'wb') as f:
    pickle.dump(bm25, f)
print("âœ… Ä�Ã£ lÆ°u BM25 Model (bm25_model.pkl)")

# 4. LÆ°u SBERT Embeddings (Náº·ng nháº¥t, lÆ°u dáº¡ng numpy)
np.save(f'{MODEL_DIR}/sbert_embeddings.npy', embeddings)
print("âœ… Ä�Ã£ lÆ°u Embeddings (sbert_embeddings.npy)")

# 5. NÃ©n láº¡i thÃ nh ZIP Ä‘á»ƒ dá»… táº£i vá»� mÃ¡y (Náº¿u cáº§n)
shutil.make_archive(MODEL_DIR, 'zip', MODEL_DIR)
print(f"táº£i file '{MODEL_DIR}.zip' vá»� mÃ¡y.")


from IPython.display import FileLink
import os

# Ä�áº£m báº£o file zip Ä‘Ã£ Ä‘Æ°á»£c táº¡o
zip_file = 'hm_images_50k_optimized.zip'

if os.path.exists(zip_file):
    print(f"ğŸ‘‡ Báº¥m vÃ o dÃ²ng chá»¯ xanh bÃªn dÆ°á»›i Ä‘á»ƒ táº£i '{zip_file}' vá»� mÃ¡y:")
    display(FileLink(zip_file))
else:
    print("âš ï¸� ChÆ°a tháº¥y file zip. Bro cháº¡y cell nÃ©n file á»Ÿ trÃªn chÆ°a?")


import os
import pandas as pd
import zipfile
from tqdm import tqdm
from PIL import Image

# 1. Ä�á»�c dá»¯ liá»‡u (Sá»­a Ä‘Æ°á»�ng dáº«n file pkl cá»§a bro trÃªn Kaggle)
df = pd.read_pickle('/kaggle/input/df-products/pytorch/default/1/df_products.pkl')
sample_ids = df['article_id'].head(100000).astype(str).str.zfill(10).tolist()

base_path = '/kaggle/input/h-and-m-personalized-fashion-recommendations/images'
output_dir = '/kaggle/working/static_images'
os.makedirs(output_dir, exist_ok=True)

print("ğŸš€ Ä�ang nÃ©n áº£nh siÃªu nhá»�...")
count = 0
for aid in tqdm(sample_ids):
    sub_folder = aid[:3]
    img_path = os.path.join(base_path, sub_folder, f"{aid}.jpg")
    
    if os.path.exists(img_path):
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                img.thumbnail((200, 200)) # Cho áº£nh nhá»� láº¡i
                # LÆ°u áº£nh vá»›i cháº¥t lÆ°á»£ng tháº¥p Ä‘á»ƒ tá»‘i Æ°u dung lÆ°á»£ng
                img.save(os.path.join(output_dir, f"{aid}.jpg"), optimize=True, quality=60)
                count += 1
        except: continue
    if count >= 10000: break

# 2. NÃ©n thÃ nh file ZIP
zip_name = 'hm_10k_compressed.zip'
with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for file in os.listdir(output_dir):
        zipf.write(os.path.join(output_dir, file), arcname=file)

print(f"âœ¨ XONG! Táº£i file nÃ y vá»� vÃ  up lÃªn HF: {zip_name} (Sá»‘ lÆ°á»£ng: {count})")


import os
import pandas as pd
import zipfile
from tqdm import tqdm
from PIL import Image
import shutil

# 1. Ä�á»�c dá»¯ liá»‡u vÃ  láº¥y danh sÃ¡ch ID
df =  pd.read_pickle('/kaggle/input/df-products/pytorch/default/1/df_products.pkl')
# Láº¥y 50,000 áº£nh Ä‘áº§u tiÃªn (hoáº·c bá»� .head() náº¿u muá»‘n láº¥y háº¿t - sáº½ lÃ¢u hÆ¡n)
all_ids = df['article_id'].astype(str).str.zfill(10).unique().tolist()
target_ids = all_ids[:100000] 

base_path = '/kaggle/input/h-and-m-personalized-fashion-recommendations/images'
temp_dir = '/kaggle/working/temp_images_compressed'
os.makedirs(temp_dir, exist_ok=True)

print(f"ğŸš€ Ä�ang xá»­ lÃ½ vÃ  nÃ©n {len(target_ids)} áº£nh...")
count = 0
for aid in tqdm(target_ids):
    sub_folder = aid[:3]
    img_path = os.path.join(base_path, sub_folder, f"{aid}.jpg")
    
    if os.path.exists(img_path):
        try:
            # NÃ©n áº£nh nhá»� láº¡i Ä‘á»ƒ tiáº¿t kiá»‡m dung lÆ°á»£ng
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                img.thumbnail((250, 250)) # KÃ­ch thÆ°á»›c Ä‘á»§ xem demo
                # LÆ°u vÃ o thÆ° má»¥c táº¡m, tÃªn file chá»‰ lÃ  ID.jpg
                img.save(os.path.join(temp_dir, f"{aid}.jpg"), optimize=True, quality=65)
                count += 1
        except: continue

# 2. NÃ©n thÆ° má»¥c táº¡m thÃ nh file ZIP (KhÃ´ng nÃ©n folder cha)
zip_name = 'hm_images_50k_optimized.zip'
print(f"ğŸ“¦ Ä�ang Ä‘Ã³ng gÃ³i {count} áº£nh vÃ o file ZIP...")
with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for file in os.listdir(temp_dir):
        zipf.write(os.path.join(temp_dir, file), arcname=file)

# Dá»�n dáº¹p
shutil.rmtree(temp_dir)
print(f"âœ¨ XONG! Táº£i file nÃ y vá»�: {zip_name}")


# 1. Load model (Ä‘Ã£ load sáºµn trong code chÃ­nh rá»“i)
# model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Cháº¡y thá»­ 1 query
query = "Running shoes"
query_vector = sbert_model.encode([query])

# 3. In ra mÃ n hÃ¬nh
print("KÃ­ch thÆ°á»›c vector truy váº¥n:", query_vector.shape)
print("Dá»¯ liá»‡u vector (5 giÃ¡ trá»‹ Ä‘áº§u):", query_vector[0][:5])
print("-" * 20)
print("KÃ­ch thÆ°á»›c ma tráº­n embeddings toÃ n bá»™:", embeddings.shape)


# 1. Xá»­ lÃ½ query (Tokenize)
query = "Running shoes"
tokenized_query = query.lower().split() # Hoáº·c dÃ¹ng hÃ m preprocess cá»§a báº¡n

# 2. Láº¥y Ä‘iá»ƒm sá»‘ BM25
bm25_scores = bm25.get_scores(tokenized_query)

# 3. In ra
print("Sá»‘ lÆ°á»£ng Ä‘iá»ƒm sá»‘ tráº£ vá»�:", len(bm25_scores))
print("Ä�iá»ƒm sá»‘ cá»§a 10 sáº£n pháº©m Ä‘áº§u tiÃªn:", bm25_scores[:10])
print("Ä�iá»ƒm cao nháº¥t:", max(bm25_scores))


# CÃ i Ä‘áº·t pyspark náº¿u chÆ°a cÃ³
!pip install pyspark


import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

# ---------------------------------------------------------
# BÆ¯á»šC 1: KHá»�I Táº O SPARK SESSION (LOCAL MODE)
# ---------------------------------------------------------
# ---------------------------------------------------------
# BÆ¯á»šC 1: KHá»�I Táº O SPARK SESSION (Cáº¤U HÃŒNH CHO BIG DATA)
# ---------------------------------------------------------
spark = SparkSession.builder \
    .appName("HM_Recommendation_ALS_Local") \
    .config("spark.driver.memory", "14g") \
    .config("spark.executor.memory", "14g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.kryoserializer.buffer.max", "1g") \
    .config("spark.driver.maxResultSize", "4g") \
    .master("local[*]") \
    .getOrCreate()

print(f"Spark Version: {spark.version}")
print("Ä�Ã£ cáº¥u hÃ¬nh xong Kryo Buffer 1GB")

# ---------------------------------------------------------
# BÆ¯á»šC 2: Ä�á»ŒC VÃ€ TIá»€N Xá»¬ LÃ� Dá»® LIá»†U (DATA PREPROCESSING)
# ---------------------------------------------------------
# Ä�Æ°á»�ng dáº«n dataset trÃªn Kaggle
dataset_path = "/kaggle/input/h-and-m-personalized-fashion-recommendations/transactions_train.csv"

# Ä�á»�c dá»¯ liá»‡u. InferSchema=True Ä‘á»ƒ Spark tá»± hiá»ƒu kiá»ƒu dá»¯ liá»‡u (tá»‘n chÃºt thá»�i gian nhÆ°ng tiá»‡n)
# LÆ°u Ã½: H&M dataset ráº¥t lá»›n (>30 triá»‡u dÃ²ng). Ä�á»ƒ demo cháº¡y nhanh, tháº§y sáº½ láº¥y máº«u (sampling)
# hoáº·c lá»�c dá»¯ liá»‡u gáº§n nháº¥t. á»� Ä‘Ã¢y tháº§y láº¥y 1 triá»‡u dÃ²ng Ä‘áº§u tiÃªn Ä‘á»ƒ em test pipeline trÆ°á»›c.
df_raw = spark.read.csv(dataset_path, header=True, inferSchema=True).limit(100000)

# Chá»�n cÃ¡c cá»™t cáº§n thiáº¿t
df = df_raw.select("t_dat", "customer_id", "article_id")

# --- Feature Engineering: Táº¡o cá»™t 'rating' ---
# H&M lÃ  bÃ i toÃ¡n Implicit Feedback (ngÆ°á»�i dÃ¹ng khÃ´ng cháº¥m 1-5 sao).
# Ta giáº£ Ä‘á»‹nh: Sá»‘ láº§n mua 1 sáº£n pháº©m = Má»©c Ä‘á»™ quan tÃ¢m (Rating).
df_rating = df.groupBy("customer_id", "article_id") \
    .agg(count("article_id").alias("rating"))

# --- StringIndexer ---
# ALS cá»§a Spark yÃªu cáº§u input lÃ  sá»‘ nguyÃªn (integer), nhÆ°ng customer_id vÃ  article_id cá»§a H&M lÃ  chuá»—i hash.
# Ta pháº£i map chÃºng sang index sá»‘ (0, 1, 2...).

# Indexing cho User
user_indexer = StringIndexer(inputCol="customer_id", outputCol="user_idx")
print(user_indexer)
user_indexer_model = user_indexer.fit(df_rating)
print("user index model: ", user_indexer_model)
df_indexed = user_indexer_model.transform(df_rating)

# Indexing cho Item
item_indexer = StringIndexer(inputCol="article_id", outputCol="item_idx")
item_indexer_model = item_indexer.fit(df_indexed)
df_final = item_indexer_model.transform(df_indexed)

# Cache dá»¯ liá»‡u vÃ o RAM Ä‘á»ƒ training nhanh hÆ¡n
# df_final.cache()

print("Dá»¯ liá»‡u Ä‘Ã£ sáºµn sÃ ng cho ALS:")
df_final.show(5)

# ---------------------------------------------------------
# BÆ¯á»šC 3: XÃ‚Y Dá»°NG VÃ€ HUáº¤N LUYá»†N MÃ” HÃŒNH (MODELING)
# ---------------------------------------------------------
# Chia táº­p train/test (80/20)
(training, test) = df_final.randomSplit([0.8, 0.2])

# Cáº¥u hÃ¬nh ALS
# - rank: Sá»‘ lÆ°á»£ng factors tiá»�m áº©n (User Factors & Item Factors).
# - maxIter: Sá»‘ vÃ²ng láº·p tá»‘i Æ°u hÃ³a.
# - regParam: Tham sá»‘ regularization Ä‘á»ƒ trÃ¡nh overfitting.
# - implicitPrefs=True: Ráº¤T QUAN TRá»ŒNG. BÃ¡o cho Spark biáº¿t Ä‘Ã¢y lÃ  dá»¯ liá»‡u hÃ nh vi (mua/khÃ´ng mua), khÃ´ng pháº£i cháº¥m Ä‘iá»ƒm.
# - coldStartStrategy="drop": Bá»� qua cÃ¡c user/item chÆ°a tá»«ng xuáº¥t hiá»‡n trong táº­p train Ä‘á»ƒ trÃ¡nh lá»—i NaN khi predict.

als = ALS(
    maxIter=5, 
    regParam=0.01, 
    userCol="user_idx", 
    itemCol="item_idx", 
    ratingCol="rating",
    implicitPrefs=True,
    coldStartStrategy="drop" 
)

print("Ä�ang huáº¥n luyá»‡n mÃ´ hÃ¬nh ALS...")
model = als.fit(training)
print("Huáº¥n luyá»‡n hoÃ n táº¥t!")

# ---------------------------------------------------------
# BÆ¯á»šC 4: SINH Gá»¢I Ã� (RECOMMENDATION)
# ---------------------------------------------------------

# Táº¡o Top-5 gá»£i Ã½ cho Táº¤T Cáº¢ user
print("Ä�ang sinh gá»£i Ã½ cho users...")
user_recs = model.recommendForAllUsers(5)

# Hiá»ƒn thá»‹ káº¿t quáº£ (Dáº¡ng index)
user_recs.show(5, truncate=False)




# Láº¥y ra 5 user báº¥t ká»³ vÃ  xem model gá»£i Ã½ gÃ¬ cho há»�
user_recs = model.recommendForAllUsers(5)

print("--- Káº¾T QUáº¢ Gá»¢I Ã� MáºªU ---")
user_recs.show(5, truncate=False)


from pyspark.ml.evaluation import RankingEvaluator
from pyspark.sql.functions import col, collect_list

# ---------------------------------------------------------
# BÆ¯á»šC 1: CHUáº¨N Bá»Š Dá»® LIá»†U KIá»‚M THá»¬ (GROUND TRUTH)
# ---------------------------------------------------------
print("Ä�ang chuáº©n bá»‹ dá»¯ liá»‡u Ground Truth tá»« táº­p Test...")

# Sá»¬A Lá»–I: Ã‰p kiá»ƒu item_idx sang 'double' ngay tá»« Ä‘áº§u
test_casted = test.withColumn("item_idx", col("item_idx").cast("double"))

ground_truth = test_casted.groupBy("user_idx") \
    .agg(collect_list("item_idx").alias("true_items"))

# ---------------------------------------------------------
# BÆ¯á»šC 2: SINH Dá»° Ä�OÃ�N Tá»ª MODEL (PREDICTIONS)
# ---------------------------------------------------------
print("Ä�ang sinh Top-10 gá»£i Ã½...")
recommendations = model.recommendForAllUsers(10)

# Sá»¬A Lá»–I: Ã‰p kiá»ƒu máº£ng dá»± Ä‘oÃ¡n sang 'array<double>'
prediction_clean = recommendations.select(
    "user_idx", 
    col("recommendations.item_idx").cast("array<double>").alias("predicted_items")
)

# ---------------------------------------------------------
# BÆ¯á»šC 3: Káº¾T Há»¢P (JOIN) Dá»° Ä�OÃ�N VÃ€ THá»°C Táº¾
# ---------------------------------------------------------
eval_df = prediction_clean.join(ground_truth, on="user_idx")

# Cache láº¡i Ä‘á»ƒ Spark khÃ´ng pháº£i tÃ­nh toÃ¡n láº¡i khi cháº¡y 2 metric
eval_df.cache()

print("Dá»¯ liá»‡u sáºµn sÃ ng Ä‘á»ƒ Ä‘Ã¡nh giÃ¡ (Ä�Ã£ convert sang Double):")
eval_df.printSchema() # In ra Ä‘á»ƒ em kiá»ƒm tra xem nÃ³ thÃ nh double chÆ°a
eval_df.show(3, truncate=True)

# ---------------------------------------------------------
# BÆ¯á»šC 4: TÃ�NH TOÃ�N METRIC (MAP@10 vÃ  NDCG@10)
# ---------------------------------------------------------

evaluator_map = RankingEvaluator(
    predictionCol="predicted_items", 
    labelCol="true_items", 
    metricName="meanAveragePrecision", 
    k=10
)

evaluator_ndcg = RankingEvaluator(
    predictionCol="predicted_items", 
    labelCol="true_items", 
    metricName="ndcgAtK", 
    k=10
)

print("Ä�ang tÃ­nh toÃ¡n Metric...")
map_score = evaluator_map.evaluate(eval_df)
ndcg_score = evaluator_ndcg.evaluate(eval_df)

print("="*40)
print(f"ğŸ“Š Káº¾T QUáº¢ Ä�Ã�NH GIÃ� MÃ” HÃŒNH (TOP-10)")
print(f"âœ… MAP@10  : {map_score:.4f}")
print(f"âœ… NDCG@10 : {ndcg_score:.4f}")
print("="*40)


from pyspark.sql.functions import concat_ws, lit

# 1. TÃ­nh toÃ¡n 12 mÃ³n Ä‘Æ°á»£c mua nhiá»�u nháº¥t trong 3 thÃ¡ng qua
# (DÃ¹ng láº¡i df_rating hoáº·c df_recent cá»§a em)
top12_items = df_rating.groupBy("article_id").count() \
    .orderBy(col("count").desc()) \
    .limit(12) \
    .select("article_id") \
    .collect()

# 2. Chuyá»ƒn thÃ nh chuá»—i: "0751471001 0573085028 ..."
# LÆ°u Ã½: ThÃªm sá»‘ '0' á»Ÿ Ä‘áº§u náº¿u ID bá»‹ máº¥t sá»‘ 0
def format_id(aid):
    s = str(aid)
    return "0" * (10 - len(s)) + s

top12_string = " ".join([format_id(row.article_id) for row in top12_items])
print(f"Chuá»—i máº·c Ä‘á»‹nh (Best Sellers): {top12_string}")


from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

# 1. Láº¥y danh sÃ¡ch nhÃ£n (Labels) tá»« Indexer
# Ä�Ã¢y lÃ  danh sÃ¡ch [Item0, Item1, Item2...]
item_labels = item_indexer_model.labels
user_labels = user_indexer_model.labels

# Broadcast danh sÃ¡ch Item Ä‘á»ƒ cÃ¡c Worker Ä‘á»�c Ä‘Æ°á»£c
bc_item_labels = spark.sparkContext.broadcast(item_labels)
bc_user_labels = spark.sparkContext.broadcast(user_labels)

# 2. Ä�á»‹nh nghÄ©a hÃ m Map ngÆ°á»£c (UDF)
def map_ids_and_format(indices):
    # indices: list cÃ¡c item_idx [0.0, 5.0, 10.0...]
    if not indices: return ""
    
    result = []
    labels = bc_item_labels.value
    for idx in indices:
        try:
            # Láº¥y ID gá»‘c tá»« danh sÃ¡ch labels
            original_id = str(labels[int(idx)])
            # Format láº¡i cho Ä‘á»§ 10 kÃ½ tá»± (thÃªm sá»‘ 0 Ä‘áº§u)
            formatted_id = "0" * (10 - len(original_id)) + original_id
            result.append(formatted_id)
        except:
            continue
    return " ".join(result) # Ná»‘i láº¡i báº±ng dáº¥u cÃ¡ch

# Ä�Äƒng kÃ½ UDF vá»›i Spark
format_pred_udf = udf(map_ids_and_format, StringType())

# 3. Ã�p dá»¥ng UDF vÃ o DataFrame káº¿t quáº£
# recommendations: [user_idx, [(item_idx, score), ...]]
# Ta láº¥y Top 12 luÃ´n cho Ä‘Ãºng chuáº©n cuá»™c thi
recs_top12 = model.recommendForAllUsers(12)

# Map Item IDX -> Chuá»—i Article ID
df_preds = recs_top12.select(
    col("user_idx"),
    format_pred_udf(col("recommendations.item_idx")).alias("prediction")
)

# Map User IDX -> Customer ID tháº­t
# VÃ¬ User Labels quÃ¡ lá»›n, ta khÃ´ng dÃ¹ng UDF mÃ  dÃ¹ng IndexToString sáº½ an toÃ n hÆ¡n cho Driver
from pyspark.ml.feature import IndexToString

user_converter = IndexToString(inputCol="user_idx", outputCol="customer_id", labels=user_labels)
df_preds_final = user_converter.transform(df_preds).select("customer_id", "prediction")

print("Káº¿t quáº£ dá»± Ä‘oÃ¡n sau khi format:")
df_preds_final.show(3, truncate=True)


# ---------------------------------------------------------
# BÆ¯á»šC 3: JOIN Vá»šI SAMPLE SUBMISSION (Ä�Ãƒ Sá»¬A Lá»–I AMBIGUOUS)
# ---------------------------------------------------------
from pyspark.sql.functions import col, lit, coalesce

# 1. Ä�á»�c file máº«u (Sample Submission)
# Ta chá»‰ cáº§n láº¥y cá»™t 'customer_id' Ä‘á»ƒ Ä‘áº£m báº£o Ä‘á»§ user, bá»� qua cá»™t 'prediction' rÃ¡c trong Ä‘Ã³
path_sample = "/kaggle/input/h-and-m-personalized-fashion-recommendations/sample_submission.csv"
df_sample_users = spark.read.csv(path_sample, header=True).select("customer_id")

# 2. Ä�á»•i tÃªn cá»™t dá»± Ä‘oÃ¡n cá»§a Model Ä‘á»ƒ trÃ¡nh trÃ¹ng tÃªn (Quan trá»�ng!)
df_preds_renamed = df_preds_final.withColumnRenamed("prediction", "model_prediction")

# 3. Left Join
# Logic: Láº¥y danh sÃ¡ch user chuáº©n tá»« file máº«u, ghÃ©p vá»›i dá»± Ä‘oÃ¡n cá»§a model
df_joined = df_sample_users.join(df_preds_renamed, on="customer_id", how="left")

# 4. Ä�iá»�n vÃ o chá»— trá»‘ng (Fillna/Coalesce)
# Náº¿u model_prediction cÃ³ dá»¯ liá»‡u -> DÃ¹ng nÃ³
# Náº¿u model_prediction lÃ  null (Cold Start) -> DÃ¹ng chuá»—i top12_string
df_submission = df_joined.select(
    col("customer_id"),
    coalesce(col("model_prediction"), lit(top12_string)).alias("prediction")
)

# 5. LÆ°u file
print("Ä�ang lÆ°u file submission.csv (CÃ³ thá»ƒ máº¥t vÃ i phÃºt)...")
# coalesce(1): Gom vá»� 1 file duy nháº¥t Ä‘á»ƒ dá»… download
df_submission.coalesce(1).write.csv("submission_output", header=True, mode="overwrite")

print("âœ… Ä�Ã£ xong! HÃ£y vÃ o folder 'submission_output' táº£i file csv vá»� vÃ  ná»™p.")


import os
import shutil
from IPython.display import FileLink

print("â�³ Ä�ang nÃ©n dá»¯ liá»‡u... (Vui lÃ²ng Ä‘á»£i vÃ i phÃºt)")

# 1. Ä�á»‹nh nghÄ©a tÃªn file zip Ä‘áº§u ra
output_filename = "HM_Project_Artifacts"
source_dir = "./"  # ThÆ° má»¥c hiá»‡n táº¡i (Working Directory)

# 2. Táº¡o thÆ° má»¥c táº¡m Ä‘á»ƒ gom nhá»¯ng thá»© cáº§n thiáº¿t
temp_folder = "pack_to_download"
os.makedirs(temp_folder, exist_ok=True)

# Danh sÃ¡ch cÃ¡c má»¥c quan trá»�ng cáº§n táº£i vá»�
targets = [
    "als_hm_model_final",    # Model Spark (Quan trá»�ng nháº¥t cho Docker)
    "user_labels.csv",       # Map User ID
    "item_labels.csv",       # Map Item ID
    "submission_output"      # Folder chá»©a file ná»™p bÃ i
]

# Copy vÃ o thÆ° má»¥c táº¡m
for t in targets:
    if os.path.exists(t):
        # Náº¿u lÃ  thÆ° má»¥c (Model, Submission)
        if os.path.isdir(t):
            # XÃ³a náº¿u Ä‘Ã£ tá»“n táº¡i trong temp Ä‘á»ƒ trÃ¡nh lá»—i
            if os.path.exists(f"{temp_folder}/{t}"):
                shutil.rmtree(f"{temp_folder}/{t}")
            shutil.copytree(t, f"{temp_folder}/{t}")
        # Náº¿u lÃ  file (CSV)
        else:
            shutil.copy(t, f"{temp_folder}/{t}")
    else:
        print(f"âš ï¸� Cáº£nh bÃ¡o: KhÃ´ng tÃ¬m tháº¥y {t}, sáº½ bá»� qua.")

# 3. NÃ©n thÆ° má»¥c táº¡m thÃ nh file ZIP
shutil.make_archive(output_filename, 'zip', temp_folder)

# 4. XÃ³a thÆ° má»¥c táº¡m cho sáº¡ch rÃ¡c
shutil.rmtree(temp_folder)

print(f"âœ… Ä�Ã£ nÃ©n xong! File cá»§a em tÃªn lÃ : {output_filename}.zip")
print("ğŸ‘‡ Báº¥m vÃ o link dÆ°á»›i Ä‘Ã¢y Ä‘á»ƒ táº£i vá»� ngay:")
display(FileLink(f'{output_filename}.zip'))


dataset_path = "/kaggle/input/h-and-m-personalized-fashion-recommendations/transactions_train.csv"

# Ä�á»�c dá»¯ liá»‡u. InferSchema=True Ä‘á»ƒ Spark tá»± hiá»ƒu kiá»ƒu dá»¯ liá»‡u (tá»‘n chÃºt thá»�i gian nhÆ°ng tiá»‡n)
# LÆ°u Ã½: H&M dataset ráº¥t lá»›n (>30 triá»‡u dÃ²ng). Ä�á»ƒ demo cháº¡y nhanh, tháº§y sáº½ láº¥y máº«u (sampling)
# hoáº·c lá»�c dá»¯ liá»‡u gáº§n nháº¥t. á»� Ä‘Ã¢y tháº§y láº¥y 1 triá»‡u dÃ²ng Ä‘áº§u tiÃªn Ä‘á»ƒ em test pipeline trÆ°á»›c.
df_raw = spark.read.csv(dataset_path, header=True, inferSchema=True)
df_raw.show()







