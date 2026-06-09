"""
PROYEK SAINS DATA - KLASIFIKASI KOMENTAR REDDIT (SVM VERSION)
Model: Support Vector Machine (SVM)
Author: Kelompok IF 5E
"""

# ============================================================================
# 1. PERSIAPAN (KONFIGURASI & LIBRARY)
# ============================================================================

import pandas as pd
import numpy as np
import re
from sklearn.svm import SVC  # <--- PERUBAHAN: Menggunakan SVM
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# --- Path Model & Data ---
MODEL_LOCAL_PATH = '/kaggle/input/all-minilm-l6-v2/pytorch/default/1' 
DATA_PATH = '/kaggle/input/jigsaw-agile-community-rules/'
TRAIN_FILE = DATA_PATH + 'train.csv'
TEST_FILE = DATA_PATH + 'test.csv'

# ============================================================================
# 2. DATA PREPARATION (SAMA SEPERTI SEBELUMNYA)
# ============================================================================

# --- Parameter Feature Engineering ---
COMMERCIAL_KEYWORDS = [
    'buy', 'sell', 'discount', 'free', 'click', 'visit', 'check out',
    'sale', 'offer', 'deal', 'promo', 'code', 'limited', 'now',
    'subscribe', 'follow', 'join', 'sign up', 'register', 'download',
    'price', '$', 'payment', 'paypal', 'earn', 'money', 'win',
    'shop', 'store', 'checkout', 'bonus' 
]

LEGAL_KEYWORDS = [
    'sue', 'lawsuit', 'lawyer', 'attorney', 'legal', 'court', 'judge',
    'illegal', 'law', 'crime', 'police', 'arrest', 'charge', 'felony',
    'contract', 'rights', 'liable', 'damages', 'settlement', 'rape',
    'should i', 'can i', 'what should', 'advice',
    'is it legal', 'is this legal' 
]

# --- Alat Data Prep ---
print(f"ğŸ”„ Memuat alat Data Prep: SBERT model...")
try:
    SBERT_MODEL = SentenceTransformer(MODEL_LOCAL_PATH)
except Exception as e:
    print(f"â�Œ ERROR: Gagal memuat SBERT model dari {MODEL_LOCAL_PATH}. Pastikan path benar.")
    SBERT_MODEL = None

print(f"ğŸ”„ Menyiapkan alat Data Prep: StandardScaler...")
SCALER = StandardScaler()

# --- Fungsi-Fungsi Data Prep ---
def clean_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', ' URL ', text, flags=re.MULTILINE)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_advanced_features(text):
    features = {}
    if pd.isna(text): return {}
    text = str(text)
    features['length'] = len(text)
    features['word_count'] = len(text.split())
    features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
    features['has_url'] = 1 if re.search(r'http\S+|www\S+', text) else 0
    features['url_count'] = len(re.findall(r'http\S+|www\S+', text))
    features['special_char_count'] = len(re.findall(r'[!@#$%^&*()_+=\[\]{};:\'",.<>?/\\|`~]', text))
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['upper_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
    
    text_lower = text.lower()
    features['commercial_keyword_count'] = sum(1 for kw in COMMERCIAL_KEYWORDS if kw in text_lower)
    features['has_price'] = 1 if re.search(r'\$\d+|\d+\s*(dollar|usd|euro)', text_lower) else 0
    features['legal_keyword_count'] = sum(1 for kw in LEGAL_KEYWORDS if kw in text_lower)
    features['has_question_about_law'] = 1 if re.search(r'(should i|can i|is it legal)', text_lower) else 0
    
    features['sentence_count'] = len(re.findall(r'[.!?]+', text))
    words = text_lower.split()
    features['word_diversity'] = len(set(words)) / len(words) if len(words) > 0 else 0
    return features

def prepare_features(df, is_training=True):
    print(f"ğŸ”„ Mempersiapkan fitur untuk {'training' if is_training else 'prediksi'}...")
    df = df.copy()
    
    # Cleaning & Text Features
    text_cols = ['body', 'rule', 'positive_example_1', 'positive_example_2', 'negative_example_1', 'negative_example_2']
    for col in text_cols:
        if col in df.columns: df[f'{col}_clean'] = df[col].apply(clean_text)
    
    text_features = df['body'].apply(extract_advanced_features).apply(pd.Series).fillna(0)
    
    # SBERT Embeddings
    similarity_df = pd.DataFrame()
    if SBERT_MODEL:
        try:
            body_emb = SBERT_MODEL.encode(df['body_clean'].tolist(), show_progress_bar=is_training, batch_size=32)
            rule_emb = SBERT_MODEL.encode(df['rule_clean'].tolist(), show_progress_bar=False, batch_size=32)
            pos1_emb = SBERT_MODEL.encode(df['positive_example_1_clean'].tolist(), show_progress_bar=False, batch_size=32)
            pos2_emb = SBERT_MODEL.encode(df['positive_example_2_clean'].tolist(), show_progress_bar=False, batch_size=32)
            neg1_emb = SBERT_MODEL.encode(df['negative_example_1_clean'].tolist(), show_progress_bar=False, batch_size=32)
            neg2_emb = SBERT_MODEL.encode(df['negative_example_2_clean'].tolist(), show_progress_bar=False, batch_size=32)
            
            sim_feats = []
            for i in range(len(df)):
                s_pos1 = cosine_similarity([body_emb[i]], [pos1_emb[i]])[0][0]
                s_pos2 = cosine_similarity([body_emb[i]], [pos2_emb[i]])[0][0]
                s_neg1 = cosine_similarity([body_emb[i]], [neg1_emb[i]])[0][0]
                s_neg2 = cosine_similarity([body_emb[i]], [neg2_emb[i]])[0][0]
                s_rule = cosine_similarity([body_emb[i]], [rule_emb[i]])[0][0]
                
                sim_feats.append({
                    'sim_pos_max': max(s_pos1, s_pos2),
                    'sim_pos_mean': (s_pos1 + s_pos2) / 2,
                    'sim_neg_max': max(s_neg1, s_neg2),
                    'sim_neg_mean': (s_neg1 + s_neg2) / 2,
                    'sim_rule': s_rule,
                    'sim_diff': ((s_pos1 + s_pos2)/2) - ((s_neg1 + s_neg2)/2)
                })
            similarity_df = pd.DataFrame(sim_feats)
        except Exception as e:
            print(f"Error SBERT: {e}")
            similarity_df = pd.DataFrame()

    # Rule Type Features
    rule_feats = pd.DataFrame({
        'is_advertising': df['rule'].str.contains('Advertising', case=False).astype(int),
        'is_legal': df['rule'].str.contains('legal', case=False).astype(int)
    }) if 'rule' in df.columns else pd.DataFrame()

    # Combine
    final_features = pd.concat([df[['row_id']].reset_index(drop=True), similarity_df, text_features, rule_feats], axis=1).fillna(0)
    row_ids = final_features.pop('row_id')
    
    # Scaling
    if is_training:
        final_scaled = SCALER.fit_transform(final_features)
    else:
        # Ensure columns match
        final_features = final_features.reindex(columns=SCALER.feature_names_in_, fill_value=0)
        final_scaled = SCALER.transform(final_features)
        
    final_df = pd.DataFrame(final_scaled, columns=SCALER.feature_names_in_)
    final_df.insert(0, 'row_id', row_ids.values)
    return final_df

# ============================================================================
# 3. MODELING (SVM CLASSIFIER)
# ============================================================================

# --- Konfigurasi SVM ---
# C: Regularisasi. C besar = margin sempit (training akurat), C kecil = margin lebar (lebih umum)
# Kernel 'rbf': Sangat bagus untuk data embedding (non-linear)
# class_weight 'balanced': Mengatasi ketidakseimbangan data
PARAM_SVM_C = 0.5
PARAM_SVM_KERNEL = 'linear' 
PARAM_SVM_GAMMA = 'scale'

class SimpleClassifier:
    """
    Class dengan Model SVM (Support Vector Machine)
    """

    def __init__(self):
        print(f"ğŸ”µ Menginisiasi SVM Classifier...")
        # probability=True SANGAT PENTING agar bisa predict_proba
        self.classifier = SVC(
            C=PARAM_SVM_C,
            kernel=PARAM_SVM_KERNEL,
            gamma=PARAM_SVM_GAMMA,
            probability=True,  # <--- WAJIB TRUE
            class_weight='balanced',
            random_state=42,
            verbose=True
        )
        print(f"Classifier: SVM (Kernel={PARAM_SVM_KERNEL}, C={PARAM_SVM_C})")

    def train(self, X_train, y_train):
        print("\nğŸ�¯ Melatih classifier SVM...")
        X_train_clean = X_train.drop('row_id', axis=1)
        self.classifier.fit(X_train_clean, y_train)
        print("âœ… Pelatihan selesai!")

    def predict_proba(self, X_predict):
        print("ğŸ§  Memprediksi data (SVM Probabilities)...")
        X_predict_clean = X_predict.drop('row_id', axis=1)
        # SVM menghasilkan probabilitas jika probability=True
        probabilities = self.classifier.predict_proba(X_predict_clean)[:, 1]
        return probabilities

# ============================================================================
# RUNNER
# ============================================================================
def main():
    print("="*60)
    print("ğŸš€ KLASIFIKASI KOMENTAR REDDIT (SVM VERSION)")
    print("="*60)

    if SBERT_MODEL is None:
        print("âš ï¸� Peringatan: SBERT tidak dimuat. Fitur embedding akan kosong.")

    # --- Training ---
    print("\nğŸ“‚ Memuat dataset training...")
    try:
        train_df = pd.read_csv(TRAIN_FILE)
        X_train_scaled = prepare_features(train_df, is_training=True)
        y_train = train_df['rule_violation'].values
        
        model = SimpleClassifier()
        model.train(X_train_scaled, y_train)
        
        # --- Prediction ---
        print("\nğŸŸ  Memprediksi data tes...")
        test_df = pd.read_csv(TEST_FILE)
        X_test_scaled = prepare_features(test_df, is_training=False)
        
        probs = model.predict_proba(X_test_scaled)
        
        # --- Submission ---
        sub_df = pd.DataFrame({'row_id': test_df['row_id'], 'rule_violation': probs})
        sub_df.to_csv('/kaggle/working/submission.csv', index=False)
        print(f"\nâœ… File submission berhasil dibuat!")
        print(sub_df.head())
        
    except Exception as e:
        print(f"â�Œ Terjadi kesalahan: {e}")

if __name__ == "__main__":
    main()

