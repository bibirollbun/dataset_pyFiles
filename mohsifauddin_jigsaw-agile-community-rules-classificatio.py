"""
PROYEK SAINS DATA - KLASIFIKASI KOMENTAR REDDIT
MODEL: XGBOOST (GRADIENT BOOSTING)
"""

import pandas as pd
import numpy as np
import re
import warnings
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ============================================================================
# 1. KONFIGURASI & PARAMETER
# ============================================================================

# --- Path Model & Data ---
MODEL_LOCAL_PATH = '/kaggle/input/all-minilm-l6-v2/pytorch/default/1' 
DATA_PATH = '/kaggle/input/jigsaw-agile-community-rules/'
TRAIN_FILE = DATA_PATH + 'train.csv'
TEST_FILE = DATA_PATH + 'test.csv'

# --- Keyword Lists ---
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

# ============================================================================
# 2. FUNGSI FEATURE ENGINEERING
# ============================================================================

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|httpsS+', ' URL ', text, flags=re.MULTILINE)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'\\([^\]]+)\\*', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def extract_advanced_features(text):
    """Ekstrak fitur manual"""
    features = {}
    if pd.isna(text):
        return {}
    else:
        text = str(text)

    features['length'] = len(text)
    features['word_count'] = len(text.split())
    features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
    features['has_url'] = 1 if re.search(r'http\S+|www\S+', text) else 0
    features['url_count'] = len(re.findall(r'http\S+|www\S+', text))
    features['has_shortened_url'] = 1 if re.search(r'bit\.ly|goo\.gl|tinyurl|t\.co', text.lower()) else 0
    features['special_char_count'] = len(re.findall(r'[!@#$%^&*()_+=\[\]{};:\'",.<>?/\\|`~]', text))
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['upper_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
    features['has_all_caps_word'] = 1 if re.search(r'\b[A-Z]{3,}\b', text) else 0
    features['has_email'] = 1 if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text) else 0
    features['has_phone'] = 1 if re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text) else 0
    
    text_lower = text.lower()
    features['commercial_keyword_count'] = sum(1 for kw in COMMERCIAL_KEYWORDS if kw in text_lower)
    features['has_price'] = 1 if re.search(r'\$\d+|\d+\s*(dollar|usd|euro)', text_lower) else 0

    features['legal_keyword_count'] = sum(1 for kw in LEGAL_KEYWORDS if kw in text_lower)
    features['has_question_about_law'] = 1 if re.search(r'(should i|can i|is it legal|is this legal)', text_lower) else 0
    
    features['sentence_count'] = len(re.findall(r'[.!?]+', text))
    features['has_question'] = 1 if '?' in text else 0
    words = text_lower.split()
    features['word_diversity'] = len(set(words)) / len(words) if len(words) > 0 else 0
    return features

# ============================================================================
# 3. KELAS CLASSIFIER UTAMA (UPDATED: XGBOOST)
# ============================================================================

class EnhancedClassifier:
    """Menggabungkan SBERT, Fitur Manual, dan XGBoost"""

    def __init__(self):
        print(f"ğŸ”„ Memuat SBERT model dari path lokal: {MODEL_LOCAL_PATH}")
        try:
            self.model = SentenceTransformer(MODEL_LOCAL_PATH)
        except Exception as e:
            print(f"â�Œ ERROR: Gagal memuat model dari {MODEL_LOCAL_PATH}.")
            self.model = None
            
        self.scaler = StandardScaler()
        
        # --- KONFIGURASI XGBOOST ---
        print(f"Menginisiasi XGBoost Classifier...")
        self.classifier = XGBClassifier(
            n_estimators=1000,      # Jumlah pohon (High accuracy)
            learning_rate=0.05,     # Belajar pelan biar teliti
            max_depth=6,            # Kedalaman pohon
            subsample=0.8,          
            colsample_bytree=0.8,   
            n_jobs=-1,              # Pake semua CPU
            random_state=42,
            eval_metric='logloss',
            tree_method='hist'      # Mode cepat
        )
        print("Classifier dipilih: XGBoost (Gradient Boosting).")

    def prepare_features(self, df, is_training=True):
        """Mempersiapkan semua fitur (teks, SBERT, similarity)"""
        show_progress = is_training 
        df = df.copy()

        text_cols = ['body', 'rule', 'positive_example_1', 'positive_example_2', 'negative_example_1', 'negative_example_2']
        for col in text_cols:
            df[f'{col}_clean'] = df[col].apply(clean_text) if col in df.columns else ""

        # Ekstrak fitur manual
        text_features = df['body'].apply(extract_advanced_features).apply(pd.Series).fillna(0) 

        # Ekstrak fitur SBERT & Similarity
        similarity_df = pd.DataFrame()
        if self.model:
            try:
                # Batch encoding biar ga berat
                batch_size = 32
                body_embeddings = self.model.encode(df['body_clean'].tolist(), show_progress_bar=show_progress, batch_size=batch_size)
                
                # Handle kolom opsional
                def encode_col(col_name):
                    if col_name in df.columns:
                        return self.model.encode(df[col_name].tolist(), show_progress_bar=False, batch_size=batch_size)
                    return np.zeros_like(body_embeddings)

                rule_embeddings = encode_col('rule_clean')
                pos1_embeddings = encode_col('positive_example_1_clean')
                pos2_embeddings = encode_col('positive_example_2_clean')
                neg1_embeddings = encode_col('negative_example_1_clean')
                neg2_embeddings = encode_col('negative_example_2_clean')

                similarity_features = []
                for i in range(len(df)):
                    # Hitung cosine similarity per baris
                    sim_pos1 = cosine_similarity([body_embeddings[i]], [pos1_embeddings[i]])[0][0]
                    sim_pos2 = cosine_similarity([body_embeddings[i]], [pos2_embeddings[i]])[0][0]
                    sim_neg1 = cosine_similarity([body_embeddings[i]], [neg1_embeddings[i]])[0][0]
                    sim_neg2 = cosine_similarity([body_embeddings[i]], [neg2_embeddings[i]])[0][0]
                    
                    sim_pos_mean = (sim_pos1 + sim_pos2) / 2
                    sim_neg_mean = (sim_neg1 + sim_neg2) / 2
                    
                    features = {
                        'sim_pos_max': max(sim_pos1, sim_pos2),
                        'sim_pos_min': min(sim_pos1, sim_pos2),
                        'sim_pos_mean': sim_pos_mean,
                        'sim_neg_max': max(sim_neg1, sim_neg2),
                        'sim_neg_min': min(sim_neg1, sim_neg2),
                        'sim_neg_mean': sim_neg_mean,
                        'sim_rule': cosine_similarity([body_embeddings[i]], [rule_embeddings[i]])[0][0],
                        'sim_diff': sim_pos_mean - sim_neg_mean,
                        'sim_ratio': sim_pos_mean / (sim_neg_mean + 1e-6)
                    }
                    similarity_features.append(features)
                similarity_df = pd.DataFrame(similarity_features)
            except Exception as e:
                print(f"Error saat SBERT encoding: {e}")
                # Fallback kalo error (isi 0 semua)
                cols = ['sim_pos_max', 'sim_pos_min', 'sim_pos_mean', 'sim_neg_max', 'sim_neg_min', 'sim_neg_mean', 'sim_rule', 'sim_diff', 'sim_ratio']
                similarity_df = pd.DataFrame(np.zeros((len(df), len(cols))), columns=cols)
        else:
             cols = ['sim_pos_max', 'sim_pos_min', 'sim_pos_mean', 'sim_neg_max', 'sim_neg_min', 'sim_neg_mean', 'sim_rule', 'sim_diff', 'sim_ratio']
             similarity_df = pd.DataFrame(np.zeros((len(df), len(cols))), columns=cols)

        
        rule_type_features = pd.DataFrame({
            'is_advertising_rule': df['rule'].str.contains('Advertising|advertising', case=False, na=False).astype(int),
            'is_legal_rule': df['rule'].str.contains('legal advice|legal', case=False, na=False).astype(int)
        }) if 'rule' in df.columns else pd.DataFrame(columns=['is_advertising_rule', 'is_legal_rule']).fillna(0)

        # Gabung semua fitur
        final_features = pd.concat([
            df[['row_id']].reset_index(drop=True),
            similarity_df.reset_index(drop=True),
            text_features.reset_index(drop=True),
            rule_type_features.reset_index(drop=True)
        ], axis=1)

        row_ids = final_features.pop('row_id')

        if is_training:
            final_features_scaled = self.scaler.fit_transform(final_features)
        else:
            # Pastikan kolom sama dengan saat training
            final_features = final_features.reindex(columns=self.scaler.feature_names_in_, fill_value=0)
            final_features_scaled = self.scaler.transform(final_features)

        final_features_scaled = pd.DataFrame(final_features_scaled, columns=self.scaler.feature_names_in_).fillna(0) 
        final_features_scaled.insert(0, 'row_id', row_ids.values)
        return final_features_scaled

    def train(self, X_train, y_train):
        """Melatih classifier XGBoost"""
        print("\nMelatih classifier...")
        X_train = X_train.drop('row_id', axis=1).fillna(0) 
        self.classifier.fit(X_train, y_train)
        print("Pelatihan selesai")

    def predict_proba(self, X):
        """Prediksi probabilitas"""
        X_predict = X.drop('row_id', axis=1).reindex(columns=self.scaler.feature_names_in_, fill_value=0).fillna(0) 
        probabilities = self.classifier.predict_proba(X_predict)[:, 1]
        probabilities = np.nan_to_num(probabilities, nan=0.5) 
        return probabilities

# ============================================================================
# 4. EKSEKUSI UTAMA
# ============================================================================

def main():
    print("="*60)
    print("KLASIFIKASI KOMENTAR REDDIT (XGBOOST EDITION)")
    print("="*60)

    # --- Training Phase ---
    print("\n[1/4] Memuat dataset training...")
    try:
        train_df = pd.read_csv(TRAIN_FILE)
        print(f"Dataset training dimuat: {train_df.shape[0]} baris")
    except FileNotFoundError:
        print(f"Error: {TRAIN_FILE} tidak ditemukan. Cek path data.")
        return

    classifier_instance = EnhancedClassifier()
    
    if classifier_instance.model is None:
        print("CRITICAL: Model SBERT tidak ditemukan. Cek Sidebar Kaggle > Add Data.")
        return

    print("\n[2/4] Feature Engineering Training Data...")
    X_train_scaled = classifier_instance.prepare_features(train_df.copy(), is_training=True)
    y_train = train_df['rule_violation'].values
    
    classifier_instance.train(X_train_scaled, y_train)

    # --- Prediction Phase ---
    print("\n" + "="*60)
    print("[3/4] Memprediksi data tes...")
    print("="*60)
    
    try:
        test_df = pd.read_csv(TEST_FILE) 
        print(f"Dataset tes dimuat: {test_df.shape[0]} baris")
    except FileNotFoundError:
        print(f"Error: {TEST_FILE} tidak ditemukan.")
        return

    print("Feature Engineering Test Data...")
    X_test_scaled = classifier_instance.prepare_features(test_df.copy(), is_training=False)
    
    print("Sedang memprediksi...")
    final_probabilities = classifier_instance.predict_proba(X_test_scaled)
    
    print("Prediksi selesai.")

    # --- Submission ---
    print("\n" + "="*60)
    print("[4/4] Membuat file submission...")
    print("="*60)

    submission_df = pd.DataFrame({'row_id': test_df['row_id'], 'rule_violation': final_probabilities})

    if submission_df['rule_violation'].isnull().any():
        print(f"PERINGATAN: Ditemukan NaN di file submission! Mengisi dengan 0.5...")
        submission_df['rule_violation'] = submission_df['rule_violation'].fillna(0.5)

    submission_file_name = "/kaggle/working/submission.csv"
    submission_df.to_csv(submission_file_name, index=False)

    print(f"âœ… SUKSES! File tersimpan di: {submission_file_name}")
    print("5 baris pertama:")
    print(submission_df.head())

if __name__ == "__main__":
    main()


"""
PROYEK SAINS DATA - KLASIFIKASI KOMENTAR REDDIT
MODEL: CATBOOST â€” SBERT + MANUAL FEATURES
"""

import pandas as pd
import numpy as np
import re
import warnings
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from catboost import CatBoostClassifier   # ğŸ”¥ MODEL BARU

warnings.filterwarnings('ignore')

# ============================================================================
# 1. PATH DAN KONFIGURASI
# ============================================================================

MODEL_LOCAL_PATH = '/kaggle/input/all-minilm-l6-v2/pytorch/default/1'
DATA_PATH = '/kaggle/input/jigsaw-agile-community-rules/'

TRAIN_FILE = DATA_PATH + 'train.csv'
TEST_FILE  = DATA_PATH + 'test.csv'

COMMERCIAL_KEYWORDS = ['buy','sell','discount','click','visit','sale','offer','deal','promo','limited','subscribe','follow','join']
LEGAL_KEYWORDS = ['sue','lawyer','attorney','legal','court','judge','illegal','crime','police','arrest','felony','rights','damages','advice','is it legal']


# ============================================================================
# 2. CLEANING & FEATURE ENGINEERING
# ============================================================================

def clean_text(txt):
    if pd.isna(txt):
        return ""
    txt = str(txt).lower()
    txt = re.sub(r"http\S+|www\S+", " URL ", txt)
    txt = re.sub(r'\s+', ' ', txt).strip()
    return txt

def extract_advanced_features(text):
    text = "" if pd.isna(text) else str(text)
    words = text.split()
    text_lower = text.lower()

    return {
        'length': len(text),
        'word_count': len(words),
        'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
        'has_url': int("http" in text or "www" in text),
        'special_char_count': len(re.findall(r'[!@#$%^&*()?]', text)),
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'upper_ratio': sum(c.isupper() for c in text) / len(text) if text else 0,
        'commercial_keyword_count': sum(1 for kw in COMMERCIAL_KEYWORDS if kw in text_lower),
        'legal_keyword_count': sum(1 for kw in LEGAL_KEYWORDS if kw in text_lower),
    }


# ============================================================================
# 3. MODEL CLASSIFIER (CATBOOST)
# ============================================================================

class NBClassifier:

    def __init__(self):
        print("ğŸ”„ Loading SBERT...")
        try:
            self.model = SentenceTransformer(MODEL_LOCAL_PATH)
            print("âœ” SBERT Loaded")
        except:
            print("âš  SBERT gagal dimuat â†’ fitur similarity = 0")
            self.model = None

        self.scaler = StandardScaler()

        # â­� MODEL UTAMA â†’ CATBOOST CLASSIFIER
        self.classifier = CatBoostClassifier(
            iterations=600,
            depth=6,
            learning_rate=0.05,
            loss_function='Logloss',
            eval_metric='AUC',
            random_seed=42,
            verbose=False
        )

    def prepare_features(self, df, is_training=True):

        df = df.copy()
        df['body_clean'] = df['body'].apply(clean_text)
        df['rule_clean'] = df['rule'].apply(clean_text)

        manual_df = df['body_clean'].apply(extract_advanced_features).apply(pd.Series)

        # --- SBERT FEATURES ---
        if self.model:
            try:
                body_emb = self.model.encode(df['body_clean'].tolist())
                rule_emb = self.model.encode(df['rule_clean'].tolist())

                sim_list = []
                for i in range(len(df)):
                    sim_score = cosine_similarity([body_emb[i]], [rule_emb[i]])[0][0]
                    diff = np.linalg.norm(body_emb[i] - rule_emb[i])
                    body_mag = np.linalg.norm(body_emb[i])
                    rule_mag = np.linalg.norm(rule_emb[i])
                    sim_list.append({
                        'sim_rule': sim_score,
                        'emb_diff': diff,
                        'body_mag': body_mag,
                        'rule_mag': rule_mag,
                    })

                sim_df = pd.DataFrame(sim_list)

            except:
                sim_df = pd.DataFrame({
                    'sim_rule':[0]*len(df),
                    'emb_diff':[0]*len(df),
                    'body_mag':[0]*len(df),
                    'rule_mag':[0]*len(df),
                })

        else:
            sim_df = pd.DataFrame({
                'sim_rule':[0]*len(df),
                'emb_diff':[0]*len(df),
                'body_mag':[0]*len(df),
                'rule_mag':[0]*len(df),
            })

        # Gabungkan fitur
        full = pd.concat([manual_df, sim_df], axis=1).fillna(0)

        # CatBoost tidak butuh scaling tapi tetap aman digunakan
        if is_training:
            return pd.DataFrame(self.scaler.fit_transform(full),
                                columns=full.columns)
        else:
            full = full.reindex(columns=self.scaler.feature_names_in_, fill_value=0)
            return pd.DataFrame(self.scaler.transform(full),
                                columns=full.columns)

    def train(self, X, y):
        print("\nâ�³ Training CatBoost...")
        self.classifier.fit(X, y)
        preds = self.classifier.predict(X)
        acc = accuracy_score(y, preds)
        print(f"ğŸ�¯ TRAIN ACCURACY = {acc:.4f}")

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)[:, 1]


# ============================================================================
# 4. MAIN PIPELINE
# ============================================================================

def main():

    print("="*70)
    print("   KLASIFIKASI KOMENTAR REDDIT â€” CATBOOST (SBERT + FEATURES)")
    print("="*70)

    train_df = pd.read_csv(TRAIN_FILE)
    test_df  = pd.read_csv(TEST_FILE)

    clf = NBClassifier()

    print("\n[1/4] Extracting training features...")
    X_full = clf.prepare_features(train_df, is_training=True)
    y_full = train_df['rule_violation']

    # VALIDATION SPLIT
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42
    )

    clf.train(X_train, y_train)

    y_pred = clf.classifier.predict(X_val)
    val_acc = accuracy_score(y_val, y_pred)
    print(f"\nğŸ”¥ VALIDATION ACCURACY = {val_acc:.4f}")

    print("\n[2/4] Extracting test features...")
    X_test = clf.prepare_features(test_df, is_training=False)

    print("\n[3/4] Predicting probabilities...")
    preds = clf.predict_proba(X_test)

    submission = pd.DataFrame({
        "row_id": test_df["row_id"],
        "rule_violation": preds
    })

    submission.to_csv("/kaggle/working/submission.csv", index=False)

    print("\nâœ… Submission saved to /kaggle/working/submission.csv")


if __name__ == "__main__":
    main()


