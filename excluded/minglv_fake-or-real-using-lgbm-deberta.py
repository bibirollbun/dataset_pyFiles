!pip install tqdm textstat langdetect lightgbm scikit-learn transformers sentence-transformers accelerate wordcloud -q
import numpy as np
import pandas as pd
import os
import glob
import gc
import warnings
import re
import unicodedata
from tqdm.auto import tqdm
import textstat
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from langdetect import detect, DetectorFactory, LangDetectException
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
import torch.cuda.amp as amp
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords', quiet=True)


  class CFG:
    seed = 42
    n_folds = 5
    target_col = 'label'
    data_path = '/kaggle/input/fake-or-real-the-impostor-hunt/data/'
    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')
    train_csv_path = os.path.join(data_path, 'train.csv')
    output_path = './'
    model_name = 'microsoft/deberta-v3-base'
    max_length = 512
    batch_size = 6
    n_epochs = 5
    learning_rate = 2e-5
    weight_decay = 0.01
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'n_estimators': 2000,
        'learning_rate': 0.01,
        'num_leaves': 31,
        'max_depth': 7,
        'seed': seed,
        'n_jobs': -1,
        'verbose': -1,
        'device': 'gpu',
        'colsample_bytree': 0.7,
        'subsample': 0.7,
    }
    cascaded_confidence_threshold = 0.45

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    DetectorFactory.seed = seed

seed_everything(CFG.seed)
# stop_words = set(stopwords.words("english"))


def read_text_files(df, base_path):
    texts_1, texts_2 = [], []
    all_dirs = glob.glob(os.path.join(base_path, 'article_*'))
    dir_map = {int(os.path.basename(p).replace('article_', '')): p for p in all_dirs}
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Reading files from {os.path.basename(base_path)}"):
        article_id = row['id']
        dir_path = dir_map.get(article_id)
        text1_content, text2_content = "", ""
        if dir_path:
            try:
                with open(os.path.join(dir_path, 'file_1.txt'), 'r', encoding='utf-8') as f:
                    text1_content = f.read()
            except FileNotFoundError:
                pass
            try:
                with open(os.path.join(dir_path, 'file_2.txt'), 'r', encoding='utf-8') as f:
                    text2_content = f.read()
            except FileNotFoundError:
                pass
        texts_1.append(text1_content)
        texts_2.append(text2_content)
    df['text_1'] = texts_1
    df['text_2'] = texts_2
    return df

def load_data(config):
    train_df = pd.read_csv(config.train_csv_path)
    train_df = read_text_files(train_df, config.train_path)
    test_dirs = glob.glob(os.path.join(config.test_path, 'article_*'))
    test_ids = [int(os.path.basename(p).replace('article_', '')) for p in test_dirs]
    test_df = pd.DataFrame(sorted(test_ids), columns=['id'])
    test_df = read_text_files(test_df, config.test_path)
    train_df[config.target_col] = train_df['real_text_id'].apply(lambda x: 0 if x == 1 else 1)
    print(f"原始训练数据: {train_df.shape}")
    print(f"测试数据: {test_df.shape}")
    return train_df, test_df

train_df, test_df = load_data(CFG)

print("进行数据增强...")
df_swap = train_df.copy()
df_swap['text_1'], df_swap['text_2'] = df_swap['text_2'], df_swap['text_1']
df_swap['label'] = 1 - df_swap['label']
train_df_augmented = pd.concat((train_df, df_swap), axis=0).reset_index(drop=True)
print(f"增强后训练数据: {train_df_augmented.shape}")


def generate_stylometric_features(text):
    if not isinstance(text, str) or not text.strip():
        return {k: 0 for k in ['char_count', 'word_count', 'sentence_count', 'avg_word_length', 'flesch_reading_ease', 'gunning_fog', 'latin_ratio']}
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    word_count = len(words)
    if word_count == 0: return {k: 0 for k in ['char_count', 'word_count', 'sentence_count', 'avg_word_length', 'flesch_reading_ease', 'gunning_fog', 'latin_ratio']}
    features = {}
    features['char_count'] = len(text)
    features['word_count'] = word_count
    features['sentence_count'] = len(sentences)
    features['avg_word_length'] = np.mean([len(w) for w in words])
    try: features['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
    except: features['flesch_reading_ease'] = 0
    try: features['gunning_fog'] = textstat.gunning_fog(text)
    except: features['gunning_fog'] = 0
    non_space_chars = [c for c in text if c != ' ']
    if non_space_chars:
        latin_chars = [c for c in non_space_chars if 'LATIN' in unicodedata.name(c, '')]
        features['latin_ratio'] = len(latin_chars) / len(non_space_chars)
    else:
        features['latin_ratio'] = 0
    return features

sbert_model = None
try:
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("SentenceTransformer ('all-MiniLM-L6-v2') 加载成功.")
except Exception as e:
    print(f"无法加载 SentenceTransformer: {e}。语义特征将被跳过。")

def create_differential_features(df):
    print("开始提取文体学特征...")
    features_1 = df['text_1'].apply(generate_stylometric_features).apply(pd.Series)
    features_2 = df['text_2'].apply(generate_stylometric_features).apply(pd.Series)
    feature_cols = list(features_1.columns)
    for col in tqdm(feature_cols, desc="创建差分特征(diff & ratio)"):
        df[f'{col}_diff'] = features_1[col].astype(float) - features_2[col].astype(float)
        df[f'{col}_ratio'] = features_1[col].astype(float) / (features_2[col].astype(float) + 1e-9)
    if sbert_model is not None:
        print("开始计算语义特征...")
        embeddings1 = sbert_model.encode(df['text_1'].tolist(), show_progress_bar=True, batch_size=16)
        embeddings2 = sbert_model.encode(df['text_2'].tolist(), show_progress_bar=True, batch_size=16)
        df['cosine_similarity'] = [cosine_similarity([e1], [e2])[0][0] for e1, e2 in zip(embeddings1, embeddings2)]
        df['euclidean_distance'] = [np.linalg.norm(e1 - e2) for e1, e2 in zip(embeddings1, embeddings2)]
        print("语义特征计算完成。")
    final_feature_cols = [f'{col}_diff' for col in feature_cols] + \
                         [f'{col}_ratio' for col in feature_cols]
    if 'cosine_similarity' in df.columns:
        final_feature_cols.extend(['cosine_similarity', 'euclidean_distance'])
    for col in final_feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).replace([np.inf, -np.inf], 0)
    return df, final_feature_cols

train_df_features, feature_cols = create_differential_features(train_df_augmented.copy())
test_df_features, _ = create_differential_features(test_df.copy())


class SiameseNetwork(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.interaction_head = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)
        )
    def forward_one(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :] # 提取[CLS]标记的向量表示
    
    def forward(self, input_ids_A, attention_mask_A, input_ids_B, attention_mask_B, labels=None):
        vec_A = self.forward_one(input_ids_A, attention_mask_A)
        vec_B = self.forward_one(input_ids_B, attention_mask_B)
        
        # 交互特征
        diff = vec_A - vec_B
        prod = vec_A * vec_B
        
        combined_vec = torch.cat((vec_A, vec_B, diff, prod), dim=1)
        logits = self.interaction_head(combined_vec)
        
        loss = None
        if labels is not None:
            loss = nn.BCEWithLogitsLoss()(logits.view(-1), labels.float())
            
        return (loss, logits)

class TextPairDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.texts1 = df['text_1'].values
        self.texts2 = df['text_2'].values
        self.is_test = is_test
        if not self.is_test:
            self.labels = df[CFG.target_col].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        encoding_A = self.tokenizer(self.texts1[idx], add_special_tokens=True, truncation=True, max_length=self.max_len, padding='max_length', return_tensors='pt')
        encoding_B = self.tokenizer(self.texts2[idx], add_special_tokens=True, truncation=True, max_length=self.max_len, padding='max_length', return_tensors='pt')
        
        item = {
            'input_ids_A': encoding_A['input_ids'].flatten(),
            'attention_mask_A': encoding_A['attention_mask'].flatten(),
            'input_ids_B': encoding_B['input_ids'].flatten(),
            'attention_mask_B': encoding_B['attention_mask'].flatten()
        }
        
        if not self.is_test:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            
        return item


def train_and_evaluate_deberta():
    oof_deberta = np.zeros(len(train_df_augmented))
    test_preds_deberta = np.zeros(len(test_df))

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
    
    # 准备用于TTA的交换顺序的测试集
    test_df_swapped = test_df.copy()
    test_df_swapped['text_1'], test_df_swapped['text_2'] = test_df['text_2'], test_df['text_1']
    
    test_dataset = TextPairDataset(test_df, tokenizer, CFG.max_length, is_test=True)
    test_dataset_swapped = TextPairDataset(test_df_swapped, tokenizer, CFG.max_length, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False)
    test_loader_swapped = DataLoader(test_dataset_swapped, batch_size=CFG.batch_size, shuffle=False)
    
    skf = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df_augmented, train_df_augmented[CFG.target_col])):
        print(f"===== 第 {fold+1}/{CFG.n_folds} 折 =====")
        train_fold_df = train_df_augmented.iloc[train_idx]
        val_fold_df = train_df_augmented.iloc[val_idx]

        # --- DeBERTa ---
        print("开始训练 DeBERTa...")
        train_dataset = TextPairDataset(train_fold_df, tokenizer, CFG.max_length)
        val_dataset = TextPairDataset(val_fold_df, tokenizer, CFG.max_length)
        train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False)
        
        model = SiameseNetwork(CFG.model_name).to(CFG.device)
        optimizer = AdamW(model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * CFG.n_epochs)
        scaler = amp.GradScaler()
        
        for epoch in range(CFG.n_epochs):
            model.train()
            for batch in tqdm(train_loader, desc=f"第 {epoch+1} 轮训练"):
                optimizer.zero_grad()
                with amp.autocast():
                    loss, _ = model(**{k: v.to(CFG.device) for k, v in batch.items()})
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

        model.eval()
        fold_preds = []
        with torch.no_grad():
            for batch in val_loader:
                 _, logits = model(**{k: v.to(CFG.device) for k, v in batch.items() if k != 'labels'})
                 fold_preds.extend(torch.sigmoid(logits).cpu().numpy().flatten())
        oof_deberta[val_idx] = fold_preds
        
        # --- 使用 TTA 进行推理 ---
        print("在测试集上使用 TTA 进行推理...")
        # 预测 A (原始顺序)
        test_fold_preds_A = []
        with torch.no_grad():
            for batch in test_loader:
                _, logits = model(**{k: v.to(CFG.device) for k, v in batch.items() if k != 'labels'})
                test_fold_preds_A.extend(torch.sigmoid(logits).cpu().numpy().flatten())
        
        # 预测 B (交换顺序)
        test_fold_preds_B = []
        with torch.no_grad():
            for batch in test_loader_swapped:
                _, logits = model(**{k: v.to(CFG.device) for k, v in batch.items() if k != 'labels'})
                test_fold_preds_B.extend(torch.sigmoid(logits).cpu().numpy().flatten())

        test_fold_preds_A = np.array(test_fold_preds_A)
        test_fold_preds_B = np.array(test_fold_preds_B)
        
        # 集成预测结果
        integrated_preds = (test_fold_preds_A + (1 - test_fold_preds_B)) / 2.0
        test_preds_deberta += integrated_preds / CFG.n_folds
        
        del model, train_dataset, val_dataset, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()

    return oof_deberta, test_preds_deberta

oof_deberta, test_deberta = train_and_evaluate_deberta()


print(f"DeBERTa OOF 准确率: {accuracy_score(train_df_augmented[CFG.target_col], np.round(oof_deberta)):.5f}")


# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(train_df_augmented[CFG.target_col], np.round(oof_lgbm))
# print("LGBM:")
# print(cm)  #0.48  0.95789  #0.47 0.96316 # 0.46 0.96842 
# cm = confusion_matrix(train_df_augmented[CFG.target_col], oof_deberta>0.46)
# print(f"DeBERTa OOF Accuracy: {accuracy_score(train_df_augmented[CFG.target_col], oof_deberta>0.46):.5f}")
# print("DeBERTa:")
# print(cm)
# cm = confusion_matrix(train_df_augmented[CFG.target_col], np.round(oof_deberta))
# print("DeBERTa-0.5:")
# print(cm)


# # Blending
# oof_blend = 0.5 * oof_lgbm + 0.5 * oof_deberta
# test_blend = 0.5 * test_lgbm + 0.5 * test_deberta
# print(f"Blend OOF Accuracy: {accuracy_score(train_df_augmented[CFG.target_col], np.round(oof_blend)):.5f}")

# # Stacking
# meta_X_train = np.column_stack([oof_lgbm, oof_deberta])
# meta_X_test = np.column_stack([test_lgbm, test_deberta])
# # meta_model = lgb.LGBMClassifier(objective='binary', metric='binary_logloss', n_estimators=500, learning_rate=0.05, num_leaves=15, random_state=CFG.seed, verbose=-1)
# # meta_model.fit(meta_X_train, train_df_augmented[CFG.target_col])
# # final_preds_proba = meta_model.predict_proba(meta_X_test)[:, 1]

# # Logistics
# from sklearn.linear_model import LogisticRegression
# meta_model = LogisticRegression(random_state=CFG.seed)
# meta_model.fit(meta_X_train, train_df_augmented[CFG.target_col])
# final_preds_proba = meta_model.predict_proba(meta_X_test)[:, 1]

# # Submission
# final_preds_class = (final_preds_proba > 0.5).astype(int)
# submission_preds = [1 if pred == 0 else 2 for pred in final_preds_class]
# submission_df = pd.DataFrame({'id': test_df['id'], 'real_text_id': submission_preds})
# submission_df.to_csv('submission.csv', index=False)
# print("Submission file 'submission.csv' created successfully!")
# print(submission_df.head())


# Blending
# oof_blend = 0.5 * oof_lgbm + 0.5 * oof_deberta
# test_blend = 0.5 * test_lgbm + 0.5 * test_deberta
# print(f"Blend OOF Accuracy: {accuracy_score(train_df_augmented[CFG.target_col], np.round(oof_blend)):.5f}")

# # Stacking
# meta_X_train = np.column_stack([oof_lgbm, oof_deberta])
# meta_X_test = np.column_stack([test_lgbm, test_deberta])
# # meta_model = lgb.LGBMClassifier(objective='binary', metric='binary_logloss', n_estimators=500, learning_rate=0.05, num_leaves=15, random_state=CFG.seed, verbose=-1)
# # meta_model.fit(meta_X_train, train_df_augmented[CFG.target_col])
# # final_preds_proba = meta_model.predict_proba(meta_X_test)[:, 1]

# # Logistics
# from sklearn.linear_model import LogisticRegression
# meta_model = LogisticRegression(random_state=CFG.seed)
# meta_model.fit(meta_X_train, train_df_augmented[CFG.target_col])
# final_preds_proba = meta_model.predict_proba(meta_X_test)[:, 1]

# Submission
final_preds_class = (test_deberta > 0.5).astype(int)
submission_preds = [1 if pred == 0 else 2 for pred in final_preds_class]
submission_df = pd.DataFrame({'id': test_df['id'], 'real_text_id': submission_preds})
submission_df.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' created successfully!")
print(submission_df.head())


plt.figure(figsize=(8, 5))
sns.histplot(oof_deberta, bins=30, kde=True, color='lightcoral')
plt.title("DeBERTa OOF Probabilities")
plt.xlabel("Probabilities")
plt.ylabel("counts")
plt.tight_layout()
plt.show()

# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# sns.histplot(oof_lgbm, bins=30, kde=True, color='skyblue')
# plt.title("LGBM OOF Probabilities")
# plt.subplot(1, 2, 2)
# sns.histplot(oof_deberta, bins=30, kde=True, color='lightcoral')
# plt.title("DeBERTa OOF Probabilities")
# plt.tight_layout()
# plt.show()

# # # 全量数据训练一个LGBM以获取特征重要性
# # full_lgbm_model = lgb.LGBMClassifier(**CFG.lgb_params)
# # full_lgbm_model.fit(train_df_features[feature_cols], train_df_features[CFG.target_col])
# # feature_importance = pd.DataFrame({'feature': feature_cols, 'importance': full_lgbm_model.feature_importances_}).sort_values('importance', ascending=False)

# plt.figure(figsize=(10, 8))
# sns.barplot(x='importance', y='feature', data=feature_importance.head(20), palette="viridis")
# plt.title("Top 20 LightGBM Feature Importance")
# plt.tight_layout()
# plt.show()

