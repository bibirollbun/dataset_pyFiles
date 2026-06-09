!pip install tqdm textstat langdetect lightgbm scikit-learn transformers sentence-transformers accelerate wordcloud -q

import numpy as np
import pandas as pd
import os
import glob
import gc
import warnings
import re
import unicodedata
import textstat
import nltk
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.cuda.amp as amp
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.auto import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from langdetect import detect, DetectorFactory, LangDetectException
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sentence_transformers import SentenceTransformer, util

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


# code from: https://www.kaggle.com/code/minglv/fake-or-real-using-lgbm-deberta
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
    batch_size = 4
    n_epochs = 5
    learning_rate = 2e-5
    weight_decay = 0.01
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cascaded_confidence_threshold = 0.45

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    DetectorFactory.seed = seed

seed_everything(CFG.seed)
stop_words = set(stopwords.words("english"))


# make sure GPU is being used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")


# code from: https://www.kaggle.com/code/minglv/fake-or-real-using-lgbm-deberta
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
    print(f"train DF shape: {train_df.shape}")
    print(f"test DF shape: {test_df.shape}")
    return train_df, test_df


# load data
train_df, test_df = load_data(CFG)

print("Data Augmentation")
df_swap = train_df.copy()
df_swap['text_1'], df_swap['text_2'] = df_swap['text_2'], df_swap['text_1']
df_swap['label'] = 1 - df_swap['label']
train_df_augmented = pd.concat((train_df, df_swap), axis=0).reset_index(drop=True)
print(f"Enhanced Data Training: {train_df_augmented.shape}")


train_df_augmented


# look at the augmented data, samples 0-94 are the same as 95-190 (swapped)
# training samples 0 == 95, but they're swapped, including the labels
display(train_df_augmented.iloc[0])
display(train_df_augmented.iloc[95])


# code from: https://www.kaggle.com/code/minglv/fake-or-real-using-lgbm-deberta
def generate_stylometric_features(text):
    if not isinstance(text, str) or not text.strip():
        return {k: 0 for k in ['char_count', 'word_count', 'sentence_count', 'avg_word_length', 'flesch_reading_ease', 'gunning_fog', 'latin_ratio']}
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    word_count = len(words)
    if word_count == 0: 
        return {k: 0 for k in ['char_count', 'word_count', 'sentence_count', 'avg_word_length', 'flesch_reading_ease', 'gunning_fog', 'latin_ratio']}
    features = {}
    features['char_count'] = len(text)
    features['word_count'] = word_count
    features['sentence_count'] = len(sentences)
    features['avg_word_length'] = np.mean([len(w) for w in words])
    try:
        features['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
    except:
        features['flesch_reading_ease'] = 0
    try:
        features['gunning_fog'] = textstat.gunning_fog(text)
    except:
        features['gunning_fog'] = 0
    non_space_chars = [c for c in text if c != ' ']
    if non_space_chars:
        latin_chars = [c for c in non_space_chars if 'LATIN' in unicodedata.name(c, '')]
        features['latin_ratio'] = len(latin_chars) / len(non_space_chars)
    else:
        features['latin_ratio'] = 0
    return features


def create_differential_features(df, encoder_model):
    print("Starting extraction of stylometric features...")
    features_1 = df['text_1'].apply(generate_stylometric_features).apply(pd.Series)
    features_2 = df['text_2'].apply(generate_stylometric_features).apply(pd.Series)
    feature_cols = list(features_1.columns)

    for col in tqdm(feature_cols, desc="Creating differential features (diff & ratio)"):
        df[f'{col}_diff'] = features_1[col].astype(float) - features_2[col].astype(float)
        df[f'{col}_ratio'] = features_1[col].astype(float) / (features_2[col].astype(float) + 1e-9)

    print("Starting computation of semantic features...")
    embeddings1 = encoder_model.encode(df['text_1'].tolist(), show_progress_bar=True, batch_size=16)
    embeddings2 = encoder_model.encode(df['text_2'].tolist(), show_progress_bar=True, batch_size=16)
    df['cosine_similarity'] = [cosine_similarity([e1], [e2])[0][0] for e1, e2 in zip(embeddings1, embeddings2)]
    df['euclidean_distance'] = [np.linalg.norm(e1 - e2) for e1, e2 in zip(embeddings1, embeddings2)]
    print("Semantic feature computation completed.")

    final_feature_cols = [f'{col}_diff' for col in feature_cols] + \
                         [f'{col}_ratio' for col in feature_cols]
    if 'cosine_similarity' in df.columns:
        final_feature_cols.extend(['cosine_similarity', 'euclidean_distance'])

    for col in final_feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).replace([np.inf, -np.inf], 0)

    return df, final_feature_cols


# load BERT
# Hugging Face: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
print("SentenceTransformer ('all-MiniLM-L6-v2') loaded successfully.")

# encode data
train_df_features, feature_cols = create_differential_features(train_df_augmented.copy(), sbert_model)
test_df_features, _ = create_differential_features(test_df.copy(), sbert_model)


train_df_features.head()


feature_cols


test_df_features.head()


def embed_fn(df, encoder_model):
    embeddings1 = encoder_model.encode(df['text_1'].tolist(), show_progress_bar=True, batch_size=16)
    embeddings2 = encoder_model.encode(df['text_2'].tolist(), show_progress_bar=True, batch_size=16)
    return embeddings1, embeddings2

embeddings1, embeddings2 = embed_fn(train_df_augmented, sbert_model)
display(embeddings1.shape)
display(embeddings2.shape)


embed_df = train_df_augmented.copy()
embed_df["embeddings1"] = embeddings1.tolist()
embed_df["embeddings2"] = embeddings2.tolist()
embed_df.head()


from sklearn.manifold import TSNE

# limit to first 95 rows (original dataset)
n_original = 95
X = np.vstack([embeddings1[:n_original], embeddings2[:n_original]])  # shape (190, 384)
labels = np.concatenate(
    [
        train_df_augmented["label"].values[:n_original],
        train_df_augmented["label"].values[:n_original],
    ]
)

# run t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
X_tsne = tsne.fit_transform(X)

# make dataframe for plotting
plot_df = pd.DataFrame(X_tsne, columns=["TSNE1", "TSNE2"])
plot_df["label"] = labels
plot_df["source"] = ["text_1"] * len(embeddings1[:n_original]) + ["text_2"] * len(embeddings2[:n_original])


# plot
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
sns.scatterplot(
    data=plot_df,
    x="TSNE1", y="TSNE2",
    hue="label",
    style="source",
    palette="tab10",
    s=50, alpha=0.8,
    ax=ax
)
plt.title("t-SNE visualization of embeddings (colored by label)", fontsize=14, weight="bold")
plt.legend(bbox_to_anchor=(1, 1), loc="upper left", title="Legend", title_fontproperties={"weight": "bold"}, shadow=True)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.grid(color="blue", linestyle="--", linewidth=1, alpha=0.2)
for spine in ax.spines.values():
    spine.set_visible(False)
plt.tight_layout()
plt.show()



from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import xgboost as xgb
import numpy as np

def train_xgboost(train_df, test_df, feature_cols):
    oof_xgb = np.zeros(len(train_df)) # out-of-fold preds
    test_preds_xgb = np.zeros(len(test_df))

    skf = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)

    for fold, (train_idx, val_idx) in enumerate(
        tqdm(skf.split(train_df, train_df[CFG.target_col]), total=CFG.n_folds, desc="XGBoost Folds")
    ):
        train_fold_df = train_df.iloc[train_idx]
        val_fold_df = train_df.iloc[val_idx]

        xgb_model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="gpu_hist",       # GPU acceleration
            predictor="gpu_predictor",    # GPU inference
            n_estimators=2000,
            learning_rate=0.01,
            max_depth=7,
            subsample=0.7,
            colsample_bytree=0.7,
            random_state=CFG.seed,
            n_jobs=-1,
            verbosity=0
        )

        xgb_model.fit(
            train_fold_df[feature_cols], train_fold_df[CFG.target_col],
            eval_set=[(val_fold_df[feature_cols], val_fold_df[CFG.target_col])],
            early_stopping_rounds=100,
            verbose=False
        )

        oof_xgb[val_idx] = xgb_model.predict_proba(val_fold_df[feature_cols])[:, 1]
        test_preds_xgb += xgb_model.predict_proba(test_df[feature_cols])[:, 1] / CFG.n_folds

    return oof_xgb, test_preds_xgb


oof_xgb, test_xgb = train_xgboost(train_df_features, test_df_features, feature_cols)


acc_xgb = accuracy_score(train_df_features[CFG.target_col], np.round(oof_xgb))
print(f"XGBoost OOF Accuracy: {acc_xgb}")


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
        return outputs.last_hidden_state[:, 0, :] # [CLS] token
    
    def forward(self, input_ids_A, attention_mask_A, input_ids_B, attention_mask_B, labels=None):
        vec_A = self.forward_one(input_ids_A, attention_mask_A)
        vec_B = self.forward_one(input_ids_B, attention_mask_B)
        diff = vec_A - vec_B
        prod = vec_A * vec_B
        combined_vec = torch.cat((vec_A, vec_B, diff, prod), dim=1)
        logits = self.interaction_head(combined_vec)
        loss = None
        if labels is not None:
            loss = nn.BCEWithLogitsLoss()(logits.view(-1), labels.float())
        return (loss, logits)

class TextPairDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.texts1 = df['text_1'].values
        self.texts2 = df['text_2'].values
        self.labels = df[CFG.target_col].values if CFG.target_col in df.columns else None
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        encoding_A = self.tokenizer(self.texts1[idx], add_special_tokens=True, truncation=True, max_length=self.max_len, padding='max_length', return_tensors='pt')
        encoding_B = self.tokenizer(self.texts2[idx], add_special_tokens=True, truncation=True, max_length=self.max_len, padding='max_length', return_tensors='pt')
        item = {'input_ids_A': encoding_A['input_ids'].flatten(), 'attention_mask_A': encoding_A['attention_mask'].flatten(), 'input_ids_B': encoding_B['input_ids'].flatten(), 'attention_mask_B': encoding_B['attention_mask'].flatten()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def train_eval_loop(train_df, test_df):
    oof_deberta = np.zeros(len(train_df))
    test_preds_deberta = np.zeros(len(test_df))

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)

    skf = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)

    for fold, (train_idx, val_idx) in enumerate(
        tqdm(skf.split(train_df, train_df[CFG.target_col]), total=CFG.n_folds, desc="DeBERTa Folds")
    ):
        train_fold_df = train_df.iloc[train_idx]
        val_fold_df = train_df.iloc[val_idx]

        # --- DeBERTa ---
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
            for batch in train_loader:
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

        test_dataset_full = TextPairDataset(test_df, tokenizer, CFG.max_length)
        test_loader_full = DataLoader(test_dataset_full, batch_size=CFG.batch_size, shuffle=False)
        test_fold_preds = []
        with torch.no_grad():
            for batch in test_loader_full:
                _, logits = model(**{k: v.to(CFG.device) for k, v in batch.items() if k != 'labels'})
                test_fold_preds.extend(torch.sigmoid(logits).cpu().numpy().flatten())
        test_preds_deberta += np.array(test_fold_preds) / CFG.n_folds
        gc.collect()
        torch.cuda.empty_cache()

    return oof_deberta, test_preds_deberta


# # train model
# oof_deberta, test_preds_deberta = train_eval_loop(train_df_features, test_df_features)


# acc_deberta = accuracy_score(train_df_features[CFG.target_col], np.round(oof_deberta))
# print(f"DeBERTa OOF Accuracy: {acc_deberta}")
# # DeBERTa OOF Accuracy: 0.9578947368421052


# # train model
# oof_deberta_embeds, test_preds_deberta_embeds = train_eval_loop(embed_df, test_df_features)


# acc_deberta_embeds = accuracy_score(embed_df[CFG.target_col], np.round(oof_deberta_embeds))
# print(f"DeBERTa Embeds OOF Accuracy: {acc_deberta_embeds}")
# # DeBERTa Embeds OOF Accuracy: 0.9526315789473684


# init xgboost model for getting feature importance
xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="gpu_hist",       # GPU acceleration
    predictor="gpu_predictor",    # GPU inference
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=7,
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=CFG.seed,
    n_jobs=-1,
    verbosity=0
)

# fit model
xgb_model.fit(
    train_df_features[feature_cols], train_df_features[CFG.target_col],
    verbose=False
)

# dataframe
feature_importance = pd.DataFrame(
    {
        'feature': feature_cols,
        'importance': xgb_model.feature_importances_
    }
).sort_values('importance', ascending=False)


plt.figure(figsize=(8, 6), dpi=100)
sns.barplot(x='importance', y='feature', data=feature_importance.head(20), palette="viridis")
plt.title("Top 20 XGBoost Feature Importance")
plt.tight_layout()
plt.show()


# create final dataframe
cols_to_keep = ["text_1", "text_2", "label", "sentence_count_ratio", "sentence_count_diff", "word_count_diff", "cosine_similarity", "euclidean_distance", "word_count_ratio"]
final_df = train_df_features[cols_to_keep].copy()
final_df["embeddings1"] = embeddings1.tolist()
final_df["embeddings2"] = embeddings2.tolist()
final_df.head()


# train model
oof_deberta_final, test_preds_deberta_final = train_eval_loop(final_df, test_df_features)


acc_deberta_final = accuracy_score(final_df[CFG.target_col], np.round(oof_deberta_final))
print(f"DeBERTa Final OOF Accuracy: {acc_deberta_final}")


# Submission
final_preds_class = (test_preds_deberta_final > 0.5).astype(int)
submission_preds = [1 if pred == 0 else 2 for pred in final_preds_class]
submission_df = pd.DataFrame({'id': test_df['id'], 'real_text_id': submission_preds})
submission_df.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' created successfully!")
print(submission_df.head())




