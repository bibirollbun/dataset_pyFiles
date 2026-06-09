!pip install textstat

import os
import re
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import math
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from scipy.stats import ttest_ind
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import PCA
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import xgboost as xgb
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import joblib
from nltk.tokenize import word_tokenize, sent_tokenize
from textblob import TextBlob
from textstat import flesch_reading_ease, flesch_kincaid_grade
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from wordcloud import WordCloud
import textstat

# Download resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
warnings.filterwarnings("ignore")
plt.style.use("ggplot")


#Give paths for data loading
train_csv_path = '/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv'
train_dir = "/kaggle/input/fake-or-real-the-impostor-hunt/data/train"
test_dir = "/kaggle/input/fake-or-real-the-impostor-hunt/data/test"
df = pd.read_csv(train_csv_path)


display(df.head())
df.head()
print("Label distribution:")
display(df['real_text_id'].value_counts())


real_text_counts = df['real_text_id'].value_counts().sort_index()

sns.set(style='whitegrid')
fig, ax = plt.subplots(figsize=(6,5))

colors = ['#2ca02c', '#6a0dad'] 
bars = ax.bar(real_text_counts.index.astype(str), real_text_counts.values, color=colors)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + 1.5,  # smaller margin
        f'{int(height)}', ha='center', va='bottom', fontsize=12, weight='bold')


ax.set_xlabel('Real Text Number')
ax.set_ylabel('Count')
ax.set_title('Distribution of Real Texts: Text 1 vs Text 2',pad=20)
plt.tight_layout()
plt.savefig("Real_Text_.png", dpi=300, bbox_inches='tight')
plt.show()



#Data Loading
def generate_train_data(data_dir, csv_path):
    """Generate training samples with labels"""
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        folder_id = row["id"]
        folder_path = os.path.join(data_dir, f"article_{folder_id:04d}")
        with open(os.path.join(folder_path, "file_1.txt"), encoding="utf-8") as f1:
            text1 = f1.read()
        with open(os.path.join(folder_path, "file_2.txt"), encoding="utf-8") as f2:
            text2 = f2.read()
        label = 1 if row["real_text_id"] == 1 else 0
        yield {"id": folder_id, "text1": text1, "text2": text2, "label": label}

def generate_test_data(data_dir):
    """Generate test samples without labels"""
    folders = sorted([
        f for f in os.listdir(data_dir) 
        if os.path.isdir(os.path.join(data_dir, f)) and re.match(r'article_\d+', f)
    ])
    for folder in folders:
        folder_id = int(folder.split('_')[1])
        folder_path = os.path.join(data_dir, folder)
        with open(os.path.join(folder_path, "file_1.txt"), encoding="utf-8") as f1:
            text1 = f1.read()
        with open(os.path.join(folder_path, "file_2.txt"), encoding="utf-8") as f2:
            text2 = f2.read()
        yield {"id": folder_id, "text1": text1, "text2": text2}


train_dataset = Dataset.from_generator(lambda: generate_train_data(train_dir, train_csv_path))
test_dataset = Dataset.from_generator(lambda: generate_test_data(test_dir))
raw_datasets = DatasetDict({"train": train_dataset, "test": test_dataset})

#DataFrames for EDA
train_df = pd.DataFrame(raw_datasets["train"])
print("Train DataFrame shape:", train_df.shape)
display(train_df.head(5))



# Number of pairs
num_pairs = train_df.shape[0]

# Since each pair has one real and one fake text
num_real = num_pairs
num_fake = num_pairs

print(f"Total text pairs: {num_pairs}")
print(f"Total real texts: {num_real}")
print(f"Total fake texts: {num_fake}")

# Plot barplot for real vs fake counts
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.figure(figsize=(6,4))

labels = ['Fake (0)', 'Real (1)']
counts = [num_fake, num_real]

sns.barplot(x=labels, y=counts, palette=['#6a0dad', '#2ca02c'])  # purple, green

plt.title("Total Real vs Fake Texts in Dataset (Per Pair Counts)")
plt.xlabel("Text Class")
plt.ylabel("Count")
plt.ylim(0, max(counts)*1.1)

# Show counts on bars
for i, count in enumerate(counts):
    plt.text(i, count + max(counts)*0.05, str(count), ha='center', fontsize=12, weight='bold')

plt.savefig("Class_Balance.png", dpi=300, bbox_inches='tight')
plt.show()



def basic_text_stats(text):
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    word_freq = {}
    for w in words:
        word_freq[w] = word_freq.get(w, 0) + 1
    return {
        'num_chars': len(text),
        'num_words': len(words),
        'num_sentences': len(sentences),
        'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
        'avg_sentence_length': len(words)/len(sentences) if sentences else 0,
        'n_unique_words': len(set(words)),
        'ttr': len(set(words))/len(words) if words else 0,
        'hapax_legomena_ratio': len([w for w, c in word_freq.items() if c == 1])/len(words) if words else 0
    }

def readability_scores(text):
    return {
        'flesch_reading_ease': flesch_reading_ease(text),
        'flesch_kincaid_grade': flesch_kincaid_grade(text)
    }

def stylometric_features(text):
    words = word_tokenize(text)
    return {
        'stopword_ratio': len([w for w in words if w.lower() in stop_words])/len(words) if words else 0,
        'punctuation_count': sum(1 for c in text if c in '.,;:?!"\''),
        'special_chars_count': sum(1 for c in text if c in '@#$%&*')
    }

def lexical_pos_ratios(text):
    words = word_tokenize(text)
    if not words: 
        return {'noun_ratio': 0, 'verb_ratio': 0, 'adj_ratio': 0, 'adv_ratio': 0}
    tags = nltk.pos_tag(words)
    noun_count = sum(1 for _, tag in tags if tag.startswith('NN'))
    verb_count = sum(1 for _, tag in tags if tag.startswith('VB'))
    adj_count = sum(1 for _, tag in tags if tag.startswith('JJ'))
    adv_count = sum(1 for _, tag in tags if tag.startswith('RB'))
    total = len(words)
    return {
        'noun_ratio': noun_count/total,
        'verb_ratio': verb_count/total,
        'adj_ratio': adj_count/total,
        'adv_ratio': adv_count/total
    }

def sentiment_features(text):
    blob = TextBlob(text)
    return {'polarity': blob.sentiment.polarity, 'subjectivity': blob.sentiment.subjectivity}
    
def apply_eda_features(df, has_label=True):
    """Apply all EDA feature extractors to a dataframe"""
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting EDA features"):
        feats1 = {**basic_text_stats(row['text1']), 
                 **readability_scores(row['text1']), 
                 **stylometric_features(row['text1']), 
                 **lexical_pos_ratios(row['text1']), 
                 **sentiment_features(row['text1'])}
        
        feats2 = {**basic_text_stats(row['text2']), 
                 **readability_scores(row['text2']), 
                 **stylometric_features(row['text2']), 
                 **lexical_pos_ratios(row['text2']), 
                 **sentiment_features(row['text2'])}
        
        new_row = {'id': row['id']}
        if has_label: 
            new_row['label'] = row['label']
        for k, v in feats1.items():
            new_row[f't1_{k}'] = v
        for k, v in feats2.items():
            new_row[f't2_{k}'] = v
        rows.append(new_row)
    return pd.DataFrame(rows)

print("Extracting EDA features...")

eda_train = apply_eda_features(train_df, has_label=True)
eda_test = apply_eda_features(pd.DataFrame(raw_datasets["test"]), has_label=False)

print("EDA train shape:", eda_train.shape)
eda_train.head(2)




def melt_eda_df(eda_df):
    """Convert to flat format with real/fake labels"""
    real_rows = []
    fake_rows = []
    for _, row in eda_df.iterrows():
        if row['label'] == 1:
            real_text = {k.replace('t1_', ''): row[k] for k in eda_df.columns if k.startswith('t1_')}
            fake_text = {k.replace('t2_', ''): row[k] for k in eda_df.columns if k.startswith('t2_')}
        else:
            real_text = {k.replace('t2_', ''): row[k] for k in eda_df.columns if k.startswith('t2_')}
            fake_text = {k.replace('t1_', ''): row[k] for k in eda_df.columns if k.startswith('t1_')}
        real_text['label'] = 'real'
        fake_text['label'] = 'fake'
        real_text['id'] = row['id']
        fake_text['id'] = row['id']
        real_rows.append(real_text)
        fake_rows.append(fake_text)
    result_df = pd.DataFrame(real_rows + fake_rows)
    print(f"Melted EDA dataframe with {result_df.shape[0]} rows and {result_df.shape[1]} columns.")
    return result_df

# PLOTS ------------------------------------------------------------------

flat_eda_df = melt_eda_df(eda_train)

def plot_feature_distributions(df, features, title):
    for feat in features:
        plt.figure(figsize=(10, 4))
        sns.kdeplot(data=df, x=feat, hue='label', fill=True, common_norm=False, alpha=0.5)
        plt.title(f"{title}: {feat}")
        plt.show()

def plot_feature_boxplots(df, features, title):
    for feat in features:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x='label', y=feat, data=df)
        plt.title(f"{title}: {feat}")
        plt.show()

def plot_correlation_matrix(df, title):
    corr = df.drop(columns=['label']).corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title(f"{title}: Feature Correlation Matrix")
    plt.savefig("Feature Correlation Matrix.png", dpi=300, bbox_inches='tight')
    plt.show()

# Confirm all features are there
length_feats = ['num_chars', 'num_words', 'num_sentences', 'avg_word_length', 'avg_sentence_length']
lexical_feats = ['ttr', 'hapax_legomena_ratio', 'n_unique_words']
readability_feats = ['flesch_reading_ease', 'flesch_kincaid_grade']
stylo_feats = ['stopword_ratio', 'punctuation_count', 'special_chars_count']
sentiment_feats = ['polarity', 'subjectivity']
pos_feats = ['noun_ratio', 'verb_ratio','adj_ratio','adv_ratio']

all_feats = length_feats + lexical_feats + readability_feats + stylo_feats + sentiment_feats + pos_feats
assert len(all_feats) == 19, f"Expected 19 features, got {len(all_feats)}"


# ðŸŽ¨ KDE Plot Grid (histogram-style)
def plot_kde_grid(df, features, label_col='label'):
    n_feats = len(features)
    n_cols = 3
    n_rows = (n_feats + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 3))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        sns.kdeplot(data=df, x=feat, hue=label_col, ax=axes[i], fill=True, common_norm=False, alpha=0.4,palette={"real": "green", "fake": "red"})
        axes[i].set_title(feat)
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Density')
    
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    fig.suptitle("KDE Distribution of All Features by Label", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig("EDA_histogram.png", dpi=300, bbox_inches='tight')
    plt.show()
    

# ðŸ“¦ Boxplot for All Features in One Figure
def plot_boxplot_grid(df, features, label_col='label', cols=3):
    rows = math.ceil(len(features) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        sns.boxplot(data=df, x=label_col, y=feat, ax=axes[i],palette={"real": "green", "fake": "red"})
        axes[i].set_title(feat)
        axes[i].set_xlabel('Label')          
        axes[i].set_ylabel('Feature Value')

    # Turn off any unused axes
    for j in range(len(features), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig("EDA_Boxplot.png", dpi=300, bbox_inches='tight')
    plt.show()


plot_boxplot_grid(flat_eda_df, all_feats)
plot_kde_grid(flat_eda_df, all_feats)
plot_correlation_matrix(flat_eda_df, "EDA")

def generate_wordclouds(df):
    
    real_texts = []
    fake_texts = []
    
    for _, row in df.iterrows():
        if row['label'] == 1:
            real_texts.append(row['text1'])
            fake_texts.append(row['text2'])
        else:
            real_texts.append(row['text2'])
            fake_texts.append(row['text1'])
    

    real_cloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(real_texts))
    fake_cloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(fake_texts))
    
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].imshow(real_cloud, interpolation='bilinear')
    ax[0].set_title('Real Texts Word Cloud')
    ax[0].axis('off')
    
    ax[1].imshow(fake_cloud, interpolation='bilinear')
    ax[1].set_title('Fake Texts Word Cloud')
    ax[1].axis('off')
    plt.savefig("WordClouds.png", dpi=300, bbox_inches='tight')
    plt.show()

generate_wordclouds(train_df)  



from sklearn.preprocessing import StandardScaler

def select_eda_features(flat_eda_df, all_feats):
    """Select top EDA features using statistical tests and feature importance"""
    flat_eda_df['label_bin'] = flat_eda_df['label'].map({'real': 1, 'fake': 0})
    
    # 1. Feature-Label Correlation
    corrs = flat_eda_df[all_feats].corrwith(flat_eda_df['label_bin']).abs().sort_values(ascending=False)
    
    # 2. Statistical Significance Testing
    significant_features = []
    for feat in all_feats:
        real_vals = flat_eda_df[flat_eda_df['label'] == 'real'][feat]
        fake_vals = flat_eda_df[flat_eda_df['label'] == 'fake'][feat]
        _, pval = ttest_ind(real_vals, fake_vals, nan_policy='omit')
        if pval < 0.05:
            significant_features.append(feat)
    
    # 3. Feature Importance with Random Forest
    X = flat_eda_df[all_feats]
    y = LabelEncoder().fit_transform(flat_eda_df['label'])
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=all_feats).sort_values(ascending=False)

    # Combine criteria
    selected = importances[importances.index.isin(significant_features)].head(10).index.tolist()

    # 1) Top correlations
    df_corrs = corrs.head(10).to_frame(name='abs_pearson_corr')
    display(df_corrs.style.set_caption("Top 10 Feature â†” Label Correlations"))

    # 2) Significant features (t-test p < 0.05)
    df_signif = pd.DataFrame({
        'feature': significant_features,
        'p_value': [ttest_ind(
            flat_eda_df[flat_eda_df['label']=='real'][f],
            flat_eda_df[flat_eda_df['label']=='fake'][f],
            nan_policy='omit'
        )[1] for f in significant_features]
    }).sort_values('p_value')
    display(df_signif.style.set_caption(f"Statistically Significant Features (p < 0.05): {len(significant_features)}"))

    # 3) RandomForest importances
    df_imps = importances.head(10).to_frame(name='importance')
    display(df_imps.style.set_caption("Top 10 RandomForest Feature Importances"))

    # 4) Final selected
    display(
        pd.DataFrame(selected, columns=['feature'])
          .style.set_caption("Final Selected Features")
    )

    
    return selected,importances

# Feature definitions
length_feats = ['num_chars', 'num_words', 'num_sentences', 'avg_word_length', 'avg_sentence_length']
lexical_feats = ['ttr', 'hapax_legomena_ratio', 'n_unique_words']
readability_feats = ['flesch_reading_ease', 'flesch_kincaid_grade']
stylo_feats = ['stopword_ratio', 'punctuation_count', 'special_chars_count']
sentiment_feats = ['polarity', 'subjectivity']
pos_feats = ['noun_ratio', 'verb_ratio','adj_ratio','adv_ratio']
all_feats = length_feats + lexical_feats + readability_feats + stylo_feats + sentiment_feats + pos_feats

# Apply feature selection
selected_features,importances = select_eda_features(flat_eda_df, all_feats)

def create_eda_features(eda_df, selected_features):
    """Create difference features from selected EDA features"""
    diff_features = pd.DataFrame({'id': eda_df['id']})
    for feat in selected_features:
        diff_features[f'diff_{feat}'] = eda_df[f't1_{feat}'] - eda_df[f't2_{feat}']
    return diff_features


# Create EDA features
train_eda_diff = create_eda_features(eda_train, selected_features)
test_eda_diff = create_eda_features(eda_test, selected_features)

train_eda_values = train_eda_diff.drop(columns=['id']).values
test_eda_values = test_eda_diff.drop(columns=['id']).values

scaler_eda = StandardScaler()
train_eda_scaled = scaler_eda.fit_transform(train_eda_values)
test_eda_scaled = scaler_eda.transform(test_eda_values)

train_eda_sparse = csr_matrix(train_eda_scaled)
test_eda_sparse = csr_matrix(test_eda_scaled)
y_train = train_df['label'].values

print(f"\nEDA features shape: {train_eda_sparse.shape}")
print(f"Labels shape: {y_train.shape}")
print(f"Label distribution     :\n{pd.Series(y_train).value_counts()}")

# Save the EDA scaler to disk
joblib.dump(scaler_eda, 'scaler_eda.pkl')
print("EDA scaler saved to scaler_eda.pkl")


import seaborn as sns

diff_feat_df = train_eda_diff.drop(columns='id')
corr_matrix = pd.DataFrame(diff_feat_df)[[f'diff_{f}' for f in selected_features]].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation between Selected EDA Features")
plt.tight_layout()
plt.savefig("Correlation between selected feature.png", dpi=300, bbox_inches='tight')
plt.show()



importances[selected_features].sort_values().plot.barh(figsize=(8, 5))
plt.title("Selected Feature Importances")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("Selected Features RandomForest.png", dpi=300, bbox_inches='tight')
plt.show()



from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib


# ========== BERT FEATURE EXTRACTION ==========
def extract_mean_pooling_vector(text, tokenizer, model, max_len=512, stride=256, device="cuda"):
    """Extract mean-pooled BERT embeddings with sliding window"""
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        stride=stride,
        return_overflowing_tokens=True,
        padding="max_length"
    )
    input_ids_chunks = encoded["input_ids"]
    attention_mask_chunks = encoded["attention_mask"]
    all_mean_vecs = []

    model.to(device)
    model.eval()
    with torch.no_grad():
         for input_ids, attention_mask in zip(input_ids_chunks, attention_mask_chunks):
            input_ids = input_ids.unsqueeze(0).to(device)
            attention_mask = attention_mask.unsqueeze(0).to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
            masked_hidden = last_hidden_state * mask
            summed = masked_hidden.sum(dim=1)
            count = mask.sum(dim=1)
            mean_vec = summed / count
            all_mean_vecs.append(mean_vec.squeeze(0))
    return torch.stack(all_mean_vecs).mean(dim=0).cpu()

def extract_bert_features(dataset, tokenizer, model):
    """Extract BERT interaction features for all samples"""
    features = []
    ids = []
    for row in tqdm(dataset, desc="Extracting BERT features"):
        vec1 = extract_mean_pooling_vector(row['text1'], tokenizer, model)
        vec2 = extract_mean_pooling_vector(row['text2'], tokenizer, model)
        diff = vec1 - vec2
        prod = vec1 * vec2
        final_vec = torch.cat([vec1, vec2, diff, prod])
        features.append(final_vec.numpy())
        ids.append(row['id'])
    return np.array(features), ids
  
# Initialize BERT model
model_checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
bert_model = AutoModel.from_pretrained(model_checkpoint)

# Extract BERT features
X_train_bert, train_bert_ids = extract_bert_features(raw_datasets["train"], tokenizer, bert_model)
X_test_bert, test_bert_ids = extract_bert_features(raw_datasets["test"], tokenizer, bert_model)

scaler_bert = StandardScaler()
X_train_bert_scaled = scaler_bert.fit_transform(X_train_bert)
X_test_bert_scaled  = scaler_bert.transform(X_test_bert)

# Apply PCA to BERT features
n_components=50
pca = PCA(n_components=n_components)
X_train_bert_pca = pca.fit_transform(X_train_bert_scaled)
X_test_bert_pca  = pca.transform(X_test_bert_scaled)

# Save model
joblib.dump(scaler_bert, 'bert_scaler.joblib')
joblib.dump(pca, 'bert_pca.joblib')

print("Original BERT shape:", X_train_bert.shape)
print("PCA-reduced shape:", X_train_bert_pca.shape)

explained_variance = pca.explained_variance_ratio_.sum()
print(f"Total explained variance by {n_components} components: {explained_variance:.2%}")


from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

tsne_model = TSNE(n_components=2, random_state=42, perplexity=30)
X_2d = tsne_model.fit_transform(X_train_bert_pca)

# Convert to DataFrame for easier plotting
df_2d = pd.DataFrame(X_2d, columns=["x", "y"])
df_2d["label"] = raw_datasets["train"]["label"]


plt.figure(figsize=(6, 6))
sns.scatterplot(data=df_2d, x="x", y="y", hue="label", palette="Set2", alpha=0.7)
plt.title("t-SNE of BERT Embeddings")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend()
plt.tight_layout()
plt.savefig("t-SNE of BERT Embeddings.png", dpi=300, bbox_inches='tight')
plt.show()


full_pca = PCA().fit(X_train_bert)

# Plot cumulative explained variance
plt.figure(figsize=(8, 4))
plt.plot(np.cumsum(full_pca.explained_variance_ratio_), marker='o')
plt.axvline(n_components, color='red', linestyle='--', label=f'{n_components} components')
plt.title("Cumulative Explained Variance by PCA Components (BERT)")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("Cumulative Explained Variance by PCA Components.png", dpi=300, bbox_inches='tight')
plt.show()


#Cosine Similarity
hidden_size = X_train_bert.shape[1] // 4
cos_sims = [
    cosine_similarity(
        X_train_bert[i][:hidden_size].reshape(1, -1),
        X_train_bert[i][hidden_size:2*hidden_size].reshape(1, -1)
    )[0][0]
    for i in range(len(X_train_bert))
]
plt.figure(figsize=(10, 5))
scatter = plt.scatter(range(len(cos_sims)), cos_sims, c=y_train, cmap='coolwarm', alpha=0.7)

cbar = plt.colorbar(scatter, ticks=[0, 1])
cbar.ax.set_yticklabels(['Fake', 'Real']) 

plt.title("Cosine Similarities by Labels")
plt.xlabel("Sample Index")
plt.ylabel("Cosine Similarity")
plt.savefig("Cosine Similarity.png", dpi=300, bbox_inches='tight')
plt.show()



# â€” TF-IDF on [SEP]-joined pairs â€”
pairs_train = [
    f"{row['text1']} [SEP] {row['text2']}" 
    for row in raw_datasets["train"]
]
pairs_test  = [
    f"{row['text1']} [SEP] {row['text2']}"
    for row in raw_datasets["test"]
]

tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=1000)
tfidf.fit(pairs_train)

X_tfidf_train = tfidf.transform(pairs_train).toarray()
X_tfidf_test  = tfidf.transform(pairs_test).toarray()

# Save TF-IDF vectorizer and scaler
joblib.dump(tfidf, 'tfidf_vectorizer.joblib')


# Plot explained variance
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import numpy as np

svd_test = TruncatedSVD(n_components=300, random_state=42)
svd_test.fit(X_tfidf_train)

explained = np.cumsum(svd_test.explained_variance_ratio_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained) + 1), explained, marker='o')
plt.axhline(y=0.90, color='r', linestyle='--', label='90% variance')
plt.axhline(y=0.95, color='g', linestyle='--', label='95% variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('TF-IDF Reduction via Truncated SVD')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



svd = TruncatedSVD(n_components=70, random_state=42)
X_tfidf_train_reduced = svd.fit_transform(X_tfidf_train)
X_tfidf_test_reduced  = svd.transform(X_tfidf_test)

joblib.dump(svd, 'tfidf_svd_transformer.joblib')




df_eda = pd.DataFrame(train_eda_scaled,index=train_eda_diff["id"],columns=[f"eda_diff_{f}" for f in selected_features])

df_bert = pd.DataFrame(X_train_bert_pca,index=train_bert_ids,columns=[f"bert_pc{i}" for i in range(n_components)])

tf_ids = [r["id"] for r in raw_datasets["train"]]
df_tfidf = pd.DataFrame(X_tfidf_train,index=tf_ids,columns=[f"tfidf_{i}" for i in range(X_tfidf_train.shape[1])])

#Labels
raw_ids    = [r["id"] for r in raw_datasets["train"]]
raw_labels = [r["label"] for r in raw_datasets["train"]]
df_labels = pd.DataFrame({"label": raw_labels}, index=raw_ids)

#Merge all on ID
df_combined = df_labels.join([df_bert, df_tfidf, df_eda], how="inner")

#Inspect
print("ðŸ”¹ final shape:", df_combined.shape)
missing = set(raw_ids) - set(df_combined.index)
print("ðŸ”¹ any missing IDs from merge?  ", missing or "None")



# Get top 20 important n-grams
feature_names = tfidf.get_feature_names_out()
importances = np.mean(X_tfidf_train_reduced, axis=0)  # Average TF-IDF score
top_indices = np.argsort(importances)[-10:]
top_ngrams = [feature_names[i] for i in top_indices]
top_scores = importances[top_indices]

# Plot horizontal bar chart
plt.figure(figsize=(10,8))
plt.barh(top_ngrams, top_scores, color='skyblue')
plt.xlabel('TF-IDF Importance Score')
plt.title('Top Discriminative N-grams in [SEP] Pairs')
plt.tight_layout()
plt.savefig("Top Discriminative N-grams.png", dpi=300, bbox_inches='tight')
plt.show()


import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import joblib

# Combine BERT, TF-IDF, and EDA features
X_train_scaled = np.hstack([X_train_bert_pca, X_tfidf_train_reduced, train_eda_scaled])
X_test_scaled  = np.hstack([X_test_bert_pca,  X_tfidf_test_reduced,  test_eda_scaled])

models = {
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_split=10,
            min_samples_leaf=5, random_state=42),
    
    "XGBoost": XGBClassifier(n_estimators=200, eval_metric='logloss', random_state=42, n_jobs=-1),
    
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
    
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    
    "SVM": SVC(C=0.5, kernel='rbf', gamma='scale', probability=True, random_state=42),
    
    "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu',
                         solver='adam', alpha=1e-4, learning_rate_init=0.001,
                         max_iter=500, random_state=42)
}

#Evaluation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
summary = []
cv_records = []
model_scores = {}

print("\nCross-Validation Results (Average over 5 folds):")
summary = []

all_preds = {}  # To store predictions per model for ROC/PR
all_probas = {}  # For predicted probabilities
all_trues = {}   # For true labels

for name, model in models.items():
    acc_scores = []
    prec_scores = []
    rec_scores = []
    f1_scores = []

    fold_preds = []
    fold_probas = []
    fold_trues = []
    
    for fold_i, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled, y_train), 1):
        X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)

        fold_preds.extend(y_pred)
        fold_trues.extend(y_val)
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_val)[:, 1]
        else:
            y_score = model.decision_function(X_val)
        fold_probas.extend(y_score)
        
        acc = accuracy_score(y_val, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='binary')

        acc_scores.append(acc)
        prec_scores.append(prec)
        rec_scores.append(rec)
        f1_scores.append(f1)
        
        cv_records.append({
            "Model": name,"Fold": f"Fold {fold_i}","Accuracy": acc,"Precision": prec,"Recall": rec,"F1": f1})
    
    summary.append({"Model": name,"Accuracy_Mean": np.mean(acc_scores),"Precision_Mean": np.mean(prec_scores),"Recall_Mean":np.mean(rec_scores),"F1_Mean": np.mean(f1_scores),})
    
    model_scores[name] = np.mean(acc_scores)

    # Store for plotting later
    all_preds[name] = np.array(fold_preds)
    all_probas[name] = np.array(fold_probas)
    all_trues[name] = np.array(fold_trues)

cv_summary_df = pd.DataFrame(summary).set_index("Model")

display(
    cv_summary_df.sort_values(by="Accuracy_Mean", ascending=False).style.format({
        "Accuracy_Mean": "{:.4f}",
        "Precision_Mean": "{:.4f}",
        "Recall_Mean": "{:.4f}",
        "F1_Mean": "{:.4f}",
    })
)

# ========== Select and Save Best Model ==========
best_model_name = max(model_scores, key=model_scores.get)
best_model = models[best_model_name]

# Train on full data
best_model.fit(X_train_scaled, y_train)
joblib.dump(best_model, f"{best_model_name.replace(' ', '_').lower()}_model.joblib")
print(f"\nBest model selected: {best_model_name} â€” saved as .joblib")


from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

for name in models.keys():
    y_true = all_trues[name]
    y_pred = all_preds[name]

    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred, digits=4)

    print(f"\n=== {name} ===")
    print(cr)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix: {name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()



from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

for name in models.keys():
    y_true = all_trues[name]
    y_score = all_probas[name]
    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.title(f"ROC Curve - {name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



from sklearn.metrics import precision_recall_curve, auc

for name in models.keys():
    y_true = all_trues[name]
    y_score = all_probas[name]
    
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=f"AUC = {pr_auc:.2f}")
    plt.title(f"Precision-Recall Curve - {name}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



cv_df = pd.DataFrame(cv_records)
cv_melt = pd.melt(cv_df, id_vars=['Model', 'Fold'], 
                  value_vars=['Accuracy', 'Precision', 'Recall', 'F1'],
                  var_name='Metric', value_name='Score')

plt.figure(figsize=(12, 6))
sns.barplot(data=cv_melt, x='Model', y='Score', ci=None, hue='Metric', errwidth=2)
plt.title("Model Comparison Across Folds")
plt.ylim(0.7, 1.0)
plt.xticks(rotation=45)
plt.legend(title='Metric')
plt.tight_layout()
plt.show()



g = sns.catplot(
    data=cv_melt,
    x='Model', y='Score', hue='Fold',
    col='Metric', kind='bar',
    height=4, aspect=1, ci=None
)
g.set_titles("{col_name}")
g.set_xticklabels(rotation=45)
g.fig.subplots_adjust(top=0.85)
g.fig.suptitle("Model Performance by Metric and Fold")
plt.show()



import seaborn as sns
import matplotlib.pyplot as plt

cv_df = pd.DataFrame(cv_records)

# Accuracy grouped bar plot
plt.figure(figsize=(12, 6))
sns.barplot(data=cv_df, x="Model", y="Accuracy", hue="Fold")
plt.title("Model Accuracy Across Folds")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.ylim(0.0, 1.05)
plt.legend(title="Fold")
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.savefig("Fold Accuracy.png", dpi=300, bbox_inches='tight')
plt.show()

# F1 Score grouped bar plot
plt.figure(figsize=(12, 6))
sns.barplot(data=cv_df, x="Model", y="F1", hue="Fold")
plt.title("Model F1 Score Across Folds")
plt.ylabel("F1 Score")
plt.xticks(rotation=45)
plt.ylim(0.0, 1.05)
plt.legend(title="Fold")
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.savefig("Fold f1 score.png", dpi=300, bbox_inches='tight')
plt.show()



import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
    precision_recall_curve, average_precision_score
)

# 2) Predict on the full training set
y_pred = best_model.predict(X_train_scaled)
if hasattr(best_model, "predict_proba"):
    y_proba = best_model.predict_proba(X_train_scaled)[:, 1]
else:
    y_proba = best_model.decision_function(X_train_scaled)
    y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())

# 7) Feature Importances 

bert_feature_names = [f'BERT_PCA_{i+1}' for i in range(X_train_bert_pca.shape[1])]
tfidf_feature_names = tfidf.get_feature_names_out()
eda_feature_names = [c for c in train_eda_diff.columns if c != 'id']
all_feature_names = list(bert_feature_names) + list(tfidf_feature_names) + eda_feature_names


fig, ax = plt.subplots(figsize=(8,6))
if hasattr(best_model, "feature_importances_"):
    imp = best_model.feature_importances_
    names = all_feature_names  # ensure this list exists as in your importances code
    order = np.argsort(imp)[-20:]
    ax.barh(range(len(order)), imp[order], color='skyblue')
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[i] for i in order])
    ax.set_title(f"Top 20 Feature Importances ({best_model_name})")
    ax.set_xlabel("Importance")
elif hasattr(best_model, "coef_"):
    coef = best_model.coef_.ravel()
    names = all_feature_names
    order = np.argsort(np.abs(coef))[-20:]
    ax.barh(range(len(order)), coef[order], color='lightcoral')
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[i] for i in order])
    ax.set_title(f"Top 20 Coefficients ({best_model_name})")
    ax.set_xlabel("Coefficient value")
else:
    ax.text(0.5, 0.5, "No feature importances available", ha='center')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("Feature Importance of best model.png", dpi=300, bbox_inches='tight')
plt.show()



# ========== PREDICTION & SUBMISSION ==========
# Predict probabilities
test_probs = best_model.predict_proba(X_test_scaled)[:, 1]

# Create submission
submission = []
for i, pid in enumerate(test_bert_ids):
    real_text_id = 1 if test_probs[i] >= 0.5 else 2
    submission.append((pid, real_text_id))

submission_df = pd.DataFrame(submission, columns=["id", "real_text_id"])
submission_df.to_csv("submission.csv", index=False)
print("\n Submission saved!")
print(submission_df.head())


!zip -r /kaggle/working/aug_02.zip /kaggle/working/*





