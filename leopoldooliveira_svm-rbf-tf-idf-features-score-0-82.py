import os
import sys
import re
import math
import string
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

from sklearn.feature_selection import f_classif, mutual_info_classif
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pathlib import Path
from collections import Counter
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler
)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_recall_fscore_support
)
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from scipy.sparse import hstack, csr_matrix


nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('punkt')

# proporÃ§Ã£o de stopword
stopwords_set = set(stopwords.words('english'))


train_rules = pd.read_csv("data/train.csv")
train_rules.head()


def cria_caminho_article(text_id, article):
    # FunÃ§Ã£o para criar o caminho de leitura
    caminho = f"data/train/article_{article}/file_{text_id}.txt"
    return caminho
    
def leitura_arquivo(file_path):
    # FunÃ§Ã£o para ler o caminho criado
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    return text


# CriaÃ§Ã£o do dataset de treino
train_rules['fake_text_id'] = train_rules['real_text_id'].apply(lambda x: 2 if x == 1 else 1)
train_rules['article'] = train_rules['id'].apply(lambda x: str(x).zfill(4))

train_rules['real_text_file'] = train_rules[['real_text_id', 'article']].apply(lambda x: cria_caminho_article(x['real_text_id'], x['article']), axis=1)
train_rules['fake_text_file'] = train_rules[['fake_text_id', 'article']].apply(lambda x: cria_caminho_article(x['fake_text_id'], x['article']), axis=1)

train_rules['real_text'] = train_rules['real_text_file'].apply(leitura_arquivo)
train_rules['fake_text'] = train_rules['fake_text_file'].apply(leitura_arquivo)


train_rules.head()


# NormalizaÃ§Ã£o do dataset
# Artigos apenas com os textos reais
df_real = train_rules[['article', 'real_text']].copy()
df_real.columns = ['article', 'texto']  # Padroniza nomes
df_real['label'] = 1

# Artigos apenas com os textos fakes
df_fake = train_rules[['article', 'fake_text']].copy()
df_fake.columns = ['article', 'texto']
df_fake['label'] = 0

# Concatenar os dois dataframes (real + fake)
df_completo = pd.concat([df_real, df_fake], ignore_index=True)

# Exibir resultado
df_completo.head()


n_artigo = '0090'
print(f"Label: {df_completo[df_completo['article']==n_artigo].label.iloc[0]}")
df_completo[df_completo['article']==n_artigo].texto.iloc[0]


print(f"Label: {df_completo[df_completo['article']==n_artigo].label.iloc[1]}")
df_completo[df_completo['article']==n_artigo].texto.iloc[1]


# Estilo visual (opcional)
sns.set(style='whitegrid')

# Obtem contagens
contagens = df_completo['label'].value_counts().sort_index()

# Cria figura e eixo
fig, ax = plt.subplots(figsize=(6, 5))  # Altere o tamanho aqui se quiser

# Cores personalizadas
cores = ['#1f77b4', '#ff7f0e']  # azul / laranja

# Plot com altura ajustÃ¡vel
bars = ax.bar(contagens.index.astype(str), contagens.values, color=cores)

# Adiciona os valores no topo das barras
for bar in bars:
    altura = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, altura + 1,
            f'{int(altura)}', ha='center', va='bottom', fontsize=12, weight='bold')

# Labels e tÃ­tulo
ax.set_xlabel('Classe', fontsize=12)
ax.set_ylabel('Quantidade', fontsize=12)
ax.set_title('DistribuiÃ§Ã£o das classes (real vs. fake)', fontsize=14)

plt.tight_layout()
plt.show()


def extract_features(text):
    words = word_tokenize(text.lower())
    word_count = len(words)
    non_latin = len(re.findall(r'[^\x00-\x7F]', text))
    punct_count = sum(1 for c in text if c in string.punctuation)
    num_lines = text.count('\n')
    stop_count = sum(1 for w in words if w in stopwords_set)
    long_words = sum(1 for w in words if len(w) > 15)
    ent = entropy(text)
    num_sentences = text.count('.') + text.count('!') + text.count('?') + text.count('-')
    avg_sent_len = word_count / (num_sentences + 1)

    return {
        'texto_len': word_count,
        'non_latin_chars': non_latin,
        'num_pontuacao': punct_count,
        'num_linhas': num_lines,
        'stopword_ratio': stop_count / word_count if word_count > 0 else 0,
        'long_words': long_words,
        'entropia': ent,
        'num_sentencas': num_sentences,
        'avg_sentence_len': avg_sent_len
    }


def entropy(text):
    if len(text) == 0:
        return 0
    probs = [v / len(text) for v in Counter(text).values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)




feature_df = df_completo['texto'].apply(extract_features).apply(pd.Series)
df_completo = pd.concat([df_completo, feature_df], axis=1)


df_completo.head()


df_completo.describe().T


df_completo.groupby('label')[[
    'texto_len', 'non_latin_chars', 'num_pontuacao',
    'num_linhas', 'stopword_ratio', 'long_words',
    'entropia', 'num_sentencas', 'avg_sentence_len'
]].describe().T


# Mostra todas as linhas e colunas
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Agora sim a visualizaÃ§Ã£o completa:
df_completo.groupby('label')[[
    'texto_len', 'non_latin_chars', 'num_pontuacao',
    'num_linhas', 'stopword_ratio', 'long_words',
    'entropia', 'num_sentencas', 'avg_sentence_len'
]].describe().T




features = ['texto_len', 'non_latin_chars', 'num_pontuacao', 'num_linhas',
            'stopword_ratio', 'long_words', 'entropia', 'num_sentencas', 'avg_sentence_len']

# Grid 3x3
fig, axes = plt.subplots(3, 3, figsize=(18, 12))  # (linhas, colunas)
fig.suptitle('DistribuiÃ§Ã£o das Features por Classe (0=fake, 1=real)', fontsize=16)

for i, feature in enumerate(features):
    row = i // 3
    col = i % 3
    ax = axes[row, col]
    sns.boxplot(x='label', y=feature, data=df_completo, ax=ax)
    ax.set_title(feature)
    ax.set_xlabel("")
    ax.set_ylabel("")

plt.tight_layout(rect=[0, 0, 1, 0.96])  # ajuste para o tÃ­tulo principal
plt.show()



features = [
    'texto_len', 'non_latin_chars', 'num_pontuacao', 'num_linhas',
    'stopword_ratio', 'long_words', 'entropia', 'num_sentencas', 'avg_sentence_len'
]

# Define quais variÃ¡veis sÃ£o contagens
clip_zero = {
    'texto_len', 'non_latin_chars', 'num_pontuacao', 'num_linhas',
    'long_words', 'num_sentencas'
}

# Configura grid
n_cols = 3
n_rows = (len(features) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 12))
axes = axes.flatten()

for i, feature in enumerate(features):
    ax = axes[i]

    clip_range = (0, df_completo[feature].max()) if feature in clip_zero else None

    sns.kdeplot(data=df_completo[df_completo['label'] == 0], x=feature, label='Fake',
                fill=True, ax=ax, clip=clip_range)
    sns.kdeplot(data=df_completo[df_completo['label'] == 1], x=feature, label='Real',
                fill=True, ax=ax, clip=clip_range)

    ax.set_title(f'Densidade de {feature}')
    ax.legend()

# Remove subplots extras se houver
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


X = df_completo[features]
y = df_completo['label']

# ANOVA F-value
f_vals, p_vals = f_classif(X, y)

# Mutual Information
mi_scores = mutual_info_classif(X, y, random_state=42)

ranking = pd.DataFrame({
    'feature': X.columns,
    'anova_f': f_vals,
    'anova_p': p_vals,
    'mutual_info': mi_scores
}).sort_values(by='mutual_info', ascending=False)

ranking


df_corr = df_completo[features]

# 2. Calcular a matriz de correlaÃ§Ã£o
correlation_matrix = df_corr.corr()

# 3. Plotar heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Mapa de CorrelaÃ§Ã£o entre Features")
plt.tight_layout()
plt.show()


# Apenas colunas Ãºteis
colunas_uteis = ['article', 'texto', 'texto_len', 'stopword_ratio', 'entropia', 'label', 'avg_sentence_len']
df_final = df_completo[colunas_uteis].copy()
df_final.head()


columns_x = ['texto_len', 'stopword_ratio', 'entropia', 'avg_sentence_len']
columns_y = 'label'
X = df_final[columns_x]
y = df_final[columns_y]


X.head()


# Shuffle dos dados
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

# VetorizaÃ§Ã£o TF-IDF
tfidf = TfidfVectorizer(max_features=500)
X_tfidf = tfidf.fit_transform(df_final['texto'])

# NormalizaÃ§Ã£o das features manuais
scaler = StandardScaler()
X_extra = scaler.fit_transform(df_final[columns_x])

# CombinaÃ§Ã£o das features
from scipy.sparse import hstack, csr_matrix
X_combined = hstack([X_tfidf, X_extra])
X_combined = csr_matrix(X_combined)  # Permite indexaÃ§Ã£o

y = df_final['label'].values


modelos = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

# Cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for nome, modelo in modelos.items():
    accuracies, precisions, recalls, f1s = [], [], [], []

    for train_index, test_index in kf.split(X_combined, y):
        X_train, X_test = X_combined[train_index], X_combined[test_index]
        y_train, y_test = y[train_index], y[test_index]

        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')

        accuracies.append(acc)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    print(f"\nğŸ“Œ Modelo: {nome}")
    print(f"Accuracy:  {np.mean(accuracies):.4f}")
    print(f"Precision: {np.mean(precisions):.4f}")
    print(f"Recall:    {np.mean(recalls):.4f}")
    print(f"F1-score:  {np.mean(f1s):.4f}")


# 1. DefiniÃ§Ã£o do modelo e pipeline
svm = SVC(probability=True)

pipeline = Pipeline([
    ('clf', svm)
])

# 2. ParÃ¢metros a testar
param_grid = {
    'clf__kernel': ['linear', 'rbf'],
    'clf__C': [0.1, 1, 10],
    'clf__gamma': ['scale', 'auto']  # sÃ³ afeta kernel rbf
}

# 3. Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='f1_macro',
    cv=cv,
    n_jobs=-1,  # usa todos os nÃºcleos
    verbose=2
)

# 4. Executar GridSearch
grid.fit(X_combined, y)

# 5. Resultados
print("ğŸ”� Melhor combinaÃ§Ã£o de parÃ¢metros:")
print(grid.best_params_)
print("\nğŸ“ˆ Melhor F1-score mÃ©dio (validaÃ§Ã£o cruzada):")
print(grid.best_score_)


# treinamento do melhor modelo
best_model = grid.best_estimator_

# 2. Reajustar (re-treinar) com todos os dados disponÃ­veis
best_model.fit(X_combined, y)



# Dados de SubmissÃ£o
data_path = Path.cwd() / 'data' / 'test'

# Coleta todas as subpastas (apenas diretÃ³rios)
pastas = [folder for folder in data_path.iterdir() if folder.is_dir()]

# Arquivos desejados por pasta
nomes_arquivos = ["file_1.txt", "file_2.txt"]

# Lista de caminhos completos para cada arquivo
arquivos = [pasta / nome for pasta in pastas for nome in nomes_arquivos]

# Exibir como DataFrame
df_arquivos = pd.DataFrame(arquivos, columns=["caminho_completo"])
df_arquivos.head()


# CriaÃ§Ã£o do dataframe de submissÃ£o
df_submission = df_arquivos.copy()

# ExtraÃ§Ã£o de colunas a partir do caminho
df_submission["texto"] = df_submission["caminho_completo"].apply(leitura_arquivo)
df_submission["article"] = df_submission["caminho_completo"].astype(str).str.extract(r'(article_\d{4})')
df_submission["file_id"] = df_submission["caminho_completo"].astype(str).str.extract(r'file_(\d)\.txt').astype(int)

# VisualizaÃ§Ã£o inicial
df_submission.head()



# 1. Extrai features e concatena com df_submission
feature_df = df_submission["texto"].apply(extract_features).apply(pd.Series)
df_sub = pd.concat([df_submission, feature_df], axis=1)

# 2. Define as colunas Ãºteis para validaÃ§Ã£o/inferÃªncia (sem 'label', mas incluindo 'file_id')
colunas_uteis_validation = [col for col in colunas_uteis if col != 'label']
colunas_uteis_validation.append('file_id')

# 3. Seleciona apenas as colunas Ãºteis
df_final_test = df_sub[colunas_uteis_validation].copy()

# 4. VisualizaÃ§Ã£o rÃ¡pida
df_final_test.head()


# --- VetorizaÃ§Ã£o e normalizaÃ§Ã£o das features de teste ---
# 1. TF-IDF do texto (usando vetorizer treinado)
X_text = tfidf.transform(df_final_test["texto"])

# 2. NormalizaÃ§Ã£o das features manuais (usando scaler treinado)
X_manual = scaler.transform(df_final_test[columns_x])  # columns_x jÃ¡ contÃ©m as colunas numÃ©ricas Ãºteis

# 3. Combina texto e features manuais
X_combined = hstack([X_text, X_manual])

# --- GeraÃ§Ã£o dos scores (probabilidades de ser classe 1) ---
df_final_test["score"] = best_model.predict_proba(X_combined)[:, 1]




# --- GeraÃ§Ã£o do dataframe de submissÃ£o final ---
# 1. Seleciona, por artigo, o file_id com maior probabilidade de ser "real"
df_submission_result = df_final_test.loc[df_final_test.groupby("article")["score"].idxmax()].copy()

# 2. Renomeia para o formato exigido pela competiÃ§Ã£o
df_submission_result = df_submission_result[["file_id"]].rename(columns={"file_id": "real_text_id"})

# 3. Cria coluna sequencial de IDs (comeÃ§ando de 1)
df_submission_result["id"] = range(1, len(df_submission_result) + 1)

# 4. Reorganiza as colunas
df_submission_result = df_submission_result[["id", "real_text_id"]]

# 5. Exporta o arquivo final de submissÃ£o
df_submission_result.to_csv("submission.csv", index=False)



df_sub.head()


df_submission_result.head()

