# Importo Librerías

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD


# Cargar datos

train = pd.read_csv('../input/petfinder-adoption-prediction/train/train.csv').set_index("PetID")
test = pd.read_csv('../input/petfinder-adoption-prediction/test/test.csv').set_index("PetID")


# Defino el modelo

model_path = "/kaggle/input/deberta-v3-base/deberta-v3-base"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    print(f"✓ Modelo cargado: {model_path}")
except:
    # Fallback a BERT base si falla
    print("⚠️ Usando BERT base como fallback")
    model_path = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print(f"Dispositivo: {device}")


# Función para extraer embeddings (mean pooling)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def extract_text_embeddings(texts, batch_size=16):
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Extrayendo embeddings"):
        batch = texts[i:i + batch_size]
        
        # Tokenizar
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,  # 256 es suficiente para descripciones
            return_tensors='pt'
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        # Extraer embeddings
        with torch.no_grad():
            output = model(**encoded)
            embeddings = mean_pooling(output, encoded['attention_mask'])
            # Normalizar embeddings (mejora similitud coseno)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        all_embeddings.append(embeddings.cpu().numpy())
    
    return np.vstack(all_embeddings)


# Procesar textos
train_texts = train.Description.fillna("").astype(str).tolist()
test_texts = test.Description.fillna("").astype(str).tolist()

print(f"Procesando {len(train_texts)} textos de entrenamiento...")
train_embeddings = extract_text_embeddings(train_texts, batch_size=16)

print(f"Procesando {len(test_texts)} textos de prueba...")
test_embeddings = extract_text_embeddings(test_texts, batch_size=16)


# Reducir dimensionalidad (de 768 a 50 componentes)
print("Reduciendo dimensionalidad con SVD...")
svd = TruncatedSVD(n_components=50, random_state=42)
train_reduced = svd.fit_transform(train_embeddings)
test_reduced = svd.transform(test_embeddings)

print(f"Varianza explicada: {svd.explained_variance_ratio_.sum():.2%}")


# Features estadísticas adicionales
def compute_text_stats(texts):
    stats = []
    for text in tqdm(texts, desc="Calculando estadísticas"):
        words = text.split()
        stats.append({
            'text_length': len(text),
            'word_count': len(words),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'unique_word_ratio': len(set(words)) / len(words) if words else 0,
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'capital_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'has_description': int(len(text.strip()) > 0),
            'comma_count': text.count(','),
            'period_count': text.count('.'),
            'digit_count': sum(1 for c in text if c.isdigit()),
        })
    return pd.DataFrame(stats)

train_stats = compute_text_stats(train_texts)
test_stats = compute_text_stats(test_texts)


# Combinar embeddings + estadísticas

train_df = pd.DataFrame(
    train_reduced,
    columns=[f'text_emb_{i}' for i in range(50)],
    index=train.index
)
train_df = pd.concat([train_df, train_stats.set_index(train.index)], axis=1)

test_df = pd.DataFrame(
    test_reduced,
    columns=[f'text_emb_{i}' for i in range(50)],
    index=test.index
)
test_df = pd.concat([test_df, test_stats.set_index(test.index)], axis=1)


# Guardo los datasets en parquet

train_df.to_parquet("train_text.parquet")
test_df.to_parquet("test_text.parquet")

