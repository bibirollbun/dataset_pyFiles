# --- IMPORTATION DES LIBRAIRIES ---
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import tensorflow as tf
from numpy import array

# Bibliothèque pour transformer le texte en tokens
from tensorflow.keras.preprocessing.text import Tokenizer

# Installation des librairies nécessaires (Transformers pour BERT)
!pip install transformers tensorflow

# Import du tokenizer et du modèle DistilBERT
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf



# Nom du modèle pré-entraîné (DistilBERT finetuné sur SST-2 pour l’analyse de sentiment)
model_name = "distilbert-base-uncased-finetuned-sst-2-english"

# Chargement du tokenizer associé à DistilBERT
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

# Chargement du modèle de classification de séquences (binaire : positif/négatif)
model = TFDistilBertForSequenceClassification.from_pretrained(model_name)



df = pd.read_csv('/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip', delimiter='\t')
tsv_file_path = '/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip'
df_un = pd.read_csv(tsv_file_path, delimiter='\t', quoting=3)

tsv_file_path = '/kaggle/input/word2vec-nlp-tutorial/testData.tsv.zip'
df_test = pd.read_csv(tsv_file_path, delimiter='\t', quoting=3)



# Supprimer les balises HTML
TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)

def preprocess_text(sen):
    # 1. Supprimer les balises HTML
    sentence = remove_tags(sen)
    
    # 2. Supprimer tout caractère non alphabétique
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # 3. Supprimer les lettres isolées (ex: " a ")
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # 4. Supprimer les espaces multiples
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence



# Prétraitement de toutes les critiques (reviews)
X_train = []
sentences = list(df['review'])
for sen in sentences:
    X_train.append(preprocess_text(sen))

# Extraction des labels (0 = négatif, 1 = positif)
y = df['sentiment']

# Conversion en tableau numpy
y_train = np.array(list(map(lambda x: 1 if x==1 else 0, y)))



X_test = []
sentences = list(df_test['review'])
for sen in sentences:
    X_test.append(preprocess_text(sen))


le = 100  # taille du batch (nombre de phrases à traiter à la fois)
s = 0
probs = []

# Boucle sur les données de test pour tout prédire par lots
while s < len(X_test):
    # Tokenisation (retourne un dict avec 'input_ids' et 'attention_mask')
    inputs = tokenizer(X_test[s:s + le], return_tensors="tf", padding=True, truncation=True, max_length=512)
    s += le

    # Passage dans le modèle
    outputs = model(inputs)
    logits = outputs.logits

    # Transformation en probabilités via softmax
    probs.extend(tf.nn.softmax(logits, axis=1))



def convert_array(x):
    if x >= 0.5:
        return 1
    return 0

# Prendre la probabilité de la classe "positive"
y_final = [convert_array(probs[i].numpy()[1]) for i in range(len(probs))]

# Création d’un DataFrame avec les prédictions
result = pd.DataFrame({"id":df_test["id"], "sentiment":y_final})

# Sauvegarde des résultats
result.to_csv("output.csv", index=False, quoting=3, escapechar='\\')



from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# 1) Préparer les textes et labels
X_all = [preprocess_text(t) for t in df["review"].tolist()]
y_all = df["sentiment"].values

# 2) Séparer en train/validation
X_tr, X_va, y_tr, y_va = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=42)

# 3) Fonction de prédiction par lots
def predict_probs(texts, batch_size=128, max_len=256):
    probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="tf", padding=True, truncation=True, max_length=max_len)
        outputs = model(**inputs)
        logits  = outputs.logits
        p = tf.nn.softmax(logits, axis=-1)[:, 1]  # probabilité classe positive
        probs.append(p.numpy())
    return np.concatenate(probs, axis=0)

# 4) Calcul des prédictions et métriques
p_va = predict_probs(X_va)
y_pred = (p_va >= 0.5).astype(int)

acc = accuracy_score(y_va, y_pred)
auc = roc_auc_score(y_va, p_va)
print(f"Accuracy: {acc*100:.2f}%")
print(f"ROC AUC : {auc*100:.2f}%")





