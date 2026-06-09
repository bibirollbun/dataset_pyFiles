import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re


# 1. Configuración de NLTK
try:
    stop = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    stop = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """Limpia y lematiza el texto."""
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))
    text = text.lower().strip()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop])
    return text


# 2. Cargar datos crudos
print("Cargando datasets...")
train = pd.read_csv('/kaggle/input/petfinder-adoption-prediction/train/train.csv').set_index("PetID")
test = pd.read_csv('/kaggle/input/petfinder-adoption-prediction/test/test.csv').set_index("PetID")


# 3. Limpieza
print("Limpiando descripciones...")
train["Description"] = train["Description"].fillna("empty").astype(str).apply(clean_text)
test["Description"] = test["Description"].fillna("empty").astype(str).apply(clean_text)


# 4. TF-IDF (Aumentamos max_features ya que se guarda en disco)
print("Vectorizando TF-IDF...")
vectorizer = TfidfVectorizer(stop_words=stop, ngram_range=(1, 2), max_features=1000)
X_train_desc = vectorizer.fit_transform(train['Description']).toarray()
X_test_desc = vectorizer.transform(test['Description']).toarray()


# 5. Crear DataFrames con índice PetID
feat_cols = [f"text_feat_{i+1}" for i in range(X_train_desc.shape[1])]
train_text = pd.DataFrame(X_train_desc, index=train.index, columns=feat_cols)
test_text = pd.DataFrame(X_test_desc, index=test.index, columns=feat_cols)


# 6. Guardar OUTPUTS
print("Guardando parquets...")
train_text.to_parquet("train_text.parquet")
test_text.to_parquet("test_text.parquet")
print("¡Listo! Haz Commit de este notebook.")


import os, glob

print("Directorio actual:", os.getcwd())
print("Archivos parquet encontrados:")
print(glob.glob("*.parquet"))


