import transformers
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset



# Ruta del archivo en el entorno de Kaggle
train_path = "/kaggle/input/traveler-insights-utb/train.csv"
test_path = "/kaggle/input/traveler-insights-utb/test.csv"
sub_path = "/kaggle/input/traveler-insights-utb/submission.csv"
# Leer el archivo
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
df_sub = pd. read_csv(sub_path)
df_train = df_train[['Comentario', 'Sentimiento']].dropna()
df_train['Sentimiento'] = df_train['Sentimiento'].map({'positivo': 2, 'neutral': 1, 'negativo': 0})
# Mostrar las primeras filas para confirmar carga
df_train


df_train['Sentimiento'].value_counts()


!pip install scikit-learn==1.3.2
!pip install imbalanced-learn==0.11.0



from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

# Separar features y labels
X = df_train['Comentario']
y = df_train['Sentimiento']

# Convertir X a matriz porque RandomOverSampler no acepta series sueltas
X = X.values.reshape(-1, 1)

# Aplicar oversampling
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Convertir de nuevo a DataFrame
df_train_balanced = pd.DataFrame({
    'Comentario': X_resampled.flatten(),
    'Sentimiento': y_resampled
})

df_train_balanced['Sentimiento'].value_counts()



train_df, val_df = train_test_split(df_train_balanced, test_size=0.2, random_state=42, stratify=df_train_balanced['Sentimiento'])


# Textos y etiquetas del entrenamiento
train_texts = df_train['Comentario'].tolist()
train_labels = df_train['Sentimiento'].tolist()

# Textos y etiquetas del test
test_texts = df_test['Comentario'].tolist()
test_labels = df_test['Sentimiento'].tolist()


import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


# Stopwords en español
stopwords_es = stopwords.words('spanish')


# Crear vectorizador
vectorizer = TfidfVectorizer(stop_words=stopwords_es)


# Aplicar vectorización
X_train_tfidf = vectorizer.fit_transform(train_texts)
X_test_tfidf = vectorizer.transform(test_texts)


X_train = train_df['Comentario']
y_train = train_df['Sentimiento']

X_val = val_df['Comentario']
y_val = val_df['Sentimiento']


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
    ('svm', LinearSVC(class_weight=None))
])



model.fit(X_train, y_train)


from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


from sklearn.metrics import classification_report, accuracy_score

y_pred = model.predict(X_val)

print("Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))


predicciones = model.predict(df_sub['Comentario'])


df_sub['Sentimiento'] = predicciones


df_sub[['ID', 'Sentimiento']].to_csv("submission.csv", index=False)


