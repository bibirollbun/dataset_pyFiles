# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install -q transformers==4.43.3
import warnings
warnings.filterwarnings("ignore")



import datasets
import huggingface_hub
import pyarrow
import pydantic
import gradio


!pip install -U transformers==4.30.2
!pip install -U accelerate


import pandas as pd
import os

# ğŸ“Œ 1. Ver archivos disponibles en /kaggle/input
input_path = "/kaggle/input"
print("ğŸ“� Archivos disponibles en /kaggle/input:")
for root, dirs, files in os.walk(input_path):
    for file in files:
        print(os.path.join(root, file))

# ğŸ“Œ 2. Rutas automÃ¡ticas (ajusta los nombres segÃºn aparezcan arriba)
train_path = None
test_path = None
sample_path = None

for root, dirs, files in os.walk(input_path):
    for file in files:
        lower = file.lower()
        full_path = os.path.join(root, file)
        if "train" in lower and lower.endswith(".csv"):
            train_path = full_path
        elif "test" in lower and lower.endswith(".csv"):
            test_path = full_path
        elif "sample" in lower or "submission" in lower:
            sample_path = full_path

# 3. Cargar archivos si existen
if train_path:
    train_df = pd.read_csv(train_path)
    print("\nâœ… train cargado:", train_path)
    print(train_df.head())
else:
    print("\nâš  No se encontrÃ³ train.csv")

if test_path:
    test_df = pd.read_csv(test_path)
    print("\nâœ… test cargado:", test_path)
    print(test_df.head())
else:
    print("\nâš  No se encontrÃ³ test.csv")

if sample_path:
    sample_df = pd.read_csv(sample_path)
    print("\nâœ… sample_submission cargado:", sample_path)
    print(sample_df.head())
else:
    print("\nâš  No se encontrÃ³ sample_submission.csv")



import pandas as pd

# âœ… Lectura de archivos desde Kaggle con manejo de errores
train_df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv', quoting=3, on_bad_lines='skip')
test_df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv', quoting=3, on_bad_lines='skip')
sample_submission = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv', quoting=3, on_bad_lines='skip')

# âœ… Mostrar valores faltantes por archivo
print("âœ… train cargado:", train_df.shape)
print(train_df.isnull().sum())

print("\nâœ… test cargado:", test_df.shape)
print(test_df.isnull().sum())

print("\nâœ… sample_submission cargado:", sample_submission.shape)
print(sample_submission.head())



from sklearn.preprocessing import LabelEncoder

# âœ… 1. Corregir nombre de columna si viene con caracteres extra (solo por seguridad)
if "Misconcpetion;;;" in train_df.columns:
    train_df.rename(columns={"Misconcpetion;;;": "Misconception"}, inplace=True)

# âœ… 2. Crear la columna objetivo combinando Category y Misconception
train_df["target"] = train_df["Category"] + ":" + train_df["Misconception"]

# âœ… 3. Codificar las clases con LabelEncoder
le = LabelEncoder()
train_df["label"] = le.fit_transform(train_df["target"])

# âœ… 4. NÃºmero total de clases
num_labels = len(le.classes_)
print("Total de clases:", num_labels)

# âœ… 5. Verificar columnas disponibles
print(train_df.columns.tolist())



# âœ… Texto de entrada (solo explicaciÃ³n del estudiante)
X = train_df["StudentExplanation"]

# âœ… Etiquetas numÃ©ricas ya codificadas
y = train_df["label"]

print("Ejemplo de texto:", X.iloc[0])
print("Ejemplo de etiqueta:", y.iloc[0])



train_df['StudentExplanation'] = train_df['StudentExplanation'].str.lower()




import numpy as np

# TamaÃ±o del vector (puedes cambiar 300 por otro valor si quieres)
vector_size = 300

# Crear matriz de ceros con tantas filas como train_df
X_vectors = np.zeros((len(train_df), vector_size))

# Variable objetivo
y = train_df['label']

print("X_vectors shape:", X_vectors.shape)
print("y shape:", y.shape)



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X_vectors, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_vectors, y, test_size=0.2, random_state=42
)



from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)



import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report



# Rellenar valores vacÃ­os
train_df['StudentExplanation'] = train_df['StudentExplanation'].fillna('')

# Convertir a minÃºsculas
train_df['StudentExplanation'] = train_df['StudentExplanation'].str.lower()



from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(train_df['StudentExplanation'])



y = train_df['label']



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)




print(type(X_train))
print(type(y_train))
print(X_train.shape)
print(y_train[:10])




model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")



print(train_df['label'].value_counts(dropna=False))
print("Valores nulos:", train_df['label'].isnull().sum())



# Eliminar clases con menos de 2 ejemplos
label_counts = train_df['label'].value_counts()
train_df = train_df[train_df['label'].isin(label_counts[label_counts > 1].index)]

print("NÃºmero de clases despuÃ©s de limpieza:", train_df['label'].nunique())
print(train_df['label'].value_counts().tail(10))



label_counts = train_df['label'].value_counts()
train_df = train_df[train_df['label'].isin(label_counts[label_counts > 1].index)]



from sklearn.model_selection import train_test_split

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df['StudentExplanation'], 
    train_df['label'],
    test_size=0.2, 
    random_state=42, 
    stratify=train_df['label']
)



# Importaciones principales
from transformers import BertTokenizerFast, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW  # âœ… ahora AdamW viene de torch.optim
import torch

# Cargar el modelo y el tokenizer desde Hugging Face
try:
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    print("âœ… Modelo y tokenizer cargados correctamente desde Hugging Face.")
except Exception as e:
    raise RuntimeError(
        "â�Œ No se pudo importar 'transformers' o cargar el modelo/tokenizer. "
        "AsegÃºrate de que el paquete estÃ© instalado en el entorno Kaggle "
        "o sube el modelo/tokenizer a /kaggle/input/bert-base-uncased. "
        f"\nError original:\n{e}"
    )

# Definir el optimizador AdamW (desde torch)
optimizer = AdamW(model.parameters(), lr=2e-5)

# Programador de tasa de aprendizaje (scheduler)
num_epochs = 3
num_training_steps = 1000  # puedes ajustarlo segÃºn tu tamaÃ±o de dataset
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# Enviar el modelo a GPU si estÃ¡ disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"âœ… Modelo listo para entrenamiento en: {device}")





# BLOQUE COMPLETO: BERT multiclass para Kaggle (num_labels=138)
# Ajusta hyperparÃ¡metros mÃ¡s abajo si quieres

import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tqdm.auto import tqdm

# ---------------------------
# Config
# ---------------------------
SEED = 42
BATCH_SIZE = 16            # reducir si te quedas sin memoria
EPOCHS = 3                 # empieza con 1-3 para probar
LR = 2e-5
MAX_LEN = 128
NUM_LABELS = 138           # confirmado por ti
MODEL_CACHE_LOCAL = "/kaggle/input/bert-base-uncased"  # si subiste el modelo aquÃ­
OUTPUT_DIR = "/kaggle/working/bert_misconception_model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---------------------------
# Instalar / importar transformers (si no estÃ¡ instalado fallarÃ¡)
# ---------------------------
try:
    from transformers import BertTokenizerFast, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
except Exception as e:
    raise RuntimeError(
        "No se puede importar 'transformers'. AsegÃºrate de que el paquete estÃ© instalado en el entorno Kaggle "
        "o sube el modelo/tokenizer a /kaggle/input/bert-base-uncased. Error original:\n" + str(e)
    )

# ---------------------------
# Cargar tokenizer (intento local primero)
# ---------------------------
tokenizer = None
if os.path.isdir(MODEL_CACHE_LOCAL):
    try:
        print("Intentando cargar tokenizer desde", MODEL_CACHE_LOCAL)
        tokenizer = BertTokenizerFast.from_pretrained(MODEL_CACHE_LOCAL)
        print("Tokenizer cargado desde /kaggle/input")
    except Exception as e:
        print("Fallo cargar tokenizer desde carpeta local:", e)
        tokenizer = None

if tokenizer is None:
    try:
        print("Intentando cargar tokenizer desde HuggingFace (internet).")
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        print("Tokenizer descargado desde HuggingFace.")
    except Exception as e:
        raise RuntimeError(
            "No fue posible obtener el tokenizer. Si no tienes internet en Kaggle, sube el modelo/tokenizer "
            "a /kaggle/input/bert-base-uncased y vuelve a ejecutar. Error original:\n" + str(e)
        )

# ---------------------------
# Tokenize helper
# ---------------------------
def tokenize(texts):
    """
    texts: list-like of strings
    return: dict of tensors (input_ids, attention_mask)
    """
    # tokenizer returns dict of lists (or numpy) -- use return_tensors=None to process into lists then convert
    enc = tokenizer(
        texts.tolist() if hasattr(texts, "tolist") else list(texts),
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_attention_mask=True,
        return_tensors="pt"
    )
    return enc

# ---------------------------
# Dataset class
# ---------------------------
class MisconceptionDataset(Dataset):
    def __init__(self, texts, labels=None):
        # texts: pd.Series or list
        self.encodings = tokenize(texts)
        self.labels = labels.values.tolist() if hasattr(labels, "values") else (labels if labels is not None else None)

    def __len__(self):
        return self.encodings["input_ids"].size(0)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# ---------------------------
# Preparar datos (usa train_df con StudentExplanation y label)
# ---------------------------
# AsegÃºrate de haber cargado train_df previamente en la celda anterior.
assert "train_df" in globals() or "train_df" in locals(), "train_df no encontrado. Carga tu CSV antes de ejecutar este bloque."

# Preprocesado mÃ­nimo
train_df['StudentExplanation'] = train_df['StudentExplanation'].fillna('').astype(str)

# Stratified split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df['StudentExplanation'],
    train_df['label'],
    test_size=0.2,
    random_state=SEED,
    stratify=train_df['label']
)

print("Train examples:", len(train_texts), "Val examples:", len(val_texts))

# Crear datasets y dataloaders
train_dataset = MisconceptionDataset(train_texts, train_labels)
val_dataset = MisconceptionDataset(val_texts, val_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------------------
# Cargar modelo (intentando local primero)
# ---------------------------
model = None
if os.path.isdir(MODEL_CACHE_LOCAL):
    try:
        print("Intentando cargar modelo desde", MODEL_CACHE_LOCAL)
        model = BertForSequenceClassification.from_pretrained(MODEL_CACHE_LOCAL, num_labels=NUM_LABELS)
        print("Modelo cargado desde /kaggle/input")
    except Exception as e:
        print("Fallo cargar modelo desde carpeta local:", e)
        model = None

if model is None:
    try:
        print("Intentando descargar modelo 'bert-base-uncased' desde HuggingFace")
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=NUM_LABELS)
        print("Modelo descargado desde HuggingFace.")
    except Exception as e:
        raise RuntimeError(
            "No fue posible obtener el modelo Bert. Si no tienes internet en Kaggle, sube el modelo a /kaggle/input/bert-base-uncased. "
            "Error original:\n" + str(e)
        )

model.to(device)

# ---------------------------
# Optimizer & Scheduler
# ---------------------------
optimizer = AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06*total_steps), num_training_steps=total_steps)

# ---------------------------
# Entrenamiento (loop simple)
# ---------------------------
def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    losses = []
    for batch in tqdm(loader, desc="Train", leave=False):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(loss.item())
    return float(sum(losses)/len(losses)) if losses else 0.0

def eval_epoch(model, loader, device):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=-1).cpu().numpy()
            preds.extend(batch_preds.tolist())
            trues.extend(labels.cpu().numpy().tolist())
    return preds, trues

# Entrenar
best_acc = 0.0
os.makedirs(OUTPUT_DIR, exist_ok=True)

for epoch in range(1, EPOCHS + 1):
    print(f"\n=== Epoch {epoch}/{EPOCHS} ===")
    train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
    print(f"Train loss: {train_loss:.4f}")

    preds, trues = eval_epoch(model, val_loader, device)
    acc = accuracy_score(trues, preds)
    print(f"Val accuracy: {acc:.4f}")
    print("Classification report (val):")
    print(classification_report(trues, preds, zero_division=0))

    # Guardar si mejora
    if acc > best_acc:
        best_acc = acc
        ckpt_path = os.path.join(OUTPUT_DIR, f"best_model_epoch{epoch}.pt")
        # Guardar peso y tokenizer config
        torch.save(model.state_dict(), ckpt_path)
        try:
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
        except Exception:
            pass
        print("Guardado modelo en:", OUTPUT_DIR)

print("Entrenamiento finalizado. Mejor acc:", best_acc)



import pandas as pd

test_df = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")
print(test_df.head())
print("\nColumnas disponibles:", test_df.columns.tolist())



import pandas as pd

# Cargar el archivo de test
test_df = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")

# Verificamos las columnas para saber cuÃ¡l contiene el texto
print(test_df.columns)

# Supongamos que la columna que contiene el texto se llama 'prompt'
test_texts = test_df["StudentExplanation"].astype(str).tolist()


# Tokenizamos los textos del test
test_encodings = tokenizer(
    test_texts,
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors="pt"
)

# Creamos el dataloader para el test
from torch.utils.data import DataLoader, TensorDataset

test_dataset = TensorDataset(
    test_encodings["input_ids"],
    test_encodings["attention_mask"]
)

test_dataloader = DataLoader(test_dataset, batch_size=16)



import torch
from torch.utils.data import DataLoader, TensorDataset

# 1ï¸�âƒ£ Tomamos los textos del test
test_texts = test_df["StudentExplanation"].astype(str).tolist()

# 2ï¸�âƒ£ Tokenizamos
test_encodings = tokenizer(
    test_texts,
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors="pt"
)

# 3ï¸�âƒ£ Creamos el dataset y dataloader
test_dataset = TensorDataset(
    test_encodings["input_ids"],
    test_encodings["attention_mask"]
)
test_dataloader = DataLoader(test_dataset, batch_size=16)

# 4ï¸�âƒ£ Predicciones
model.eval()
test_preds = []

for batch in test_dataloader:
    input_ids, attention_mask = [b.to(device) for b in batch]
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        test_preds.extend(preds.cpu().numpy())

# 5ï¸�âƒ£ Guardamos el archivo de salida final (compatible con Kaggle)
submission = pd.DataFrame({
    "row_id": test_df["row_id"],
    "prediction": test_preds
})

submission.to_csv("/kaggle/working/submission.csv", index=False)
print("âœ… Archivo 'submission.csv' guardado correctamente en /kaggle/working/")

# Confirmamos que existe
import os
if os.path.exists("/kaggle/working/submission.csv"):
    print("ğŸ“„ Confirmado: archivo listo para enviar.")
else:
    print("â�Œ No se encontrÃ³ el archivo, revisa el path.")




# Ver las primeras filas del archivo de salida
import pandas as pd

submission = pd.read_csv("/kaggle/working/submission.csv")
submission.head(10)


