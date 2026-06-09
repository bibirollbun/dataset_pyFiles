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
# 读取 CSV 成 pandas DataFrame
train_df = pd.read_csv("/kaggle/input/llm-detect-ai-generated-text/train_essays.csv")

# 查看一下前几行
train_df.head()
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Celda 1 - Instalación de dependencias
!pip uninstall -y torch torchvision torchaudio
!pip cache purge
!pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
!pip install -q peft transformers[torch] datasets scipy accelerate


# Celda 2 - Importaciones y verificación de GPU
import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoModelForCausalLM,
    AutoModelForMaskedLM
)
from datasets import Dataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Verificar GPU
print("GPU disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Dispositivo GPU:", torch.cuda.get_device_name(0))
    print("Memoria GPU total:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")



import torch
torch.cuda.empty_cache()
print(torch.cuda.memory_summary())


# Celda 3 - Clase del detector
class ImprovedMultiModelDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando dispositivo: {self.device}")

        # 1. DeBERTa
        self.deberta_name = "microsoft/deberta-v3-base"
        self.deberta_tokenizer = AutoTokenizer.from_pretrained(self.deberta_name)
        self.deberta_model = AutoModelForSequenceClassification.from_pretrained(
            self.deberta_name,
            num_labels=1,
            trust_remote_code=True
        ).to(self.device)

        #distilroberta
        self.disroberta_name = "albert/albert-base-v2"
        self.disroberta_tokenizer = AutoTokenizer.from_pretrained(self.disroberta_name)
        self.disroberta_model = AutoModelForSequenceClassification.from_pretrained(
            self.disroberta_name,
            num_labels=1,
            trust_remote_code=True
        ).to(self.device)

    
        # 2. TinyLlama
        self.tiny_llama_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.tiny_llama_tokenizer = AutoTokenizer.from_pretrained(self.tiny_llama_name)
        self.tiny_llama = AutoModelForCausalLM.from_pretrained(
            self.tiny_llama_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)

    def get_training_args(self):
        return TrainingArguments(
            output_dir="./results_distilroberta",
            num_train_epochs=3,
            per_device_train_batch_size=4,  # Reducido para evitar OOM
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=8,  # Aumentado para compensar batch size menor
            warmup_ratio=0.1,
            learning_rate=2e-5,
            weight_decay=0.01,
            evaluation_strategy="steps",
            eval_steps=100,
            save_steps=100,
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="auc",
            remove_unused_columns=True,
            fp16=torch.cuda.is_available(),  # Solo usar fp16 si hay GPU
            report_to=["none"]
        )

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        print(f"Predictions shape: {predictions.shape}")
        print(f"Labels shape: {labels.shape}")
        predictions_probs = torch.sigmoid(torch.tensor(predictions)).numpy()

        # Buscar mejor umbral
        thresholds = np.linspace(0.1, 0.7, 30)
        best_accuracy = 0
        best_threshold = 0.5
        best_precision = 0
        best_recall = 0
        best_f1 = 0

        for threshold in thresholds:
            # Convertir predicciones a binario usando el umbral actual
            predictions_binary = (predictions_probs > threshold).astype(int)

            tp = np.sum((predictions_binary == 1) & (labels == 1))
            fp = np.sum((predictions_binary == 1) & (labels == 0))
            tn = np.sum((predictions_binary == 0) & (labels == 0))
            fn = np.sum((predictions_binary == 0) & (labels == 1))

            accuracy = (tp + tn) / (tp + tn + fp + fn)
            curr_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            curr_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            curr_f1 = 2 * (curr_precision * curr_recall) / (curr_precision + curr_recall) if (curr_precision + curr_recall) > 0 else 0

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
                best_precision = curr_precision
                best_recall = curr_recall
                best_f1 = curr_f1

        metrics = {
            "auc": roc_auc_score(labels, predictions_probs),
            "accuracy": best_accuracy,
            "precision": best_precision,
            "recall": best_recall,
            "f1": best_f1,
            "best_threshold": best_threshold
        }
        print(f"Validation metrics: {metrics}")
        return metrics

    def prepare_data(self, df, is_train=True):
        print(f"Preparando {'training' if is_train else 'validation'} data...")
        texts = df['text'].str.strip().str.lower().tolist()

        # Procesar en batches para evitar OOM
        batch_size = 32
        all_input_ids = []
        all_attention_mask = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            tokenized = self.disroberta_tokenizer(
                batch_texts,
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            all_input_ids.append(tokenized['input_ids'])
            all_attention_mask.append(tokenized['attention_mask'])

        dataset_dict = {
            'input_ids': torch.cat(all_input_ids),
            'attention_mask': torch.cat(all_attention_mask)
        }

        if is_train:
            dataset_dict['labels'] = torch.tensor(df['generated'].astype(float).values)

        return Dataset.from_dict(dataset_dict)

    def train(self, train_df, val_df):
        print("Iniciando entrenamiento...")
        train_dataset = self.prepare_data(train_df)
        val_dataset = self.prepare_data(val_df)

        trainer = Trainer(
            model=self.disroberta_model,
            args=self.get_training_args(),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            tokenizer=self.disroberta_tokenizer,
        )

        trainer.train()

    def get_perplexity(self, text):
        inputs = self.tiny_llama_tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.tiny_llama(**inputs)
            return outputs.loss.item() if outputs.loss is not None else 0.0

    @torch.no_grad()
    def predict(self, text):
        # DeBERTa prediction
        inputs = self.disroberta_tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        disberta_output = self.disroberta_model(**inputs)
        disberta_pred = torch.sigmoid(disberta_output.logits).cpu().numpy()[0][0]

        # TinyLlama perplexity
        perplexity = self.get_perplexity(text)
        llama_pred = 1 / (1 + np.exp(-perplexity))

        # Ensemble weighted prediction
        final_pred = (0.7 * disberta_pred + 0.3 * llama_pred)

        # Calibración
        if final_pred > 0.7:
            return 1.0
        elif final_pred < 0.3:
            return 0.0
        return final_pred





from datasets import load_dataset
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# 读取 CSV 成 pandas DataFrame
kaggle_dataset = pd.read_csv("/kaggle/input/llm-detect-ai-generated-text/train_essays.csv")

# 查看一下前几行
kaggle_dataset.head()
hf_dataset = load_dataset("dmitva/human_ai_generated_text")
hf_dataset['train'][0]


from datasets import load_dataset
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# 读取 CSV 成 pandas DataFrame
kaggle_dataset = pd.read_csv("/kaggle/input/llm-detect-ai-generated-text/train_essays.csv")

# 查看一下前几行
kaggle_dataset.head()
hf_dataset = load_dataset("dmitva/human_ai_generated_text")

# Determinar el número de muestras a tomar
n_samples = min(15000 // 2, len(hf_dataset['train']['human_text']))
print(f"Tomando {n_samples} muestras de cada clase...")

# human_texts
human_text_series = pd.concat([
    pd.Series(hf_dataset['train']['human_text'])[:n_samples],
    kaggle_dataset[kaggle_dataset['generated'] == 0]['text'][:n_samples]
], ignore_index=True)

human_texts = pd.DataFrame({
    'text': human_text_series,
    'generated': 0
})

# ai_texts
ai_text_series = pd.concat([
    pd.Series(hf_dataset['train']['ai_text'])[:n_samples],
    kaggle_dataset[kaggle_dataset['generated'] == 1]['text'][:n_samples]
], ignore_index=True)

ai_texts = pd.DataFrame({
    'text': ai_text_series,
    'generated': 1
})

# Combinar datasets
train_df = pd.concat([human_texts, ai_texts], ignore_index=True)
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

from sklearn.model_selection import train_test_split

train, test = train_test_split(
    train_df,
    test_size=0.2,
    random_state=42
)

# Limpiar y preparar textos
train['text'] = train['text'].fillna("").str.strip()
test['text'] = test['text'].fillna("").str.strip()

# Mostrar estadísticas
print("\nEstadísticas del dataset de entrenamiento:")
print(f"Total muestras: {len(train)}")
print(f"Textos AI: {sum(train['generated'] == 1)}")
print(f"Textos humanos: {sum(train['generated'] == 0)}")
print(f"Ratio AI/Humano: {sum(train['generated'] == 1)/sum(train['generated'] == 0):.3f}")

# Verificar memoria
print("\nUso de memoria:")
print(f"Train DataFrame: {train.memory_usage().sum() / 1024 / 1024:.2f} MB")
print(f"Test DataFrame: {test.memory_usage().sum() / 1024 / 1024:.2f} MB")


# Celda 4 - Función de entrenamiento y evaluación
def train_and_evaluate():
    print("Iniciando proceso completo...")
    detector = ImprovedMultiModelDetector()

    print("Cargando datos...")
    train_df = train
    test_df = test
    print("Realizando split de datos...")
    train_df_final, val_df = train_test_split(
        train_df,
        test_size=0.2,
        random_state=42,
        stratify=train_df['generated']
    )

    try:
        detector.train(train_df_final, val_df)
    except Exception as e:
        print(f"Error durante el entrenamiento: {str(e)}")
        raise

    return detector





# from tqdm import tqdm
# # Celda 5 - Ejecución
# if __name__ == "__main__":
#     try:
#         detector = train_and_evaluate()

#         # Guardar modelos
#         OUTPUT_DIR = "/kaggle/working/output"
#         os.makedirs(OUTPUT_DIR, exist_ok=True)

#         detector.deberta_model.save_pretrained(f"{OUTPUT_DIR}/deberta")
#         detector.tiny_llama.save_pretrained(f"{OUTPUT_DIR}/tiny_llama")
#         for name, model in tqdm(
#         # Test de ejemplo
#         texto_prueba = "This is an example of a test detector"
#         probabilidad = detector.predict(texto_prueba)
#         print(f"Probabilidad de ser generado por IA: {probabilidad:.2%}")

#     except Exception as e:
#         print(f"Error en la ejecución principal: {str(e)}")
#         raise
# Cell 5 - Main execution with progress bar
from tqdm import tqdm
import os

if __name__ == "__main__":
    try:
        print("Training the detector...")
        detector = train_and_evaluate()

        # Save trained models
        OUTPUT_DIR = "/kaggle/working/output1"
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        print("Saving trained models...")
        for name, model in tqdm(
            [("disroberta", detector.disroberta_model), ("tiny_llama", detector.tiny_llama)],
            desc="Saving models"
        ):
            model.save_pretrained(f"{OUTPUT_DIR}/{name}")

        # Run a single prediction test
        print("\nRunning inference on a sample text...")
        sample_text = "This is an example of a test detector"
        probability = detector.predict(sample_text)
        print(f"Probability of being AI-generated: {probability:.2%}")

    except Exception as e:
        print(f"Error during main execution: {str(e)}")
        raise



# Generar predicciones
test_predictions = []
for text in test['text']:
    pred = detector.predict(text)
    test_predictions.append(pred)

# Crear submission
submission = pd.DataFrame({
    'generated': test_predictions
})

# Evaluar métricas
predictions_binary = [1 if pred > 0.5 else 0 for pred in test_predictions]

print("\nMétricas en conjunto de test:")
print(f"Accuracy: {accuracy_score(test['generated'], predictions_binary):.4f}")
print(f"Precision: {precision_score(test['generated'], predictions_binary):.4f}")
print(f"Recall: {recall_score(test['generated'], predictions_binary):.4f}")
print(f"F1-score: {f1_score(test['generated'], predictions_binary):.4f}")
print(f"AUC-ROC: {roc_auc_score(test['generated'], test_predictions):.4f}")


# Save each model and tokenizer separately
# Assuming 'tokenizer' was intended for the deberta tokenizer:
detector.deberta_model.save_pretrained('/kaggle/working/deberta')
detector.deberta_tokenizer.save_pretrained('/kaggle/working/deberta')

detector.tiny_llama.save_pretrained('/kaggle/working/tinyllama')
detector.tiny_llama_tokenizer.save_pretrained('/kaggle/working/tinyllama')

print("Modelos y tokenizadores guardados en './my_model_V6'")


!zip -r /kaggle/working/results_distilroberta.zip /kaggle/working/results_distilroberta




