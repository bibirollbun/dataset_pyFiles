# %% [code]
# 1. Instalar dependencias específicas
# (Asegúrate de que estas sean todas las que necesitas, incluyendo torch si usas modelos de Transformers)
!pip install -U transformers sentencepiece textblob imbalanced-learn datasets accelerate

# %% [code]
# 2. Clonar tu repositorio de GitHub y navegar a la raíz del proyecto
!git clone https://github.com/HugoMojicaAngarita/ToxicCommentDetection.git

# Verificar la estructura y navegar al directorio correcto
# Asumiendo que la estructura es ToxicCommentDetection/ToxicCommentDetection/...
# Si es ToxicCommentDetection/data, ToxicCommentDetection/src, etc. directamente,
# usa solo %cd ToxicCommentDetection
%cd ToxicCommentDetection/ToxicCommentDetection 

# Ahora estás en /kaggle/working/ToxicCommentDetection/ToxicCommentDetection (o similar)
# Puedes verificar con !pwd y !ls -R

# %% [code]
# 3. Crear estructura de carpetas (si main.py no las crea o si necesitas asegurar su existencia)
# Estas rutas son relativas a donde estás ahora (la raíz de tu proyecto clonado)
import os
os.makedirs("data/raw", exist_ok=True)
os.makedirs("models", exist_ok=True)

# %% [code]
# 4. Copiar datos de la competencia a la estructura requerida
import shutil
from pathlib import Path

# La ruta CORRECTA para los datos de la competencia Jigsaw
kaggle_input_path = Path("/kaggle/input/jigsaw-unintended-bias-in-toxicity-classification")
project_data_path = Path("data/raw/")

files_to_copy = [
    "train.csv",
    "test.csv",
    "sample_submission.csv",
    "identity_individual_annotations.csv"
]

for file in files_to_copy:
    src = kaggle_input_path / file
    dst = project_data_path / file
    if src.exists():
        shutil.copy(src, dst)
        print(f"Copied: {src} -> {dst}")
    else:
        print(f"Warning: File not found - {src}. Check dataset availability.")

# %% [code]
# 5. Ejecutar tu pipeline principal
# Asume que main.py leerá de data/raw y escribirá a models/kaggle_submission.csv
# Ya estamos en el directorio raíz de tu proyecto (gracias al %cd anterior)
!python src/main.py

# %% [code]
# 6. Preparar archivo de submission para Kaggle
import pandas as pd
from pathlib import Path

# Leer el archivo generado por tu sistema
# La ruta es relativa al directorio actual (la raíz de tu proyecto clonado)
submission_path = Path("models/kaggle_submission.csv")

# Asegúrate de que el archivo exista antes de intentar leerlo
if not submission_path.exists():
    raise FileNotFoundError(f"El archivo de submission esperado no se encontró en: {submission_path}")

submission_df = pd.read_csv(submission_path)

# Guardar en el formato y ubicación que Kaggle espera para la submission
# Este es el paso final crucial para que Kaggle detecte tu envío
submission_df.to_csv("/kaggle/working/submission.csv", index=False)
print("Submission file created at /kaggle/working/submission.csv")
print(submission_df.head())

