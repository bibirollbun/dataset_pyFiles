import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from transformers import BeitImageProcessor, BeitForImageClassification
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# ==== 1. CONFIGURA TUS RUTAS ====
# Carpeta donde está el modelo exportado
model_checkpoint_path = "/kaggle/input/modelos-edema-multilabel/modelos/modelo_multilabel_4_clases"

# Carpeta donde están las imágenes nuevas para inferencia
images_folder_path = "/kaggle/input/dataset-edema-eyepacs-labels/eyepacs/split/test"  # <-- modifica este path

# ==== 2. CARGAR MODELO Y PROCESSOR ====
model = BeitForImageClassification.from_pretrained(model_checkpoint_path)
model.eval()

processor = BeitImageProcessor.from_pretrained(model_checkpoint_path)

# ==== 3. DEFINIR TRANSFORMACIONES ====
transforms = Compose([
    Resize((processor.size['height'], processor.size['width'])),
    ToTensor(),
    Normalize(mean=processor.image_mean, std=processor.image_std)
])

# ==== 4. ETIQUETAS MULTIETIQUETA (EN ORDEN) ====
label_columns = ['Normal', 'Diabetic', 'Edema1',"Edema2"]

# ==== 5. FUNCIONES ====
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = transforms(image).unsqueeze(0)  # (1, 3, H, W)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        preds = (probs > 0.5).astype(int)

    return preds, probs

# ==== 6. PROCESAR TODAS LAS IMÁGENES DE LA CARPETA ====
results = []

for filename in os.listdir(images_folder_path):
    if filename.lower().endswith((".png", ".jpg", ".jpeg",".tiff",".tif")):
        full_path = os.path.join(images_folder_path, filename)
        preds, probs = predict_image(full_path)

        result = {
            "filename": filename,
            **{f"Pred_{label}": int(p) for label, p in zip(label_columns, preds)},
            **{f"Prob_{label}": float(pr) for label, pr in zip(label_columns, probs)}
        }
        results.append(result)

# ==== 7. GUARDAR RESULTADOS EN CSV ====
df = pd.DataFrame(results)
df.to_csv("resultados_inferencia_4_clases.csv", index=False)
print("✅ Resultados guardados en 'resultados_inferencia4.csv'")



import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from transformers import BeitImageProcessor, BeitForImageClassification
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# ==== 1. CONFIGURA TUS RUTAS ====
# Carpeta donde está el modelo exportado
model_checkpoint_path = "/kaggle/input/modelos-edema-multilabel/modelos/multilabel_edema_3_clases"

# Carpeta base donde están las imágenes
images_folder_path = "/kaggle/input/dataset-multilabel-edema-3-clases/todo/todo"  # <-- tu carpeta base

# CSV con la columna 'file' (nombres de archivo de imagen)
csv_path = "/kaggle/input/dataset-multilabel-edema-3-clases/test_split_20_eyepacs3.csv"  # <-- pon aquí tu CSV

# ==== 2. CARGAR MODELO Y PROCESSOR ====
model = BeitForImageClassification.from_pretrained(model_checkpoint_path)
model.eval()
processor = BeitImageProcessor.from_pretrained(model_checkpoint_path)

# ==== 3. DEFINIR TRANSFORMACIONES ====
transforms = Compose([
    Resize((processor.size['height'], processor.size['width'])),
    ToTensor(),
    Normalize(mean=processor.image_mean, std=processor.image_std)
])

# ==== 4. ETIQUETAS MULTIETIQUETA (EN ORDEN) ====
label_columns = ['Normal', 'Diabetic', 'Edema1', 'Edema2']

# ==== 5. FUNCIONES ====
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = transforms(image).unsqueeze(0)  # (1, 3, H, W)
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        preds = (probs > 0.5).astype(int)
    return preds, probs

# ==== 6. LEER LISTA DE IMÁGENES DESDE CSV (columna 'file') ====
df_input = pd.read_csv(csv_path)
if 'file' not in df_input.columns:
    raise ValueError("El CSV no contiene la columna requerida 'file'.")

# Normaliza a str y elimina espacios/NaN
filenames = df_input['file'].astype(str).str.strip().tolist()

# ==== 7. PROCESAR TODAS LAS IMÁGENES LISTADAS EN EL CSV ====
results = []
for filename in filenames:
    full_path = os.path.join(images_folder_path, filename)
    if not os.path.isfile(full_path):
        # Si falta el archivo, registra fila con NaN para trazabilidad
        results.append({
            "file": filename,
            **{f"Pred_{label}": np.nan for label in label_columns},
            **{f"Prob_{label}": np.nan for label in label_columns}
        })
        continue

    preds, probs = predict_image(full_path)
    result = {
        "file": filename,
        **{f"Pred_{label}": int(p) for label, p in zip(label_columns, preds)},
        **{f"Prob_{label}": float(pr) for label, pr in zip(label_columns, probs)}
    }
    results.append(result)

# ==== 8. GUARDAR RESULTADOS EN CSV ====
df_out = pd.DataFrame(results)
df_out.to_csv("resultados_inferencia_3_clases.csv", index=False)
print("✅ Resultados guardados en 'resultados_inferencia_3_clases.csv'")


