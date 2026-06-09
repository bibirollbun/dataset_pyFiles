import os
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError, ImageOps
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn import set_config
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import torchvision.transforms as T
from glob import glob
from torchvision import transforms as T
from PIL import ImageOps
from sklearn.cluster import KMeans
from tensorflow.keras.applications import ConvNeXtBase
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications.convnext import preprocess_input
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B3
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score


set_config(transform_output="pandas")
seed_global = 27912


def build_cluster_dataset_hierarchical(image_paths, y_hier, image_size=(167, 250), training=True):
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    hier_ds = tf.data.Dataset.from_tensor_slices(y_hier)

    def load_img(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.resize_with_pad(img, target_height=167, target_width=250)
        img = tf.keras.applications.convnext.preprocess_input(img)
        return img

    image_ds = path_ds.map(load_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds = tf.data.Dataset.zip((image_ds, hier_ds))
    if training:
        ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(16, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return ds


train_cluster_df = pd.read_json("/kaggle/input/metadatos-aumentados/train_cluster_augmented_df (1).json")
val_cluster_df = pd.read_json("/kaggle/input/cluster-dataframes/val_cluster_df.json")


# Umbral mÃ­nimo de instancias por combinaciÃ³n
umbral_minimo = 750

# Agrupar por cluster y familia
conteo_cluster_familia = train_cluster_df.groupby(["cluster", "familia"]).size().reset_index(name="count")

# Filtrar combinaciones por debajo del umbral (pero mayor que 0)
conteos_bajos_sin_cat = conteo_cluster_familia[
    (conteo_cluster_familia["count"] > 0) & (conteo_cluster_familia["count"] < umbral_minimo)
]

# Filtrar combinaciones por debajo del umbral (pero mayor que 0)
conteos_altos_sin_cat = conteo_cluster_familia[
    (conteo_cluster_familia["count"] > 0) & (conteo_cluster_familia["count"] > umbral_minimo)
]

# NÃºmero de combinaciones
total_bajos = len(conteos_bajos_sin_cat)

total_altos = len(conteos_altos_sin_cat)

# Total de instancias sumadas en esas combinaciones
total_instancias_bajas = conteos_bajos_sin_cat["count"].sum()
total_instancias_altas = conteos_altos_sin_cat["count"].sum()

print(f"Total de combinaciones 'cluster + familia' con menos de {umbral_minimo} instancias: {total_bajos}")
print(f"Estas combinaciones suman {total_instancias_bajas} instancias")

print(f"Total de combinaciones 'cluster + familia' con mÃ¡s de {umbral_minimo} instancias: {total_altos}")
print(f"Estas combinaciones suman {total_instancias_altas} instancias")


# definir un umbral mÃ­nimo
umbral_minimo = 750

# conteo por cluster + familia + categorÃ­a
conteo_cfc = (
    train_cluster_df
    .groupby(["cluster", "familia", "categoria"])
    .size()
    .reset_index(name="count")
)

# obtener las combinaciones cluster+familia bajas (sin categorÃ­a)
conteo_cf = (
    train_cluster_df
    .groupby(["cluster", "familia"])
    .size()
    .reset_index(name="count")
)
bajas_cf = conteo_cf[
    (conteo_cf["count"] > 0) &
    (conteo_cf["count"] < umbral_minimo)
][["cluster", "familia"]]

# filtrar el conteo con categorÃ­a para quedarnos sÃ³lo con esas parejas bajas
conteos_bajos_filtrados = conteo_cfc.merge(
    bajas_cf,
    on=["cluster", "familia"],
    how="inner"
)

# ya incluye columna 'categoria' y 'count'
print(f"Total de filas filtradas: {len(conteos_bajos_filtrados)}")
print(conteos_bajos_filtrados.head())



# Crear un diccionario con clave (cluster, familia) y valor el nÃºmero de instancias
pares_bajos_dict = {
    (row["cluster"], row["familia"]): row["count"]
    for _, row in conteos_bajos_sin_cat.iterrows()
}

print(f"ğŸ”§ Diccionario de pares (cluster, familia) con menos de {umbral_minimo} instancias creado.")


@tf.function
def cosine_embedding_loss(y_true, y_pred, label_embeddings):
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.math.l2_normalize(y_pred, axis=1)
    target_embeds = tf.gather(label_embeddings, y_true)
    target_embeds = tf.math.l2_normalize(target_embeds, axis=1)
    sim = tf.reduce_sum(y_pred * target_embeds, axis=1)
    sim = tf.clip_by_value(sim, -1.0, 1.0)  # Seguridad adicional
    return 1 - tf.reduce_mean(sim)
    
@tf.function
def train_step(x, y, model, label_embeddings, optimizer):
    with tf.GradientTape() as tape:
        preds = model(x, training=True)
        loss = cosine_embedding_loss(y, preds, label_embeddings)

        # CÃ¡lculo de predicciones
        sims = tf.matmul(preds, tf.transpose(label_embeddings))
        pred_ids = tf.argmax(sims, axis=1)
        acc = tf.reduce_mean(tf.cast(tf.equal(pred_ids, tf.cast(y, tf.int64)), tf.float32))

    grads = tape.gradient(loss, model.trainable_variables + [label_embeddings])
    optimizer.apply_gradients(zip(grads, model.trainable_variables + [label_embeddings]))

    return loss, acc


@tf.function
def val_step(x, y, model, label_embeddings):
    preds = model(x, training=False)
    target_embeds = tf.gather(label_embeddings, tf.cast(y, tf.int32))
    target_embeds = tf.math.l2_normalize(target_embeds, axis=1)
    val_loss = 1 - tf.reduce_mean(tf.reduce_sum(preds * target_embeds, axis=1))
    sims = tf.matmul(preds, tf.transpose(label_embeddings))
    preds_idx = tf.argmax(sims, axis=1)
    acc = tf.reduce_mean(tf.cast(tf.equal(preds_idx, tf.cast(y, tf.int64)), tf.float32))
    return val_loss, acc


import os
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

def get_dataset_por_cluster_familia_(
    cluster_id,
    fam_id,
    df_train,
    df_val,
    conteos_bajos,
    label_col="categoria",
    training=True,
    label_map=None
):
    # Filtrar por cluster y familia
    df_tr = df_train[(df_train["cluster"] == cluster_id) & (df_train["familia"] == fam_id)].copy()
    df_vl = df_val  [(df_val  ["cluster"] == cluster_id) & (df_val  ["familia"] == fam_id)].copy()

    # UniÃ³n de clases en train y val
    clases_union = sorted(set(df_tr[label_col].unique()) | set(df_vl[label_col].unique()))
    if training:
        print(f"Usando {len(clases_union)} clases de '{label_col}' (train âˆª val)")

    # Mapas de labels
    if label_map is None:
        label_map = {val: idx for idx, val in enumerate(clases_union)}
    inv_label_map = {idx: val for val, idx in label_map.items()}

    # AÃ±adir ejemplos de conteos bajos solo al train
    if training:
        df_extra = conteos_bajos[
            (conteos_bajos["cluster"] == cluster_id) &
            (conteos_bajos["familia"] == fam_id) &
            (conteos_bajos[label_col].isin(clases_union))
        ]
        print(f"AÃ±adiendo {len(df_extra)} ejemplos extra desde conteos_bajos al TRAIN")
        df_tr = pd.concat([df_tr, df_extra], ignore_index=True)

    # Filtrar rutas que realmente existen (solo imprime en train)
    def filter_existing(df, name):
        mask = df["path"].apply(lambda p: isinstance(p, str) and os.path.exists(p))
        if training and name == "TRAIN":
            removed = (~mask).sum()
            print(f"{removed} rutas inexistentes en {name}, eliminadas")
        return df[mask].reset_index(drop=True)

    df_tr = filter_existing(df_tr, "TRAIN")
    df_vl = filter_existing(df_vl, "VALIDATION")

    # Elegir subset
    target_df = df_tr if training else df_vl

    # Mapear etiquetas a Ã­ndices
    target_df["label"] = target_df[label_col].map(label_map)
    paths = target_df["path"].values
    labels = target_df["label"].values

    # Crear el tf.data.Dataset
    ds = build_cluster_dataset_hierarchical(
        image_paths=paths,
        y_hier=labels,
        training=training
    )

    # Calcular y devolver pesos de clase solo para train
    if training:
        clases_presentes = np.unique(labels)
        weights = compute_class_weight(
            class_weight="balanced",
            classes=clases_presentes,
            y=labels
        )
        cw = dict(zip(clases_presentes, weights))
        total = sum(cw.values())
        cw = {k: v / total for k, v in cw.items()}
        custom_weights = {i: 1.0 + 0.35 * (cw.get(i, 0) - 1.0) for i in clases_union}
        return ds, custom_weights, len(clases_union), inv_label_map, label_map

    return ds, None, len(clases_union), inv_label_map, label_map


from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(
    monitor='val_macro_f1',
    mode = 'max',
    patience=5,
    restore_best_weights=True,
    verbose=1
)


import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model

class L2Normalization(layers.Layer):
    def __init__(self, axis=1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config

def adapt_embedding_model_for_categoria(base_model, embedding_dim=128, extra_trainable_ratio=0.3):
    # Si base_model es ya el backbone, lo usamos directamente
    from tensorflow.keras import Model as KerasModel
    if isinstance(base_model, KerasModel) and 'efficientnet' in base_model.name.lower():
        backbone = base_model
    else:
        backbone = None
        for layer in base_model.layers:
            if isinstance(layer, KerasModel) and 'efficientnet' in layer.name.lower():
                backbone = layer
                break
        if backbone is None:
            raise ValueError("No se encontrÃ³ EfficientNet dentro de base_model")

    # Congelar capas
    n = len(backbone.layers)
    m = int(n * extra_trainable_ratio)
    for layer in backbone.layers[:-m]:
        layer.trainable = False
    for layer in backbone.layers[-m:]:
        layer.trainable = True

    # Data augmentation
    augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
    ], name="augmentation")

    # ConstrucciÃ³n del modelo
    inputs = layers.Input(shape=(167, 250, 3), name="input_image")
    x = augmentation(inputs)
    x = backbone(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    x = layers.Dense(512, activation="relu", name="fc1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Dropout(0.4, name="drop1")(x)

    x = layers.Dense(256, activation="relu", name="fc2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Dropout(0.4, name="drop2")(x)

    x = layers.Dense(
        embedding_dim,
        kernel_regularizer=regularizers.l2(1e-4),
        name="embed_dense"
    )(x)

    outputs = L2Normalization(axis=1, name="l2_normalize")(x)

    return Model(inputs=inputs, outputs=outputs, name="adapted_embedding_model")


cluster_id = 0
df_cluster = train_cluster_df[train_cluster_df["cluster"] == cluster_id]
familias_cluster = df_cluster["familia"].unique()

# Lista para guardar combinaciones que no estÃ¡n en conteos_bajos
combinaciones_faltantes = []

for fam_id in familias_cluster:
    existe = ((conteos_bajos_sin_cat["cluster"] == cluster_id) & (conteos_bajos_sin_cat["familia"] == fam_id)).any()
    if not existe:
        combinaciones_faltantes.append((cluster_id, fam_id))

print(f"Total combinaciones (cluster={cluster_id}, familia) que NO estÃ¡n en conteos_bajos: {len(combinaciones_faltantes)}")
print("Combinaciones faltantes:")
for comb in combinaciones_faltantes:
    print(comb)


tf.debugging.set_log_device_placement(False)
print("GPUs disponibles:", tf.config.list_physical_devices('GPU'))


from keras.config import enable_unsafe_deserialization
enable_unsafe_deserialization()


import tensorflow as tf
import math

class CosineDecayWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, warmup_steps, total_steps, alpha=0.0):
        self.initial_lr = tf.cast(initial_lr, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.total_steps = tf.cast(total_steps, tf.float32)
        self.alpha = tf.cast(alpha, tf.float32)
        self.pi = tf.constant(math.pi, dtype=tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)

        cosine_decay = 0.5 * (1 + tf.cos(self.pi * (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)))
        decayed = (1 - self.alpha) * cosine_decay + self.alpha

        lr = tf.where(
            step < self.warmup_steps,
            self.initial_lr * (step / self.warmup_steps),
            self.initial_lr * decayed
        )
        return lr


import gc
import json
import pickle
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight


os.makedirs("/kaggle/working/modelos", exist_ok=True)
inv_label_cluster_genus = {}

embedding_dim = 128
patience = 5
epochs = 30
clusters = [5]

for cluster_id in clusters:
    df_cluster = train_cluster_df[train_cluster_df["cluster"] == cluster_id]

    for fam_id in df_cluster["familia"].unique():
        df_filtrado = df_cluster[df_cluster["familia"] == fam_id]
        if ((conteos_bajos_sin_cat["cluster"] == cluster_id) &
            (conteos_bajos_sin_cat["familia"] == fam_id)).any():
            continue
        if df_filtrado["categoria"].nunique() <= 1:
            print(f"Descartando cluster={cluster_id}, familia={fam_id}: "
                  f"sÃ³lo {df_filtrado['categoria'].nunique()} categorÃ­a")
            continue

        train_ds, class_weights, num_clases, inv_label_map, label_map = get_dataset_por_cluster_familia_(
            cluster_id, fam_id,
            train_cluster_df, val_cluster_df,
            conteos_bajos_filtrados,
            training=True
        )
        val_ds, _, _, _, _ = get_dataset_por_cluster_familia_(
            cluster_id, fam_id,
            train_cluster_df, val_cluster_df,
            conteos_bajos_filtrados,
            training=False,
            label_map=label_map
        )

        label_embeddings = tf.Variable(
            tf.random.normal((num_clases, embedding_dim)),
            trainable=True, name="label_embeddings"
        )

        backbone = tf.keras.applications.EfficientNetV2B3(
            include_top=False, weights="imagenet", input_shape=(167, 250, 3)
        )
        model = adapt_embedding_model_for_categoria(
            backbone,
            embedding_dim=embedding_dim,
            extra_trainable_ratio=0.6
        )

        # Scheduler con warmup
        steps_per_epoch = len(train_ds)
        total_steps = epochs * steps_per_epoch
        warmup_steps = int(0.1 * total_steps)
        lr_schedule = CosineDecayWithWarmup(
            initial_lr=5e-4,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            alpha=0.01
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        # Warm-up inicial
        dummy_x = tf.zeros((1, 167, 250, 3))
        dummy_y = tf.zeros((1,), dtype=tf.int32)
        with tf.GradientTape() as tape:
            preds = model(dummy_x, training=True)
            loss = cosine_embedding_loss(dummy_y, preds, label_embeddings)
        grads = tape.gradient(loss, model.trainable_variables + [label_embeddings])
        optimizer.apply_gradients(zip(grads, model.trainable_variables + [label_embeddings]))

        best_val_acc = 0
        epochs_no_improve = 0

        print(f"Dataset de entrenamiento: {steps_per_epoch} batches")
        print(f"Dataset de validaciÃ³n: {len(val_ds)} batches")

        for epoch in range(epochs):
            # Obtener el LR actual
            current_lr = tf.keras.backend.get_value(optimizer.learning_rate)
            print(f"Epoch {epoch+1}/{epochs}  LR={current_lr:.2e}")

            # Entrenamiento
            train_losses, train_accs = [], []
            for x_batch, y_batch in tqdm(train_ds, desc="Training"):
                loss, acc = train_step(x_batch, y_batch, model, label_embeddings, optimizer)
                train_losses.append(loss.numpy())
                train_accs.append(acc.numpy())
            avg_train_loss = np.mean(train_losses)
            avg_train_acc = np.mean(train_accs)

            # ValidaciÃ³n
            val_losses, val_accs = [], []
            for x_val, y_val in tqdm(val_ds, desc="Validation"):
                v_loss, v_acc = val_step(x_val, y_val, model, label_embeddings)
                val_losses.append(v_loss.numpy())
                val_accs.append(v_acc.numpy())
            avg_val_loss = np.mean(val_losses)
            avg_val_acc = np.mean(val_accs)

            print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f}")

            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                epochs_no_improve = 0

                save_model_path = (
                    f"/kaggle/working/modelos/"
                    f"modelo_categoria_cluster_{cluster_id}_familia_{fam_id}.keras"
                )
                model.save(save_model_path)
                np.save(
                    f"/kaggle/working/modelos/"
                    f"embeddings_categoria_cluster_{cluster_id}_familia_{fam_id}.npy",
                    label_embeddings.numpy()
                )
                with open(
                    f"/kaggle/working/modelos/"
                    f"inv_label_map_categoria_cluster_{cluster_id}_familia_{fam_id}.pkl", "wb"
                ) as f:
                    pickle.dump(inv_label_map, f)
                print(f"Modelo guardado en {save_model_path}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print("Early stopping")
                    break

        # Liberar recursos
        del model, train_ds, val_ds, label_embeddings
        tf.keras.backend.clear_session()
        gc.collect()

        inv_label_cluster_genus[cluster_id] = inv_label_map


def predict_taxonomy_dual(image_path, feature_extractor, scaler, kmeans,
                          models_familia, models_genero, models_especie,
                          le_fam, le_gen, le_esp):
    
    # --- 1. Cargamos y preprocesamos cada imagen ---
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize_with_pad(img, target_height=333, target_width=500)
    img = tf.keras.applications.convnext.preprocess_input(img)
    img = tf.expand_dims(img, 0)

    # --- 2. extraemos el vector de caracterÃ­sticas y lo escalamos ---
    features = feature_extractor.predict(img, verbose=0)
    features_scaled = scaler.transform(features)

    # --- 3. Obtenemos los dos clusteres mÃ¡s probables para seguir dos predicciones ---
    distances = kmeans.transform(features_scaled)[0]  # Distancia a cada centro
    top_clusters = distances.argsort()[:2]  # Los dos mÃ¡s cercanos

    predictions = {}

    # --- 4. CLASIFICACIÃ“N JERÃ�RQUICA ---
    for i, cluster_id in enumerate(top_clusters):
        label = "primera" if i == 0 else "segunda"

        # --- FAMILIA ---
        if cluster_id not in models_familia:
            predictions[label] = {"familia": "UNKNOWN", "genero": "UNKNOWN", "especie": "UNKNOWN"}
            continue

        model_fam = models_familia[cluster_id]
        pred_fam_probs = model_fam.predict(img, verbose=0)[0]
        pred_fam_id = pred_fam_probs.argmax()
        pred_fam_conf = float(pred_fam_probs[pred_fam_id]) * 100
        pred_fam = le_fam.inverse_transform([pred_fam_id])[0]

        # --- GÃ‰NERO ---
        key_gen = (cluster_id, pred_fam_id)
        if key_gen in models_genero:
            model_gen = models_genero[key_gen]
            pred_gen_probs = model_gen.predict(img, verbose=0)[0]
            pred_gen_id = pred_gen_probs.argmax()
            pred_gen_conf = float(pred_gen_probs[pred_gen_id]) * 100
            pred_gen = le_gen.inverse_transform([pred_gen_id])[0]
        else:
            pred_gen, pred_gen_conf = "UNKNOWN", 0.0
            pred_gen_id = None

        # --- ESPECIE ---
        key_esp = (cluster_id, pred_fam_id, pred_gen_id)
        if key_esp in models_especie:
            model_esp = models_especie[key_esp]
            pred_esp_probs = model_esp.predict(img, verbose=0)[0]
            pred_esp_id = pred_esp_probs.argmax()
            pred_esp_conf = float(pred_esp_probs[pred_esp_id]) * 100
            pred_esp = le_esp.inverse_transform([pred_esp_id])[0]
        else:
            pred_esp, pred_esp_conf = "UNKNOWN", 0.0

        # Guardar predicciÃ³n
        predictions[label] = {
            "familia": f"{pred_fam} ({pred_fam_conf:.2f}%)",
            "genero": f"{pred_gen} ({pred_gen_conf:.2f}%)",
            "especie": f"{pred_esp} ({pred_esp_conf:.2f}%)"
        }

    return predictions


def predict_taxonomy_dual_batch(test_ds, feature_extractor, scaler, kmeans,
                                 models_familia, models_genero, models_especie,
                                 le_fam, le_gen, le_esp, batch_size=32):
    
    all_preds = []
    all_true = []
    
    for batch in tqdm(test_ds.batch(batch_size)):
        images, labels = batch
    
        # Redimensionar y preprocesar imÃ¡genes
        images_resized = tf.image.resize_with_pad(images, target_height=333, target_width=500)
        imgs_proc = tf.keras.applications.convnext.preprocess_input(images_resized)
    
        # Extraer caracterÃ­sticas
        features = feature_extractor.predict(imgs_proc, verbose=0)
        features_scaled = scaler.transform(features)

        # CALCULO DE CLUSTERS, LOS DOS MÃ�S CERCANOS
        
        distances = kmeans.transform(features_scaled)  # shape: (batch, n_clusters)
        top_clusters = np.argsort(distances, axis=1)[:, :2]  # shape: (batch, 2)

        for i in range(len(images)):

            # SE INICIALIZA UN DICCIONARIO PARA ALMACENAR LAS PREDICCIONES
            # Y SE ALMACENAN LOS CLUSTERS
            
            pred_dict = {}
            image_features = features[i].reshape(1, -1)
            cluster_pair = top_clusters[i]

            for j, cluster_id in enumerate(cluster_pair):
                label = "primera" if j == 0 else "segunda"

                if cluster_id not in models_familia:
                    pred_dict[label] = {"familia": "UNKNOWN", "genero": "UNKNOWN", "especie": "UNKNOWN"}
                    continue

                # FAMILIA
                model_fam = models_familia[cluster_id]
                pred_fam_probs = model_fam.predict(images[i:i+1], verbose=0)[0]
                pred_fam_id = pred_fam_probs.argmax()
                pred_fam = le_fam.inverse_transform([pred_fam_id])[0]

                # GÃ‰NERO
                key_gen = (cluster_id, pred_fam_id)
                if key_gen in models_genero:
                    model_gen = models_genero[key_gen]
                    pred_gen_probs = model_gen.predict(images[i:i+1], verbose=0)[0]
                    pred_gen_id = pred_gen_probs.argmax()
                    pred_gen = le_gen.inverse_transform([pred_gen_id])[0]
                else:
                    pred_gen, pred_gen_id = "UNKNOWN", None

                # ESPECIE
                key_esp = (cluster_id, pred_fam_id, pred_gen_id)
                if key_esp in models_especie:
                    model_esp = models_especie[key_esp]
                    pred_esp_probs = model_esp.predict(images[i:i+1], verbose=0)[0]
                    pred_esp_id = pred_esp_probs.argmax()
                    pred_esp = le_esp.inverse_transform([pred_esp_id])[0]
                else:
                    pred_esp = "UNKNOWN"

                pred_dict[label] = {
                    "familia": pred_fam,
                    "genero": pred_gen,
                    "especie": pred_esp
                }

            # Ground truth para esta imagen
            true_labels = {
                "familia": le_fam.inverse_transform([labels["familia"][i].numpy()])[0],
                "genero": le_gen.inverse_transform([labels["genero"][i].numpy()])[0],
                "especie": le_esp.inverse_transform([labels["especie"][i].numpy()])[0],
            }

            all_preds.append(pred_dict)
            all_true.append(true_labels)

    return all_preds, all_true


# EJEMPLO DE EJECUCIÃ“N

all_preds, all_true = predict_taxonomy_dual_batch(
    test_ds=test_ds,
    feature_extractor=feature_extractor,
    scaler=scaler,
    kmeans=kmeans,
    models_familia=models_familia,
    models_genero=models_genero,
    models_especie=models_especie,
    le_fam=le_fam,
    le_gen=le_gen,
    le_esp=le_esp,
    batch_size=32  # puedes ajustar el tamaÃ±o del batch segÃºn tu memoria disponible
)


def evaluate_predictions(all_preds, all_true, le_fam, le_gen, le_esp):
    results = {}

    for nivel in ["familia", "genero", "especie"]:
        y_true = []
        y_pred_top1 = []
        y_pred_top2 = []

        for pred, true in zip(all_preds, all_true):
            true_label = true[nivel]
            pred1 = pred["primera"][nivel]
            pred2 = pred["segunda"][nivel]

            y_true.append(true_label)
            y_pred_top1.append(pred1)

            # En top-2, si el verdadero label aparece en cualquiera, usamos el correcto. Si no, marcamos el top-1 (para F1 y accuracy)
            if true_label == pred1 or true_label == pred2:
                y_pred_top2.append(true_label)  # cuenta como acierto
            else:
                y_pred_top2.append(pred1)  # sigue siendo una predicciÃ³n errÃ³nea

        # Convertimos a ids para sklearn
        le = {"familia": le_fam, "genero": le_gen, "especie": le_esp}[nivel]
        y_true_ids = le.transform(y_true)
        y_pred_top1_ids = le.transform(y_pred_top1)
        y_pred_top2_ids = le.transform(y_pred_top2)

        results[nivel] = {
            "accuracy_top1": accuracy_score(y_true_ids, y_pred_top1_ids),
            "f1_macro_top1": f1_score(y_true_ids, y_pred_top1_ids, average='macro'),
            "accuracy_top2": accuracy_score(y_true_ids, y_pred_top2_ids),
            "f1_macro_top2": f1_score(y_true_ids, y_pred_top2_ids, average='macro'),
        }

    return results


results = evaluate_predictions(all_preds, all_true, le_fam, le_gen, le_esp)

for nivel, mÃ©tricas in results.items():
    print(f"\nJerarquÃ­a: {nivel.upper()}")
    for nombre_metrica, valor in mÃ©tricas.items():
        print(f"{nombre_metrica}: {valor:.4f}")

