# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

!pip install exifread
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, metrics, losses, optimizers
from sklearn.model_selection import train_test_split
import exifread
import cv2
import os


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

TRAINING_CSV_PATH = '/kaggle/input/detect-ai-vs-human-generated-images/train.csv'
main_dir = '/kaggle/input/ai-vs-human-generated-dataset'

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    print(dirname)
    #for filename in filenames:
        #print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df = pd.read_csv(TRAINING_CSV_PATH, index_col='Unnamed: 0')
train_df['pair_id'] = train_df.index // 2 

print(train_df.head())


# Verificar uma amostra de caminhos
sample_path = main_dir + '/' + train_df['file_name'].iloc[0]
print(f"Exemplo de caminho: {sample_path}")
print(f"Arquivo existe? {os.path.exists(sample_path)}")

# Listar alguns arquivos no diretório (para debug)
print("\nPrimeiros arquivos em train_data/:")
print(os.listdir(main_dir + '/train_data/')[:5])


def plot_image_pair(df, pair_id):

    pair_df = df[df['pair_id'] == pair_id]
    
    if len(pair_df) != 2:
        print(f"Pair ID {pair_id} não encontrado ou incompleto")
        return
    
    # IA (label 1), Humana (label 0)
    ai_image_row = pair_df[pair_df['label'] == 1].iloc[0]
    human_image_row = pair_df[pair_df['label'] == 0].iloc[0]

    ai_img_path = main_dir + '/' + ai_image_row['file_name']
    human_img_path = main_dir + '/' + human_image_row['file_name']
    
    # Carregar imagens diretamente dos caminhos completos
    ai_img = plt.imread(ai_img_path)
    human_img = plt.imread(human_img_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'Comparação do Pair ID: {pair_id}', fontsize=14, y=1.05)
    
    axes[0].imshow(ai_img)
    axes[0].set_title(f'IA Gerada (Label 1)\n{ai_image_row["file_name"]}', fontsize=10)
    axes[0].axis('off')
    
    axes[1].imshow(human_img)
    axes[1].set_title(f'Humana (Label 0)\n{human_image_row["file_name"]}', fontsize=10)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()


# Plotar um par genérico
plot_image_pair(train_df, pair_id=42)


from collections import defaultdict

# Contar frequência de tags EXIF em uma amostra de imagens
exif_counts = defaultdict(int)
sample_size = 50  

for _, row in train_df.sample(sample_size).iterrows():
    try:
        with open(os.path.join(main_dir, row['file_name']), 'rb') as f:
            print (os.path.join(main_dir, row['file_name']))
            tags = exifread.process_file(f, details=False)
            for tag in tags:
                exif_counts[tag] += 1
    except:
        pass

print("Tags EXIF mais frequentes:")
for tag, count in sorted(exif_counts.items(), key=lambda x: -x[1]):
    print(f"{tag}: {count}/{sample_size}")


from PIL import Image

for _, row in train_df.sample(10).iterrows():
    try:
        img = Image.open(os.path.join(main_dir, row['file_name']))
        print(f"{row['file_name']}: OK ({img.size}, {img.mode})")
    except Exception as e:
        print(f"{row['file_name']}: ERRO - {str(e)}")


for pair_id in [0, 1, 5, 10, 100]:
    plot_image_pair(train_df, pair_id)


# Comparar métricas entre classes para uma amostra
sample_df = train_df.groupby('label').sample(10)

for _, row in sample_df.iterrows():
    img = cv2.imread(os.path.join(main_dir, row['file_name']))
    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
    print(f"Label {row['label']}: Variância do Laplaciano = {laplacian_var:.2f}")


for i in range(10):
    shape = plt.imread(f'{main_dir}/{train_df.file_name[i]}').shape
    print(f"Shape of the Image: {shape}")


from skimage.feature import graycomatrix, graycoprops

def calculate_glcm_features(image_gray):
    glcm = graycomatrix(image_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    return {'glcm_contrast': contrast, 'glcm_dissimilarity': dissimilarity}


def calculate_color_stats(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    saturation = hsv[:,:,1].flatten()
    saturation_entropy = -np.sum(np.histogram(saturation, bins=20)[0] * np.log2(np.histogram(saturation, bins=20)[0] + 1e-6))
    
    red_channel = image[:,:,0].flatten()
    skewness_red = pd.Series(red_channel).skew()
    
    return {'saturation_entropy': saturation_entropy, 'skewness_red': skewness_red}


def calculate_edge_features(image_gray):
    laplacian_var = cv2.Laplacian(image_gray, cv2.CV_64F).var()
    edges = cv2.Canny(image_gray, 100, 200)
    edge_density = np.mean(edges)  # proporção de pixels de borda
    
    return {'laplacian_var': laplacian_var, 'edge_density': edge_density}


def calculate_fft_features(image_gray):
    fft = np.fft.fftshift(np.fft.fft2(image_gray))
    magnitude = np.log(np.abs(fft) + 1e-6)
    
    # foco em altas frequências (bordas do espectro)
    h, w = magnitude.shape
    crop_size = 50
    high_freq = magnitude[h//2 - crop_size:h//2 + crop_size, w//2 - crop_size:w//2 + crop_size]
    return {'fft_high_freq_mean': np.mean(high_freq)}


def extract_all_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    features = {}
    features.update(calculate_glcm_features(gray))
    features.update(calculate_color_stats(image))
    features.update(calculate_edge_features(gray))
    features.update(calculate_fft_features(gray))
    
    return features


from concurrent.futures import ThreadPoolExecutor

def batch_feature_extraction(paths, workers=4):
    with ThreadPoolExecutor(max_workers=workers) as executor:
        features = list(executor.map(extract_all_features, paths))
    return pd.DataFrame(features)


import seaborn as sns

path_list = train_df['file_name'].tolist()

full_paths = [
    os.path.join(main_dir, filename)
    for filename in path_list
]
path_list = train_df['file_name'].tolist()

full_paths = [
    os.path.join(main_dir, filename)
    for filename in path_list
]

print("Exemplos de caminhos completos:")
for path in full_paths[:3]:
    print(f" - {path} → Existe? {os.path.exists(path)}")


feature_df = batch_feature_extraction(full_paths)  # Amostra inicial

for feature in ['glcm_contrast', 'saturation_entropy', 'fft_high_freq_mean']:
    plt.figure(figsize=(10, 4))
    sns.boxplot(x=train_df['label'].iloc[:1000], y=feature_df[feature])
    plt.title(f'Distribuição de {feature} por Classe')
    plt.show()


feature_df = feature_df.drop(columns=['saturation_entropy'])
feature_df.head()


# plotar antes/depois das transformações
def plot_distribution(feature, title):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(feature, kde=True)
    plt.title(f'Original: {title}')
    plt.subplot(1, 2, 2)
    sns.histplot(np.log1p(feature), kde=True)
    plt.title(f'Log Transform: {title}')
    plt.show()

plot_distribution(feature_df['glcm_contrast'], 'GLCM Contrast')
plot_distribution(feature_df['fft_high_freq_mean'], 'FFT High Freq Mean')


from scipy.stats import skew

for col in feature_df.columns:
    s = skew(feature_df[col])
    print(f"{col}: Skewness = {s:.2f}")


feature_df['fft_glcm_ratio'] = feature_df['fft_high_freq_mean'] / (feature_df['glcm_contrast'] + 1e-6)
feature_df['log_glcm_contrast'] = np.log1p(feature_df['glcm_contrast'])


plt.figure(figsize=(10, 4))
sns.boxplot(x=train_df['label'].iloc[:1000], y=feature_df['log_glcm_contrast'])
plt.title(f'Distribuição de log_glcm_contrast por Classe')
plt.show()


full_train_df = pd.concat([train_df, feature_df], axis=1)
full_train_df.head()


IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
INIT_LR = 1e-4
FOLDS = 5


from tensorflow.keras import applications
from sklearn.model_selection import GroupKFold

def build_hybrid_model(num_features):

    # Branch de Imagens
    image_input = layers.Input(shape=(*IMG_SIZE, 3), name='image_input')
    
    # Data Augmentation integrado
    x = layers.RandomFlip("horizontal")(image_input)
    x = layers.RandomRotation(0.1)(x)
    
    # Base pré-treinada
    base_model = applications.EfficientNetV2B0(
        include_top=False,
        weights='imagenet',
        input_tensor=x
    )
    base_model.trainable = False 
    
    # pooling
    image_features = layers.GlobalAveragePooling2D(name='image_pool')(base_model.output)

    # Branch de Features Manuais
    feature_input = layers.Input(shape=(num_features,), name='feature_input')
    f = layers.Dense(64, activation='relu')(feature_input)
    f = layers.BatchNormalization()(f)
    
    # Combina as representações das imagens e features manuais em um único vetor
    combined = layers.concatenate([image_features, f])
    x = layers.Dense(128, activation='relu')(combined)
    x = layers.Dropout(0.5)(x)
    
    output = layers.Dense(1, activation='sigmoid')(x)
    
    return models.Model(
        inputs=[image_input, feature_input],
        outputs=output,
        name='Hybrid_AI_Detector'
    )


full_train_df["full_path"] = full_train_df["file_name"].apply(
    lambda x: os.path.join(main_dir, x)
)

# Verificar caminhos
print(full_train_df[["file_name", "full_path"]].head())
print("\nExemplo de caminho válido?", os.path.exists(full_train_df["full_path"].iloc[0]))


# Garantir que pares não sejam divididos entre folds
gkf = GroupKFold(n_splits=FOLDS)

# Features manuais + caminhos das imagens
features = feature_df.values  # DataFrame com suas features processadas
image_paths = full_train_df['full_path'].values
labels = full_train_df['label'].values
groups = full_train_df['pair_id'].values  # Para GroupKFold


def create_hybrid_dataset(df, feature_columns, batch_size=32):
    # dados brutos
    image_paths = df["full_path"].values
    manual_features = df[feature_columns].values.astype(np.float32)
    labels = df["label"].values.astype(np.float32)
    
    # carregar e pré-processar uma imagem
    def load_and_preprocess(image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        return img
    
    # imagens
    image_ds = tf.data.Dataset.from_tensor_slices(image_paths).map(load_and_preprocess)
    
    # features manuais
    feature_ds = tf.data.Dataset.from_tensor_slices(manual_features)
    
    # labels
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    
    # combinar todos os componentes
    combined_ds = tf.data.Dataset.zip(
        ((image_ds, feature_ds), label_ds)
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return combined_ds


# Definir colunas de features manuais
FEATURE_COLS = ["glcm_contrast", "glcm_dissimilarity", "laplacian_var", "edge_density", "fft_high_freq_mean"]

# Configurar KFold estratificado por pair_id
gkf = GroupKFold(n_splits=5)
fold = 0

for train_idx, val_idx in gkf.split(
    full_train_df, 
    groups=full_train_df["pair_id"]
):
    fold += 1
    print(f"\n=== Fold {fold}/5 ===")
    
    # Dividir dados
    train_df = full_train_df.iloc[train_idx]
    val_df = full_train_df.iloc[val_idx]
    
    # Criar datasets
    train_ds = create_hybrid_dataset(train_df, FEATURE_COLS)
    val_ds = create_hybrid_dataset(val_df, FEATURE_COLS)
    
    # Construir e treinar modelo (mesma arquitetura)
    model = build_hybrid_model(num_features=len(FEATURE_COLS))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
        ]
    )


from sklearn.metrics import roc_auc_score, precision_score, recall_score

# o modelo precisa estar carregado na memória
test_ds = create_hybrid_dataset(full_train_df, FEATURE_COLS)

# gerar predições
y_probs = model.predict(test_ds).flatten()
y_pred = (y_probs > 0.5).astype(int) # valor não-optimal
y_true = full_train_df['label'].values

# métricas
print(f"AUC: {roc_auc_score(y_true, y_probs):.4f}")
print(f"Precision: {precision_score(y_true, y_pred):.4f}")
print(f"Recall: {recall_score(y_true, y_pred):.4f}")


from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

print(f"""
=== Métricas do Modelo ===
- AUC: {roc_auc_score(y_true, y_probs):.4f}
- Precision: {precision_score(y_true, y_pred):.4f}
- Recall: {recall_score(y_true, y_pred):.4f}
- F1-Score: {f1_score(y_true, y_pred):.4f}
""")


cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Humana', 'IA'], yticklabels=['Humana', 'IA'])
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()


from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Curva ROC
RocCurveDisplay.from_predictions(y_true, y_probs, ax=ax1)
ax1.set_title("Curva ROC")

# Curva Precision-Recall
PrecisionRecallDisplay.from_predictions(y_true, y_probs, ax=ax2)
ax2.set_title("Curva Precision-Recall")

plt.show()


# DF com metadados e predições
error_df = full_train_df.copy()
error_df['pred_prob'] = y_probs
error_df['pred_label'] = y_pred

# Falsos negativos (IAs classificadas como humanas)
false_negatives = error_df[(error_df['label'] == 1) & (error_df['pred_label'] == 0)]
print("Falsos Negativos (Top 5):")
print(false_negatives[['file_name', 'pred_prob']].head())

# Falsos positivos (Humanas classificadas como IA)
false_positives = error_df[(error_df['label'] == 0) & (error_df['pred_label'] == 1)]
print("\nFalsos Positivos (Top 5):")
print(false_positives[['file_name', 'pred_prob']].head())


from sklearn.metrics import precision_recall_curve

# threshold ótimo para F1-Score
precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
best_threshold = thresholds[np.argmax(f1_scores)]

print(f"Threshold Ideal: {best_threshold:.4f}")
print(f"F1-Score no Threshold Ideal: {np.max(f1_scores):.4f}")

# aplicar novo threshold
y_pred_optimized = (y_probs > best_threshold).astype(int)
print("\n=== Métricas com Threshold Otimizado ===")
print(f"Precision: {precision_score(y_true, y_pred_optimized):.4f}")
print(f"Recall: {recall_score(y_true, y_pred_optimized):.4f}")


model.save('/kaggle/working/final_model.keras')  
print("Arquivos após salvamento manual:", os.listdir('/kaggle/working/'))


# Definir colunas de features manuais (novamente para não ter que treinar o modelo de novo)
FEATURE_COLS = ["glcm_contrast", "glcm_dissimilarity", "laplacian_var", "edge_density", "fft_high_freq_mean"]

# carregar o modelo
model = build_hybrid_model(num_features=len(FEATURE_COLS))
model.load_weights('/kaggle/working/final_model.keras')


TEST_CSV_PATH = '/kaggle/input/ai-vs-human-generated-dataset/test.csv'
TEST_DATA_PATH = '/kaggle/input/ai-vs-human-generated-dataset'

test_2_predict = pd.read_csv(TEST_CSV_PATH)
test_2_predict.head


test_2_predict['full_path'] = test_2_predict['id'].apply(
    lambda x: os.path.join(TEST_DATA_PATH, x)  
)



test_path_list = test_2_predict['full_path'].tolist()

feature_df = batch_feature_extraction(test_path_list) 


#Normalizações
feature_df['fft_glcm_ratio'] = feature_df['fft_high_freq_mean'] / (feature_df['glcm_contrast'] + 1e-6)
feature_df['log_glcm_contrast'] = np.log1p(feature_df['glcm_contrast'])
feature_df.drop(columns=['saturation_entropy'], inplace=True)
feature_df.head()


assert len(test_2_predict) == len(feature_df), "DataFrames têm tamanhos diferentes!"
test_2_predict.index
feature_df.head()


full_pred_df = test_2_predict.join(feature_df, how="inner")
full_pred_df.head()


def predict_with_hybrid_model(model, df, feature_columns, batch_size=32):
    """
    Realiza predição com o modelo híbrido passando os inputs como uma lista
    na ordem correta definida no modelo.
    """
    # Listas com dados brutos
    image_paths = df["full_path"].values
    manual_features = df[feature_columns].values.astype(np.float32)
    
    # Função para carregar e pré-processar uma imagem
    def load_and_preprocess(image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        return img.numpy()
    
    # Carregar todas as imagens
    print("Carregando imagens...")
    images = []
    for path in image_paths:
        img = load_and_preprocess(path)
        images.append(img)
    
    # Converter para arrays numpy
    images_array = np.stack(images)
    features_array = manual_features
    
    print(f"Images array shape: {images_array.shape}")
    print(f"Features array shape: {features_array.shape}")
    
    # Realizar predição passando uma LISTA com os inputs na mesma ordem que o modelo foi definido
    print("Realizando predição...")
    predictions = model.predict(
        [images_array, features_array],  # Lista em vez de dicionário
        batch_size=batch_size
    )
    
    return predictions


result_from_function = predict_with_hybrid_model(model, full_pred_df, FEATURE_COLS, batch_size=32)


result_from_function


# plotando a distribuição das probabilidades obtidas pelo modelo

import seaborn as sns

data = pd.Series(result_from_function.flatten())

plt.figure(figsize=(10, 6))

# histograma e KDE 
sns.histplot(data, bins=50, kde=True, color='royalblue', edgecolor='black', alpha=0.7)

plt.title('Distribuição das Predições do Modelo', fontsize=14, fontweight='bold')
plt.xlabel('Probabilidade', fontsize=12)
plt.ylabel('Número de Casos', fontsize=12)
plt.xlim(0, 1)  # range de 0 a 1
plt.show()


binary_predictions = (result_from_function > 0.3).astype(int) # treshold ideal era 0.6415, mas o 0.5 pontuou melhor
binary_predictions


np.sum(binary_predictions == 1)


def convert_to_submission(df, column_name, predictions):

    # Converte uma coluna de um DataFrame e um array de previsões em um df no formato de submissão.

    submission_df = pd.DataFrame({
        "id": df[column_name].values,  # Pegando os nomes dos arquivos
        "label": predictions.flatten().astype(int)  # Convertendo previsões para inteiros
    })
    
    return submission_df



submission = convert_to_submission(full_pred_df, "id", binary_predictions)
print(submission)



submission.to_csv('submission.csv', index=False)


!ls -la


from IPython.display import HTML

def create_download_link(title = "Download CSV file", filename = "data.csv"):  
    html = '<a href={filename}>{title}</a>'
    html = html.format(title=title,filename=filename)
    return HTML(html)

# create a link to download the dataframe which was saved with .to_csv method
create_download_link(filename='submission.csv')

# from https://www.kaggle.com/code/arkaung/download-csv-file

