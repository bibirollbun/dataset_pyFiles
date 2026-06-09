input_cache_path = "/kaggle/input/petfinder-processed-features"


#pip install numpy==1.25.0



#pip install lightgbm


#!pip install prince


#!pip install transformers torch




import json
import os
import numpy as np
import pandas as pd
import torch as torch 
import optuna
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import matplotlib.pyplot as plt
import seaborn as sns
#import prince
from transformers import pipeline

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score, StratifiedKFold
from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor
import lightgbm as lgb
import glob
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score
import scipy.optimize as spopt
from functools import partial

import joblib


train_df = pd.read_csv("/kaggle/input/petfinder-adoption-prediction/train/train.csv")
breed = pd.read_csv("/kaggle/input/petfinder-adoption-prediction/BreedLabels.csv")
color = pd.read_csv("/kaggle/input/petfinder-adoption-prediction/ColorLabels.csv")
state = pd.read_csv("/kaggle/input/petfinder-adoption-prediction/StateLabels.csv")
test_df = pd.read_csv("/kaggle/input/petfinder-adoption-prediction/test/test.csv")
print(train_df.info())
print(test_df.info())


tabla_completa = pd.DataFrame({
    'Nulos (abs)': train_df.isna().sum(),
    'Nulos (%)': (train_df.isna().mean() * 100).round(2),
    'Ceros (abs)': (train_df == 0).sum(),
    'Ceros (%)': ((train_df == 0).mean() * 100).round(2)
})

tabla_completa.sort_values('Nulos (%)', ascending=False)



# Filtrar registros con Age = 0
df_age0 = train_df[train_df["Age"] == 0]

print("Cantidad de registros con Age = 0:", df_age0.shape[0])

# Palabras clave que indican que son recién nacidos o de pocas semanas
keywords = [
    "week", "weeks", "newborn", "puppy", "pup", "kittens", "kitten",
    "days", "day old", "few days", "baby", "babies"
]

# Filtrar registros donde la descripción sugiere edad menor a 1 mes
mask = df_age0["Description"].str.lower().str.contains("|".join(keywords), na=False)
df_newborn = df_age0[mask]

print("\nRegistros de Age=0 cuya descripción indica que son recién nacidos o menores a 1 mes:")
df_newborn[["Name", "Age", "Quantity", "Description"]].head(20)



def preprocess_dataframe(df, breed, color, state, is_train=True):
    """
    Pipeline de preprocesamiento unificado para train y test.
    Esto asegura que ambos datasets tengan el mismo tratamiento.
    """
    df = df.copy()
    
    # ---- 4.1 Variables derivadas ----
    def procesar_nombre(x):
        if pd.isna(x):
            return 0
        x = str(x).strip().lower()
        if x == "" or "name" in x:
            return 0
        palabras = x.split()
        if len(palabras) > 3:
            return 0
        return 1
    
    df["has_name"] = df["Name"].apply(procesar_nombre)
    df["has_description"] = df["Description"].fillna("").str.strip().ne("").astype(int)
    
    # ---- 4.2 Renombrar columnas de las tablas auxiliares ----
    breed = breed.rename(columns={
        "BreedID": "BreedID_label",
        "BreedName": "BreedName_label"
    })
    
    color = color.rename(columns={
        "ColorID": "ColorID_label",
        "ColorName": "ColorName_label"
    })
    
    state = state.rename(columns={
        "StateID": "StateID_label",
        "StateName": "StateName_label"
    })
    
    # ---- 4.3 Merge con descripciones ----
    # Breed1
    df = df.merge(
        breed[["BreedID_label", "BreedName_label"]],
        how="left",
        left_on="Breed1",
        right_on="BreedID_label"
    ).rename(columns={"BreedName_label": "Breed1_desc"}).drop(columns=["BreedID_label", "Breed1"])
    
    # Breed2
    df = df.merge(
        breed[["BreedID_label", "BreedName_label"]],
        how="left",
        left_on="Breed2",
        right_on="BreedID_label"
    ).rename(columns={"BreedName_label": "Breed2_desc"}).drop(columns=["BreedID_label", "Breed2"])
    
    # Color1
    df = df.merge(
        color[["ColorID_label", "ColorName_label"]],
        how="left",
        left_on="Color1",
        right_on="ColorID_label"
    ).rename(columns={"ColorName_label": "Color1_desc"}).drop(columns=["ColorID_label", "Color1"])
    
    # Color2
    df = df.merge(
        color[["ColorID_label", "ColorName_label"]],
        how="left",
        left_on="Color2",
        right_on="ColorID_label"
    ).rename(columns={"ColorName_label": "Color2_desc"}).drop(columns=["ColorID_label", "Color2"])
    
    # Color3
    df = df.merge(
        color[["ColorID_label", "ColorName_label"]],
        how="left",
        left_on="Color3",
        right_on="ColorID_label"
    ).rename(columns={"ColorName_label": "Color3_desc"}).drop(columns=["ColorID_label", "Color3"])
    
    # State
    df = df.merge(
        state[["StateID_label", "StateName_label"]],
        how="left",
        left_on="State",
        right_on="StateID_label"
    ).rename(columns={"StateName_label": "State_desc"}).drop(columns=["StateID_label", "State"])
   
    # ---- 4.4 Mapeo variables categóricas a texto ----
    df["Type_txt"] = df["Type"].map({1: "Dog", 2: "Cat"})
    df["Gender_txt"] = df["Gender"].map({1: "Male", 2: "Female", 3: "Mixed"})
    df["MaturitySize_txt"] = df["MaturitySize"].map({
        0: "Not Specified", 1: "Small", 2: "Medium", 3: "Large", 4: "Extra Large"
    })
    df["FurLength_txt"] = df["FurLength"].map({
        0: "Not Specified", 1: "Short", 2: "Medium", 3: "Long"
    })
    df["Vaccinated_txt"] = df["Vaccinated"].map({1: "Yes", 2: "No", 3: "Not Sure"})
    df["Dewormed_txt"] = df["Dewormed"].map({1: "Yes", 2: "No", 3: "Not Sure"})
    df["Sterilized_txt"] = df["Sterilized"].map({1: "Yes", 2: "No", 3: "Not Sure"})
    df["Health_txt"] = df["Health"].map({
        0: "Not Specified", 1: "Healthy", 2: "Minor Injury", 3: "Serious Injury"
    })
    
    # ---- 4.5 Eliminación de columnas dummies originales ----
    df = df.drop(columns=[
        "Type", "Gender", "MaturitySize", "FurLength",
        "Vaccinated", "Dewormed", "Sterilized", "Health"
    ], errors='ignore')
    
    # Eliminamos Name (no RescuerID, lo usaremos después)
    df = df.drop(columns=["Name"], errors='ignore')
    
    # ---- 4.6 NUEVAS FEATURES DERIVADAS ----
    df['is_pure_breed'] = (df['Breed1_desc'].notna() & df['Breed2_desc'].isna()).astype(int)
    df['breed_count'] = df[['Breed1_desc', 'Breed2_desc']].notna().sum(axis=1)
    df['color_count'] = df[['Color1_desc', 'Color2_desc', 'Color3_desc']].notna().sum(axis=1)
    
    # Interacciones importantes
    df['age_photo_interaction'] = df['Age'] * df['PhotoAmt']
    df['fee_photo_interaction'] = df['Fee'] * df['PhotoAmt']
    df['photo_video_ratio'] = df['PhotoAmt'] / (df['VideoAmt'] + 1)
    df['total_media'] = df['PhotoAmt'] + df['VideoAmt']
    
    # Health score (suma de condiciones positivas)
    df['health_score'] = (df['Vaccinated_txt'] == 'Yes').astype(int) + \
                         (df['Dewormed_txt'] == 'Yes').astype(int) + \
                         (df['Sterilized_txt'] == 'Yes').astype(int)
    
    print(f"Procesamiento listo")
    return df



train_df = preprocess_dataframe(train_df, breed, color, state, is_train=True)
test_df = preprocess_dataframe(test_df, breed, color, state, is_train=False)


def extract_metadata_features(df, metadata_path):
    """
    Extrae features de los archivos JSON de metadata 
    """
    
    features_list = []
    
    for pet_id in tqdm(df['PetID'], desc="Procesando metadata"):
        json_file = os.path.join(metadata_path, f"{pet_id}-1.json")
        
        feat = {'PetID': pet_id}
        
        if os.path.exists(json_file):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
    
                if 'labelAnnotations' in data:
                    labels = data['labelAnnotations']
                    feat['meta_label_count'] = len(labels)
                    
                    if labels:
                        feat['meta_label_score_mean'] = np.mean([l['score'] for l in labels])
                        feat['meta_label_score_max'] = np.max([l['score'] for l in labels])
                        feat['meta_label_score_min'] = np.min([l['score'] for l in labels])
                
                if 'imagePropertiesAnnotation' in data:
                    colors = data['imagePropertiesAnnotation'].get('dominantColors', {}).get('colors', [])
                    feat['meta_color_count'] = len(colors)
                    
                    if colors:
                        feat['meta_dominant_color_score'] = colors[0]['score']
                        feat['meta_dominant_color_pixelfrac'] = colors[0]['pixelFraction']
                

                if 'cropHintsAnnotation' in data:
                    hints = data['cropHintsAnnotation'].get('cropHints', [])
                    if hints:
                        feat['meta_crop_confidence'] = hints[0]['confidence']
                
            except Exception as e:
                pass
        
        features_list.append(feat)
    
    # Creamos DataFrame con las features
    meta_df = pd.DataFrame(features_list)
    
    # Post creación, mergeamos con el DataFrame original
    df = df.merge(meta_df, on='PetID', how='left')
    
    # Llenar nulos con 0
    meta_cols = [c for c in meta_df.columns if c != 'PetID']
    df[meta_cols] = df[meta_cols].fillna(0)
    
    print(f"Features de metadata agregadas")
    return df


train_metadata_path = "/kaggle/input/petfinder-adoption-prediction/train_metadata"
test_metadata_path = "/kaggle/input/petfinder-adoption-prediction/test_metadata"



if os.path.exists(train_metadata_path):
    train_df = extract_metadata_features(train_df, train_metadata_path)
    test_df = extract_metadata_features(test_df, test_metadata_path)
else:
    print("No está el archivo")


def add_rescuer_features(train_df, test_df):
    """
    Versión CORREGIDA: Sin Data Leakage.
    Solo agregamos estadísticas de variables INDEPENDIENTES.
    """
    
    # Calcular estadísticas por rescuer SOLO de features input (NO AdoptionSpeed)
    rescuer_stats = train_df.groupby('RescuerID').agg({
        'PhotoAmt': ['mean', 'sum'],
        'Fee': ['mean'],
        'Age': ['mean'],
        'VideoAmt': ['mean', 'sum'],
        'Quantity': ['mean', 'sum']
    }).reset_index()
    
    # Aplanar nombres de columnas
    rescuer_stats.columns = ['RescuerID', 
                             'rescuer_photo_mean', 'rescuer_photo_sum',
                             'rescuer_fee_mean', 
                             'rescuer_age_mean', 
                             'rescuer_video_mean', 'rescuer_video_sum',
                             'rescuer_qty_mean', 'rescuer_qty_sum']
    
    # Categorizar rescuers por experiencia (cantidad de mascotas publicadas)
    # Esto es seguro porque es conteo de filas, no target
    counts = train_df['RescuerID'].value_counts().reset_index()
    counts.columns = ['RescuerID', 'rescuer_count']
    
    rescuer_stats = rescuer_stats.merge(counts, on='RescuerID', how='left')
    
    rescuer_stats['rescuer_experience'] = pd.cut(
        rescuer_stats['rescuer_count'], 
        bins=[-1, 5, 15, 50, 10000],
        labels=['novice', 'intermediate', 'experienced', 'expert']
    )
    
    # Merge con train y test
    train_df = train_df.merge(rescuer_stats, on='RescuerID', how='left')
    test_df = test_df.merge(rescuer_stats, on='RescuerID', how='left')
    
    # Llenar nulos (rescuers nuevos en test) con la mediana global
    numeric_cols = [c for c in rescuer_stats.columns if c not in ['RescuerID', 'rescuer_experience']]
    
    for col in numeric_cols:
        global_median = train_df[col].median()
        train_df[col] = train_df[col].fillna(global_median)
        test_df[col] = test_df[col].fillna(global_median)
    
    # Para categoricas
    train_df['rescuer_experience'] = train_df['rescuer_experience'].fillna('novice')
    test_df['rescuer_experience'] = test_df['rescuer_experience'].fillna('novice')
    
    print(f"✅ Features de rescuerID agregadas (SIN LEAK)")
    return train_df, test_df


train_df, test_df = add_rescuer_features(train_df, test_df)



all_text = pd.concat([train_df['Description'].fillna("none"), test_df['Description'].fillna("none")])


tfidf = TfidfVectorizer(
    min_df=3, max_features=10000, 
    strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
    ngram_range=(1, 2), use_idf=1, smooth_idf=1, sublinear_tf=1,
    stop_words='english'
)

tfidf.fit(all_text)


train_desc = tfidf.transform(train_df['Description'].fillna("none"))
test_desc = tfidf.transform(test_df['Description'].fillna("none"))


svd = TruncatedSVD(n_components=120, random_state=42)
train_svd = svd.fit_transform(train_desc)
test_svd = svd.transform(test_desc)


svd_cols = [f'svd_{i}' for i in range(120)]
train_svd_df = pd.DataFrame(train_svd, columns=svd_cols)
test_svd_df = pd.DataFrame(test_svd, columns=svd_cols)


train_df = pd.concat([train_df.reset_index(drop=True), train_svd_df], axis=1)
test_df = pd.concat([test_df.reset_index(drop=True), test_svd_df], axis=1)


input_cache_path = "/kaggle/input/petfinder-processed-features"



train_sent_file = os.path.join(input_cache_path, "train_sentiment_features.csv")
test_sent_file = os.path.join(input_cache_path, "test_sentiment_features.csv")
train_feat_file = os.path.join(input_cache_path, "train_img_features_deep.csv")
test_feat_file = os.path.join(input_cache_path, "test_img_features_deep.csv")

train_sent_features = pd.read_csv(train_sent_file)
test_sent_features = pd.read_csv(test_sent_file)
train_img_features = pd.read_csv(train_feat_file)
test_img_features = pd.read_csv(test_feat_file)

#train_df = train_df.merge(train_img_features, on='PetID', how='left')
train_df = train_df.merge(train_sent_features, on='PetID', how='left')

#test_df = test_df.merge(test_img_features, on='PetID', how='left')
test_df = test_df.merge(test_sent_features, on='PetID', how='left')


image_cols = [col for col in train_df.columns if col.startswith('img_deep_')]
sentiment_cols = [col for col in train_df.columns if col.startswith('emotion_')]


train_df[image_cols + sentiment_cols] = train_df[image_cols + sentiment_cols].fillna(0)
test_df[image_cols + sentiment_cols] = test_df[image_cols + sentiment_cols].fillna(0)


base_numeric_cols = ["Age", "Quantity", "Fee", "VideoAmt", "PhotoAmt"]


categorical_cols = [
    "Type_txt", "Gender_txt",
    "Color1_desc", "Color2_desc", "Color3_desc",
    "Breed1_desc", "Breed2_desc",
    "State_desc",
    "MaturitySize_txt", "FurLength_txt",
    "Vaccinated_txt", "Dewormed_txt",
    "Sterilized_txt", "Health_txt",
    "rescuer_experience" 
]


train_dummies = pd.get_dummies(
    train_df[categorical_cols],
    drop_first=False,
    prefix_sep='_',
    dummy_na=True
)

test_dummies = pd.get_dummies(
    test_df[categorical_cols],
    drop_first=False,
    prefix_sep='_',
    dummy_na=True
)


train_dummies, test_dummies = train_dummies.align(test_dummies, join='left', axis=1, fill_value=0)



train_df = pd.concat([train_df, train_dummies], axis=1)
test_df = pd.concat([test_df, test_dummies], axis=1)


numeric_cols = [
    "Age", "Quantity", "Fee", "VideoAmt", "PhotoAmt",
    "age_photo_interaction", "fee_photo_interaction", 
    "photo_video_ratio", "total_media", "health_score"
]


scaler = StandardScaler()



train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])


cols_to_drop = [
    'AdoptionSpeed', 'PetID', 'Description', 'RescuerID',
    'Type_txt', 'Gender_txt', 'Color1_desc', 'Color2_desc', 'Color3_desc',
    'Breed1_desc', 'Breed2_desc', 'State_desc', 'MaturitySize_txt',
    'FurLength_txt', 'Vaccinated_txt', 'Dewormed_txt', 'Sterilized_txt', 
    'Health_txt', 'rescuer_experience'
]


X = train_df.drop(columns=cols_to_drop, errors='ignore')
y = train_df['AdoptionSpeed']


X_test_kaggle = test_df.drop(columns=cols_to_drop, errors='ignore')
X_test_kaggle = X_test_kaggle.reindex(columns=X.columns, fill_value=0)


print("Total de features para entrenar:", X.shape[1])


def quadratic_weighted_kappa(y_true, y_pred):
    """
    Métrica oficial de la competencia
    """
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def lgb_qwk_metric(preds, train_data):
    """
    Wrapper para usar QWK como métrica en LightGBM
    """
    labels = train_data.get_label()
    
    # Para multiclass, preds viene como probabilidades
    # Necesitamos convertir a clases
    preds_reshaped = preds.reshape(5, -1).T
    preds_class = np.argmax(preds_reshaped, axis=1)
    
    score = quadratic_weighted_kappa(labels, preds_class)
    return 'qwk', score, True


#generamos un json para guardar los mejores hiperparametros asi no tenemos que volver a correr cada vez que ejecutamos
params_file = "best_hyperparameters.json"


with open('/kaggle/input/petfinder-processed-features/best_hyperparameters.json', 'r') as f:
   mejores_params = json.load(f)


print(mejores_params)




mejores_params = {
    'n_estimators': 989,      
    'learning_rate': 0.005973986924704114,    
    'num_leaves': 60,       
    'max_depth': 9,           
    'min_child_samples': 10,
    'subsample': 0.8563703251023174,
    'colsample_bytree': 0.626245306749440,
    'reg_alpha': 0.0622683174587296,
    'reg_lambda': 0.7208934757152172,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
    'objective': 'multiclass',
    'num_class': 5,
    'metric': 'multi_logloss'
}


class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4

        ll = cohen_kappa_score(y, X_p, weights='quadratic')
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = hue.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p.astype(int)



input_path = "/kaggle/input/petfinder-processed-features"

folds_preds = []


for fold in range(1, 6):
    # Construimos la ruta del archivo del modelo
    model_filename = f"lgbm_reg_fold_{fold}.txt"
    model_path = os.path.join(input_path, model_filename)
    
    if os.path.exists(model_path):
        print(f"\n⚡ Cargando {model_filename}...")
        
    
        model = lgb.Booster(model_file=model_path)
        
        # Predecir en Test
        pred = model.predict(X_test_kaggle)
        folds_preds.append(pred)
        
        print(f"Predicción del Fold {fold} completada.")
    else:
        print(f"ERROR: No se encontró el archivo {model_path}")

# --- promediamos predicciones ---
if len(folds_preds) > 0:
    avg_preds = np.mean(folds_preds, axis=0)
    print("\nPredicciones promediadas correctamente.")
else:
    print("No se generaron predicciones.")
    avg_preds = np.zeros(len(X_test_kaggle))

# --- se cargan los coeficientes---
coef_path = os.path.join(input_path, "coeficientes_optimos.npy")

if os.path.exists(coef_path):
    best_coefficients = np.load(coef_path)
    print(f"Coeficientes cargados desde archivo: {best_coefficients}")
else:
    # VALORES BACKUP (Úsalos solo si falló la carga del archivo)
    best_coefficients = [0.488, 2.134, 2.544, 2.982] 
    print(f"Archivo de coeficientes no encontrado. Usando valores manuales: {best_coefficients}")

# Instanciar el Rounder y aplicar los cortes
optR = OptimizedRounder()
final_predictions = optR.predict(avg_preds, best_coefficients)

# Guardar Submission
submission = pd.DataFrame({
    'PetID': test_df['PetID'],
    'AdoptionSpeed': final_predictions
})

submission.to_csv('submission.csv', index=False)
print(submission.head())

