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


import pandas as pd
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold


train_df = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
test_df = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")


train_df['Correctness']           = train_df['Category'].apply(lambda x: x.split('_')[0] == 'True')
train_df['Nature_of_Explanation'] = train_df['Category'].apply(lambda x: x.split('_')[1])
print(train_df.columns)


print(torch.cuda.is_available())
print(torch.cuda.device_count())


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

MODEL_NAME = '/kaggle/input/all-minilm-l6-v2-offline/all-MiniLM-L6-v2_local'
MAX_LENGTH = 96

transformer = SentenceTransformer(MODEL_NAME, device=device)

SEP_TOKEN = transformer.tokenizer.sep_token
CLS_TOKEN = transformer.tokenizer.cls_token


def creer_embeddings(texts, batch_size: int = 32) -> np.ndarray:
    embeddings = transformer.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True, 
        truncate=True,
        show_progress_bar=False
    )
    
    return embeddings


# ----------------------------------------------------------------------
# Concaténation des Données (création du contexte)
# ----------------------------------------------------------------------
full_context = train_df.apply(
    lambda row: (
        f"{CLS_TOKEN} {row['QuestionText']} {SEP_TOKEN} "
        f"{row['MC_Answer']} {SEP_TOKEN} "
        f"{row['StudentExplanation']} {SEP_TOKEN}"
    ), axis=1)
texts = full_context.tolist()

all_context_embeddings = creer_embeddings(texts, batch_size=64)


scaler_embeddings_global = StandardScaler()
all_context_embeddings_scaled = scaler_embeddings_global.fit_transform(all_context_embeddings)


N_COMPONENTS = 256 # Conserver 256 dimensions
pca = PCA(n_components=N_COMPONENTS, random_state=42)

all_context_embeddings_pca = pca.fit_transform(all_context_embeddings_scaled)
print(f" X_train_embeddings_pca généré. Forme : {all_context_embeddings_pca.shape}")
print(f" Variance Totale Conservée : {pca.explained_variance_ratio_.sum():.4f}")


nature_encoder = LabelEncoder()
category_encoder = LabelEncoder()


X_features = all_context_embeddings_pca
y_correctness = train_df['Correctness'].astype(int).values
y_nature = train_df['Nature_of_Explanation']
y_nature_enc = nature_encoder.fit_transform(y_nature)
y_meta_full = category_encoder.fit_transform(train_df['Category'].astype(str))


base_models = {
    "RF": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
    "MLP": MLPClassifier(
        hidden_layer_sizes=(100, 50),  
        max_iter=500,                  
        alpha=0.001,                   
        early_stopping=True,           
        random_state=42
    ),
    "LGBM": LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1)
}

model_nat = LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)


# ----- KFold OOF -----
n_samples = X_features.shape[0]  # embeddings PCA
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Initialiser arrays pour stocker les OOF
oof_RF = np.zeros((n_samples, 2))
oof_MLP = np.zeros((n_samples, 2))
oof_LGBM = np.zeros((n_samples, 2))
oof_Nature = np.zeros((n_samples, len(nature_encoder.classes_)))

for train_idx, val_idx in kf.split(X_features):
    X_tr, X_val = X_features[train_idx], X_features[val_idx]
    y_tr_corr, y_val_corr = y_correctness[train_idx], y_correctness[val_idx]
    y_tr_nat, y_val_nat = y_nature_enc[train_idx], y_nature_enc[val_idx]
    
    # ----- Entraîner modèles Correctness -----
    for name, model in base_models.items():
        model.fit(X_tr, y_tr_corr)
        oof_preds = model.predict_proba(X_val)
        if name == 'RF':
            oof_RF[val_idx] = oof_preds
        elif name == 'MLP':
            oof_MLP[val_idx] = oof_preds
        else:  # LGBM
            oof_LGBM[val_idx] = oof_preds
    
    # ----- Entraîner modèle Nature -----
    model_nat.fit(X_tr, y_tr_nat)
    oof_Nature[val_idx] = model_nat.predict_proba(X_val)

# ----- Concaténer OOF pour former X_meta_train -----
X_meta_train = np.hstack([oof_RF, oof_MLP, oof_LGBM, oof_Nature])
y_meta_train = y_meta_full  

print("Shape X_meta_train :", X_meta_train.shape)
print("Shape y_meta_train :", y_meta_train.shape)
print("Classes Category :", category_encoder.classes_)


# Split train/validation pour le meta-modele
X_meta_tr, X_meta_val, y_meta_tr, y_meta_val = train_test_split(
    X_meta_train, y_meta_train, test_size=0.2, random_state=42, stratify=y_meta_train
)

# Meta-modele
meta_model = LGBMClassifier(
    objective='multiclass',
    num_class=len(category_encoder.classes_),
    random_state=42,
    n_jobs=-1
)

meta_model.fit(X_meta_tr, y_meta_tr)

# Prediction sur validation
y_meta_val_pred = meta_model.predict(X_meta_val)
y_meta_val_proba = meta_model.predict_proba(X_meta_val)

# --- Accuracy ---
acc = accuracy_score(y_meta_val, y_meta_val_pred)
print(f"Accuracy du meta-modele : {acc:.4f}")

# --- Rapport détaillé (précision, rappel, F1) ---
print("\nRapport de classification :")
print(classification_report(y_meta_val, y_meta_val_pred, target_names=category_encoder.classes_))


# ----- Calcul des embeddings pour test_df -----
test_full_context = test_df.apply(
    lambda row: (
        f"{CLS_TOKEN} {row['QuestionText']} {SEP_TOKEN} "
        f"{row['MC_Answer']} {SEP_TOKEN} "
        f"{row['StudentExplanation']} {SEP_TOKEN}"
    ), axis=1)

# Transforme en embeddings avec le même encodeur
X_test_embeddings = creer_embeddings(test_full_context.tolist(), batch_size=64)

X_test_embeddings_scaled = scaler_embeddings_global.transform(X_test_embeddings)

X_test_embeddings_pca = pca.transform(X_test_embeddings_scaled)

# ----- Prédictions des modèles base (Correctness) -----
oof_RF_test = base_models["RF"].predict_proba(X_test_embeddings_pca)
oof_MLP_test = base_models["MLP"].predict_proba(X_test_embeddings_pca)
oof_LGBM_test = base_models["LGBM"].predict_proba(X_test_embeddings_pca)

# ----- Prédictions du modèle Nature -----
oof_Nature_test = model_nat.predict_proba(X_test_embeddings_pca)

# ----- Construction des features pour le méta-modèle -----
X_meta_test = np.hstack([
    oof_RF_test, 
    oof_MLP_test, 
    oof_LGBM_test, 
    oof_Nature_test
])

# ----- Prédictions finales du méta-modèle -----
y_meta_test_proba = meta_model.predict_proba(X_meta_test)

top3_idx = np.argsort(y_meta_test_proba, axis=1)[:, -3:][:, ::-1]

top3_labels = category_encoder.inverse_transform(top3_idx.ravel()).reshape(top3_idx.shape)

top3_joined = [",".join(labels) for labels in top3_labels]

submission_df = pd.DataFrame({
    "row_id": test_df["QuestionId"],
    "Category": top3_joined
})

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Fichier de soumission créé : {submission_path}")
print(submission_df.head(5))

